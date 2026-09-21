"""Run one command with the frozen summed-RSS cap and detailed PID memory."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
limit = 12 * 1024**3
start = time.perf_counter()
proc = subprocess.Popen(a.command, stdout=(a.output / "stdout.log").open("w"),
                        stderr=(a.output / "stderr.log").open("w"), start_new_session=True)
root = psutil.Process(proc.pid)
samples, aborted = [], False
while proc.poll() is None:
    try:
        members = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        members = []
    rows = []
    for member in members:
        try:
            basic = member.memory_info()
            full = member.memory_full_info()
            rows.append({"pid": member.pid, "ppid": member.ppid(), "name": member.name(),
                         "cmdline": member.cmdline(), "threads": member.num_threads(),
                         "rss": basic.rss, "vms": basic.vms,
                         "uss": getattr(full, "uss", None), "pss": getattr(full, "pss", None)})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    summed = sum(row["rss"] for row in rows)
    samples.append({"elapsed": time.perf_counter() - start, "summed_rss": summed,
                    "summed_uss": sum(row["uss"] or 0 for row in rows),
                    "summed_pss": sum(row["pss"] or 0 for row in rows), "processes": rows})
    if summed > limit:
        aborted = True
        os.killpg(proc.pid, signal.SIGKILL)
        break
    time.sleep(0.02)
code = proc.wait()
(a.output / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")
(a.output / "outcome.json").write_text(json.dumps({"exit_code": code, "aborted": aborted,
    "limit": limit, "peak_summed_rss": max(x["summed_rss"] for x in samples),
    "peak_summed_uss": max(x["summed_uss"] for x in samples),
    "peak_summed_pss": max(x["summed_pss"] for x in samples)}, indent=2) + "\n")
