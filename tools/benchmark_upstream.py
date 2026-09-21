"""Randomized serial CPU thread characterization on the frozen small fixture.

This is a P0 baseline experiment, not a candidate speedup or full release matrix.
The protocol is written before any trial. Existing output roots are rejected.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = args.run.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    fixture = root / "tests/fixtures/generated/own_met_small"
    kwargs = root / "reports/initial_kwargs.json"
    runner = root / "tools/run_reference.py"
    trials = [{"threads": threads, "repetition": index} for index in range(5) for threads in (None, 1, 4)]
    random.Random(20260918).shuffle(trials)
    hardware = {"platform": platform.platform(), "logical_cpus": os.cpu_count()}
    if sys.platform == "darwin":
        hardware["sysctl_cpu_ram_cores"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize", "hw.logicalcpu"], text=True).splitlines()
    protocol = {
        "schema_version": 1, "scope": "P0 small-fixture native-thread characterization",
        "source_commit": "0d7fe742abeeddd890dd58fc76ed7f78bd47faec",
        "fixture_sha256": json.loads((fixture / "manifest.json").read_text())["fixture_sha256"],
        "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "kwargs_sha256": hashlib.sha256(kwargs.read_bytes()).hexdigest(),
        "hardware": hardware, "trial_order": trials,
        "cache_state": "Fresh process and fresh scene each trial, geometry absent; OS file cache uncontrolled",
        "jit_state": "Upstream Numba vectorized helper import cost included; no candidate JIT",
        "timing": "Child wall time includes imports, validation and full thermal_comfort; fixture copy excluded",
        "memory": "20ms sampled summed process-tree resident sets; may double-count shared pages and miss short peaks",
        "cpu_settings": "Default process worker policy unchanged; explicit native budgets set Torch/OMP/MKL/OpenBLAS/NumExpr",
        "limitations": ["Development host, not a dedicated performance machine",
                        "Tiny workload only; selected budget not assumed strongest for larger scenes",
                        "No candidate or CUDA timing; no speedup gate inferred",
                        "Geometry-warm, larger workloads and kernel-only matrices remain pending"],
    }
    dump(destination / "protocol.json", protocol)
    results = []
    for index, trial in enumerate(trials):
        if hashlib.sha256(runner.read_bytes()).hexdigest() != protocol["runner_sha256"]:
            raise RuntimeError("Runner changed during frozen experiment; preserve partial trials and rerun")
        name = f"trial_{index:02d}_threads_{trial['threads'] or 'default'}"
        run = destination / name
        command = [sys.executable, str(runner), "--fixture", str(fixture), "--kwargs", str(kwargs), "--run", str(run)]
        if trial["threads"] is not None:
            command.extend(["--threads", str(trial["threads"])])
        completed = subprocess.run(command, capture_output=True, text=True)
        result = dict(trial, run=str(run), command=command, exit_code=completed.returncode)
        result["runner_unchanged"] = hashlib.sha256(runner.read_bytes()).hexdigest() == protocol["runner_sha256"]
        for file in ("measurement.json", "outcome.json"):
            if (run / file).exists():
                result[file[:-5]] = json.loads((run / file).read_text())
        if completed.returncode == 0 and result["runner_unchanged"]:
            verification = subprocess.run([sys.executable, str(root / "tools/check_reference_repeatability.py"),
                str(root / "reports/runs/dependencies_cpu/scene"), str(run / "scene"),
                "--report", str(run / "repeatability.json")], capture_output=True, text=True)
            result["exact_tiff_repeatability_exit_code"] = verification.returncode
            result["verification_output"] = verification.stdout + verification.stderr
        else:
            result["failure_output"] = completed.stdout + completed.stderr
        results.append(result)
        dump(destination / "trials.json", results)
        print(name, "exit", completed.returncode, flush=True)
    summaries = {}
    for threads in (None, 1, 4):
        valid = [r for r in results if r["threads"] == threads and r["exit_code"] == 0
                 and r.get("exact_tiff_repeatability_exit_code") == 0]
        times = [r["measurement"]["elapsed_seconds_including_imports"] for r in valid]
        summaries[str(threads or "default")] = {"valid_trials": len(valid), "raw_seconds": times,
            "median_seconds": statistics.median(times) if times else None,
            "min_seconds": min(times) if times else None, "max_seconds": max(times) if times else None,
            "sample_stdev_seconds": statistics.stdev(times) if len(times) > 1 else None,
            "raw_peak_rss_bytes": [r["measurement"]["sampled_process_tree_peak_rss_bytes"] for r in valid]}
    report = {"protocol": protocol, "trials": results, "summaries": summaries,
              "status": "passed" if all(s["valid_trials"] == 5 for s in summaries.values()) else "failed_trials_present"}
    dump(args.report, report)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
