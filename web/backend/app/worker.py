"""Run one scenario job in its own process.

    python -m app.worker <job_id>

The job folder must already contain request.json (written by JobManager).
All progress is written to status.json so the API process only reads files
and never runs GDAL/Numba code in a thread of the web server.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

from . import config, runner, scene

STATUS_FILE = "status.json"
REQUEST_FILE = "request.json"


def read_status(job_dir: Path) -> dict | None:
    try:
        return json.loads((job_dir / STATUS_FILE).read_text())
    except (OSError, ValueError):
        return None


def write_status(job_dir: Path, status: dict) -> None:
    tmp = job_dir / (STATUS_FILE + ".tmp")
    tmp.write_text(json.dumps(status))
    tmp.replace(job_dir / STATUS_FILE)


def _update(job_dir: Path, record: dict, **fields) -> None:
    record.update(fields)
    record["updated"] = time.time()
    write_status(job_dir, record)


def run_job(job_dir: Path) -> int:
    status = read_status(job_dir) or {"id": job_dir.name, "status": "queued", "phase": "queued"}
    request = json.loads((job_dir / REQUEST_FILE).read_text())
    trees = request["trees"]
    work = job_dir / "scene"
    try:
        _update(job_dir, status, status="running", phase="preparing", started=time.time(), pid=os.getpid())
        if work.exists():
            shutil.rmtree(work)
        geometry = scene.read_geometry()
        scene.copy_inputs(config.SCENE_DIR, work)
        changed = scene.paint_trees(work / "Trees.tif", geometry, trees)
        _update(job_dir, status, changed_pixels=changed, phase="geometry")

        stop = threading.Event()

        def monitor() -> None:
            while not stop.wait(2.0):
                if runner.geometry_done(work):
                    _update(job_dir, status, phase="simulating", timesteps_done=runner.committed_timesteps(work))
                else:
                    _update(job_dir, status, phase="geometry")

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            elapsed = runner.run_model(work)
        finally:
            stop.set()
            thread.join(timeout=5)

        _update(job_dir, status, phase="rendering", timesteps_done=status.get("timesteps_total", 24))
        runner.postprocess(work, job_dir / "result", config.BASELINE_DIR, keep_arrays=False)
        shutil.rmtree(work, ignore_errors=True)
        _update(job_dir, status, status="done", phase="done", finished=time.time(), model_seconds=elapsed)
        return 0
    except Exception as error:  # noqa: BLE001 - reported to the client via status.json
        traceback.print_exc()
        _update(job_dir, status, status="failed", phase="failed", finished=time.time(),
                error=f"{type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m app.worker <job_id>")
    sys.exit(run_job(config.JOBS_DIR / sys.argv[1]))
