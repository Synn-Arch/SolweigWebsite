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


_WRITE_LOCK = threading.Lock()


def read_status(job_dir: Path) -> dict | None:
    """Read status.json; tolerate a momentarily torn file by retrying briefly."""
    path = job_dir / STATUS_FILE
    for attempt in range(4):
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            time.sleep(0.05 * (attempt + 1))
    return None


def write_status(job_dir: Path, status: dict) -> None:
    """Atomic replace with a writer-unique temp name so concurrent writers never collide."""
    tmp = job_dir / f"{STATUS_FILE}.{os.getpid()}.{threading.get_ident()}.tmp"
    with _WRITE_LOCK:
        tmp.write_text(json.dumps(status))
        tmp.replace(job_dir / STATUS_FILE)


def _update(job_dir: Path, record: dict, **fields) -> None:
    with _WRITE_LOCK:
        record.update(fields)
        record["updated"] = time.time()
        snapshot = dict(record)
    write_status(job_dir, snapshot)


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
