"""Job queue: one scenario at a time, each run in a separate worker process.

Job state lives in <job_dir>/status.json so the API process only reads
files; it survives uvicorn reloads and never runs the model itself.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from queue import Queue

from . import config, scene
from .worker import REQUEST_FILE, STATUS_FILE, read_status, write_status

_BACKEND_DIR = Path(__file__).resolve().parent.parent
log = logging.getLogger("uvicorn.error")


def _process_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


class JobManager:
    def __init__(self, geometry: scene.SceneGeometry):
        self.geometry = geometry
        self.queue: Queue[str] = Queue()
        self.expected_seconds: float | None = None
        config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self._recover()
        threading.Thread(target=self._loop, name="solweig-job-loop", daemon=True).start()

    # -- persistence ---------------------------------------------------------
    def _job_dir(self, job_id: str) -> Path:
        return config.JOBS_DIR / job_id

    def _recover(self) -> None:
        """After a server restart: re-queue queued jobs, keep running jobs whose
        worker process is still alive, and fail the rest."""
        for status_path in sorted(config.JOBS_DIR.glob(f"*/{STATUS_FILE}")):
            status = read_status(status_path.parent)
            if not status:
                continue
            if status.get("status") == "queued":
                self.queue.put(status["id"])
            elif status.get("status") == "running" and not _process_alive(status.get("pid")):
                status.update(status="failed", phase="failed", finished=time.time(),
                              error="server restarted while the job was running; please run again")
                write_status(status_path.parent, status)

    # -- public API ---------------------------------------------------------
    def submit(self, trees: list[dict]) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / REQUEST_FILE).write_text(json.dumps({"trees": trees}))
        status = {
            "id": job_id, "status": "queued", "phase": "queued", "trees": trees,
            "created": time.time(), "started": None, "finished": None, "error": None,
            "timesteps_done": 0, "timesteps_total": 24, "changed_pixels": 0,
        }
        write_status(job_dir, status)
        self.queue.put(job_id)
        return status

    def get(self, job_id: str) -> dict | None:
        if not job_id.isalnum():
            return None
        status = read_status(self._job_dir(job_id))
        if status is None and self._job_dir(job_id).exists():
            log.warning("job %s: directory exists but status.json is unreadable (%s)",
                        job_id, sorted(p.name for p in self._job_dir(job_id).iterdir()))
        return status

    def job_dir(self, job_id: str) -> Path:
        return self._job_dir(job_id)

    def queue_position(self, job_id: str) -> int:
        queued = sorted(
            (s for s in (read_status(p.parent) for p in config.JOBS_DIR.glob(f"*/{STATUS_FILE}")) if s and s.get("status") == "queued"),
            key=lambda s: s.get("created", 0),
        )
        return next((i for i, s in enumerate(queued) if s.get("id") == job_id), 0)

    def to_response(self, status: dict) -> dict:
        now = time.time()
        started, finished = status.get("started"), status.get("finished")
        elapsed = ((finished or now) - started) if started else 0.0
        expected = self.expected_seconds
        if expected is None and (config.BASELINE_DIR / "meta.json").exists():
            try:
                expected = json.loads((config.BASELINE_DIR / "meta.json").read_text()).get("model_seconds")
            except (OSError, ValueError):
                expected = None
        return {
            "id": status["id"],
            "status": status["status"],
            "phase": status.get("phase"),
            "trees": status.get("trees", []),
            "created": status.get("created"),
            "elapsed_seconds": round(elapsed, 1),
            "expected_seconds": expected,
            "timesteps_done": status.get("timesteps_done", 0),
            "timesteps_total": status.get("timesteps_total", 24),
            "changed_pixels": status.get("changed_pixels", 0),
            "error": status.get("error"),
            "result_url": f"/results/jobs/{status['id']}/" if status["status"] == "done" else None,
            "queue_position": self.queue_position(status["id"]) if status["status"] == "queued" else 0,
        }

    # -- worker loop ---------------------------------------------------------
    def _loop(self) -> None:
        while True:
            job_id = self.queue.get()
            try:
                self._run(job_id)
            finally:
                self._prune()
                self.queue.task_done()

    def _run(self, job_id: str) -> None:
        job_dir = self._job_dir(job_id)
        started = time.time()
        log = open(job_dir / "worker.log", "ab")  # noqa: SIM115 - handed to the subprocess
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "app.worker", job_id],
                cwd=_BACKEND_DIR, stdout=log, stderr=subprocess.STDOUT,
            )
            code = process.wait()
        finally:
            log.close()
        status = read_status(job_dir)
        if status is None or status.get("status") not in ("done", "failed"):
            # The worker died without reporting (native crash, OOM kill, ...).
            tail = ""
            try:
                tail = (job_dir / "worker.log").read_text(errors="replace")[-2000:]
            except OSError:
                pass
            status = status or {"id": job_id, "trees": []}
            status.update(status="failed", phase="failed", finished=time.time(),
                          error=f"worker exited with code {code}; see worker.log. {tail[-300:]}")
            write_status(job_dir, status)
            return
        if status["status"] == "done":
            elapsed = status.get("model_seconds") or (time.time() - started)
            self.expected_seconds = elapsed if self.expected_seconds is None else 0.5 * (self.expected_seconds + elapsed)

    def _prune(self) -> None:
        finished = []
        for status_path in config.JOBS_DIR.glob(f"*/{STATUS_FILE}"):
            status = read_status(status_path.parent)
            if status and status.get("status") in ("done", "failed"):
                finished.append((status.get("created", 0), status_path.parent))
        finished.sort()
        for _, old_dir in finished[: max(0, len(finished) - config.MAX_KEPT_JOBS)]:
            shutil.rmtree(old_dir, ignore_errors=True)
