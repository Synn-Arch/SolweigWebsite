"""Single-worker job queue: one SOLWEIG scenario run at a time."""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

from . import config, runner, scene

STATUS_FILE = "status.json"


@dataclass
class Job:
    id: str
    trees: list[dict]
    created: float = field(default_factory=time.time)
    status: str = "queued"  # queued | running | done | failed
    phase: str = "queued"
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    timesteps_done: int = 0
    timesteps_total: int = 24
    changed_pixels: int = 0

    @property
    def dir(self) -> Path:
        return config.JOBS_DIR / self.id

    def to_dict(self, expected_seconds: float | None) -> dict:
        now = time.time()
        elapsed = (self.finished or now) - self.started if self.started else 0.0
        return {
            "id": self.id,
            "status": self.status,
            "phase": self.phase,
            "trees": self.trees,
            "created": self.created,
            "elapsed_seconds": round(elapsed, 1),
            "expected_seconds": expected_seconds,
            "timesteps_done": self.timesteps_done,
            "timesteps_total": self.timesteps_total,
            "changed_pixels": self.changed_pixels,
            "error": self.error,
            "result_url": f"/results/jobs/{self.id}/" if self.status == "done" else None,
        }


class JobManager:
    def __init__(self, geometry: scene.SceneGeometry):
        self.geometry = geometry
        self.jobs: dict[str, Job] = {}
        self.queue: Queue[Job] = Queue()
        self.lock = threading.Lock()
        self.expected_seconds: float | None = None
        config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
        self._load_finished()
        threading.Thread(target=self._worker, name="solweig-worker", daemon=True).start()

    # -- persistence of finished jobs across restarts -----------------------
    def _load_finished(self) -> None:
        for status_path in sorted(config.JOBS_DIR.glob(f"*/{STATUS_FILE}")):
            try:
                record = json.loads(status_path.read_text())
                job = Job(**record)
                if job.status == "done":
                    self.jobs[job.id] = job
            except (OSError, ValueError, TypeError):
                continue

    def _save(self, job: Job) -> None:
        job.dir.mkdir(parents=True, exist_ok=True)
        (job.dir / STATUS_FILE).write_text(json.dumps(job.__dict__))

    # -- public API ---------------------------------------------------------
    def submit(self, trees: list[dict]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], trees=trees)
        with self.lock:
            self.jobs[job.id] = job
        self.queue.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def queue_position(self, job: Job) -> int:
        with self.lock:
            queued = [j for j in self.jobs.values() if j.status == "queued"]
        queued.sort(key=lambda j: j.created)
        return next((i for i, j in enumerate(queued) if j.id == job.id), 0)

    # -- worker -------------------------------------------------------------
    def _worker(self) -> None:
        while True:
            job = self.queue.get()
            try:
                self._run(job)
            except Exception as error:  # noqa: BLE001 - surfaced to the client
                job.status, job.phase = "failed", "failed"
                job.error = f"{type(error).__name__}: {error}"
                traceback.print_exc()
            finally:
                job.finished = time.time()
                self._save(job)
                self._prune()
                self.queue.task_done()

    def _run(self, job: Job) -> None:
        job.status, job.phase, job.started = "running", "preparing", time.time()
        work = job.dir / "scene"
        if work.exists():
            shutil.rmtree(work)
        scene.copy_inputs(config.SCENE_DIR, work)
        job.changed_pixels = scene.paint_trees(work / "Trees.tif", self.geometry, job.trees)

        stop = threading.Event()
        monitor = threading.Thread(target=self._monitor, args=(job, work, stop), daemon=True)
        monitor.start()
        try:
            elapsed = runner.run_model(work)
        finally:
            stop.set()
            monitor.join(timeout=5)

        job.phase = "rendering"
        runner.postprocess(work, job.dir / "result", config.BASELINE_DIR, keep_arrays=False)
        # Keep the folder small: outputs are rendered, inputs are reproducible.
        shutil.rmtree(work, ignore_errors=True)
        job.timesteps_done = job.timesteps_total
        job.status, job.phase = "done", "done"
        self.expected_seconds = elapsed if self.expected_seconds is None else 0.5 * (self.expected_seconds + elapsed)

    def _monitor(self, job: Job, work: Path, stop: threading.Event) -> None:
        while not stop.wait(2.0):
            if not runner.geometry_done(work):
                job.phase = "geometry"
            else:
                job.phase = "simulating"
                job.timesteps_done = runner.committed_timesteps(work)

    def _prune(self) -> None:
        with self.lock:
            done = sorted((j for j in self.jobs.values() if j.status in ("done", "failed")), key=lambda j: j.created)
            for old in done[: max(0, len(done) - config.MAX_KEPT_JOBS)]:
                shutil.rmtree(old.dir, ignore_errors=True)
                self.jobs.pop(old.id, None)
