"""FastAPI service: scene metadata, baseline overlays, scenario jobs, frontend."""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, render, runner, scene
from .jobs import JobManager

app = FastAPI(title="Cool Choices API")
geometry = scene.read_geometry()
manager = JobManager(geometry)
_last_request = time.monotonic()


@app.middleware("http")
async def _track_activity(request: Request, call_next):
    global _last_request
    if request.url.path != "/healthz":  # Fly's health checks must not keep the machine awake
        _last_request = time.monotonic()
    return await call_next(request)


def _idle_watchdog(limit_seconds: float) -> None:
    while True:
        time.sleep(30)
        if manager.busy():
            continue
        if time.monotonic() - _last_request > limit_seconds:
            logging.getLogger("uvicorn.error").info("idle for %.0f min with no jobs; exiting", limit_seconds / 60)
            # Exit code 0, so Fly's on-failure restart policy leaves the machine stopped.
            # (uvicorn re-raises SIGTERM after a graceful shutdown, which exits 143.)
            # Nothing is in flight: no job is queued or running and no request arrived.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if config.IDLE_EXIT_MINUTES > 0:
    threading.Thread(target=_idle_watchdog, args=(config.IDLE_EXIT_MINUTES * 60,), name="idle-watchdog", daemon=True).start()


class Tree(BaseModel):
    lon: float
    lat: float
    height: float = Field(12.0, ge=config.TREE_HEIGHT_MIN, le=config.TREE_HEIGHT_MAX)
    crown_radius: float = Field(4.0, ge=config.TREE_RADIUS_MIN, le=config.TREE_RADIUS_MAX)


class RunRequest(BaseModel):
    trees: list[Tree] = Field(default_factory=list, max_length=config.MAX_TREES)


def _baseline_ready() -> bool:
    return (config.BASELINE_DIR / "result.json").exists()


@app.get("/api/config")
def get_config():
    return {"mapbox_token": config.MAPBOX_TOKEN}


@app.get("/api/scene")
def get_scene():
    corners = geometry.corners_lonlat()
    meta = {}
    if (config.BASELINE_DIR / "meta.json").exists():
        meta = json.loads((config.BASELINE_DIR / "meta.json").read_text())
    return {
        "rows": geometry.rows,
        "cols": geometry.cols,
        "pixel_size_m": geometry.pixel_size,
        "corners": corners,
        "center": [sum(c[0] for c in corners) / 4, sum(c[1] for c in corners) / 4],
        "date": config.SIM_DATE,
        "variables": render.VARIABLES,
        "diff_range": render.DIFF_RANGE,
        "change_threshold_c": runner.CHANGE_THRESHOLD_C,
        "baseline_ready": _baseline_ready(),
        "baseline_url": "/results/baseline/",
        "baseline_model_seconds": meta.get("model_seconds"),
        "limits": {
            "max_trees": config.MAX_TREES,
            "height": [config.TREE_HEIGHT_MIN, config.TREE_HEIGHT_MAX],
            "crown_radius": [config.TREE_RADIUS_MIN, config.TREE_RADIUS_MAX],
        },
    }


@app.get("/api/baseline")
def get_baseline():
    if not _baseline_ready():
        raise HTTPException(503, "baseline not computed yet")
    return json.loads((config.BASELINE_DIR / "result.json").read_text())


@app.post("/api/jobs", status_code=202)
def create_job(request: RunRequest):
    if not _baseline_ready():
        raise HTTPException(503, "baseline not computed yet")
    if not request.trees:
        raise HTTPException(400, "place at least one tree before running")
    outside = [t for t in request.trees if not geometry.contains(t.lon, t.lat)]
    if outside:
        raise HTTPException(400, f"{len(outside)} tree(s) fall outside the scene extent")
    status = manager.submit([t.model_dump() for t in request.trees])
    return JSONResponse(manager.to_response(status))


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    status = manager.get(job_id)
    if status is None:
        raise HTTPException(404, "unknown job")
    return JSONResponse(manager.to_response(status))


@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str):
    status = manager.get(job_id)
    if status is None or status["status"] != "done":
        raise HTTPException(404, "result not available")
    return json.loads((manager.job_dir(job_id) / "result" / "result.json").read_text())


@app.get("/healthz")
def healthz():
    return {"ok": True, "baseline_ready": _baseline_ready()}


# Rendered overlays: /results/baseline/tmrt_14.png, /results/jobs/<id>/tmrt_diff_14.png
config.DATA_DIR.mkdir(parents=True, exist_ok=True)
config.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
config.JOBS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/results/baseline", StaticFiles(directory=config.BASELINE_DIR), name="baseline")


@app.api_route("/results/jobs/{job_id}/{filename}", methods=["GET", "HEAD"])
def job_file(job_id: str, filename: str):
    status = manager.get(job_id)
    if status is None or status["status"] != "done" or "/" in filename or not filename.endswith((".png", ".json")):
        raise HTTPException(404)
    path = manager.job_dir(job_id) / "result" / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


if config.FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="frontend")
