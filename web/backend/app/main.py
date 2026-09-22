"""FastAPI service: scene metadata, baseline overlays, scenario jobs, frontend."""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, render, runner, scene
from .jobs import JobManager

app = FastAPI(title="Cool Choices API")
geometry = scene.read_geometry()
manager = JobManager(geometry)


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
    job = manager.submit([t.model_dump() for t in request.trees])
    return _job_response(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return _job_response(job)


@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str):
    job = manager.get(job_id)
    if job is None or job.status != "done":
        raise HTTPException(404, "result not available")
    return json.loads((job.dir / "result" / "result.json").read_text())


def _job_response(job):
    expected = manager.expected_seconds
    if expected is None and (config.BASELINE_DIR / "meta.json").exists():
        expected = json.loads((config.BASELINE_DIR / "meta.json").read_text()).get("model_seconds")
    payload = job.to_dict(expected)
    payload["queue_position"] = manager.queue_position(job) if job.status == "queued" else 0
    return JSONResponse(payload)


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
    job = manager.get(job_id)
    if job is None or job.status != "done" or "/" in filename or not filename.endswith((".png", ".json")):
        raise HTTPException(404)
    path = job.dir / "result" / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


if config.FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIST, html=True), name="frontend")
