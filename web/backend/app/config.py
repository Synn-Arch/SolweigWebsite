"""Runtime configuration for the SOLWEIG web service (environment driven)."""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WEB = _HERE.parent.parent


def _load_dotenv(*paths: Path) -> None:
    """Read KEY=VALUE lines into os.environ without overriding real env vars."""
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


# Local secrets/settings: web/.env (git-ignored). On Fly.io use `fly secrets set`.
_load_dotenv(_WEB / ".env", _HERE.parent / ".env")

# Static scene inputs (committed): Building_DSM/DEM/Trees/Landcover TIFFs + met file.
SCENE_DIR = Path(os.environ.get("SOLWEIG_SCENE_DIR", _WEB / "scene")).resolve()
# Writable working area: baseline outputs and per-job scenario folders.
DATA_DIR = Path(os.environ.get("SOLWEIG_DATA_DIR", _WEB / "data")).resolve()
BASELINE_DIR = DATA_DIR / "baseline"
JOBS_DIR = DATA_DIR / "jobs"
# Built frontend (Vite dist). Served at / when present.
FRONTEND_DIST = Path(os.environ.get("SOLWEIG_FRONTEND_DIST", _WEB / "frontend" / "dist")).resolve()

MET_FILE = "ownmet_Forcing_data.txt"
LANDCOVER_FILE = "Landcover.tif"
SIM_DATE = os.environ.get("SOLWEIG_DATE", "2020-08-13")

# Runtime budget handed to solweig_light; never above the CPUs actually present.
CPU_THREADS = max(1, min(int(os.environ.get("SOLWEIG_THREADS", "4")), os.cpu_count() or 1))
MEMORY_BUDGET_BYTES = int(float(os.environ.get("SOLWEIG_MEMORY_GB", "6")) * 1024**3)
BLOCK_PIXELS = int(os.environ.get("SOLWEIG_BLOCK_PIXELS", "1024"))

MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN", "")

# How many finished jobs to keep on disk before the oldest are removed.
MAX_KEPT_JOBS = int(os.environ.get("SOLWEIG_MAX_KEPT_JOBS", "10"))

# Tree placement limits (metres).
TREE_HEIGHT_MIN, TREE_HEIGHT_MAX = 2.0, 40.0
TREE_RADIUS_MIN, TREE_RADIUS_MAX = 1.0, 15.0
MAX_TREES = 200
