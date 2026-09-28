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
            key, value = key.strip(), value.strip()
            if value[:1] in ("'", '"') and value[-1:] == value[:1] and len(value) > 1:
                value = value[1:-1]
            elif " #" in value:  # unquoted value with an inline comment
                value = value.split(" #", 1)[0].rstrip()
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


def _physical_ram() -> int:
    try:
        pages, page_size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):  # e.g. native Windows
        return 0
    return pages * page_size if pages > 0 and page_size > 0 else 0


# Memory budget for solweig_light's admission check. SolweigLight2 charges a 512 px
# job ~1.55 GiB plus 5% of physical RAM (GDAL cache) plus 0.4 GiB for the parent, so
# a fixed 6 GiB is refused on hosts with more than ~81 GiB of RAM, while half of RAM
# alone is refused below ~4.3 GiB. max(6 GiB, half of RAM) is admitted at every size.
# Inside a memory-limited container sysconf reports the host's RAM, so set
# SOLWEIG_MEMORY_GB explicitly there (fly.toml does).
_MEMORY_GB = os.environ.get("SOLWEIG_MEMORY_GB", "").strip()
MEMORY_BUDGET_BYTES = (int(float(_MEMORY_GB) * 1024**3) if _MEMORY_GB
                       else max(6 * 1024**3, _physical_ram() // 2))
BLOCK_PIXELS = int(os.environ.get("SOLWEIG_BLOCK_PIXELS", "1024"))

MAPBOX_TOKEN = os.environ.get("MAPBOX_TOKEN", "")

# How many finished jobs to keep on disk before the oldest are removed.
MAX_KEPT_JOBS = int(os.environ.get("SOLWEIG_MAX_KEPT_JOBS", "10"))

# Tree placement limits (metres).
TREE_HEIGHT_MIN, TREE_HEIGHT_MAX = 2.0, 40.0
TREE_RADIUS_MIN, TREE_RADIUS_MAX = 1.0, 15.0
MAX_TREES = 200
