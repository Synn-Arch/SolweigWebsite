# Cool Choices — SOLWEIG tree-scenario web app

Interactive map (Mapbox GL) over a 512×512 px, 2 m Austin scene. The baseline
SOLWEIG-light result is shown as an hourly overlay; users drop trees on the
map, press **Run analysis**, and get the scenario and its difference against
the baseline, hour by hour.

```
web/
  scene/      512×512 crop of solweig_scene_small (inputs + met file)
  backend/    FastAPI service wrapping solweig_light (app/)
  frontend/   Vite + React + mapbox-gl UI
  data/       generated at runtime: baseline/ and jobs/<id>/ (git-ignored)
```

Each scenario run copies the scene, paints the trees into `Trees.tif`
(canopy height = max(existing, tree height) inside the crown radius), and
runs the unchanged `thermal_comfort` workflow. Sky view factors and all 24
timesteps are recomputed every run because the geometry cache is keyed on
the whole raster, so one run takes a few minutes (about 6 min on 4 cores).
Jobs are executed one at a time, each in its own worker process
(`python -m app.worker <id>`), with progress written to `data/jobs/<id>/status.json`;
the UI polls that state. The web server never runs the model itself, so it
survives reloads and worker crashes.

## Local development

Backend (needs GDAL 3.13 Python bindings and the pinned numerical stack;
see `backend/requirements-core.txt` and the Dockerfile for the exact recipe):

```sh
pip install -r web/backend/requirements-core.txt
pip install --no-build-isolation "gdal==$(gdal-config --version)"
pip install -r web/backend/requirements.txt
pip install --no-deps .                 # solweig_light from this repo
cd web/backend
python -m app.precompute                # one-off baseline (~6 min)
uvicorn app.main:app --reload --port 8000
```

Put your Mapbox token in `web/.env` (copy `web/.env.example`); the backend
reads that file at start-up. Real environment variables take precedence,
so `MAPBOX_TOKEN=pk.xxx uvicorn ...` still works too.

Frontend (proxies `/api` and `/results` to :8000):

```sh
cd web/frontend
npm install
npm run dev
```

Environment variables (or `web/.env` entries): `MAPBOX_TOKEN` (required), `SOLWEIG_THREADS` (default 4,
clamped to available CPUs), `SOLWEIG_MEMORY_GB` (6), `SOLWEIG_DATA_DIR`,
`SOLWEIG_DATE` (2020-08-13), `SOLWEIG_MAX_KEPT_JOBS` (10).

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/config` | Mapbox token for the browser |
| GET | `/api/scene` | Extent corners (WGS84), colour ranges, limits, baseline status |
| GET | `/api/baseline` | Baseline hourly stats and band timestamps |
| POST | `/api/jobs` | `{trees:[{lon,lat,height,crown_radius}]}` → job (202) |
| GET | `/api/jobs/{id}` | Status, phase, committed timesteps, elapsed/expected seconds |
| GET | `/api/jobs/{id}/result` | Scenario stats, baseline stats, difference stats |
| GET | `/results/baseline/{var}_{hh}.png` | Baseline overlay PNG (`tmrt`/`utci`) |
| GET | `/results/jobs/{id}/{var}_{hh}.png` | Scenario overlay; `{var}_diff_{hh}.png` for the difference |

## Deploying to Fly.io

One-time setup:

```sh
fly apps create solweig-scenarios          # or edit `app` in fly.toml
fly secrets set MAPBOX_TOKEN=pk.xxx
fly tokens create deploy -x 999999h        # paste into GitHub secret FLY_API_TOKEN
```

Then every push to `main` that touches `src/`, `web/`, the Dockerfile or
`fly.toml` runs `.github/workflows/fly-deploy.yml`, which builds the image on
Fly's remote builder and deploys it. The image build runs the baseline
simulation once (also warming the Numba JIT cache), so builds take several
minutes. The machine is `performance-4x` / 8 GB and auto-stops when idle.

Manual deploy from a laptop: `fly deploy`.
