"""Run one frozen portable-profile candidate fixture from the installed wheel."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=Path, required=True)
parser.add_argument("--outcome", type=Path, required=True)
parser.add_argument("--fixture-manifest", type=Path, required=True)
args = parser.parse_args()

import solweig_light
from solweig_light.runtime import RuntimeOptions, runtime_options

origin = Path(solweig_light.__file__).resolve()
if "m1_admission/venv" not in str(origin):
    raise RuntimeError(f"candidate is not the installed admission wheel: {origin}")
for forbidden in ("torch", "mkl", "pandas", "xarray", "rasterio", "geopandas", "ee", "geemap", "osmnx"):
    if importlib.util.find_spec(forbidden) is not None:
        raise RuntimeError(f"forbidden runtime package available: {forbidden}")
manifest = json.loads(args.fixture_manifest.read_text())
case = args.scene.name
record = manifest["cases"][case]
for item in record["files"]:
    path = args.scene / item["name"]
    if not path.is_file() or digest(path) != item["sha256"]:
        raise RuntimeError(f"fixture mismatch: {path}")
kwargs = json.loads((args.scene / "kwargs.json").read_text())
kwargs["base_path"] = str(args.scene.resolve())
kwargs["own_met_file"] = str((args.scene / "met.txt").resolve())
started = time.perf_counter()
result = {"candidate_origin": str(origin), "fixture_manifest_sha256": digest(args.fixture_manifest),
          "kwargs": kwargs, "torch_imported": "torch" in sys.modules}
try:
    options = RuntimeOptions(memory_budget_bytes=12 * 1024**3, cpu_budget=10,
                             workers=1, threads_per_worker=10)
    result["runtime_options"] = options.as_dict()
    with runtime_options(options):
        returned = solweig_light.thermal_comfort(**kwargs)
except BaseException:
    result.update(status="failed", traceback=traceback.format_exc(),
                  workflow_seconds=time.perf_counter() - started)
    args.outcome.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    raise
result.update(status="executed_not_yet_compared", return_repr=repr(returned),
              workflow_seconds=time.perf_counter() - started,
              torch_imported_after="torch" in sys.modules)
args.outcome.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
