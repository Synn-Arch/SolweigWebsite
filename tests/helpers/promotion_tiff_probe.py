"""Run one installed-wheel TIFF-to-TIFF case in a fresh Python process."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback


RAW_FILES = ("Building_DSM.tif", "DEM.tif", "Trees.tif", "met.txt", "manifest.json")
OUTPUT_FIELDS = ("UTCI", "TMRT", "Kup", "Kdown", "Lup", "Ldown", "Shadow", "WBGT", "Ta", "Wind")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--kwargs", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    args.scene.mkdir(parents=True, exist_ok=False)
    for name in RAW_FILES:
        shutil.copy2(args.fixture / name, args.scene / name)
    kwargs = json.loads(args.kwargs.read_text())
    kwargs["base_path"] = str(args.scene.resolve())
    kwargs["own_met_file"] = str((args.scene / "met.txt").resolve())
    for key in ("building_dsm_filename", "dem_filename", "trees_filename"):
        kwargs[key] = Path(kwargs[key]).name
    for field in OUTPUT_FIELDS:
        kwargs["save_" + field.lower()] = True

    try:
        import solweig_light

        options = solweig_light.RuntimeOptions(
            cpu_budget=1,
            workers=1,
            threads_per_worker=1,
            memory_budget_bytes=12 * 1024**3,
            block_pixels=128,
            cache_dir=str(args.scene / ".runtime_cache"),
            cache_enabled=True,
            checkpoint_interval=1,
            legacy_cache_policy="recompute",
            resume=False,
        )
        with solweig_light.runtime_options(options):
            solweig_light.thermal_comfort(**kwargs)
        outputs = sorted((args.scene / "output_folder").glob("*/*.tif"))
        value = {
            "status": "passed",
            "python": sys.executable,
            "package": str(Path(solweig_light.__file__).resolve()),
            "outputs": {str(path.relative_to(args.scene)): digest(path) for path in outputs},
        }
        if len(outputs) != len(OUTPUT_FIELDS):
            raise RuntimeError(f"expected {len(OUTPUT_FIELDS)} output TIFFs, found {len(outputs)}")
        args.result.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        return 0
    except BaseException as error:
        args.result.write_text(json.dumps({
            "status": "failed",
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }, indent=2, sort_keys=True) + "\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
