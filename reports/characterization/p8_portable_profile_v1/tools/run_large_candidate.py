"""Run one frozen 1024 portable-profile correctness case through the installed API."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


FILES = ("Building_DSM.tif", "Trees.tif", "DEM.tif", "met.txt", "manifest.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", choices=("dense1024", "vegetation1024"), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    fixture = root / "fixtures" / args.case
    run = root / "large_runs" / args.case
    scene = run / "scene"
    if run.exists():
        raise RuntimeError(f"run directory already exists: {run}")
    scene.mkdir(parents=True)
    for name in FILES:
        shutil.copy2(fixture / name, scene / name)

    import solweig_light
    from solweig_light import thermal_comfort
    from solweig_light.radiation._math_profile import profile_identity
    if "torch" in sys.modules:
        raise RuntimeError("installed candidate imported torch")

    started = time.perf_counter()
    result = thermal_comfort(
        str(scene), "2020-07-18",
        building_dsm_filename="Building_DSM.tif",
        dem_filename="DEM.tif",
        trees_filename="Trees.tif",
        landcover_filename=None,
        ERA_5_z0_find=False,
        tile_size=3600,
        overlap=20,
        use_own_met=True,
        own_met_file=str(scene / "met.txt"),
        use_uhi=False,
        save_tmrt=True,
        save_svf=True,
        save_kup=True,
        save_kdown=True,
        save_lup=True,
        save_ldown=True,
        save_shadow=True,
        save_wbgt=True,
        save_ta=True,
        save_wind=True,
    )
    elapsed = time.perf_counter() - started
    if result is not None:
        raise RuntimeError(f"thermal_comfort returned {result!r}")
    outputs = sorted((scene / "output_folder" / "0_0").glob("*.tif"))
    # SVF is a required processed geometry artifact, while UTCI is the tenth
    # requested output raster in output_folder.
    expected = {"TMRT", "UTCI", "Kup", "Kdown", "Lup", "Ldown", "Shadow", "WBGT", "Ta", "Wind"}
    actual = {path.name.split("_0_0.tif")[0] for path in outputs}
    if actual != expected:
        raise RuntimeError(f"output fields differ: {sorted(actual)}")
    manifest = {
        "schema": "portable-profile-v1-linux-large-candidate.v1",
        "case": args.case,
        "package_origin": solweig_light.__file__,
        "profile": profile_identity(),
        "torch_imported": "torch" in sys.modules,
        "elapsed_diagnostic_seconds": elapsed,
        "performance_claim": False,
        "outputs": [
            {"path": str(path.relative_to(run)), "bytes": path.stat().st_size, "sha256": digest(path)}
            for path in outputs
        ],
    }
    (run / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"case": args.case, "outputs": len(outputs), "elapsed": elapsed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
