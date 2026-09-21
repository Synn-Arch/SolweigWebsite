#!/usr/bin/env python3
"""Run one current source-bound wheel admission case under the frozen policy."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time


FIELDS = ("UTCI", "TMRT", "Kup", "Kdown", "Lup", "Ldown", "Shadow", "WBGT", "Ta", "Wind")


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("dense1024", "vegetation1024"), required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--kwargs", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--wheel-target", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.scene = args.scene.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Keep the target explicit so this process cannot accidentally import the
    # source tree or a prior wheel installation.
    sys.path.insert(0, str(args.wheel_target.resolve()))
    import solweig_light
    from solweig_light import thermal_comfort
    from solweig_light.runtime import RuntimeOptions, runtime_options

    origin = Path(solweig_light.__file__).resolve()
    target = args.wheel_target.resolve()
    if not origin.is_relative_to(target):
        raise RuntimeError(f"unexpected package origin: {origin}")
    if "torch" in sys.modules:
        raise RuntimeError("candidate imported torch before execution")

    fixture = json.loads(args.fixture_manifest.read_text())
    expected_case = "dense_urban_1024" if args.case == "dense1024" else "vegetation_rich_1024"
    if fixture.get("fixture_id") != expected_case:
        raise RuntimeError(f"fixture case mismatch: {fixture.get('fixture_id')!r}")
    # The frozen fixture manifest records source-generation kwargs; the run
    # kwargs file is the canonical original invocation and has its own hash.
    for name, expected in fixture["files_sha256"].items():
        if name == "kwargs.json":
            continue
        path = args.scene / name
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"fixture hash mismatch: {path}")

    kwargs = json.loads(args.kwargs.read_text())
    allowed = set(inspect.signature(thermal_comfort).parameters)
    unknown = set(kwargs) - allowed
    if unknown:
        raise RuntimeError(f"candidate kwargs are not accepted: {sorted(unknown)}")
    kwargs["base_path"] = str(args.scene)
    kwargs["own_met_file"] = str(args.scene / "met.txt")
    if kwargs.get("tile_size") != 3600 or kwargs.get("overlap") != 20:
        raise RuntimeError("frozen large-scene tile settings changed")
    if any(kwargs.get(key) is not True for key in (
        "save_tmrt", "save_svf", "save_kup", "save_kdown", "save_lup", "save_ldown",
        "save_shadow", "save_wbgt", "save_ta", "save_wind")):
        raise RuntimeError("all ten output flags must be enabled")

    options = RuntimeOptions(memory_budget_bytes=12 * 1024**3, cpu_budget=10,
                             workers=1, threads_per_worker=10, checkpoint_interval=1)
    started = time.perf_counter()
    error = None
    returned = None
    try:
        with runtime_options(options):
            returned = thermal_comfort(**kwargs)
    except BaseException:
        import traceback
        error = traceback.format_exc()
    elapsed = time.perf_counter() - started
    output_dir = args.scene / "output_folder" / "0_0"
    actual = sorted(path.name for path in output_dir.glob("*.tif")) if output_dir.is_dir() else []
    expected = sorted(f"{field}_0_0.tif" for field in FIELDS)
    result = {
        "schema": "local-cpu-optimization-e0-candidate-run.v1",
        "case": args.case,
        "status": "failed" if error else "executed_not_yet_compared",
        "error": error,
        "return_repr": repr(returned),
        "workflow_seconds": elapsed,
        "package_origin": str(origin),
        "wheel_target": str(target),
        "torch_imported_before": False,
        "torch_imported_after": "torch" in sys.modules,
        "fixture_manifest": str(args.fixture_manifest),
        "fixture_manifest_sha256": sha(args.fixture_manifest),
        "kwargs_path": str(args.kwargs),
        "kwargs": kwargs,
        "runtime_options": options.as_dict(),
        "output_names": actual,
        "output_names_expected": expected,
        "outputs_complete": actual == expected,
        "performance_claim": None
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if error:
        raise RuntimeError(error)
    if result["torch_imported_after"] or returned is not None or actual != expected:
        raise RuntimeError(f"candidate admission invariant failed: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
