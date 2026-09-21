#!/usr/bin/env python3
"""Guarded full-day cProfile capture of the real candidate runtime worker."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import pstats
import shutil
import subprocess
import sys
import time


FLAGS = ("tmrt", "svf", "kup", "kdown", "lup", "ldown", "shadow", "wbgt", "ta", "wind")
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
              "NUMBA_NUM_THREADS")
EXPECTED_OUTPUTS = ("UTCI", "TMRT", "Kup", "Kdown", "Lup", "Ldown",
                    "Shadow", "WBGT", "Ta", "Wind")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".nbi", ".nbc"}}


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def prepared_job(scene: Path) -> dict:
    prepared = scene / "processed_inputs"
    relative = {
        "Building_DSM": "Building_DSM/Building_DSM_0_0.tif",
        "Trees": "Trees/Trees_0_0.tif",
        "DEM": "DEM/DEM_0_0.tif",
        "metfiles": "metfiles/metfile_0_0.txt",
        "walls": "walls/walls_0_0.tif",
        "aspect": "aspect/aspect_0_0.tif",
    }
    paths = {name: prepared / suffix for name, suffix in relative.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"prepared fixture is incomplete: {missing}")
    met_lines = [line for line in paths["metfiles"].read_text().splitlines() if line.strip()]
    if len(met_lines) - 1 != 24:
        raise ValueError(f"full-day fixture requires 24 meteorological rows, found {len(met_lines) - 1}")
    return {
        "base_path": str(scene),
        "preprocess_dir": str(prepared),
        "selected_date_str": "2020-07-18",
        "tile": "0_0",
        "paths": {name: str(path) for name, path in paths.items()},
        "flags": {f"save_{name}": True for name in FLAGS},
    }


def profile_summary(raw: Path, text_path: Path, json_path: Path) -> None:
    stats = pstats.Stats(str(raw))
    with text_path.open("w") as stream:
        pstats.Stats(str(raw), stream=stream).strip_dirs().sort_stats("cumulative").print_stats()
    rows = []
    for (filename, line, function), (primitive, calls, total, cumulative, _callers) in stats.stats.items():
        rows.append({"file": filename, "line": line, "function": function,
                     "primitive_calls": primitive, "calls": calls,
                     "total_seconds": total, "cumulative_seconds": cumulative})
    rows.sort(key=lambda item: (-item["cumulative_seconds"], -item["total_seconds"],
                                item["file"], item["line"], item["function"]))
    dump(json_path, {"sort": "cumulative_seconds_desc", "functions": rows})


def validate_outputs(scene: Path) -> dict:
    from osgeo import gdal
    gdal.UseExceptions()
    template = gdal.Open(str(scene / "processed_inputs" / "Building_DSM" /
                             "Building_DSM_0_0.tif"), gdal.GA_ReadOnly)
    if template is None:
        raise FileNotFoundError("prepared Building_DSM template is missing")
    expected_shape = (template.RasterYSize, template.RasterXSize, 24)
    template = None
    output = scene / "output_folder" / "0_0"
    records = []
    for name in EXPECTED_OUTPUTS:
        path = output / f"{name}_0_0.tif"
        dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
        if dataset is None:
            raise FileNotFoundError(path)
        record = {"path": str(path.relative_to(scene)), "rows": dataset.RasterYSize,
                  "cols": dataset.RasterXSize, "bands": dataset.RasterCount,
                  "times": [dataset.GetRasterBand(index).GetMetadata().get("Time")
                            for index in range(1, dataset.RasterCount + 1)]}
        dataset = None
        if (record["rows"], record["cols"], record["bands"]) != expected_shape:
            raise ValueError(f"unexpected output schema: {record}")
        if any(value is None for value in record["times"]):
            raise ValueError(f"missing Time metadata: {path}")
        records.append(record)
    extras = [scene / "processed_inputs" / "SVF" / name for name in
              ("SkyViewFactor_0_0.tif", "svfs_0_0.zip", "shadowmats_0_0.npz")]
    if not all(path.is_file() for path in extras):
        raise FileNotFoundError("cold full-physics run did not publish all SVF artifacts")
    return {"passed": True, "outputs": records,
            "svf_artifacts": [str(path.relative_to(scene)) for path in extras]}


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path,
                        default=root / "tests/reference/small_original_cpu/scene")
    parser.add_argument("--source", type=Path, default=root / "src")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", action="store_true",
                        help="run one unprofiled complete worker first, then profile a fresh run reusing its JIT/geometry caches")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    fixture, source, run = args.fixture.resolve(), args.source.resolve(), args.run.resolve()
    # Preserve a virtualenv launcher path; resolving its symlink can select the
    # base interpreter and lose the environment's site-packages.
    executable = Path(os.path.abspath(sys.executable))
    executable_realpath = os.path.realpath(executable)
    if not fixture.is_dir() or not (source / "solweig_light/runtime_worker.py").is_file():
        parser.error("fixture or candidate source is missing")
    if run.is_relative_to(fixture) or run.is_relative_to(source):
        parser.error("--run must not be nested inside the immutable fixture or source tree")
    if args.threads < 1:
        parser.error("--threads must be positive")

    fixture_inventory = inventory(fixture)
    source_inventory = inventory(source)
    plan = {"status": "validated_not_executed", "fixture": str(fixture),
            "fixture_inventory": fixture_inventory, "source": str(source),
            "source_inventory": source_inventory, "tool_sha256": digest(Path(__file__)),
            "run": str(run), "threads": args.threads, "block_pixels": 128,
            "timeline_rows": 24, "patches": 153, "all_output_flags": True,
            "validated_job_template": prepared_job(fixture),
            "profile_scope": "cProfile around solweig_light.runtime_worker in its numerical process"}
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    run.mkdir(parents=True, exist_ok=False)
    setup = run / "setup"
    artifacts = run / "profile"
    jit = run / "jit"
    scene = setup / "scene"
    setup.mkdir(); artifacts.mkdir(); jit.mkdir()
    shutil.copytree(fixture, scene)
    for target in (scene / "output_folder", scene / ".solweig-light"):
        if target.exists():
            shutil.rmtree(target)
    svf = scene / "processed_inputs" / "SVF"
    if svf.exists():
        shutil.rmtree(svf)
    svf.mkdir(parents=True)

    job = prepared_job(scene)
    options = {"cache_dir": str(setup / "geometry_cache"), "legacy_cache_policy": "recompute",
               "cache_enabled": True, "memory_budget_bytes": 4 * 1024**3,
               "cpu_budget": args.threads, "workers": 1, "threads_per_worker": args.threads,
               "block_pixels": 128, "checkpoint_interval": 1, "resume": False}
    dump(setup / "job.json", job); dump(setup / "options.json", options)
    raw = artifacts / "worker.pstats"
    command = [str(executable), "-m", "cProfile", "-o", str(raw), "-m",
               "solweig_light.runtime_worker", "--job", str(setup / "job.json"),
               "--options", json.dumps(options, sort_keys=True)]
    environment = dict(os.environ, PYTHONPATH=str(source), PYTHONNOUSERSITE="1",
                       CUDA_VISIBLE_DEVICES="", NUMBA_CACHE_DIR=str(jit))
    for name in THREAD_ENV:
        environment[name] = str(args.threads)
    warmup_command = [str(executable), "-m", "solweig_light.runtime_worker", "--job", str(setup / "job.json"),
                      "--options", json.dumps(options, sort_keys=True)]
    if args.warmup:
        warmup_dir = run / "warmup"
        warmup_dir.mkdir()
        warmup_env = dict(environment)
        with (warmup_dir / "stdout.log").open("w") as stdout, (warmup_dir / "stderr.log").open("w") as stderr:
            warmup = subprocess.run(warmup_command, cwd=run, env=warmup_env, stdout=stdout, stderr=stderr)
        dump(warmup_dir / "provenance.json", {"command": warmup_command, "returncode": warmup.returncode,
                                               "options": options, "purpose": "unprofiled cache/JIT warmup"})
        if warmup.returncode != 0:
            dump(run / "outcome.json", {"status": "warmup_failed", "returncode": warmup.returncode})
            return warmup.returncode
        # Keep warmup products for audit, while giving the profiled process a fresh
        # output/transaction namespace. Geometry cache and NUMBA cache remain.
        for name in ("output_folder", ".solweig-light"):
            target = scene / name
            if target.exists():
                shutil.move(str(target), str(warmup_dir / name))
    provenance = dict(plan, status="executing_profile_not_benchmark", command=command,
                      warmup=args.warmup, warmup_command=warmup_command if args.warmup else None,
                      python=sys.version, executable=str(executable),
                      executable_realpath=executable_realpath, platform=platform.platform(),
                      packages={item.metadata["Name"]: item.version
                                for item in importlib.metadata.distributions()},
                      thread_environment={name: environment.get(name) for name in THREAD_ENV},
                      job=job, runtime_options=options, copied_input_inventory=inventory(scene),
                      jit_dir=str(jit), geometry_cache_dir=options["cache_dir"],
                      profile_artifact_dir=str(artifacts))
    dump(run / "provenance.json", provenance)
    start = time.perf_counter()
    with (artifacts / "stdout.log").open("w") as stdout, (artifacts / "stderr.log").open("w") as stderr:
        completed = subprocess.run(command, cwd=run, env=environment, stdout=stdout, stderr=stderr)
    elapsed = time.perf_counter() - start
    failure_artifact = setup / "job.failure.json"
    if failure_artifact.is_file():
        dump(run / "outcome.json", {"status": "profile_worker_failure_artifact",
                                    "returncode": completed.returncode,
                                    "profiled_wall_seconds": elapsed,
                                    "failure_artifact": str(failure_artifact),
                                    "immutable_inputs": {"fixture_unchanged": False,
                                                          "source_unchanged": False}})
        return 1
    final_fixture_inventory = inventory(fixture)
    final_source_inventory = inventory(source)
    immutable_inputs = {
        "fixture_unchanged": final_fixture_inventory == fixture_inventory,
        "source_unchanged": final_source_inventory == source_inventory,
        "final_fixture_inventory": final_fixture_inventory,
        "final_source_inventory": final_source_inventory,
    }
    if completed.returncode != 0:
        dump(run / "outcome.json", {"status": "failed", "returncode": completed.returncode,
                                    "profiled_wall_seconds": elapsed,
                                    "immutable_inputs": immutable_inputs})
        return completed.returncode
    if not all((immutable_inputs["fixture_unchanged"], immutable_inputs["source_unchanged"])):
        dump(run / "outcome.json", {"status": "failed_immutable_inputs_changed",
                                    "returncode": completed.returncode,
                                    "profiled_wall_seconds": elapsed,
                                    "immutable_inputs": immutable_inputs})
        return 1
    if not raw.is_file():
        raise RuntimeError("worker succeeded without producing cProfile data")
    profile_summary(raw, artifacts / "worker_cumulative.txt", artifacts / "worker_functions.json")
    validation = validate_outputs(scene)
    dump(run / "outcome.json", {"status": "profile_completed_not_a_benchmark",
                                "returncode": completed.returncode,
                                "profiled_wall_seconds": elapsed,
                                "output_validation": validation,
                                "immutable_inputs": immutable_inputs,
                                "final_scene_inventory": inventory(scene),
                                "limitations": ["cProfile overhead perturbs wall time and call costs",
                                                "native/Numba kernel internals are opaque to Python cProfile",
                                                "single candidate run supports attribution, not speedup claims"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
