#!/usr/bin/env python3
"""Characterize the integrated P6 runtime on the prepared small fixture.

The harness is deliberately an instrument, not a benchmark claim generator.
It records process-tree RSS, elapsed process lifetime, output schemas, and
failures.  It will not execute unless both ``--source-go`` and ``--execute``
are supplied after the runtime integration has been reviewed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks/protocols/p6_runtime_v1.json"
DEFAULT_FIXTURE = ROOT / "tests/reference/small_original_cpu/scene"
DEFAULT_OUTPUT = ROOT / "reports/characterization/p6_runtime_v1"
MEMORY_BUDGET = 4 * 1024**3
SAMPLE_INTERVAL = 0.02
SAVE_FLAGS = ("tmrt", "svf", "kup", "kdown", "lup", "ldown", "shadow", "wbgt", "ta", "wind")
OUTPUT_NAMES = ("UTCI", "TMRT", "Kup", "Kdown", "Lup", "Ldown", "Shadow", "WBGT", "Ta", "Wind")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def candidate_source_hashes() -> dict[str, str]:
    """Hash candidate source/config files used by the measured process."""
    files = []
    for base in (ROOT / "src", ROOT / "compat"):
        if not base.is_dir():
            continue
        files.extend(
            path for path in base.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json", ".txt"}
        )
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(files)}


def assert_source_unchanged(expected: dict[str, str], run_dir: Path) -> None:
    observed = candidate_source_hashes()
    write_json(run_dir / "candidate_source_hashes_after.json", observed)
    if observed != expected:
        raise RuntimeError("candidate source/configuration changed during P6 characterization")


def protocol_hash() -> str:
    return sha256(PROTOCOL)


def candidate_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "uncommitted repository without HEAD"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_met(scene: Path) -> tuple[str, np.ndarray]:
    path = scene / "met.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"forcing file has no data rows: {path}")
    values = np.loadtxt(path, skiprows=1, delimiter=" ")
    if values.shape != (24, 24):
        raise ValueError(f"expected the 24x24 small fixture forcing, got {values.shape}")
    return lines[0], values


def extend_met(scene: Path, cycles: int, seed_scene: Path | None = None) -> dict[str, object]:
    """Repeat the 24-hour forcing while advancing calendar fields exactly."""
    header, source = load_met(seed_scene or scene)
    rows = []
    for cycle in range(cycles):
        for value in source:
            year = int(round(value[0]))
            day = int(round(value[1]))
            hour = int(round(value[2]))
            minute = int(round(value[3]))
            instant = dt.datetime(year, 1, 1) + dt.timedelta(days=day - 1, hours=hour, minutes=minute)
            instant += dt.timedelta(days=cycle)
            item = value.copy()
            item[:4] = (instant.year, instant.timetuple().tm_yday, instant.hour, instant.minute)
            rows.append(item)
    extended = np.asarray(rows, dtype=np.float64)
    if extended.shape != (24 * cycles, 24):
        raise AssertionError(f"unexpected extended forcing shape: {extended.shape}")
    for relative in (Path("met.txt"), Path("processed_inputs/metfiles/metfile_0_0.txt")):
        target = scene / relative
        if target.is_file():
            np.savetxt(target, extended, fmt="%.8f", header=header, comments="")
    return {
        "source_rows": 24,
        "rows": int(extended.shape[0]),
        "cycles": cycles,
        "first_timestamp": [int(v) for v in extended[0, :4]],
        "last_timestamp": [int(v) for v in extended[-1, :4]],
        "day_fields": sorted({int(v) for v in extended[:, 1]}),
    }


def duplicate_prepared_tile(scene: Path) -> None:
    """Add a second independent copy of prepared tile 0_0 as tile 0_1."""
    prepared = scene / "processed_inputs"
    for directory in sorted(prepared.iterdir()):
        if not directory.is_dir():
            continue
        for source in sorted(directory.glob("*0_0*")):
            target = source.with_name(source.name.replace("0_0", "0_1"))
            if not target.exists():
                shutil.copy2(source, target)


def remove_known_outputs(scene: Path) -> None:
    for relative in ("output_folder", ".solweig-light/transactions", ".solweig-light-locks", "p6_child_outcome.json"):
        target = scene / relative
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()


def native_cache_manifest(cache_dir: Path) -> Path | None:
    manifests = sorted(cache_dir.rglob("manifest.json")) if cache_dir.is_dir() else []
    return manifests[0] if manifests else None


def prepare_scene(source: Path, target: Path, cycles: int, cold: bool, two_tiles: bool, seed_scene: Path | None = None) -> dict[str, object]:
    shutil.copytree(source, target)
    remove_known_outputs(target)
    forcing = extend_met(target, cycles, seed_scene=seed_scene)
    if two_tiles:
        duplicate_prepared_tile(target)
    cache_dir = target / ".runtime_cache"
    if cold:
        for relative in ("processed_inputs/SVF", ".runtime_cache"):
            path = target / relative
            if path.exists():
                shutil.rmtree(path)
    return {
        "forcing": forcing,
        "cold_geometry": cold,
        "two_tiles": two_tiles,
        "cache_dir": str(cache_dir),
        "native_cache_manifest_before": str(native_cache_manifest(cache_dir)) if native_cache_manifest(cache_dir) else None,
    }


def completion_artifacts(scene: Path, expected_tiles: tuple[str, ...]) -> dict[str, object]:
    """Validate durable completion manifests and their artifact hashes."""
    records = []
    errors = []
    transaction_root = scene / ".solweig-light" / "transactions"
    for path in sorted(transaction_root.rglob("complete.json")) if transaction_root.is_dir() else []:
        item: dict[str, object] = {"path": str(path.relative_to(scene))}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            signature = payload["signature"]
            identity = signature["identity"]
            artifacts = payload["artifacts"]
            if not identity or not isinstance(artifacts, list) or not artifacts:
                raise ValueError("completion manifest lacks identity or artifacts")
            checked = []
            for artifact in artifacts:
                final = Path(artifact["final"])
                digest = artifact["sha256"]
                if not final.is_file() or sha256(final) != digest:
                    raise ValueError(f"missing or hash-mismatched artifact: {final}")
                checked.append({"final": str(final), "sha256": digest})
            item.update({"identity": identity, "signature": signature, "artifacts": checked, "valid": True})
        except Exception as error:
            item.update({"valid": False, "error": repr(error)})
            errors.append(str(error))
        records.append(item)
    manifest_tiles = []
    for record in records:
        for artifact in record.get("artifacts") or []:
            final = Path(artifact["final"])
            if final.parent.parent.name == "output_folder":
                manifest_tiles.append(final.parent.name)
    if sorted(set(manifest_tiles)) != sorted(expected_tiles):
        errors.append(f"completion manifests cover tiles {sorted(set(manifest_tiles))}, expected {list(expected_tiles)}")
    return {"manifests": records, "valid": len(records) == len(expected_tiles) and not errors, "errors": errors}


def output_schema(scene: Path, cycles: int, two_tiles: bool) -> dict[str, object]:
    output = scene / "output_folder"
    expected_tiles = ("0_0", "0_1") if two_tiles else ("0_0",)
    expected_names = set(OUTPUT_NAMES)
    allowed_names = expected_names | {"SVF"}
    tile_records = []
    errors = []
    for tile in expected_tiles:
        directory = output / tile
        files = sorted(directory.glob("*.tif")) if directory.is_dir() else []
        by_name = {}
        for path in files:
            prefix = path.stem[: -len("_" + tile)] if path.stem.endswith("_" + tile) else path.stem
            by_name[prefix] = path
        observed_names = set(by_name)
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - allowed_names)
        records = []
        for name, path in sorted(by_name.items()):
            record: dict[str, object] = {"name": name, "path": str(path.relative_to(scene)), "bytes": path.stat().st_size}
            try:
                from osgeo import gdal

                dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
                if dataset is None:
                    raise RuntimeError("GDAL could not open output")
                expected_bands = 1 if name == "SVF" else 24 * cycles
                times = [dataset.GetRasterBand(i).GetMetadataItem("Time") for i in range(1, dataset.RasterCount + 1)]
                record.update({"rows": dataset.RasterYSize, "cols": dataset.RasterXSize,
                               "bands": dataset.RasterCount, "expected_bands": expected_bands,
                               "time_metadata": times})
                if (dataset.RasterYSize, dataset.RasterXSize) != (32, 35):
                    raise ValueError(f"expected 32x35, got {dataset.RasterYSize}x{dataset.RasterXSize}")
                if dataset.RasterCount != expected_bands:
                    raise ValueError(f"expected {expected_bands} bands, got {dataset.RasterCount}")
                if name != "SVF" and len(times) != expected_bands or name != "SVF" and any(not value for value in times):
                    raise ValueError("missing per-band Time metadata")
                dataset = None
                record["valid"] = True
            except Exception as error:
                record.update({"valid": False, "schema_error": repr(error)})
                errors.append(f"{tile}/{name}: {error}")
            records.append(record)
        tile_records.append({"tile": tile, "expected_fields": sorted(expected_names), "observed_fields": sorted(observed_names),
                             "missing_fields": missing, "unexpected_fields": unexpected, "records": records,
                             "complete": not missing and not unexpected and not errors})
    completion = completion_artifacts(scene, expected_tiles)
    return {
        "expected_tiles": list(expected_tiles),
        "expected_outputs": list(OUTPUT_NAMES),
        "timeline_rows": 24 * cycles,
        "tiles": tile_records,
        "completion_artifacts": completion,
        "complete": bool(tile_records) and all(item["complete"] for item in tile_records) and completion["valid"] and not errors,
        "errors": errors,
    }


def environment_record(options: dict[str, object], source_hashes: dict[str, str] | None = None) -> dict[str, object]:
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()},
        "thread_environment": {name: os.environ.get(name) for name in names},
        "runtime_options": options,
        "candidate_commit": candidate_revision(),
        "protocol_sha256": protocol_hash(),
        "harness_sha256": sha256(Path(__file__).resolve()),
        "candidate_source_hashes": source_hashes or candidate_source_hashes(),
    }


def child(args: argparse.Namespace) -> int:
    scene = args.scene.resolve()
    options = {
        "cache_dir": str(scene / ".runtime_cache"),
        "legacy_cache_policy": "recompute",
        "cache_enabled": True,
        "memory_budget_bytes": MEMORY_BUDGET,
        "cpu_budget": args.workers,
        "workers": args.workers,
        "threads_per_worker": 1,
        "block_pixels": 128,
        "checkpoint_interval": 1,
        "resume": False,
    }
    outcome: dict[str, object] = {
        "status": "failed",
        "case": args.case_name,
        "scene": str(scene),
        "environment": environment_record(options),
    }
    start = time.perf_counter()
    try:
        from solweig_light import run_utci_tiles
        from solweig_light.runtime import RuntimeOptions, runtime_options

        flags = {f"save_{name}": True for name in SAVE_FLAGS}
        with runtime_options(RuntimeOptions(**options)):
            run_utci_tiles(
                base_path=str(scene),
                preprocess_dir=str(scene / "processed_inputs"),
                selected_date_str="2020-07-18",
                tile_keys=None,
                **flags,
            )
        outcome["status"] = "completed"
    except BaseException:
        outcome["traceback"] = traceback.format_exc()
    outcome["elapsed_seconds_child"] = time.perf_counter() - start
    write_json(scene / "p6_child_outcome.json", outcome)
    return 0 if outcome["status"] == "completed" else 1


def monitor(command: list[str], run_dir: Path, env: dict[str, str]) -> dict[str, object]:
    import psutil

    stdout_path, stderr_path = run_dir / "stdout.log", run_dir / "stderr.log"
    start = time.perf_counter()
    peak = 0
    samples = 0
    exceeded = False
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr)
        root = psutil.Process(process.pid)
        while process.poll() is None:
            rss = 0
            try:
                members = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                members = []
            for member in members:
                try:
                    rss += member.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            peak = max(peak, rss)
            samples += 1
            if rss > MEMORY_BUDGET and not exceeded:
                exceeded = True
                process.terminate()
            time.sleep(SAMPLE_INTERVAL)
        code = process.wait()
    return {
        "command": command,
        "exit_code": code,
        "elapsed_seconds_including_imports": time.perf_counter() - start,
        "sampled_process_tree_peak_rss_bytes": peak,
        "sample_count": samples,
        "sample_interval_seconds": SAMPLE_INTERVAL,
        "memory_budget_bytes": MEMORY_BUDGET,
        "memory_budget_exceeded": exceeded,
        "rss_caveat": "Summed resident sets can count shared pages more than once; between-sample peaks can be missed.",
        "measurement_class": "P6 characterization; no release benchmark or speed claim",
    }


def run_case(args: argparse.Namespace, source: Path, root_output: Path, cycles: int, cold: bool, workers: int,
             two_tiles: bool, seed_scene: Path, source_hashes: dict[str, str]) -> dict[str, object]:
    suffix = "_two_tiles" if two_tiles else ""
    name = f"timeline_{24 * cycles}h_{'cold' if cold else 'warm'}_workers{workers}{suffix}"
    run_dir = root_output / name
    run_dir.mkdir(parents=True, exist_ok=False)
    scene = run_dir / "scene"
    preparation = prepare_scene(source, scene, cycles, cold, two_tiles, seed_scene=seed_scene)
    write_json(run_dir / "fixture_hashes.json", tree_hashes(scene))
    write_json(run_dir / "preparation.json", preparation)
    options = {
        "workers": workers,
        "threads_per_worker": 1,
        "cpu_budget": workers,
        "memory_budget_bytes": MEMORY_BUDGET,
        "block_pixels": 128,
        "patches": 153,
        "windchannels": 12,
    }
    write_json(run_dir / "environment.json", environment_record(options, source_hashes))
    command = [sys.executable, str(Path(__file__).resolve()), "--child", "--scene", str(scene), "--case-name", name, "--workers", str(workers)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=os.pathsep.join((str(ROOT / "src"), os.environ.get("PYTHONPATH", ""))))
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[variable] = "1"
    measurement = monitor(command, run_dir, env)
    assert_source_unchanged(source_hashes, run_dir)
    write_json(run_dir / "measurement.json", measurement)
    child_outcome = run_dir / "scene/p6_child_outcome.json"
    outcome = json.loads(child_outcome.read_text(encoding="utf-8")) if child_outcome.is_file() else {"status": "missing_child_outcome"}
    schema = output_schema(scene, cycles, two_tiles)
    if outcome.get("status") == "completed" and not schema["complete"]:
        outcome["status"] = "completed_with_output_schema_failure"
    write_json(run_dir / "outcome.json", outcome)
    write_json(run_dir / "output_schema.json", schema)
    return {"name": name, "status": outcome.get("status"), "run": str(run_dir.relative_to(ROOT)), "measurement": measurement}


def execute(args: argparse.Namespace) -> int:
    source = args.fixture.resolve()
    if not source.is_dir():
        raise SystemExit(f"fixture directory does not exist: {source}")
    root_output = args.output.resolve()
    root_output.mkdir(parents=True, exist_ok=False)
    source_hashes = candidate_source_hashes()
    write_json(root_output / "protocol.json", json.loads(PROTOCOL.read_text(encoding="utf-8")))
    write_json(root_output / "provenance.json", {
        "protocol_sha256": protocol_hash(),
        "harness_sha256": sha256(Path(__file__).resolve()),
        "fixture_root": str(source),
        "fixture_hashes": tree_hashes(source),
        "candidate_commit": candidate_revision(),
        "candidate_source_hashes": source_hashes,
        "execution_started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    summaries = []
    for cycles in (1, 4, 12):
        summaries.append(run_case(args, source, root_output, cycles, cold=True, workers=1, two_tiles=False,
                                  seed_scene=source, source_hashes=source_hashes))
        # Warm execution is only valid after the cold run creates a verifiable
        # native cache manifest.  This harness records unavailable otherwise.
        cold_scene = root_output / f"timeline_{24 * cycles}h_cold_workers1/scene"
        if native_cache_manifest(cold_scene / ".runtime_cache") is None:
            summaries.append({"name": f"timeline_{24 * cycles}h_warm_workers1", "status": "not_available_native_cache_manifest"})
        else:
            summaries.append(run_case(args, cold_scene, root_output, cycles, cold=False, workers=1, two_tiles=False,
                                      seed_scene=source, source_hashes=source_hashes))
    for workers in (1, 2):
        summaries.append(run_case(args, source, root_output, 1, cold=True, workers=workers, two_tiles=True,
                                  seed_scene=source, source_hashes=source_hashes))
    if candidate_source_hashes() != source_hashes:
        raise RuntimeError("candidate source/configuration changed during P6 characterization")
    write_json(root_output / "summary.json", {"status": "characterization_complete_with_limits", "cases": summaries})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-go", action="store_true", help="Confirm reviewed runtime integration is ready")
    parser.add_argument("--execute", action="store_true", help="Run the frozen characterization")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scene", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--case-name", default="child", help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        if args.scene is None:
            parser.error("--child requires --scene")
        return child(args)
    if args.execute and not args.source_go:
        parser.error("--execute requires explicit --source-go after runtime integration review")
    if not args.execute:
        print(json.dumps({"status": "protocol_authored_not_executed", "protocol": str(PROTOCOL), "protocol_sha256": protocol_hash()}, indent=2))
        return 0
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
