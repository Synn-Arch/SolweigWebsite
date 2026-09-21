#!/usr/bin/env python3
"""Paired full-pipeline harness for the isolated angular-moment candidate.

Dry-run validation is the default.  Execution is intentionally admitted only
after the protocol status is changed to ``reviewed_frozen`` and the reviewed
harness digest is populated.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "reports/characterization/p7_angular_moment_experiment/paired_protocol_v3.json"
DEFAULT_OUTPUT = ROOT / "reports/characterization/p7_angular_moment_experiment/pairs_v1"
REGIME_FIRST = "first_use_empty_private_jit_and_geometry_cache"
REGIME_WARM = "geometry_and_compatible_jit_warm"
FLAGS = ("tmrt", "svf", "kup", "kdown", "lup", "ldown", "shadow", "wbgt", "ta", "wind")
THREAD_ENV = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)
IGNORED_SUFFIXES = {".pyc", ".nbc", ".nbi"}


def digest(path: Path | str) -> str:
    value = Path(path)
    hasher = hashlib.sha256()
    with value.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def value_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _included(path: Path) -> bool:
    return "__pycache__" not in path.parts and path.suffix not in IGNORED_SUFFIXES


def tree_inventory(root: Path | str) -> dict[str, str]:
    base = Path(root)
    if not base.exists():
        return {}
    return {
        str(path.relative_to(base)): digest(path)
        for path in sorted(base.rglob("*"))
        if path.is_file() and _included(path)
    }


def cache_inventory(root: Path | str) -> dict[str, Any]:
    base = Path(root)
    files = {}
    if base.exists():
        for path in sorted(base.rglob("*")):
            if path.is_file():
                relative = str(path.relative_to(base))
                files[relative] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return {
        "exists": base.is_dir(),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files.values()),
        "files": files,
    }


def fixture_inventory(case: dict[str, Any]) -> dict[str, str]:
    scene = ROOT / case["scene"]
    return {name: digest(scene / name) for name in sorted(case["files"])}


def package_environment() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name] = distribution.version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def hardware_inventory() -> dict[str, Any]:
    import psutil

    cpu_model = platform.processor()
    memory = psutil.virtual_memory().total
    if sys.platform == "darwin":
        try:
            cpu_model = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpus": os.cpu_count(),
        "physical_ram_bytes": memory,
        "affinity": sorted(psutil.Process().cpu_affinity()) if hasattr(psutil.Process(), "cpu_affinity") else None,
    }


def _resolve_rule(document: dict[str, Any], location: str) -> dict[str, Any]:
    section, key = location.split("/", 1)
    value = document[section][key]
    if not isinstance(value, dict):
        raise ValueError(f"Comparison rule is not an object: {location}")
    return value


def comparison_rules(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specification = ROOT / protocol["comparison"]["protocol"]
    if digest(specification) != protocol["comparison"]["protocol_sha256"]:
        raise ValueError("comparison_v1 protocol digest mismatch")
    document = json.loads(specification.read_text())
    rules = {
        field: _resolve_rule(document, location)
        for field, location in protocol["comparison"]["rule_sources"].items()
    }
    if set(rules) != set(protocol["matrix"]["output_fields"]):
        raise ValueError("Every output field must map to exactly one frozen comparison_v1 rule")
    return rules


def _load_comparison_utility(protocol: dict[str, Any]):
    path = ROOT / protocol["comparison"]["utility"]
    if digest(path) != protocol["comparison"]["utility_sha256"]:
        raise ValueError("Validated comparison utility digest mismatch")
    spec = importlib.util.spec_from_file_location("p7_validated_comparison_utility", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load comparison utility: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_schedule(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    schedule=[]
    for item in protocol["pair_order"]["frozen_pairs"]:
        schedule.append({"case":item["scene"],"regime":item["regime"],"budget":item["threads"],
                         "repetition":item["repetition"],"order":["candidate" if x=="variant" else x for x in item["order"]]})
    if len(schedule)!=protocol["matrix"]["expected_pair_count"]:raise ValueError("Frozen schedule length mismatch")
    for key in {(x["case"],x["regime"],x["budget"]) for x in schedule}:
        orders=[x["order"][0] for x in schedule if (x["case"],x["regime"],x["budget"])==key]
        if sorted((orders.count("baseline"),orders.count("candidate"))) != [2,3]:raise ValueError(f"Unbalanced subgroup: {key}")
    return schedule


def _validate_fixture(name: str, case: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    from osgeo import gdal

    scene = ROOT / case["scene"]
    kwargs_path = ROOT / case["kwargs"]
    if not scene.is_dir() or not kwargs_path.is_file():
        raise ValueError(f"Fixture {name} paths are missing")
    actual = fixture_inventory(case)
    if actual != case["files"] or digest(kwargs_path) != case["kwargs_sha256"]:
        raise ValueError(f"Fixture {name} differs from its protocol inventory")
    kwargs = json.loads(kwargs_path.read_text())
    for flag in FLAGS:
        if kwargs.get("save_" + flag) is not True:
            raise ValueError(f"Fixture {name} does not enable save_{flag}")
    met_lines = (scene / "met.txt").read_text().splitlines()
    if len(met_lines) - 1 != protocol["matrix"]["timesteps"]:
        raise ValueError(f"Fixture {name} is not a 24-timestep workload")
    gdal.UseExceptions()
    raster = gdal.Open(str(scene / "Building_DSM.tif"))
    shape = [raster.RasterYSize, raster.RasterXSize]
    raster = None
    if shape != case["shape"]:
        raise ValueError(f"Fixture {name} shape mismatch: {shape}")
    return {"files": actual, "kwargs_sha256": digest(kwargs_path), "shape": shape, "timesteps": len(met_lines) - 1}


def validate_candidate(protocol: dict[str, Any], *, run_check: bool = True) -> dict[str, Any]:
    spec=protocol["sources"];source=ROOT/spec["candidate"];manifest_path=ROOT/spec["candidate_manifest"]
    admission_path=ROOT/spec["candidate_report"]
    if digest(manifest_path)!=spec["candidate_manifest_sha256"] or digest(admission_path)!=spec["candidate_report_sha256"]:raise ValueError("Angular candidate evidence digest mismatch")
    manifest=json.loads(manifest_path.read_text());actual=tree_inventory(source)
    if manifest.get("source_identity")!=spec.get("source_identity") or manifest.get("math_profile")!=spec.get("math_profile"):
        raise ValueError("Runtime source identity/math profile is not mapped by the frozen manifest")
    expected={name:item["sha256"] for name,item in manifest["files"].items()}
    if actual!=expected:raise ValueError("Angular candidate differs from complete source manifest")
    admission=json.loads(admission_path.read_text())
    if admission.get("status")!="pass":raise ValueError("Angular candidate admission is not passing")
    return {"source_inventory":actual,"manifest":str(manifest_path),"report":str(admission_path),"body_check":None}


def validate_protocol(protocol_path: Path, *, for_execution: bool = False, run_candidate_check: bool = True) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("schema_version") != 2 or protocol.get("experiment") != "p7_angular_moment":
        raise ValueError("Unsupported protocol schema")
    allowed = {"preexecution_frozen_pending_lead_review", "reviewed_frozen"}
    if protocol.get("status") not in allowed:
        raise ValueError("Invalid protocol status")
    if for_execution and protocol["status"] != "reviewed_frozen":
        raise ValueError("Execution requires root-reviewed protocol status reviewed_frozen")
    matrix = protocol["matrix"]
    expected = {
        "scenes": ["small", "repeated_block_256"],
        "regimes": [REGIME_FIRST, REGIME_WARM],
        "native_budgets": [1, 4],
        "repetitions": 5,
        "expected_pair_count": 40,
        "timesteps": 24,
        "patches": 153,
        "block_pixels": 128,
    }
    for key, value in expected.items():
        if matrix.get(key) != value:
            raise ValueError(f"Protocol matrix mismatch for {key}")
    if len(matrix.get("output_fields", [])) != 10 or len(set(matrix["output_fields"])) != 10:
        raise ValueError("Protocol must require ten unique output fields")
    runtime = protocol["runtime"]
    if runtime["workers"] != 1 or runtime["memory_budget_bytes"] != 12 * 1024**3 or runtime["rss_sample_interval_seconds"] != 0.02:
        raise ValueError("Runtime must use one worker, 12 GiB, and 20 ms process-tree sampling")
    if runtime["cache_enabled"] is not True or runtime["legacy_cache_policy"] != "recompute":
        raise ValueError("Private geometry-cache semantics are not enabled")
    harness_hash = digest(__file__)
    baseline_inventory = tree_inventory(ROOT / protocol["sources"]["baseline_current_src"])
    if for_execution and protocol.get("reviewed_harness_sha256") != harness_hash:
        raise ValueError("Harness digest differs from the root-reviewed binding")
    if for_execution and protocol.get("reviewed_baseline_inventory_sha256") != value_digest(baseline_inventory):
        raise ValueError("Current baseline source differs from the root-reviewed inventory")
    comparison = ROOT / protocol["comparison"]["utility"]
    if digest(comparison) != protocol["comparison"]["utility_sha256"]:
        raise ValueError("Comparison utility differs from protocol binding")
    rules = comparison_rules(protocol)
    fixtures = {name: _validate_fixture(name, protocol["fixtures"][name], protocol) for name in matrix["scenes"]}
    candidate = validate_candidate(protocol, run_check=run_candidate_check)
    schedule = make_schedule(protocol)
    return {
        "protocol": protocol,
        "protocol_sha256": digest(protocol_path),
        "harness_sha256": harness_hash,
        "comparison_utility_sha256": digest(comparison),
        "comparison_rules": rules,
        "fixtures": fixtures,
        "baseline_current_inventory": baseline_inventory,
        "candidate": candidate,
        "schedule": schedule,
    }


def _copy_source(source: Path, destination: Path) -> dict[str, str]:
    before = tree_inventory(source)

    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name == "__pycache__" or Path(name).suffix in IGNORED_SUFFIXES}

    shutil.copytree(source, destination, ignore=ignore)
    after = tree_inventory(source)
    copied = tree_inventory(destination)
    if before != after or copied != before:
        raise RuntimeError(f"Source changed while creating immutable snapshot: {source}")
    for path in sorted(destination.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    destination.chmod(0o555)
    return copied


def create_source_snapshots(root: Path, protocol: dict[str, Any]) -> tuple[dict[str, Path], dict[str, dict[str, str]]]:
    container = root / "frozen_sources"
    container.mkdir()
    sources = {
        "baseline": ROOT / protocol["sources"]["baseline_current_src"],
        "candidate": ROOT / protocol["sources"]["candidate"],
    }
    snapshots: dict[str, Path] = {}
    inventories: dict[str, dict[str, str]] = {}
    for label, source in sources.items():
        destination = container / label / "src"
        destination.parent.mkdir()
        inventories[label] = _copy_source(source, destination)
        snapshots[label] = destination
        write_json(container / label / "source_manifest.json", {
            "schema_version": 1,
            "label": label,
            "copied_from": str(source.resolve()),
            "files": inventories[label],
        })
    admitted=json.loads((ROOT / protocol["sources"]["candidate_manifest"]).read_text())["files"]
    admitted={name:(item["sha256"] if isinstance(item,dict) else item) for name,item in admitted.items()}
    if inventories["candidate"] != admitted:
        raise RuntimeError("Frozen candidate snapshot is not the admitted candidate")
    return snapshots, inventories


def assert_run_guards(guards: dict[str, Any], protocol_path: Path, sources: dict[str, Path], protocol: dict[str, Any]) -> None:
    observed = {
        "protocol_sha256": digest(protocol_path),
        "harness_sha256": digest(__file__),
        "comparison_utility_sha256": digest(ROOT / protocol["comparison"]["utility"]),
        "sources": {name: tree_inventory(path) for name, path in sources.items()},
        "fixtures": {name: fixture_inventory(case) for name, case in protocol["fixtures"].items()},
        "fixture_kwargs": {name: digest(ROOT / case["kwargs"]) for name, case in protocol["fixtures"].items()},
    }
    if observed != guards:
        raise RuntimeError("Protocol, harness, source snapshot, or fixture changed during execution")


def prepare_scene(case: dict[str, Any], destination: Path) -> tuple[Path, dict[str, Any]]:
    source = ROOT / case["scene"]
    destination.mkdir(parents=True)
    for name in sorted(case["files"]):
        shutil.copy2(source / name, destination / name)
    kwargs = json.loads((ROOT / case["kwargs"]).read_text())
    kwargs["base_path"] = str(destination.resolve())
    kwargs["own_met_file"] = str((destination / "met.txt").resolve())
    for key in ("building_dsm_filename", "dem_filename", "trees_filename"):
        kwargs[key] = Path(kwargs[key]).name
    for flag in FLAGS:
        kwargs["save_" + flag] = True
    return destination, kwargs


def runtime_options(protocol: dict[str, Any], scene: Path, budget: int) -> dict[str, Any]:
    runtime = protocol["runtime"]
    return {
        "cpu_budget": budget,
        "workers": 1,
        "threads_per_worker": budget,
        "memory_budget_bytes": runtime["memory_budget_bytes"],
        "block_pixels": protocol["matrix"]["block_pixels"],
        "cache_dir": str(scene / ".runtime_cache"),
        "cache_enabled": True,
        "checkpoint_interval": runtime["checkpoint_interval"],
        "legacy_cache_policy": runtime["legacy_cache_policy"],
        "resume": False,
    }


def warm_kwargs(full_kwargs: dict[str, Any], scene: Path) -> dict[str, Any]:
    result = {
        key: value for key, value in full_kwargs.items()
        if key.startswith("save_") or key in {"base_path", "selected_date_str"}
    }
    result["preprocess_dir"] = str(scene / "processed_inputs")
    result["tile_keys"] = None
    return result


def _child(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text())
    run = Path(spec["process_run"])
    try:
        source = Path(spec["source"]).resolve()
        sys.path.insert(0, str(source))
        import solweig_light

        package = Path(solweig_light.__file__).resolve()
        if not package.is_relative_to(source):
            raise RuntimeError(f"Wrong source imported: {package}; expected under {source}")
        write_json(run / "environment.json", {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "package": str(package),
            "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV},
            "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
            "dependencies": package_environment(),
            "invocation": spec,
        })
        with solweig_light.runtime_options(solweig_light.RuntimeOptions(**spec["runtime"])):
            getattr(solweig_light, spec["entrypoint"])(**spec["kwargs"])
        write_json(run / "outcome.json", {"status": "executed_not_yet_compared"})
        return 0
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "exception_type": type(error).__name__,
            "exception_module": type(error).__module__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(run / "job.failure.json", failure)
        write_json(run / "outcome.json", {"status": "failed", "failure": failure})
        return 1


def process_tree_rss(process: Any) -> int:
    import psutil

    try:
        members = [process, *process.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for member in members:
        try:
            total += member.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def run_monitored(
    command: list[str], run: Path, *, cwd: Path, env: dict[str, str],
    memory_limit: int, sample_interval: float,
) -> dict[str, Any]:
    import psutil

    run.mkdir(parents=True, exist_ok=False)
    samples: list[list[float | int]] = []
    peak = 0
    aborted = False
    started = time.perf_counter()
    with (run / "stdout.log").open("w") as stdout, (run / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
            start_new_session=True,
        )
        root_process = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                rss = process_tree_rss(root_process)
                elapsed = time.perf_counter() - started
                peak = max(peak, rss)
                samples.append([elapsed, rss])
                if rss > memory_limit:
                    aborted = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    break
                time.sleep(sample_interval)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            returncode = process.wait()
    measurement = {
        "command": command,
        "returncode": returncode,
        "elapsed_seconds": time.perf_counter() - started,
        "sampled_process_tree_peak_rss_bytes": peak,
        "memory_limit_bytes": memory_limit,
        "memory_limit_aborted": aborted,
        "sample_interval_seconds": sample_interval,
        "rss_caveat": "Summed process-tree RSS can double-count shared pages and 20 ms sampling can miss shorter peaks.",
    }
    write_json(run / "rss_samples.json", samples)
    write_json(run / "measurement.json", measurement)
    return measurement


def invoke_worker(
    source: Path, entrypoint: str, kwargs: dict[str, Any], runtime: dict[str, Any],
    process_run: Path, jit_dir: Path, protocol: dict[str, Any], python: str,
) -> dict[str, Any]:
    process_run.parent.mkdir(parents=True, exist_ok=True)
    spec = {
        "source": str(source.resolve()),
        "process_run": str(process_run.resolve()),
        "entrypoint": entrypoint,
        "kwargs": kwargs,
        "runtime": runtime,
    }
    spec_path = process_run.parent / (process_run.name + ".spec.json")
    write_json(spec_path, spec)
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(source.resolve()),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "NUMBA_CACHE_DIR": str(jit_dir),
    })
    for key in THREAD_ENV:
        env[key] = str(runtime["threads_per_worker"])
    command = [python, str(Path(__file__).resolve()), "--child", str(spec_path.resolve())]
    return run_monitored(
        command, process_run, cwd=process_run.parent, env=env,
        memory_limit=protocol["runtime"]["memory_budget_bytes"],
        sample_interval=protocol["runtime"]["rss_sample_interval_seconds"],
    )


def failure_artifacts(process_run: Path, scene: Path) -> list[str]:
    found = []
    for root in (process_run, scene):
        if root.exists():
            found.extend(str(path) for path in root.rglob("*.failure.json"))
    return sorted(set(found))


def execution_status(measurement: dict[str, Any], process_run: Path, scene: Path) -> dict[str, Any]:
    failures = failure_artifacts(process_run, scene)
    outcome_path = process_run / "outcome.json"
    outcome = json.loads(outcome_path.read_text()) if outcome_path.is_file() else None
    passed = (
        measurement.get("returncode") == 0
        and measurement.get("memory_limit_aborted") is False
        and outcome == {"status": "executed_not_yet_compared"}
        and not failures
    )
    return {"passed": passed, "outcome": outcome, "failure_artifacts": failures}


def validate_outputs(scene: Path, fields: list[str], bands: int) -> dict[str, Any]:
    from osgeo import gdal

    gdal.UseExceptions()
    output = scene / "output_folder"
    paths = sorted(output.glob("*/*.tif")) if output.is_dir() else []
    records = []
    tiles: dict[str, set[str]] = {}
    passed = bool(paths)
    for path in paths:
        tile = path.parent.name
        field = path.stem.split("_", 1)[0]
        tiles.setdefault(tile, set()).add(field)
        error = None
        try:
            dataset = gdal.Open(str(path))
            observed_bands = dataset.RasterCount if dataset else None
            valid = dataset is not None and observed_bands == bands
        except Exception as exception:
            dataset = None
            observed_bands = None
            valid = False
            error = f"{type(exception).__name__}: {exception}"
        records.append({
            "path": str(path.relative_to(scene)),
            "field": field,
            "tile": tile,
            "bands": observed_bands,
            "passed": valid,
            "error": error,
        })
        passed &= valid
        dataset = None
    expected = set(fields)
    passed &= bool(tiles) and all(values == expected for values in tiles.values())
    passed &= len(paths) == len(tiles) * len(fields)
    return {
        "passed": bool(passed),
        "expected_fields": fields,
        "expected_bands_per_tiff": bands,
        "tiles": {tile: sorted(values) for tile, values in sorted(tiles.items())},
        "records": records,
    }


def retained_geometry_snapshot(scene: Path) -> dict[str, Any]:
    processed = cache_inventory(scene / "processed_inputs")
    native = cache_inventory(scene / ".runtime_cache")
    return {
        "processed_inputs": processed,
        "native_geometry_cache": native,
        "total_bytes": processed["total_bytes"] + native["total_bytes"],
    }


def reset_warm_state(scene: Path) -> dict[str, Any]:
    before_geometry = retained_geometry_snapshot(scene)
    removed = []
    for relative in ("output_folder", ".solweig-light", ".solweig-light-locks"):
        target = scene / relative
        if not target.resolve().is_relative_to(scene.resolve()):
            raise RuntimeError("Warm reset escaped trial scene")
        if target.is_dir():
            shutil.rmtree(target)
            removed.append(relative)
        elif target.exists():
            target.unlink()
            removed.append(relative)
    after_geometry = retained_geometry_snapshot(scene)
    passed = before_geometry == after_geometry and all(not (scene / item).exists() for item in removed)
    return {"passed": passed, "removed": removed, "geometry_before": before_geometry, "geometry_after": after_geometry}


def cache_proof(
    regime: str, jit_before: dict[str, Any], jit_after: dict[str, Any],
    geometry_before: dict[str, Any], geometry_after: dict[str, Any],
) -> dict[str, Any]:
    if regime == REGIME_FIRST:
        passed = (
            jit_before["file_count"] == 0
            and geometry_before["total_bytes"] == 0
            and jit_after["file_count"] > 0
            and geometry_after["total_bytes"] > 0
        )
        reason = "empty private JIT and geometry caches before timed first-use worker; both populated by that worker"
    else:
        passed = (
            jit_before["file_count"] > 0
            and geometry_before["total_bytes"] > 0
            and jit_before == jit_after
            and geometry_before == geometry_after
        )
        reason = "nonempty compatible JIT and geometry cache inventories unchanged by fresh measured worker"
    return {
        "passed": bool(passed),
        "reason": reason,
        "jit_before": jit_before,
        "jit_after": jit_after,
        "geometry_before": geometry_before,
        "geometry_after": geometry_after,
    }


def pair_comparison(
    left_scene: Path, right_scene: Path, protocol: dict[str, Any], rules: dict[str, Any],
) -> dict[str, Any]:
    utility = _load_comparison_utility(protocol)
    result = utility.compare(
        left_scene, right_scene, rules, protocol["comparison"]["artifact_globs"]
    )
    result["evidence_class"] = protocol["evidence_class"]
    result["warmup_only_geometry_artifacts_excluded"] = True
    return result


def summarize_pairs(pairs: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = protocol["matrix"]
    summaries = []
    for case in matrix["scenes"]:
        for regime in matrix["regimes"]:
            for budget in matrix["native_budgets"]:
                group = [item for item in pairs if item["case"] == case and item["regime"] == regime and item["budget"] == budget]
                valid = [item for item in group if item.get("passed")]
                subgroup_passed = len(group) == matrix["repetitions"] and len(valid) == matrix["repetitions"]
                ratios = [item["candidate_seconds"] / item["baseline_seconds"] for item in valid]
                summaries.append({
                    "case": case,
                    "regime": regime,
                    "budget": budget,
                    "pair_count": len(group),
                    "valid_pairs": len(valid),
                    "failed_pairs": len(group) - len(valid),
                    "subgroup_passed": subgroup_passed,
                    "performance_claim_eligible": subgroup_passed,
                    "raw_paired_ratios": ratios,
                    "median_ratio": statistics.median(ratios) if subgroup_passed else None,
                    "range_ratio": [min(ratios), max(ratios)] if subgroup_passed else None,
                })
    return summaries


def angular_acceptance(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    all_medians=len(summaries)==8 and all(item.get("subgroup_passed") is True and item.get("median_ratio") is not None and item["median_ratio"] <= 1.0 for item in summaries)
    priority=[item for item in summaries if item["case"]=="repeated_block_256" and item["regime"]==REGIME_WARM]
    priority_pass=len(priority)==2 and all(item.get("subgroup_passed") is True and item.get("median_ratio") is not None and item["median_ratio"]<=.970 and sum(r<1 for r in item.get("raw_paired_ratios",()))>=4 for item in priority)
    return {"ratio_definition":"candidate_elapsed / baseline_elapsed","all_eight_medians_le_1":all_medians,
            "priority_256_warm_median_le_0_970_and_four_of_five_lt_1":priority_pass,
            "passed":bool(all_medians and priority_pass),"engineering_rule_not_significance_claim":True}


def execute(protocol_path: Path, output: Path, validation: dict[str, Any], python: str) -> int:
    protocol = validation["protocol"]
    if output.exists():
        raise FileExistsError(f"Run output already exists: {output}")
    if shutil.disk_usage(ROOT).free < protocol["runtime"]["minimum_free_disk_bytes"]:
        raise RuntimeError("Insufficient free disk headroom for the 40-pair retained-artifact matrix")
    output.mkdir(parents=True)
    sources, source_inventories = create_source_snapshots(output, protocol)
    if source_inventories["baseline"] != validation["baseline_current_inventory"]:
        raise RuntimeError("Current baseline changed between protocol validation and immutable snapshot creation")
    guards = {
        "protocol_sha256": digest(protocol_path),
        "harness_sha256": digest(__file__),
        "comparison_utility_sha256": digest(ROOT / protocol["comparison"]["utility"]),
        "sources": source_inventories,
        "fixtures": {name: fixture_inventory(case) for name, case in protocol["fixtures"].items()},
        "fixture_kwargs": {name: digest(ROOT / case["kwargs"]) for name, case in protocol["fixtures"].items()},
    }
    frozen = {
        "schema_version": 1,
        "status": "frozen_before_execution",
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": digest(protocol_path),
        "evidence_class": protocol["evidence_class"],
        "guards": guards,
        "schedule": validation["schedule"],
        "hardware": hardware_inventory(),
        "environment": {"python": sys.version, "executable": python, "packages": package_environment()},
        "candidate_admission": validation["candidate"],
    }
    write_json(output / "frozen.json", frozen)
    trials: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    rules = validation["comparison_rules"]
    for scheduled in validation["schedule"]:
        assert_run_guards(guards, protocol_path, sources, protocol)
        pair_root = output / "pairs" / scheduled["case"] / scheduled["regime"] / f"n{scheduled['budget']}" / f"r{scheduled['repetition']}"
        pair: dict[str, dict[str, Any]] = {}
        for backend in scheduled["order"]:
            assert_run_guards(guards, protocol_path, sources, protocol)
            trial_root = pair_root / backend
            scene, full_kwargs = prepare_scene(protocol["fixtures"][scheduled["case"]], trial_root / "scene")
            jit = trial_root / "private_jit_cache"
            geometry_empty = retained_geometry_snapshot(scene)
            jit_empty = cache_inventory(jit)
            runtime = runtime_options(protocol, scene, scheduled["budget"])
            setup = None
            setup_status = None
            setup_outputs = None
            reset = None
            if scheduled["regime"] == REGIME_WARM:
                setup_run = trial_root / "setup_process"
                setup = invoke_worker(sources[backend], "thermal_comfort", full_kwargs, runtime, setup_run, jit, protocol, python)
                setup_status = execution_status(setup, setup_run, scene)
                setup_outputs = validate_outputs(scene, protocol["matrix"]["output_fields"], protocol["matrix"]["timesteps"])
                setup_jit = cache_inventory(jit)
                setup_geometry = retained_geometry_snapshot(scene)
                if setup_status["passed"] and setup_outputs["passed"]:
                    reset = reset_warm_state(scene)
                else:
                    reset = {"passed": False, "reason": "setup execution or setup outputs failed"}
                jit_before = cache_inventory(jit)
                geometry_before = retained_geometry_snapshot(scene)
                entrypoint = "run_utci_tiles"
                kwargs = warm_kwargs(full_kwargs, scene)
            else:
                setup_jit = None
                setup_geometry = None
                jit_before = jit_empty
                geometry_before = geometry_empty
                entrypoint = "thermal_comfort"
                kwargs = full_kwargs
            measurement_run = trial_root / "measurement_process"
            if scheduled["regime"] == REGIME_WARM and not reset["passed"]:
                measurement = {"returncode": None, "memory_limit_aborted": False, "not_started": "warm setup/reset failed"}
                status = {"passed": False, "outcome": None, "failure_artifacts": failure_artifacts(trial_root / "setup_process", scene)}
            else:
                assert_run_guards(guards, protocol_path, sources, protocol)
                measurement = invoke_worker(sources[backend], entrypoint, kwargs, runtime, measurement_run, jit, protocol, python)
                status = execution_status(measurement, measurement_run, scene)
            outputs = validate_outputs(scene, protocol["matrix"]["output_fields"], protocol["matrix"]["timesteps"])
            jit_after = cache_inventory(jit)
            geometry_after = retained_geometry_snapshot(scene)
            proof = cache_proof(scheduled["regime"], jit_before, jit_after, geometry_before, geometry_after)
            trial_passed = status["passed"] and outputs["passed"] and proof["passed"]
            if scheduled["regime"] == REGIME_WARM:
                trial_passed &= bool(setup_status["passed"] and setup_outputs["passed"] and reset["passed"])
            record = {
                **scheduled,
                "backend": backend,
                "scene": str(scene),
                "source": str(sources[backend]),
                "setup": setup,
                "setup_status": setup_status,
                "setup_output_schema": setup_outputs,
                "setup_cache_accounting": {
                    "jit": setup_jit,
                    "geometry": setup_geometry,
                } if setup is not None else None,
                "warm_reset": reset,
                "measurement": measurement,
                "execution_status": status,
                "output_schema": outputs,
                "cache_proof": proof,
                "passed_before_pair_comparison": bool(trial_passed),
            }
            trials.append(record)
            pair[backend] = record
            write_json(output / "trials.json", trials)
        both_executed = all(pair[name]["passed_before_pair_comparison"] for name in ("baseline", "candidate"))
        if both_executed:
            try:
                comparison = pair_comparison(Path(pair["baseline"]["scene"]), Path(pair["candidate"]["scene"]), protocol, rules)
            except BaseException:
                comparison = {
                    "passed": False,
                    "comparison_failed": True,
                    "traceback": traceback.format_exc(),
                    "warmup_only_geometry_artifacts_excluded": True,
                }
        else:
            comparison = {"passed": False, "execution_failed": True, "warmup_only_geometry_artifacts_excluded": True}
        write_json(pair_root / "comparison.json", comparison)
        passed = both_executed and comparison["passed"]
        pairs.append({
            **scheduled,
            "passed": bool(passed),
            "comparison": comparison,
            "baseline_seconds": pair["baseline"]["measurement"].get("elapsed_seconds"),
            "candidate_seconds": pair["candidate"]["measurement"].get("elapsed_seconds"),
        })
        write_json(output / "pairs.json", pairs)
    assert_run_guards(guards, protocol_path, sources, protocol)
    summaries = summarize_pairs(pairs, protocol)
    acceptance=angular_acceptance(summaries)
    complete = len(pairs) == protocol["matrix"]["expected_pair_count"] and all(item["subgroup_passed"] for item in summaries) and acceptance["passed"]
    write_json(output / "summary.json", {
        "status": "passed" if complete else "failed",
        "evidence_class": protocol["evidence_class"],
        "pair_count": len(pairs),
        "all_source_fixture_harness_guards_passed": True,
        "performance_claim_eligible": bool(complete),
        "frozen_acceptance": acceptance,
        "summaries": summaries,
        "limitations": [
            "OS page cache is uncontrolled.",
            "Process-tree RSS is sampled and may double-count shared pages or miss sub-20-ms peaks.",
            "This compares two candidate source snapshots and is not upstream speedup evidence.",
        ],
    })
    return 0 if complete else 1


def dry_run_report(protocol_path: Path, output: Path, validation: dict[str, Any], python: str) -> dict[str, Any]:
    protocol = validation["protocol"]
    ready_command = [
        python,
        str(Path(__file__).resolve()),
        "--protocol", str(protocol_path.resolve()),
        "--output", str(output.resolve()),
        "--execute",
    ]
    return {
        "status": "validated_not_executed",
        "protocol_status": protocol["status"],
        "execution_admitted": (
            protocol["status"] == "reviewed_frozen"
            and protocol.get("reviewed_harness_sha256") == validation["harness_sha256"]
            and protocol.get("reviewed_baseline_inventory_sha256")
            == value_digest(validation["baseline_current_inventory"])
        ),
        "review_binding_required": {
            "status": "reviewed_frozen",
            "reviewed_harness_sha256": validation["harness_sha256"],
            "reviewed_baseline_inventory_sha256": value_digest(validation["baseline_current_inventory"]),
        },
        "protocol_sha256": validation["protocol_sha256"],
        "harness_sha256": validation["harness_sha256"],
        "baseline_current_inventory": validation["baseline_current_inventory"],
        "candidate_admission": validation["candidate"],
        "fixture_admission": validation["fixtures"],
        "schedule": validation["schedule"],
        "pair_count": len(validation["schedule"]),
        "ready_command_after_root_review": ready_command,
        "execution_semantics": {
            "first_use": "new scene, empty private JIT cache, empty private geometry cache, full thermal_comfort timed",
            "warm": "full setup accounted separately, output/transaction/state reset, fresh measured run_utci_tiles worker, unchanged retained JIT and geometry inventories required",
            "time_parallelism": False,
            "comparison": "ten 24-band TIFFs only; setup-only SVF sidecars excluded from paired output comparison",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--child", type=Path)
    args = parser.parse_args(argv)
    if args.child is not None:
        return _child(args.child)
    protocol_path = args.protocol.resolve()
    validation = validate_protocol(protocol_path, for_execution=args.execute)
    if not args.execute:
        report = dry_run_report(protocol_path, args.output, validation, args.python)
        report_path = args.output.parent / "dryrun_plan.json"
        write_json(report_path, report)
        print(json.dumps({
            "status": report["status"],
            "plan": str(report_path),
            "pair_count": report["pair_count"],
            "review_binding_required": report["review_binding_required"],
            "ready_command_after_root_review": report["ready_command_after_root_review"],
        }, indent=2))
        return 0
    try:
        return execute(protocol_path, args.output.resolve(), validation, args.python)
    except BaseException:
        if args.output.exists():
            write_json(args.output / "harness.failure.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
