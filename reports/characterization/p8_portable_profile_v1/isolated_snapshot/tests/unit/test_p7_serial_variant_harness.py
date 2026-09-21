"""Executable semantics for the persistent-JIT paired harness."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from osgeo import gdal


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "p7_serial_variant_harness", ROOT / "tools/run_p7_serial_variant.py"
)
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)
PROTOCOL_PATH = ROOT / "benchmarks/protocols/p7_persistent_jit_serial_variant/protocol.json"


def protocol():
    return json.loads(PROTOCOL_PATH.read_text())


def test_schedule_is_seeded_complete_and_sequentially_paired():
    value = protocol()
    first = HARNESS.make_schedule(value)
    second = HARNESS.make_schedule(value)
    assert first == second
    assert len(first) == 40
    assert {tuple(item["order"]) for item in first} == {
        ("baseline", "candidate"), ("candidate", "baseline")
    }
    keys = {
        (item["case"], item["regime"], item["budget"], item["repetition"])
        for item in first
    }
    assert len(keys) == 40


def test_protocol_resolves_every_rule_from_unchanged_comparison_v1():
    value = protocol()
    rules = HARNESS.comparison_rules(value)
    assert set(rules) == set(value["matrix"]["output_fields"])
    assert rules["Shadow"]["rule"] == "exact_value_and_masks"
    assert rules["TMRT"]["max_abs"] == 0.01
    assert rules["UTCI"]["max_abs"] == 0.02
    assert rules["Kdown"] == {"unit": "W m-2", "atol": 0.05, "rtol": 1e-5}


def test_full_dry_validation_admits_distinct_candidate_and_two_fixtures():
    result = HARNESS.validate_protocol(PROTOCOL_PATH, run_candidate_check=True)
    assert len(result["schedule"]) == 40
    assert result["candidate"]["body_check"]["returncode"] == 0
    assert result["fixtures"]["small"]["shape"] == [32, 35]
    assert result["fixtures"]["repeated_block_256"]["shape"] == [256, 256]
    assert all(item["timesteps"] == 24 for item in result["fixtures"].values())


def test_execution_requires_review_status_and_bound_harness(tmp_path):
    value = protocol()
    value["status"] = "draft_for_root_review"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="reviewed_frozen"):
        HARNESS.validate_protocol(path, for_execution=True, run_candidate_check=False)
    value["status"] = "reviewed_frozen"
    value["reviewed_harness_sha256"] = "wrong"
    value["reviewed_baseline_inventory_sha256"] = "wrong"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Harness digest"):
        HARNESS.validate_protocol(path, for_execution=True, run_candidate_check=False)


def test_source_snapshot_is_content_identical_and_read_only(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("VALUE = 1\n")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "module.pyc").write_bytes(b"ignored")
    destination = tmp_path / "snapshot"
    inventory = HARNESS._copy_source(source, destination)
    assert inventory == {"module.py": HARNESS.digest(source / "module.py")}
    assert HARNESS.tree_inventory(destination) == inventory
    assert not (destination / "__pycache__").exists()
    assert destination.stat().st_mode & 0o222 == 0
    assert (destination / "module.py").stat().st_mode & 0o222 == 0


def test_prepare_scene_copies_only_declared_raw_inputs(tmp_path):
    case = protocol()["fixtures"]["small"]
    scene, kwargs = HARNESS.prepare_scene(case, tmp_path / "scene")
    assert {path.name for path in scene.iterdir()} == set(case["files"])
    assert not (scene / "processed_inputs").exists()
    assert not (scene / "output_folder").exists()
    assert kwargs["base_path"] == str(scene.resolve())
    assert kwargs["own_met_file"] == str((scene / "met.txt").resolve())
    assert all(kwargs["save_" + flag] is True for flag in HARNESS.FLAGS)


def test_run_monitored_executes_command_and_records_process_tree(tmp_path):
    run = tmp_path / "success"
    result = HARNESS.run_monitored(
        [sys.executable, "-c", "print('worker-ok')"], run,
        cwd=tmp_path, env=dict(**__import__("os").environ),
        memory_limit=1024**3, sample_interval=0.005,
    )
    assert result["returncode"] == 0
    assert result["memory_limit_aborted"] is False
    assert (run / "stdout.log").read_text().strip() == "worker-ok"
    assert json.loads((run / "measurement.json").read_text())["returncode"] == 0


def test_run_monitored_kills_process_tree_over_memory_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(HARNESS, "process_tree_rss", lambda _process: 2)
    result = HARNESS.run_monitored(
        [sys.executable, "-c", "import time; time.sleep(5)"], tmp_path / "limited",
        cwd=tmp_path, env=dict(**__import__("os").environ),
        memory_limit=1, sample_interval=0.001,
    )
    assert result["memory_limit_aborted"] is True
    assert result["returncode"] != 0
    assert result["sampled_process_tree_peak_rss_bytes"] == 2


def test_return_code_outcome_and_job_failure_are_all_required(tmp_path):
    run = tmp_path / "process"
    scene = tmp_path / "scene"
    run.mkdir(); scene.mkdir()
    (run / "outcome.json").write_text('{"status":"executed_not_yet_compared"}')
    measurement = {"returncode": 0, "memory_limit_aborted": False}
    assert HARNESS.execution_status(measurement, run, scene)["passed"]
    (run / "job.failure.json").write_text("{}")
    assert not HARNESS.execution_status(measurement, run, scene)["passed"]
    (run / "job.failure.json").unlink()
    assert not HARNESS.execution_status({**measurement, "returncode": 1}, run, scene)["passed"]


def test_child_failure_is_retained_as_job_failure(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "src"))
    spec = {
        "source": str(ROOT / "src"),
        "process_run": str(tmp_path),
        "entrypoint": "entrypoint_that_does_not_exist",
        "kwargs": {},
        "runtime": {},
    }
    path = tmp_path / "job.spec.json"
    path.write_text(json.dumps(spec))
    assert HARNESS._child(path) == 1
    failure = json.loads((tmp_path / "job.failure.json").read_text())
    assert failure["exception_type"] == "AttributeError"
    assert json.loads((tmp_path / "outcome.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("fails", [False, True])
def test_invoked_child_records_outcome_in_its_process_directory(tmp_path, fails):
    # Instrumentation-only fake package: exercise the real parent/child boundary.
    source = tmp_path / "source"
    package = source / "solweig_light"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from contextlib import nullcontext\n"
        "def RuntimeOptions(**kwargs): return kwargs\n"
        "def runtime_options(options): return nullcontext()\n"
        "def workflow(fails=False):\n"
        "    if fails: raise ValueError('instrumented failure')\n"
    )
    scene = tmp_path / "scene"
    scene.mkdir()
    run = tmp_path / "trial" / "measurement_process"
    result = HARNESS.invoke_worker(
        source, "workflow", {"fails": fails}, {"threads_per_worker": 1},
        run, tmp_path / "jit", protocol(), sys.executable,
    )
    status = HARNESS.execution_status(result, run, scene)
    assert status["passed"] is (not fails)
    assert (run / "environment.json").is_file()
    assert (run / "outcome.json").is_file()
    assert (run / "job.failure.json").exists() is fails
    assert not (run.parent / "outcome.json").exists()


def test_warm_reset_preserves_geometry_and_removes_only_state(tmp_path):
    scene = tmp_path / "scene"
    for relative in ("processed_inputs/SVF", ".runtime_cache/key", "output_folder/0_0", ".solweig-light/transactions", ".solweig-light-locks"):
        (scene / relative).mkdir(parents=True)
    (scene / "processed_inputs/SVF/geometry.bin").write_bytes(b"geometry")
    (scene / ".runtime_cache/key/cache.bin").write_bytes(b"native")
    (scene / "output_folder/0_0/UTCI.tif").write_bytes(b"output")
    (scene / ".solweig-light/transactions/state.json").write_text("state")
    (scene / ".solweig-light-locks/owner.lock").write_text("lock")
    result = HARNESS.reset_warm_state(scene)
    assert result["passed"]
    assert (scene / "processed_inputs/SVF/geometry.bin").read_bytes() == b"geometry"
    assert (scene / ".runtime_cache/key/cache.bin").read_bytes() == b"native"
    assert not (scene / "output_folder").exists()
    assert not (scene / ".solweig-light").exists()
    assert not (scene / ".solweig-light-locks").exists()


def test_cache_proofs_distinguish_first_use_and_retained_warm_state():
    empty = {"file_count": 0, "total_bytes": 0, "files": {}, "exists": False}
    jit = {"file_count": 1, "total_bytes": 3, "files": {"x.nbc": {"sha256": "a", "bytes": 3}}, "exists": True}
    geometry = {
        "processed_inputs": {**empty, "file_count": 1, "total_bytes": 2},
        "native_geometry_cache": {**empty, "file_count": 1, "total_bytes": 4},
        "total_bytes": 6,
    }
    empty_geometry = {"processed_inputs": empty, "native_geometry_cache": empty, "total_bytes": 0}
    assert HARNESS.cache_proof(HARNESS.REGIME_FIRST, empty, jit, empty_geometry, geometry)["passed"]
    assert HARNESS.cache_proof(HARNESS.REGIME_WARM, jit, jit, geometry, geometry)["passed"]
    changed = {**jit, "total_bytes": 5}
    assert not HARNESS.cache_proof(HARNESS.REGIME_WARM, jit, changed, geometry, geometry)["passed"]


def _write_output(path: Path, bands: int = 24):
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = gdal.GetDriverByName("GTiff").Create(str(path), 2, 2, bands, gdal.GDT_Float32)
    for index in range(1, bands + 1):
        dataset.GetRasterBand(index).WriteArray(np.full((2, 2), index, dtype=np.float32))
    dataset = None


def test_output_schema_requires_exact_ten_fields_and_24_bands(tmp_path):
    fields = protocol()["matrix"]["output_fields"]
    for field in fields:
        _write_output(tmp_path / "output_folder/0_0" / f"{field}_0_0.tif")
    assert HARNESS.validate_outputs(tmp_path, fields, 24)["passed"]
    (tmp_path / "output_folder/0_0/Wind_0_0.tif").unlink()
    assert not HARNESS.validate_outputs(tmp_path, fields, 24)["passed"]
    _write_output(tmp_path / "output_folder/0_0/Wind_0_0.tif", bands=23)
    assert not HARNESS.validate_outputs(tmp_path, fields, 24)["passed"]


def test_setup_only_svf_sidecar_difference_does_not_fail_measured_output_pair(tmp_path):
    value = protocol()
    for variant in ("baseline", "candidate"):
        scene = tmp_path / variant
        for field in value["matrix"]["output_fields"]:
            _write_output(scene / "output_folder/0_0" / f"{field}_0_0.tif", bands=1)
        sidecar = scene / "processed_inputs/SVF/SkyViewFactor_0_0.tif"
        sidecar.parent.mkdir(parents=True)
        sidecar.write_bytes(variant.encode())
    result = HARNESS.pair_comparison(
        tmp_path / "baseline", tmp_path / "candidate", value,
        HARNESS.comparison_rules(value),
    )
    assert result["passed"]
    assert result["warmup_only_geometry_artifacts_excluded"] is True
    assert all(item["path"].startswith("output_folder/") for item in result["artifacts"])


def test_any_pair_failure_invalidates_five_pair_subgroup():
    value = protocol()
    template = {
        "case": "small", "regime": HARNESS.REGIME_FIRST, "budget": 1,
        "baseline_seconds": 2.0, "candidate_seconds": 1.0,
    }
    pairs = [{**template, "repetition": index, "passed": index != 4} for index in range(5)]
    summaries = HARNESS.summarize_pairs(pairs, value)
    summary = next(item for item in summaries if item["case"] == "small" and item["regime"] == HARNESS.REGIME_FIRST and item["budget"] == 1)
    assert summary["valid_pairs"] == 4
    assert summary["subgroup_passed"] is False
    assert summary["performance_claim_eligible"] is False
    assert summary["median_ratio"] is None


def test_source_fixture_harness_guard_detects_snapshot_change(tmp_path):
    value = protocol()
    protocol_copy = tmp_path / "protocol.json"
    protocol_copy.write_text(json.dumps(value))
    baseline = tmp_path / "baseline"; candidate = tmp_path / "candidate"
    baseline.mkdir(); candidate.mkdir()
    (baseline / "a.py").write_text("a=1\n")
    (candidate / "b.py").write_text("b=1\n")
    sources = {"baseline": baseline, "candidate": candidate}
    guards = {
        "protocol_sha256": HARNESS.digest(protocol_copy),
        "harness_sha256": HARNESS.digest(ROOT / "tools/run_p7_serial_variant.py"),
        "comparison_utility_sha256": HARNESS.digest(ROOT / value["comparison"]["utility"]),
        "sources": {name: HARNESS.tree_inventory(path) for name, path in sources.items()},
        "fixtures": {name: HARNESS.fixture_inventory(case) for name, case in value["fixtures"].items()},
        "fixture_kwargs": {name: HARNESS.digest(ROOT / case["kwargs"]) for name, case in value["fixtures"].items()},
    }
    HARNESS.assert_run_guards(guards, protocol_copy, sources, value)
    (candidate / "b.py").write_text("b=2\n")
    with pytest.raises(RuntimeError, match="changed during execution"):
        HARNESS.assert_run_guards(guards, protocol_copy, sources, value)
