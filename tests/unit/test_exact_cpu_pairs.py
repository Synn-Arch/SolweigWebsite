"""Negative checks for the optimization evidence harness, not model mocks."""
import json
import os
from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest


TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
import run_exact_cpu_pairs as harness


def test_finite_bit_and_special_value_contract():
    value = np.array([0., -0., 1., np.inf, -np.inf, np.nan], dtype=np.float32)
    assert harness.exact_array(value, value.copy())
    other = value.copy()
    other[0] = -0.
    assert not harness.exact_array(value, other)
    other = value.copy()
    other[2] = np.nextafter(other[2], np.float32(2))
    assert not harness.exact_array(value, other)
    other = value.copy()
    other[3] = -np.inf
    assert not harness.exact_array(value, other)
    other = value.copy()
    other.view(np.uint32)[-1] = 0x7fc01234
    assert harness.exact_array(value, other)  # Payload is outside the NaN-mask contract.
    assert not harness.exact_array(value, value.astype(np.float64))
    assert not harness.exact_array(value, value.reshape(2, 3))


def write_tiff(path, array, timestamp="2020-07-18 00:00"):
    from osgeo import gdal
    dataset = gdal.GetDriverByName("GTiff").Create(str(path), 3, 2, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform((1, 2, 0, 4, 0, -2))
    band = dataset.GetRasterBand(1)
    band.SetMetadata({"Time": timestamp})
    band.SetNoDataValue(float("nan"))
    band.WriteArray(array)
    dataset = band = None


def test_tiff_comparison_rejects_bit_and_timestamp_changes(tmp_path):
    left, right = tmp_path / "left.tif", tmp_path / "right.tif"
    array = np.array([[0., -0., np.nan], [1., np.inf, -np.inf]], dtype=np.float32)
    write_tiff(left, array)
    write_tiff(right, array)
    assert harness.compare_tiff(left, right)["passed"]
    other = array.copy()
    other[0, 0] = -0.
    write_tiff(right, other)
    assert not harness.compare_tiff(left, right)["passed"]
    write_tiff(right, array, timestamp="wrong timestamp")
    assert not harness.compare_tiff(left, right)["passed"]


def test_schedule_is_reproducible_balanced_and_preserves_input():
    protocol = {"seed": 20260920, "cells": [{"fixture": "small", "repetitions": 5},
                                           {"fixture": "large", "repetitions": 6}]}
    before = json.dumps(protocol, sort_keys=True)
    result = harness.schedule(protocol)
    assert result == harness.schedule(protocol)
    for cell in result:
        baseline_first = sum(order[0] == "baseline" for order in cell["orders"])
        assert abs(2 * baseline_first - cell["repetitions"]) <= 1
    assert json.dumps(protocol, sort_keys=True) == before


def test_existing_run_is_not_overwritten_even_on_failure(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "failure.json"
    sentinel.write_text("earlier evidence")
    monkeypatch.setattr(harness, "validate", lambda *args, **kwargs: {})
    with pytest.raises(FileExistsError):
        harness.main(["--protocol", str(tmp_path / "protocol.json"), "--output", str(output), "--execute"])
    assert sentinel.read_text() == "earlier evidence"


def test_checkpoint_corruption_and_incomplete_cursor_are_rejected(tmp_path):
    transaction = tmp_path / ".solweig-light/transactions/key"
    state = transaction / "state-123"
    state.mkdir(parents=True)
    payload = state / "array-0.npy"
    np.save(payload, np.array([1., -0.], dtype=np.float32))
    manifest = {"schema": 1, "fields": {name: {"type": "none"} for name in harness.STATE_FIELDS}}
    manifest["fields"]["Tgmap1"] = {"type": "array", "file": payload.name, "sha256": harness.base.digest(payload)}
    harness.base.write_json(state / "state.json", manifest)
    outputs = []
    for field in harness.FIELDS:
        path = tmp_path / (field + "_0_0.tif")
        path.write_bytes(b"unit-test publication payload")
        outputs.append(str(path))
    record = {"signature": {"outputs": outputs, "timestamps": list(range(24))},
              "checkpoint": {"next_timestep": 24, "generation": state.name,
                             "state_sha256": harness.base.digest(state / "state.json")}}
    harness.base.write_json(transaction / "transaction.json", record)
    (transaction / "complete.json").write_text("{}")
    with pytest.raises(ValueError, match="Completion signature"):
        harness.checkpoint_inventory(tmp_path, 24)
    completed = {"signature": record["signature"], "artifacts": [
        {"final": name, "sha256": harness.base.digest(name)} for name in outputs]}
    harness.base.write_json(transaction / "complete.json", completed)
    assert harness.checkpoint_inventory(tmp_path, 24)
    Path(outputs[0]).write_bytes(b"changed after publication")
    with pytest.raises(ValueError, match="artifact hash"):
        harness.checkpoint_inventory(tmp_path, 24)
    Path(outputs[0]).write_bytes(b"unit-test publication payload")
    payload.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="array hash"):
        harness.checkpoint_inventory(tmp_path, 24)
    record["checkpoint"]["next_timestep"] = 23
    harness.base.write_json(transaction / "transaction.json", record)
    with pytest.raises(ValueError, match="Incomplete chronological"):
        harness.checkpoint_inventory(tmp_path, 24)


def _runtime(seconds=30):
    return {"maximum_trial_seconds": seconds}


def test_monitored_rejects_and_kills_descendant_after_launcher_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        harness.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=harness.RESERVE + 1),
    )
    run = tmp_path / "lingering-child"
    child_pid_path = tmp_path / "child.pid"
    launcher = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
        "time.sleep(.1)"
    )

    result = harness.monitored(
        [sys.executable, "-c", launcher, str(child_pid_path)],
        run,
        dict(os.environ),
        _runtime(),
    )

    assert result["returncode"] == 0
    assert result["abort_reason"] == "launcher_exited_with_descendants"
    assert result["rss_valid"] is True
    assert result["monitoring_passed"] is False
    assert result["cleanup"]["ownership"]["verified"] is True
    assert result["cleanup"]["launcher_exited_with_live_group"] is True
    assert int(child_pid_path.read_text()) in result["cleanup"]["pre_cleanup_owned_pids"]
    assert result["cleanup"]["group_signal"] == "sent_sigkill"
    assert result["cleanup"]["post_cleanup_owned_pids"] == []
    assert result["cleanup"]["post_cleanup_group_exists"] is False
    assert result["cleanup"]["passed"] is True
    assert json.loads((run / "measurement.json").read_text()) == result


def test_monitored_exception_retains_partial_samples_and_cleanup(tmp_path, monkeypatch):
    class MonitorFailure(RuntimeError):
        pass

    def fail_disk_check(_path):
        raise MonitorFailure("injected disk monitor failure")

    monkeypatch.setattr(harness.shutil, "disk_usage", fail_disk_check)
    run = tmp_path / "monitor-failure"
    with pytest.raises(MonitorFailure, match="injected disk monitor failure"):
        harness.monitored(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            run,
            dict(os.environ),
            _runtime(),
        )

    result = json.loads((run / "measurement.json").read_text())
    samples = (run / "rss_samples.jsonl").read_text().splitlines()
    assert result["abort_reason"] == "monitor_exception"
    assert result["monitor_error"]["type"] == "MonitorFailure"
    assert result["monitor_error"]["message"] == "injected disk monitor failure"
    assert result["samples"] >= 1
    assert len(samples) == result["samples"]
    assert result["sampled_process_tree_peak_rss_bytes"] > 0
    assert result["cleanup"]["ownership"]["verified"] is True
    assert result["cleanup"]["group_signal"] == "sent_sigkill"
    assert result["cleanup"]["post_cleanup_owned_pids"] == []
    assert result["cleanup"]["post_cleanup_group_exists"] is False
    assert result["cleanup"]["passed"] is True
    assert result["monitoring_passed"] is False


def test_successful_worker_timing_excludes_cleanup_scans(tmp_path, monkeypatch):
    import psutil
    import time

    original = psutil.pids

    def slow_pid_scan():
        time.sleep(.08)
        return original()

    monkeypatch.setattr(psutil, "pids", slow_pid_scan)
    monkeypatch.setattr(harness.shutil, "disk_usage", lambda _path: SimpleNamespace(free=harness.RESERVE + 1))
    result = harness.monitored([sys.executable, "-c", "import time; time.sleep(.1)"],
                               tmp_path / "successful", dict(os.environ), _runtime())
    assert result["monitoring_passed"]
    assert result["cleanup_elapsed_seconds"] >= .16
    assert result["supervisor_elapsed_seconds"] >= result["elapsed_seconds"] + result["cleanup_elapsed_seconds"]
