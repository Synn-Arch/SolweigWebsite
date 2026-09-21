"""Statistical decisions must use paired ratios, not ratios of pooled means."""

import importlib.util
import json
import math
from pathlib import Path

import pytest


_PATH = Path(__file__).resolve().parents[2] / "tools/summarize_exact_cpu_pairs.py"
_SPEC = importlib.util.spec_from_file_location("exact_pair_statistics", _PATH)
statistics_tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(statistics_tool)


def test_constant_ratio_has_exact_degenerate_enumerated_interval():
    result = statistics_tool.paired_statistics([2., 4., 6., 8., 10.], [1., 2., 3., 4., 5.])
    assert result["median_baseline_over_candidate"] == 2.
    assert result["bootstrap"]["resamples"] == 3125
    assert result["bootstrap"]["interval"] == [2., 2.]


def test_pairing_is_preserved_and_three_pairs_are_only_diagnostic():
    result = statistics_tool.paired_statistics([1., 10., 100.], [2., 5., 100.])
    assert result["paired_baseline_over_candidate"] == [0.5, 2., 1.]
    assert result["median_baseline_over_candidate"] == 1.
    assert result["geometric_mean_baseline_over_candidate"] == 1.
    assert result["bootstrap"] is None


def test_linear_quantile_interpolates_and_exchange_inverts_geometric_mean():
    assert statistics_tool.linear_quantile([1., 2., 3., 4.], 0.25) == 1.75
    left = [1., 2., 4., 8., 16.]
    right = [2., 2., 5., 6., 12.]
    forward = statistics_tool.paired_statistics(left, right)
    reverse = statistics_tool.paired_statistics(right, left)
    assert math.isclose(forward["geometric_mean_baseline_over_candidate"]
                        * reverse["geometric_mean_baseline_over_candidate"], 1.)
    assert forward["bootstrap"]["interval"][0] < forward["bootstrap"]["interval"][1]


@pytest.mark.parametrize("baseline,candidate", [([], []), ([1.], []), ([0.], [1.]),
                                                  ([1.], [float("nan")]), ([1.], [float("inf")])])
def test_invalid_observations_are_not_summarized(baseline, candidate):
    with pytest.raises(ValueError):
        statistics_tool.paired_statistics(baseline, candidate)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _evidence(tmp_path, purpose="development_selection", repetitions=3):
    """Synthetic evidence tests the analyzer, not the numerical pipeline."""
    root = tmp_path / "run"
    protocol_path = tmp_path / "original_protocol.json"
    protocol = {"schema_version": 1, "status": "reviewed_frozen", "purpose": purpose,
                "seed": 20260920, "cells": [{"variant": "candidate", "fixture": "synthetic",
                                             "repetitions": repetitions}]}
    _write(protocol_path, protocol)
    schedule = statistics_tool.expected_schedule(protocol)
    _write(root / "frozen.json", {"protocol": protocol,
                                  "protocol_sha256": statistics_tool.digest(protocol_path),
                                  "schedule": schedule})
    _write(root / "summary.json", {"status": "passed", "source_and_fixture_guards_passed": True,
                                   "cells": [{"cell": schedule[0], "passed": True,
                                              "pair_count": repetitions, "ratios": [2.] * repetitions}]})
    for rep in range(repetitions):
        pair = {"repetition": rep, "order": schedule[0]["orders"][rep],
                "passed": True, "comparison": {"passed": True}, "trials": {}}
        for side, elapsed in (("baseline", 2.), ("candidate", 1.)):
            pair["trials"][side] = {
                "status": {"passed": True}, "cache_valid": True,
                "measurement": {"monitoring_passed": True, "rss_valid": True,
                                "cleanup": {"passed": True}, "elapsed_seconds": elapsed,
                                "sampled_process_tree_peak_rss_bytes": 100 * 1024**2,
                                "samples": 2, "sample_interval_seconds": .02,
                                "rss_caveat": "Synthetic samples; no pipeline or performance evidence."}}
            samples = root / "cell-000" / side / f"r{rep:02d}" / "rss_samples.jsonl"
            samples.parent.mkdir(parents=True)
            samples.write_text(json.dumps([.01, 0, []]) + "\n"
                               + json.dumps([.04, 100 * 1024**2, [1]]) + "\n")
        _write(root / "cell-000" / f"pair-{rep:02d}.json", pair)
    return root, protocol_path


def test_complete_synthetic_matrix_preserves_sampling_and_scope(tmp_path):
    root, protocol = _evidence(tmp_path)
    result = statistics_tool.summarize(root, protocol)
    assert result["purpose"] == "development_selection"
    assert result["promotion_approved"] is False
    cell = result["cells"][0]
    assert cell["bootstrap_eligible_for_promotion"] is False
    assert cell["bootstrap"] is None
    sample = cell["sampling"]["baseline"][0]
    assert sample["configured_interval_seconds"] == .02
    assert sample["observed_interval_seconds"]["median"] == .03
    assert "Synthetic" in sample["rss_caveat"]


def test_forged_summary_and_pair_order_do_not_replace_frozen_schedule(tmp_path):
    root, protocol = _evidence(tmp_path)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["cells"][0]["cell"]["orders"][0].reverse()
    _write(summary_path, summary)
    pair_path = root / "cell-000/pair-00.json"
    pair = json.loads(pair_path.read_text())
    pair["order"].reverse()
    _write(pair_path, pair)
    with pytest.raises(ValueError, match="frozen schedule"):
        statistics_tool.summarize(root, protocol)


@pytest.mark.parametrize("kind", ["duplicate", "missing", "order"])
def test_corrupt_frozen_schedule_is_rejected(tmp_path, kind):
    root, protocol = _evidence(tmp_path)
    path = root / "frozen.json"
    frozen = json.loads(path.read_text())
    if kind == "duplicate":
        frozen["schedule"].append(frozen["schedule"][0])
    elif kind == "missing":
        frozen["schedule"].clear()
    else:
        frozen["schedule"][0]["orders"][0].reverse()
    _write(path, frozen)
    with pytest.raises(ValueError, match="Frozen schedule"):
        statistics_tool.summarize(root, protocol)


@pytest.mark.parametrize("kind", ["trial", "cache", "cleanup", "exactness", "incomplete", "samples"])
def test_failed_or_incomplete_pair_is_rejected(tmp_path, kind):
    root, protocol = _evidence(tmp_path)
    path = root / "cell-000/pair-00.json"
    pair = json.loads(path.read_text())
    trial = pair["trials"]["candidate"]
    if kind == "trial":
        trial["status"]["passed"] = False
    elif kind == "cache":
        trial["cache_valid"] = False
    elif kind == "cleanup":
        trial["measurement"]["cleanup"]["passed"] = False
    elif kind == "exactness":
        pair["comparison"]["passed"] = False
    elif kind == "samples":
        trial["measurement"]["samples"] = 3
    else:
        pair["passed"] = False
    _write(path, pair)
    with pytest.raises(ValueError):
        statistics_tool.summarize(root, protocol)


def test_original_protocol_byte_binding_is_required(tmp_path):
    root, protocol = _evidence(tmp_path)
    protocol.write_text(protocol.read_text() + " ")
    with pytest.raises(ValueError, match="Original protocol bytes"):
        statistics_tool.summarize(root, protocol)


def test_unsupported_promotion_count_is_rejected(tmp_path):
    root, protocol = _evidence(tmp_path, purpose="promotion", repetitions=6)
    with pytest.raises(ValueError, match="exactly five"):
        statistics_tool.summarize(root, protocol)
