#!/usr/bin/env python3
"""Summarize complete local CPU pairs without granting promotion or speed claims.

Reads retained evidence only; no simulation, profiling or reference generation.
The five-pair interval is the fully enumerated paired percentile bootstrap
specified in local_cpu_optimization_v1/promotion_method.md.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def linear_quantile(ordered, probability):
    index = (len(ordered) - 1) * probability
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def paired_statistics(baseline, candidate):
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("Complete nonempty matched observations are required")
    if any(not math.isfinite(x) or x <= 0 for x in (*baseline, *candidate)):
        raise ValueError("Elapsed times must be positive finite values")
    ratios = [a / b for a, b in zip(baseline, candidate)]
    logs = [math.log(ratio) for ratio in ratios]
    result = {
        "baseline_seconds": baseline,
        "candidate_seconds": candidate,
        "baseline_median_seconds": statistics.median(baseline),
        "candidate_median_seconds": statistics.median(candidate),
        "paired_baseline_over_candidate": ratios,
        "median_baseline_over_candidate": statistics.median(ratios),
        "median_candidate_over_baseline": statistics.median([1 / ratio for ratio in ratios]),
        "geometric_mean_baseline_over_candidate": math.exp(statistics.fmean(logs)),
        "ratio_range": [min(ratios), max(ratios)],
        "bootstrap": None,
    }
    if len(ratios) == 5:
        resamples = sorted(
            math.exp(statistics.fmean(sample))
            for sample in itertools.product(logs, repeat=5)
        )
        result["bootstrap"] = {
            "method": "enumerated_ordered_paired_log_ratio_percentile_linear_quantile",
            "resamples": len(resamples),
            "confidence": 0.95,
            "interval": [linear_quantile(resamples, p) for p in (0.025, 0.975)],
            "limitation": "Five observed pairs; interval does not model thermal drift or cross-host variability.",
        }
    return result


def expected_schedule(protocol):
    """Reconstruct schema-v1 order from the authenticated seed and cell list."""
    generator = random.Random(protocol["seed"])
    result = []
    for index, cell in enumerate(protocol["cells"]):
        initial = generator.randrange(2)
        orders = [["baseline", "candidate"] if (rep + initial) % 2 == 0
                  else ["candidate", "baseline"] for rep in range(cell["repetitions"])]
        result.append({"cell": index, **cell, "orders": orders})
    generator.shuffle(result)
    return result


def sample_statistics(path, measurement):
    """Verify retained samples and distinguish configured and observed spacing."""
    count, peak, previous, gaps = 0, 0, None, []
    with path.open() as stream:
        for line in stream:
            elapsed, rss, _pids = json.loads(line)
            if not math.isfinite(elapsed) or elapsed < 0 or not isinstance(rss, int) or rss < 0:
                raise ValueError("Invalid retained RSS sample")
            if previous is not None:
                if elapsed < previous:
                    raise ValueError("RSS sample time moved backwards")
                gaps.append(elapsed - previous)
            previous = elapsed
            peak = max(peak, rss)
            count += 1
    if count != measurement["samples"] or peak != measurement["sampled_process_tree_peak_rss_bytes"]:
        raise ValueError("Retained RSS samples disagree with measurement")
    return {
        "samples": count,
        "configured_interval_seconds": measurement["sample_interval_seconds"],
        "observed_interval_seconds": ({"minimum": min(gaps), "median": statistics.median(gaps),
                                        "maximum": max(gaps)} if gaps else None),
        "rss_caveat": measurement["rss_caveat"],
        "interval_caveat": "Configured sleep is not an upper bound on sampling gaps; monitor work adds delay.",
    }


def summarize(root, protocol_path):
    root = Path(root).resolve()
    if (root / "failure.json").exists():
        raise ValueError("A failed matrix cannot become successful through aggregation")
    frozen_path = root / "frozen.json"
    summary_path = root / "summary.json"
    frozen = json.loads(frozen_path.read_text())
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "passed" or not summary.get("source_and_fixture_guards_passed"):
        raise ValueError("A completed matrix with passing source/fixture guards is required")
    protocol = frozen["protocol"]
    protocol_path = Path(protocol_path).resolve()
    if (digest(protocol_path) != frozen["protocol_sha256"]
            or json.loads(protocol_path.read_text()) != protocol):
        raise ValueError("Original protocol bytes do not authenticate the frozen protocol")
    if (protocol.get("schema_version") != 1 or protocol.get("status") != "reviewed_frozen"
            or protocol.get("purpose") not in {"development_selection", "promotion"}):
        raise ValueError("Unsupported or unfrozen protocol")
    if frozen["schedule"] != expected_schedule(protocol):
        raise ValueError("Frozen schedule differs from authenticated seed/cells")
    frozen_cells = {cell["cell"]: cell for cell in frozen["schedule"]}
    cells = summary["cells"]
    if len(cells) != len(protocol["cells"]):
        raise ValueError("Incomplete cell matrix")
    indexes = [entry["cell"]["cell"] for entry in cells]
    if sorted(indexes) != list(range(len(protocol["cells"]))):
        raise ValueError("Missing or duplicate cell identities")
    evidence = {str(path.relative_to(root)): digest(path) for path in (frozen_path, summary_path)}
    results = []
    for entry in sorted(cells, key=lambda value: value["cell"]["cell"]):
        cell = entry["cell"]
        if cell != frozen_cells[cell["cell"]]:
            raise ValueError("Summary cell differs from frozen schedule")
        expected = protocol["cells"][cell["cell"]]
        if any(cell.get(name) != value for name, value in expected.items()):
            raise ValueError("Cell differs from frozen workload")
        repetitions = expected["repetitions"]
        if protocol["purpose"] == "promotion" and repetitions != 5:
            raise ValueError("This promotion method requires exactly five pairs per cell")
        if not entry.get("passed") or entry["pair_count"] != repetitions:
            raise ValueError("Incomplete or failed cell")
        times = {"baseline": [], "candidate": []}
        rss = {"baseline": [], "candidate": []}
        sampling = {"baseline": [], "candidate": []}
        orders = []
        for repetition in range(repetitions):
            path = root / f"cell-{cell['cell']:03d}" / f"pair-{repetition:02d}.json"
            pair = json.loads(path.read_text())
            evidence[str(path.relative_to(root))] = digest(path)
            if (pair["repetition"] != repetition or not pair.get("passed")
                    or not pair["comparison"].get("passed")
                    or pair["order"] != cell["orders"][repetition]):
                raise ValueError("Pair lacks exactness or schedule agreement")
            orders.append(pair["order"])
            for side in ("baseline", "candidate"):
                trial = pair["trials"][side]
                measurement = trial["measurement"]
                if (not trial["status"].get("passed") or not trial.get("cache_valid")
                        or not measurement.get("monitoring_passed")
                        or not measurement.get("rss_valid")
                        or not measurement["cleanup"].get("passed")):
                    raise ValueError("Invalid trial, cache proof, monitoring or cleanup")
                times[side].append(measurement["elapsed_seconds"])
                peak = measurement["sampled_process_tree_peak_rss_bytes"]
                if not isinstance(peak, int) or not 0 < peak <= 12 * 1024**3:
                    raise ValueError("Invalid or over-limit process-tree RSS")
                rss[side].append(peak)
                samples_path = path.parent / side / f"r{repetition:02d}" / "rss_samples.jsonl"
                sampling[side].append(sample_statistics(samples_path, measurement))
                evidence[str(samples_path.relative_to(root))] = digest(samples_path)
        stats = paired_statistics(times["baseline"], times["candidate"])
        if stats["paired_baseline_over_candidate"] != entry["ratios"]:
            raise ValueError("Stored ratios disagree with raw elapsed times")
        results.append({
            "cell": cell,
            "pair_count": repetitions,
            "orders": orders,
            **stats,
            "sampled_process_tree_peak_rss_bytes": rss,
            "sampling": sampling,
            "bootstrap_eligible_for_promotion": protocol["purpose"] == "promotion" and repetitions == 5,
            "rss_guard_passed": statistics.median(rss["candidate"])
            <= 1.10 * statistics.median(rss["baseline"]) + 64 * 1024**2,
        })
    return {
        "status": "complete_matrix_summarized",
        "run": str(root),
        "protocol_sha256": frozen["protocol_sha256"],
        "original_protocol_bytes": str(protocol_path),
        "purpose": protocol["purpose"],
        "summarizer_sha256": digest(__file__),
        "evidence_sha256": evidence,
        "cells": results,
        "promotion_approved": False,
        "performance_claim_eligible": False,
        "upstream_speedup_claim_eligible": False,
        "decision_required": "Independent review against all predeclared benefit, regression, numerical and memory gates.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--protocol", type=Path, required=True,
                        help="Retained original protocol bytes; must match the frozen SHA and content")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Retain previous analyses rather than replacing them silently.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
