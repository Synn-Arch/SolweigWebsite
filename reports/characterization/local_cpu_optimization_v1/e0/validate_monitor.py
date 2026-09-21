#!/usr/bin/env python3
"""Validate that an E0 process-tree monitor captured usable samples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--outcome", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = json.loads(args.samples.read_text())
    outcome = json.loads(args.outcome.read_text())
    elapsed = [float(row["elapsed"]) for row in samples]
    rss = [int(row["summed_rss"]) for row in samples]
    positive = all(value > 0 for value in rss)
    monotonic = all(b >= a for a, b in zip(elapsed, elapsed[1:]))
    cap = int(outcome["limit"])
    within_cap = all(value <= cap for value in rss)
    result = {"schema": "local-cpu-optimization-e0-monitor-validation.v1",
              "sample_count": len(samples), "positive_rss": positive,
              "elapsed_monotonic": monotonic, "elapsed_seconds": elapsed[-1] if elapsed else 0.0,
              "peak_summed_rss": max(rss) if rss else 0, "within_cap": within_cap,
              "outcome_exit_code": outcome.get("exit_code"), "aborted": outcome.get("aborted"),
              "passed": bool(samples and positive and monotonic and within_cap
                              and outcome.get("exit_code") == 0 and not outcome.get("aborted"))}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["passed"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
