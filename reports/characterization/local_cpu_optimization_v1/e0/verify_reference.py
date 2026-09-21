#!/usr/bin/env python3
"""Verify canonical original output and fixture hashes before comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reference-scene", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    prefix = f"large/{args.case}/run/scene/"
    expected = {path.removeprefix(prefix): value for path, value in summary["files_sha256"].items()
                if path.startswith(prefix)}
    checked = []
    mismatches = []
    for relative, digest in sorted(expected.items()):
        path = args.reference_scene / relative
        if not path.is_file():
            mismatches.append({"path": str(path), "reason": "missing"})
            continue
        actual = sha(path)
        checked.append({"path": str(path), "sha256": actual, "expected": digest, "passed": actual == digest})
        if actual != digest:
            mismatches.append({"path": str(path), "expected": digest, "actual": actual})
    fixture_manifest = json.loads((args.fixture / "manifest.json").read_text())
    fixture_expected = summary["files_sha256"][f"large/{args.case}/fixture/manifest.json"]
    fixture_actual = sha(args.fixture / "manifest.json")
    result = {
        "schema": "local-cpu-optimization-e0-reference-verification.v1",
        "case": args.case,
        "summary": str(args.summary),
        "summary_sha256": sha(args.summary),
        "reference_scene": str(args.reference_scene),
        "reference_files_checked": len(checked),
        "fixture_manifest": str(args.fixture / "manifest.json"),
        "fixture_manifest_sha256": fixture_actual,
        "fixture_manifest_expected_sha256": fixture_expected,
        "fixture_id": fixture_manifest.get("fixture_id"),
        "checked": checked,
        "mismatches": mismatches,
        "passed": not mismatches and fixture_actual == fixture_expected,
        "performance_claim": None,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["passed"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
