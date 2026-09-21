#!/usr/bin/env python3
"""Check exact equality of cProfile and retained unprofiled warmup outputs."""
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
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    warmup = args.run / "warmup" / "output_folder"
    profiled = args.run / "setup" / "scene" / "output_folder"
    warm = {str(path.relative_to(warmup)): sha(path) for path in sorted(warmup.rglob("*")) if path.is_file()}
    profile = {str(path.relative_to(profiled)): sha(path) for path in sorted(profiled.rglob("*")) if path.is_file()}
    # The profiling helper deliberately writes only the ten requested model
    # outputs into the profiled output folder.  Its unprofiled warmup also
    # retains the generated SVF TIFF there for provenance; SVF is validated as
    # processed geometry by the helper and is not a profiled output.  Compare
    # the shared output set exactly and record this expected warmup-only file.
    warm_only = sorted(set(warm) - set(profile))
    profile_only = sorted(set(profile) - set(warm))
    paths = sorted(set(warm) & set(profile))
    rows = [{"path": p, "warmup_sha256": warm[p], "profile_sha256": profile[p],
             "passed": warm[p] == profile[p]} for p in paths]
    expected_warmup_only = [p for p in warm_only if p.endswith("/SVF_0_0.tif")]
    result = {"schema": "local-cpu-optimization-e0-profile-output-comparison.v1",
              "run": str(args.run), "warmup_outputs": len(warm), "profiled_outputs": len(profile),
              "shared_outputs": len(paths), "warmup_only": warm_only,
              "profile_only": profile_only,
              "expected_warmup_only": expected_warmup_only,
              "files": rows,
              "passed": bool(rows) and all(row["passed"] for row in rows)
              and warm_only == expected_warmup_only and not profile_only,
              "performance_claim": None}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["passed"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
