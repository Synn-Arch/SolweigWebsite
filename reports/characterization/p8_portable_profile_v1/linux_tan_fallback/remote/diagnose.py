#!/usr/bin/env python3
"""Compare the installed-profile Numba fallback with vector NumPy tan."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import llvmlite
import numba
import numpy as np

from solweig_light.radiation._sleef_classifier import tan_array


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assess(name: str, values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    expected = np.asarray(np.tan(values), dtype=np.float32)
    actual = tan_array(values)
    eb, ab = expected.view(np.uint32), actual.view(np.uint32)
    both_nan = np.isnan(expected) & np.isnan(actual)
    mismatch = (eb != ab) & ~both_nan
    fast = np.isfinite(values) & (np.abs(values) < np.float32(125.0))
    fallback = ~fast
    indices = np.flatnonzero(mismatch)
    examples = []
    for i in indices[:20]:
        examples.append({
            "index": int(i), "input_bits": int(values.view(np.uint32)[i]),
            "expected_bits": int(eb[i]), "actual_bits": int(ab[i]),
        })
    return {
        "name": name, "count": int(values.size),
        "fast_domain_count": int(np.sum(fast)),
        "fallback_domain_count": int(np.sum(fallback)),
        "bit_mismatches_excluding_both_nan": int(indices.size),
        "fast_domain_bit_mismatches": int(np.sum(mismatch & fast)),
        "fallback_domain_bit_mismatches": int(np.sum(mismatch & fallback)),
        "nan_masks_equal": bool(np.array_equal(np.isnan(expected), np.isnan(actual))),
        "positive_inf_masks_equal": bool(np.array_equal(np.isposinf(expected), np.isposinf(actual))),
        "negative_inf_masks_equal": bool(np.array_equal(np.isneginf(expected), np.isneginf(actual))),
        "examples": examples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    args = ap.parse_args()
    seed = 20260920
    rng = np.random.default_rng(seed)
    positive_bits = rng.integers(0x42FA0000, 0x7F800000, 100_000, dtype=np.uint32)
    negative_bits = positive_bits | np.uint32(0x80000000)
    mixed_bits = rng.integers(0, 0x100000000, 100_000, dtype=np.uint32)
    special_bits = np.array([
        0x00000000, 0x80000000, 0x42F9FFFF, 0x42FA0000,
        0xC2F9FFFF, 0xC2FA0000, 0x7F7FFFFF, 0xFF7FFFFF,
        0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000,
        0x7F800001, 0xFF800001,
    ], dtype=np.uint32)
    cases = [
        assess("wide_positive_fallback", positive_bits.view(np.float32)),
        assess("wide_negative_fallback", negative_bits.view(np.float32)),
        assess("mixed_all_float32_bits", mixed_bits.view(np.float32)),
        assess("boundaries_nonfinite_signed_zero", special_bits.view(np.float32)),
    ]
    source = args.source.resolve()
    report = {
        "schema": "portable-profile-v1-linux-tan-fallback-parity.v1",
        "seed": seed,
        "comparison": "exact float32 result bits; pairs of NaNs compare by mask because payload preservation is not an API gate",
        "source": {"path": str(source), "sha256": digest(source)},
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "machine": platform.machine(), "numpy": np.__version__,
            "numba": numba.__version__, "llvmlite": llvmlite.__version__,
        },
        "cases": cases,
        "passed": all(c["fallback_domain_bit_mismatches"] == 0 and c["nan_masks_equal"] and c["positive_inf_masks_equal"] and c["negative_inf_masks_equal"] for c in cases),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "mismatches": {c["name"]: c["bit_mismatches_excluding_both_nan"] for c in cases}}, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
