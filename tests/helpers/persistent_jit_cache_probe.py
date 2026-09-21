"""Fresh-process probe for persistent radiation JIT cache isolation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def arguments():
    rng = np.random.default_rng(20260919)
    pixels, patches = 7, 5
    binary = lambda: rng.integers(0, 2, (pixels, patches), dtype=np.int8).astype(np.float32)
    sh, vs, vb = binary(), binary(), binary()
    diff = rng.normal(size=(pixels, patches)).astype(np.float32)
    sun = rng.integers(0, 2, (pixels, patches), dtype=np.int8).astype(np.bool_)
    shade = np.logical_not(sun)
    vector = lambda: rng.normal(size=patches).astype(np.float32)
    solid, sine, cosine, luminance = vector(), vector(), vector(), vector()
    directions = rng.normal(size=(patches, 4)).astype(np.float32)
    gate = rng.integers(0, 2, (patches, 4), dtype=np.int8).astype(np.bool_)
    box_gate = rng.integers(0, 2, patches, dtype=np.int8).astype(np.bool_)
    shortwave = (
        sh, vs, vb, diff, sun, shade, luminance, solid, cosine, directions,
        gate, np.logical_not(gate), box_gate, np.float32(1.25),
        np.float32(0.75), True,
    )
    longwave = (
        sh, vs, vb, sun, shade, solid, sine, cosine, directions, gate,
        box_gate, vector(), vector(), np.float32(1.25), np.float32(0.75),
        rng.normal(size=pixels).astype(np.float32), np.float32(0.31),
    )
    return shortwave, longwave


def metadata(function):
    overload = next(iter(function.overloads.values()))
    stored = overload.metadata or {}
    parfors = stored.get("parfors", {})
    return {
        "parallel_target": function.targetoptions.get("parallel") is True,
        "parfors_present": bool(parfors),
        "parfors_keys": sorted(parfors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("serial", "parallel", "verify"), required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    from solweig_light.radiation.patch_radiation import (
        _longwave, _longwave_serial, _shortwave, _shortwave_serial,
    )

    shortwave, longwave = arguments()
    if args.mode == "serial":
        _shortwave_serial(*shortwave)
        _longwave_serial(*longwave)
        result = {
            "mode": args.mode,
            "metadata": {
                "shortwave_serial": metadata(_shortwave_serial),
                "longwave_serial": metadata(_longwave_serial),
            },
        }
    elif args.mode == "parallel":
        _shortwave(*shortwave)
        _longwave(*longwave)
        result = {
            "mode": args.mode,
            "metadata": {
                "shortwave_parallel": metadata(_shortwave),
                "longwave_parallel": metadata(_longwave),
            },
        }
    else:
        short_parallel = _shortwave(*shortwave)
        short_serial = _shortwave_serial(*shortwave)
        long_parallel = _longwave(*longwave)
        long_serial = _longwave_serial(*longwave)
        result = {
            "mode": args.mode,
            "shortwave_exact": bool(np.array_equal(short_parallel, short_serial)),
            "longwave_exact": bool(np.array_equal(long_parallel, long_serial)),
            "shortwave_max_abs": float(np.max(np.abs(short_parallel - short_serial))),
            "longwave_max_abs": float(np.max(np.abs(long_parallel - long_serial))),
            "target_options": {
                "shortwave_parallel": metadata(_shortwave)["parallel_target"],
                "shortwave_serial": metadata(_shortwave_serial)["parallel_target"],
                "longwave_parallel": metadata(_longwave)["parallel_target"],
                "longwave_serial": metadata(_longwave_serial)["parallel_target"],
            },
        }
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
