"""Fresh-process exact-output and cache probe for the isolated E4 packet."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


def descriptor(value):
    array = np.asarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "bytes_hex": array.tobytes(order="C").hex(),
        "nan": np.isnan(array).ravel().tolist() if array.dtype.kind == "f" else [],
        "posinf": np.isposinf(array).ravel().tolist() if array.dtype.kind == "f" else [],
        "neginf": np.isneginf(array).ravel().tolist() if array.dtype.kind == "f" else [],
        "signbit": np.signbit(array).ravel().tolist() if array.dtype.kind == "f" else [],
    }


def origin(module):
    actual = Path(module.__file__).resolve()
    expected = Path(os.environ["EXPECTED_SOURCE_ROOT"]).resolve()
    if not actual.is_relative_to(expected):
        raise AssertionError(f"wrong source origin: {actual} outside {expected}")
    return str(actual.relative_to(expected))


def wall(order):
    import solweig_light
    from solweig_light.radiation import wall_shadows as ws

    a = np.array([
        [-0.0, 1.0, 2.0, 1.0, 0.0],
        [0.0, 3.0, 7.0, 4.0, 1.0],
        [1.0, 2.0, 5.0, 9.0, 2.0],
        [0.0, 1.0, 3.0, 2.0, 0.0],
    ], dtype=np.float32)
    walls = np.array([
        [0, 2, 0, 1, 0], [1, 0, 4, 0, 2],
        [0, 3, 1, 2, 0], [2, 0, 0, 1, 3],
    ], dtype=np.float32)
    aspect = np.linspace(np.float32(0), np.float32(2 * np.pi), a.size,
                         endpoint=False, dtype=np.float32).reshape(a.shape)
    veg = a + np.array([
        [0, 2, 0, 4, 0], [3, 0, 5, 0, 1],
        [0, 2, 0, 3, 0], [1, 0, 4, 0, 2],
    ], dtype=np.float32)
    veg2 = veg - np.float32(1.25)
    bush = np.array([
        [0, 2, 0, 3, 0], [2, 0, 0, 2, 0],
        [0, 3, 0, 0, 2], [2, 0, 3, 0, 0],
    ], dtype=np.float32)
    parallel = order == "parallel-first"
    modes = (parallel, not parallel)
    results = {}
    for use_parallel in modes:
        key = "parallel" if use_parallel else "serial"
        results[key] = {
            "exact_13": [descriptor(x) for x in ws.exact_13(
                a, np.float32(137.25), np.float32(31.5), np.float32(1.0),
                walls, aspect, parallel=use_parallel)],
            "exact_23": [descriptor(x) for x in ws.exact_23(
                a, veg, veg2, np.float32(137.25), np.float32(31.5),
                np.float32(1.0), np.float32(np.max(veg)), bush, walls, aspect,
                parallel=use_parallel)],
        }
    return {
        "origins": {"package": origin(solweig_light), "module": origin(ws)},
        "separate_py_funcs": {
            "wall13": ws._wall13.py_func is not ws._wall13_serial.py_func,
            "wall23": ws._wall23.py_func is not ws._wall23_serial.py_func,
        },
        "order": order,
        "results": results,
    }


def math_profile():
    import solweig_light
    from solweig_light.radiation import _math_profile as mp
    from solweig_light.radiation import _sleef_acos, _sleef_classifier

    bits = np.array([
        0x80000000, 0x00000000, 0x00000001, 0x3F000000, 0x3F800000,
        0xBF800000, 0x7F800000, 0xFF800000, 0x7FC12345,
    ], dtype=np.uint32)
    values = bits.view(np.float32)
    wide_bits = np.array([
        0xC2FA0000, 0x42FA0000, 0x60AD78EC, 0xE0AD78EC,
        0x7F800000, 0xFF800000, 0x7FC54321,
    ], dtype=np.uint32)
    wide = np.concatenate((
        np.array([-124.999, -0.0, 0.0, 124.999], dtype=np.float32),
        wide_bits.view(np.float32),
    ))
    bits64 = np.array([
        0x8000000000000000, 0, 0x405F400000000000, 0xC05F400000000000,
        0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8123456789ABC,
    ], dtype=np.uint64)
    wide64 = bits64.view(np.float64)
    outputs = {
        "asvf32": descriptor(mp.asvf(values)),
        "asvf64": descriptor(mp.asvf(values.astype(np.float64))),
        "tan32_mixed_fallback": descriptor(mp.tan32(wide)),
        "tan64_numpy_fallback": descriptor(mp.tan32(wide64)),
        "atan32": descriptor(mp.atan32(values)),
        "atan64": descriptor(mp.atan32(wide64)),
    }
    lowering = {}
    for name, dispatcher in (
        ("asvf_fma", _sleef_acos.asvf_fma),
        ("tan_array", _sleef_classifier.tan_array),
        ("atan_array", _sleef_classifier.atan_array),
    ):
        llvm = dispatcher.inspect_llvm(dispatcher.signatures[0])
        lowering[name] = {
            "explicit_fma": "llvm.fma.f32" in llvm,
            "fast_flag_absent": " fast " not in llvm,
        }
    identity = mp.profile_identity()
    return {
        "origins": {
            "package": origin(solweig_light), "profile": origin(mp),
            "acos": origin(_sleef_acos), "classifier": origin(_sleef_classifier),
        },
        "profile_id": identity["id"],
        "profile_fingerprint": identity["fingerprint"],
        "source_sha256": identity["implementation"]["source_sha256"],
        "lowering": lowering,
        "outputs": outputs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("wall", "math"), required=True)
    parser.add_argument("--order", choices=("serial-first", "parallel-first"),
                        default="serial-first")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    payload = wall(args.order) if args.mode == "wall" else math_profile()
    args.result.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
