"""C_native adapter (B7-20): exact ISPC port of the longwave primary reducer.

Python boundary for the first-island candidate C_native. The kernel is the
ISPC file lw_primary.ispc in this directory; each shared library
(liblw_native_g4.dylib / liblw_native_g8.dylib) exposes two C-ABI entry
points that differ only in the surface-scalar specialization:

  lw_primary_f32: surface_sun/surface_sh are float32 scalars; the vegetation
                  and sun/shade chains stay float32 (synthetic specialization).
  lw_primary_f64: surface_sun/surface_sh are float64 scalars; chains stay
                  float64 and round to float32 only at the accumulator add
                  (real-pipeline specialization).

Both schedule identically (pixels across ISPC lanes, ordered patch sweeps per
lane), so parallel-label and serial-label calls map to the same kernel.

Admission (mirrors the frozen B7-02 contract, guard_domains.kernel_admission):
  sh/vs/vb        float32 [B,P] C-contiguous, P = 1..609, B >= 0
  sun/shade       bool    [B,P] C-contiguous, same shape
  solid/sine/cosine float32 [P] C-contiguous
  solar_gate      bool    [P] stride-1
  sky_down/side   float32 [P], any element stride (pipeline uses stride-12
                  column views); the element stride is passed natively
  directions/gate float32/bool [P,4] (unused by the reducer body; dtype/shape
                  checked, layout unrestricted because they are never read)
  surface_sun/sh  np.float32 scalar -> f32 entry; np.float64 scalar or Python
                  float -> f64 entry (numba types Python float as float64)
  reflection_factor np.float32 scalar or 0-d float32 ndarray
  lup             float32 [B] C-contiguous

Anything else raises UnsupportedInput BEFORE the native call (the caller keeps
the unchanged reference fallback). Inputs are never written; the kernel is
synchronous, so no input can be unmapped or mutated while in flight.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_LIB_DIR = Path(__file__).resolve().parent

MAX_PATCHES = 609


class UnsupportedInput(TypeError):
    """Pre-launch rejection; the caller must keep the reference fallback."""


_PROTO = [
    ctypes.c_void_p,                   # sh
    ctypes.c_void_p,                   # vs
    ctypes.c_void_p,                   # vb
    ctypes.c_void_p,                   # sun
    ctypes.c_void_p,                   # shade
    ctypes.c_void_p,                   # solid
    ctypes.c_void_p,                   # sine
    ctypes.c_void_p,                   # cosine
    ctypes.c_void_p,                   # solar_gate
    ctypes.c_void_p,                   # sky_down
    ctypes.c_int64,                   # sky_down element stride
    ctypes.c_void_p,                   # sky_side
    ctypes.c_int64,                   # sky_side element stride
    ctypes.c_double,                  # surface_sun  (float entry re-pins this)
    ctypes.c_double,                  # surface_sh
    ctypes.c_void_p,                   # lup
    ctypes.c_float,                   # reflection_factor
    ctypes.c_int64,                   # B
    ctypes.c_int64,                   # P
    ctypes.c_void_p,                   # out
]

_LIBS: dict[int, tuple[ctypes.CDLL, dict[str, ctypes._FuncPointer]]] = {}


def _load(gang: int):
    if gang in _LIBS:
        return _LIBS[gang]
    if gang not in (4, 8):
        raise UnsupportedInput(f"gang must be 4 or 8, got {gang!r}")
    path = _LIB_DIR / f"liblw_native_g{gang}.dylib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} missing; run ./build.sh in {_LIB_DIR} first")
    lib = ctypes.CDLL(str(path))
    entries = {}
    for spec in ("f32", "f64"):
        fn = getattr(lib, f"lw_primary_{spec}")
        proto = list(_PROTO)
        if spec == "f32":  # surface scalars pinned to float for the f32 entry
            proto[13] = ctypes.c_float
            proto[14] = ctypes.c_float
        fn.argtypes = proto
        fn.restype = None
        entries[spec] = fn
    _LIBS[gang] = (lib, entries)
    return _LIBS[gang]


def _need_array(x, name, dtype, ndim, shape=None):
    if not isinstance(x, np.ndarray):
        raise UnsupportedInput(f"{name}: expected ndarray, got {type(x)!r}")
    if x.dtype != dtype:
        raise UnsupportedInput(f"{name}: dtype {x.dtype} != {np.dtype(dtype)}")
    if x.ndim != ndim:
        raise UnsupportedInput(f"{name}: ndim {x.ndim} != {ndim}")
    if shape is not None and x.shape != shape:
        raise UnsupportedInput(f"{name}: shape {x.shape} != {shape}")


def _need_c_contig(x, name, itemsize):
    strides = x.strides
    expected = (itemsize * x.shape[1], itemsize) if x.ndim == 2 else (itemsize,)
    if tuple(strides) != expected:
        raise UnsupportedInput(
            f"{name}: strides {strides} != C-contiguous {expected}")


def _scalar_spec(x, name):
    """Return 'f32' | 'f64' for the admitted surface-scalar provenances."""
    if isinstance(x, float):
        return "f64"  # numba types Python float as float64
    if isinstance(x, np.generic) and x.dtype == np.float32:
        return "f32"
    if isinstance(x, np.generic) and x.dtype == np.float64:
        return "f64"
    raise UnsupportedInput(
        f"{name}: unsupported scalar provenance {type(x)!r} "
        f"(admitted: np.float32 / np.float64 scalar, Python float)")


def _reflection_spec(x):
    # numba saw np.float32 scalars and 0-d float32 arrays; both compute as
    # float32. A Python float would promote the reflection to float64 and is
    # NOT admitted.
    if isinstance(x, np.generic) and x.dtype == np.float32:
        return float(x)
    if isinstance(x, np.ndarray) and x.ndim == 0 and x.dtype == np.float32:
        return float(x[()])
    raise UnsupportedInput(
        f"reflection_factor: unsupported {type(x)!r} "
        f"(admitted: np.float32 scalar or 0-d float32 ndarray)")


def _buffer(x):
    return x.ctypes.data


def _ranges_overlap(a, b) -> bool:
    a0, a1 = a
    b0, b1 = b
    return a0 < b1 and b0 < a1


def _extent(x) -> tuple[int, int]:
    start = x.ctypes.data
    span = sum((d - 1) * s for d, s in zip(x.shape, x.strides)) + x.itemsize
    return (start, start + span)


def primary(sh, vs, vb, sun, shade, solid, sine, cosine, directions, gate,
            solar_gate, sky_down, sky_side, surface_sun, surface_sh, lup,
            reflection_factor, gang: int = 4, out=None) -> np.ndarray:
    """Run the exact ISPC port of _longwave_primary; returns float32 [B,7]."""
    _load(gang)

    # ---- validation (all rejections happen before any native work) ----
    _need_array(sh, "sh", np.float32, 2)
    B, P = sh.shape
    if not (1 <= P <= MAX_PATCHES):
        raise UnsupportedInput(f"P={P} outside admitted 1..{MAX_PATCHES}")
    if B < 0:
        raise UnsupportedInput(f"B={B} negative")
    if B == 0:
        # Original supports B=0: prange(0) yields the zero-filled (0,7) frame.
        return np.zeros((0, 7), dtype=np.float32)

    _need_array(vs, "vs", np.float32, 2, sh.shape)
    _need_array(vb, "vb", np.float32, 2, sh.shape)
    for name, arr in (("sh", sh), ("vs", vs), ("vb", vb)):
        _need_c_contig(arr, name, 4)
    for name, arr in (("sun", sun), ("shade", shade)):
        _need_array(arr, name, np.bool_, 2, sh.shape)
        _need_c_contig(arr, name, 1)
    for name in ("solid", "sine", "cosine"):
        _need_array(locals()[name], name, np.float32, 1, (P,))
        _need_c_contig(locals()[name], name, 4)
    _need_array(solar_gate, "solar_gate", np.bool_, 1, (P,))
    if solar_gate.strides != (1,):
        raise UnsupportedInput(f"solar_gate: strides {solar_gate.strides} != (1,)")
    for name in ("sky_down", "sky_side"):
        _need_array(locals()[name], name, np.float32, 1, (P,))
    _need_array(directions, "directions", np.float32, 2, (P, 4))
    _need_array(gate, "gate", np.bool_, 2, (P, 4))
    _need_array(lup, "lup", np.float32, 1, (B,))
    _need_c_contig(lup, "lup", 4)

    spec = _scalar_spec(surface_sun, "surface_sun")
    if spec != _scalar_spec(surface_sh, "surface_sh"):
        raise UnsupportedInput(
            f"surface scalar specialization mismatch: surface_sun is "
            f"{spec}, surface_sh is {_scalar_spec(surface_sh, 'surface_sh')}")
    refl = _reflection_spec(reflection_factor)

    if out is not None:
        _need_array(out, "out", np.float32, 2, (B, 7))
        _need_c_contig(out, "out", 4)
        out_range = _extent(out)
        for name, arr in (("sh", sh), ("vs", vs), ("vb", vb), ("sun", sun),
                          ("shade", shade), ("solid", solid), ("sine", sine),
                          ("cosine", cosine), ("solar_gate", solar_gate),
                          ("sky_down", sky_down), ("sky_side", sky_side),
                          ("lup", lup), ("directions", directions),
                          ("gate", gate)):
            if _ranges_overlap(out_range, _extent(arr)):
                raise UnsupportedInput(
                    f"out aliases input {name}; rejecting before launch")

    # bool arrays are uint8 storage with values 0/1: zero-copy views (the
    # view construction is part of adapter timing), never a copy or recast.
    sun_u8 = sun.view(np.uint8)
    shade_u8 = shade.view(np.uint8)
    gate_u8 = solar_gate.view(np.uint8)

    if out is None:
        out = np.empty((B, 7), dtype=np.float32)

    _, entries = _LIBS[gang]
    entries[spec](
        _buffer(sh), _buffer(vs), _buffer(vb),
        _buffer(sun_u8), _buffer(shade_u8),
        _buffer(solid), _buffer(sine), _buffer(cosine),
        _buffer(gate_u8),
        _buffer(sky_down), sky_down.strides[0] // 4,
        _buffer(sky_side), sky_side.strides[0] // 4,
        float(surface_sun), float(surface_sh),
        _buffer(lup),
        refl,
        B, P,
        _buffer(out),
    )
    return out
