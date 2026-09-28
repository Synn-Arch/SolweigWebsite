#SOLWEIG-GPU: GPU-accelerated SOLWEIG model for urban thermal comfort simulation
#Copyright (C) 2022–2025 Harsh Kamth and Naveen Sudharsan

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
"""N8-12 Numba B control: producer-matched AoSoA consumer for the longwave
primary reducer (dossier 02_direct_aosoa, packet gate).

This is the B arm of the A/B/C leaf comparison: "same region and layout as C,
adapter cost counted". It consumes the IDENTICAL uint32 AoSoA blocks
``experiments/optimization_v8/layout/direct_aosoa.py`` (N8-11) emits --
``[G, P, W]`` C order, native element ``((g*P+p)*W+lane)``, W=8 (W=4 also
admitted, mirroring the native gang domain), padding lanes never written by
the producer and never read here -- so the leaf comparison measures the
consumer, not the feed. No re-decode and no dense materialization ever
happen: the adapter's only data operation is the zero-copy
``uint32 -> float32`` ``view`` of the producer's own bytes (the producer's
documented float interface), which is inside the timed/admitted call.

FLOAT32-VIEW CONVENTION (review note N6, n8_30_review_n8_11_layout.md --
BINDING on N8-12/N8-13): the kernels must only ever be fed float32 views of
the produced blocks. Raw uint32 arrays would change numba's promotion
(integer compares/arithmetic on the bit patterns) and silently diverge from
the oracle -- e.g. ``sh`` bit 0x3F800000 (1.0f) no longer equals integer 1,
so the sky predicate collapses. ``lw_primary_b`` therefore constructs the
views itself inside the admitted call, and the test suite pins the
convention with a mutation check (raw-uint32 feed FAILS a discriminating
input while the view feed matches the oracle).

Kernel bodies are the frozen ``_longwave_primary`` / ``_longwave_primary_serial``
graph (N8-04 contract at
optimization_v8_native_default/evidence/contract/n8_04_typed_lw_contract.md)
transcribed with AoSoA reads at ``(gang, patch, lane)``:

* ten float32 accumulators, two ordered ``p = 0..P-1`` sweeps, reflection
  barrier on the completed ``a0 + lup`` with a true IEEE division by the
  stored float32 pi ``0x40490FDB`` (never a reciprocal multiply);
* float64 surface scalars keep the chains in float64 and round ONCE at the
  accumulator store -- ``a = RN32(f64(a) + c64)`` -- never
  ``RN32(a + RN32(c))`` (discriminator: surface ``1+2^-25`` with
  solid ``[2^24, 1]`` gives a6 ``0x4B800001``);
* predicates multiply as 0.0f/1.0f numbers (``0*Inf = NaN`` is an
  observable), ``solar_gate`` stays a REAL branch, ``fastmath=False``, no
  FMA contraction, no reassociation.

Admission mirrors the native contract
(``solweig_light.backends.native.lw_native.primary``) rejection set and order
against the same inputs, adapted only where the AoSoA boundary differs
(sh/vs/vb/sun/shade arrive as ``[G, P, W]`` and ``rows`` is explicit because
the gang count cannot recover a tail): dtype/ndim/shape/stride violations,
P outside 1..609, rows < 0, rows == 0 early return of the zero frame, gang
width outside {4, 8}, surface provenance mismatch, non-f32 reflection
factor, caller ``out`` overlapping any input byte extent. Rejections raise
``UnsupportedInput`` (a TypeError) BEFORE any kernel work. This boundary is
deliberately NOT the producer's leaf admission (review note N2): the
producer declines duck types/subclasses/lazy leaves more strictly than
``decode_block`` ever would, but that strictness lives upstream of the
buffers this adapter sees; nothing of it is inherited here, and the only
additions beyond the native set are the AoSoA-boundary rows-type and
gang-count consistency checks documented above.

``lw_primary_b_lanes_serial`` is an experimental gang-major formulation
(per-lane accumulator columns, lane-inner loops over the producer's
contiguous lane groups) kept to give LLVM's vectorizer a runtime-trip
contiguous inner loop; per-lane operation ORDER is unchanged, so it is
bitwise-gated on the same adversarial grid as the row-major control. It is
informational -- ``lw_primary_b`` dispatches the row-major bodies, the
structural twins of the frozen kernels.
"""
import numpy as np
from numba import njit, prange

MAX_PATCHES = 609


class UnsupportedInput(TypeError):
    """Pre-launch rejection; the caller must keep the reference fallback."""


@njit(cache=True, fastmath=False, parallel=True)
def lw_primary_b_parallel(sh, vs, vb, sun, shade, solid, sine, cosine,
                          directions, gate, solar_gate, sky_down, sky_side,
                          surface_sun, surface_sh, lup, reflection_factor,
                          rows, output):
    """_longwave_primary verbatim with AoSoA reads; one row = one lane.

    ``sh``/``vs``/``vb`` are float32 VIEWS of the producer's uint32 blocks
    (same bytes, same strides), so no bit is ever moved or converted.
    ``directions``/``gate`` are admitted-but-never-read, as in the frozen
    signature. ``output`` is float32 [rows, 7], fully written.
    """
    gangs, patches, width = sh.shape
    for row in prange(rows):
        gang = row // width
        lane = row - gang * width
        accum = np.zeros(10, dtype=np.float32)
        for patch in range(patches):
            sky = sh[gang, patch, lane] == 1 and vs[gang, patch, lane] == 1
            veg = vs[gang, patch, lane] == 0 or vb[gang, patch, lane] == 0
            building = np.float32(np.float32(1)-sh[gang, patch, lane])*vb[gang, patch, lane] == 1
            accum[0] = np.float32(accum[0]+np.float32(sky*sky_down[patch]))
            accum[5] = np.float32(accum[5]+np.float32(sky*sky_side[patch]))
            vegetation_side = ((surface_sh*solid[patch])*cosine[patch])*veg
            vegetation_down = ((surface_sh*solid[patch])*sine[patch])*veg
            accum[6] = np.float32(accum[6]+vegetation_side)
            accum[1] = np.float32(accum[1]+vegetation_down)
            if solar_gate[patch]:
                sun_side = ((((surface_sun*sun[gang, patch, lane])*solid[patch])*cosine[patch])*building)
                shade_side = ((((surface_sh*shade[gang, patch, lane])*solid[patch])*cosine[patch])*building)
                sun_down = ((((surface_sun*sun[gang, patch, lane])*solid[patch])*sine[patch])*building)
                shade_down = ((((surface_sh*shade[gang, patch, lane])*solid[patch])*sine[patch])*building)
                accum[8] = np.float32(accum[8]+sun_side)
                accum[7] = np.float32(accum[7]+shade_side)
                accum[3] = np.float32(accum[3]+sun_down)
                accum[2] = np.float32(accum[2]+shade_down)
            else:
                shade_side = (((surface_sh*solid[patch])*cosine[patch])*building)
                shade_down = (((surface_sh*solid[patch])*sine[patch])*building)
                accum[7] = np.float32(accum[7]+shade_side)
                accum[2] = np.float32(accum[2]+shade_down)
        # The reflection field depends on the completed ordered sky sweep.
        reflected = np.float32(np.float32(np.float32(np.float32(accum[0]+lup[row])*reflection_factor)*np.float32(.5))/np.float32(np.pi))
        for patch in range(patches):
            mask = sh[gang, patch, lane] == 0 or vs[gang, patch, lane] == 0 or vb[gang, patch, lane] == 0
            side = np.float32(np.float32(np.float32(reflected*solid[patch])*cosine[patch])*mask)
            down = np.float32(np.float32(np.float32(reflected*solid[patch])*sine[patch])*mask)
            accum[9] = np.float32(accum[9]+side)
            accum[4] = np.float32(accum[4]+down)
        output[row, 0] = np.float32(np.float32(np.float32(np.float32(accum[0]+accum[1])+accum[2])+accum[3])+accum[4])
        output[row, 1] = np.float32(np.float32(np.float32(np.float32(accum[5]+accum[6])+accum[7])+accum[8])+accum[9])
        output[row, 2:7] = accum[5:10]
    return output


@njit(cache=True, fastmath=False)
def lw_primary_b_serial(sh, vs, vb, sun, shade, solid, sine, cosine,
                        directions, gate, solar_gate, sky_down, sky_side,
                        surface_sun, surface_sh, lup, reflection_factor,
                        rows, output):
    """Serial twin of lw_primary_b_parallel; must match it bitwise."""
    gangs, patches, width = sh.shape
    for row in range(rows):
        gang = row // width
        lane = row - gang * width
        accum = np.zeros(10, dtype=np.float32)
        for patch in range(patches):
            sky = sh[gang, patch, lane] == 1 and vs[gang, patch, lane] == 1
            veg = vs[gang, patch, lane] == 0 or vb[gang, patch, lane] == 0
            building = np.float32(np.float32(1)-sh[gang, patch, lane])*vb[gang, patch, lane] == 1
            accum[0] = np.float32(accum[0]+np.float32(sky*sky_down[patch]))
            accum[5] = np.float32(accum[5]+np.float32(sky*sky_side[patch]))
            vegetation_side = ((surface_sh*solid[patch])*cosine[patch])*veg
            vegetation_down = ((surface_sh*solid[patch])*sine[patch])*veg
            accum[6] = np.float32(accum[6]+vegetation_side)
            accum[1] = np.float32(accum[1]+vegetation_down)
            if solar_gate[patch]:
                sun_side = ((((surface_sun*sun[gang, patch, lane])*solid[patch])*cosine[patch])*building)
                shade_side = ((((surface_sh*shade[gang, patch, lane])*solid[patch])*cosine[patch])*building)
                sun_down = ((((surface_sun*sun[gang, patch, lane])*solid[patch])*sine[patch])*building)
                shade_down = ((((surface_sh*shade[gang, patch, lane])*solid[patch])*sine[patch])*building)
                accum[8] = np.float32(accum[8]+sun_side)
                accum[7] = np.float32(accum[7]+shade_side)
                accum[3] = np.float32(accum[3]+sun_down)
                accum[2] = np.float32(accum[2]+shade_down)
            else:
                shade_side = (((surface_sh*solid[patch])*cosine[patch])*building)
                shade_down = (((surface_sh*solid[patch])*sine[patch])*building)
                accum[7] = np.float32(accum[7]+shade_side)
                accum[2] = np.float32(accum[2]+shade_down)
        # The reflection field depends on the completed ordered sky sweep.
        reflected = np.float32(np.float32(np.float32(np.float32(accum[0]+lup[row])*reflection_factor)*np.float32(.5))/np.float32(np.pi))
        for patch in range(patches):
            mask = sh[gang, patch, lane] == 0 or vs[gang, patch, lane] == 0 or vb[gang, patch, lane] == 0
            side = np.float32(np.float32(np.float32(reflected*solid[patch])*cosine[patch])*mask)
            down = np.float32(np.float32(np.float32(reflected*solid[patch])*sine[patch])*mask)
            accum[9] = np.float32(accum[9]+side)
            accum[4] = np.float32(accum[4]+down)
        output[row, 0] = np.float32(np.float32(np.float32(np.float32(accum[0]+accum[1])+accum[2])+accum[3])+accum[4])
        output[row, 1] = np.float32(np.float32(np.float32(np.float32(accum[5]+accum[6])+accum[7])+accum[8])+accum[9])
        output[row, 2:7] = accum[5:10]
    return output


@njit(cache=True, fastmath=False)
def lw_primary_b_lanes_serial(sh, vs, vb, sun, shade, solid, sine, cosine,
                              directions, gate, solar_gate, sky_down, sky_side,
                              surface_sun, surface_sh, lup, reflection_factor,
                              rows, output):
    """EXPERIMENTAL gang-major formulation (informational; not dispatched).

    Same per-lane operation order as the row-major control -- sweep 1
    ``p = 0..P-1``, reflection barrier, sweep 2 ``p = 0..P-1`` -- with the
    lane loops innermost over the producer's contiguous ``W``-lane groups and
    accumulators held as ``[10, W]`` columns, so each lane's arithmetic
    sequence is unchanged while the inner loop trips over contiguous data
    (runtime trip count ``live``, branchless lane predicates; the
    ``solar_gate`` branch stays real, per contract). Only live lanes are
    visited, so padding lanes are never read. Gated bitwise on the full
    adversarial grid alongside the row-major control.
    """
    gangs, patches, width = sh.shape
    for gang in range(gangs):
        base = gang * width
        live = min(width, rows - base)
        accum = np.zeros((10, width), dtype=np.float32)
        for patch in range(patches):
            for lane in range(live):
                sky = sh[gang, patch, lane] == 1 and vs[gang, patch, lane] == 1
                veg = vs[gang, patch, lane] == 0 or vb[gang, patch, lane] == 0
                accum[0, lane] = np.float32(accum[0, lane]+np.float32(sky*sky_down[patch]))
                accum[5, lane] = np.float32(accum[5, lane]+np.float32(sky*sky_side[patch]))
                vegetation_side = ((surface_sh*solid[patch])*cosine[patch])*veg
                vegetation_down = ((surface_sh*solid[patch])*sine[patch])*veg
                accum[6, lane] = np.float32(accum[6, lane]+vegetation_side)
                accum[1, lane] = np.float32(accum[1, lane]+vegetation_down)
            if solar_gate[patch]:
                for lane in range(live):
                    building = np.float32(np.float32(1)-sh[gang, patch, lane])*vb[gang, patch, lane] == 1
                    sun_side = ((((surface_sun*sun[gang, patch, lane])*solid[patch])*cosine[patch])*building)
                    shade_side = ((((surface_sh*shade[gang, patch, lane])*solid[patch])*cosine[patch])*building)
                    sun_down = ((((surface_sun*sun[gang, patch, lane])*solid[patch])*sine[patch])*building)
                    shade_down = ((((surface_sh*shade[gang, patch, lane])*solid[patch])*sine[patch])*building)
                    accum[8, lane] = np.float32(accum[8, lane]+sun_side)
                    accum[7, lane] = np.float32(accum[7, lane]+shade_side)
                    accum[3, lane] = np.float32(accum[3, lane]+sun_down)
                    accum[2, lane] = np.float32(accum[2, lane]+shade_down)
            else:
                for lane in range(live):
                    building = np.float32(np.float32(1)-sh[gang, patch, lane])*vb[gang, patch, lane] == 1
                    shade_side = (((surface_sh*solid[patch])*cosine[patch])*building)
                    shade_down = (((surface_sh*solid[patch])*sine[patch])*building)
                    accum[7, lane] = np.float32(accum[7, lane]+shade_side)
                    accum[2, lane] = np.float32(accum[2, lane]+shade_down)
        for lane in range(live):
            row = base + lane
            # The reflection field depends on the completed ordered sky sweep.
            reflected = np.float32(np.float32(np.float32(np.float32(accum[0, lane]+lup[row])*reflection_factor)*np.float32(.5))/np.float32(np.pi))
            for patch in range(patches):
                mask = sh[gang, patch, lane] == 0 or vs[gang, patch, lane] == 0 or vb[gang, patch, lane] == 0
                side = np.float32(np.float32(np.float32(reflected*solid[patch])*cosine[patch])*mask)
                down = np.float32(np.float32(np.float32(reflected*solid[patch])*sine[patch])*mask)
                accum[9, lane] = np.float32(accum[9, lane]+side)
                accum[4, lane] = np.float32(accum[4, lane]+down)
            output[row, 0] = np.float32(np.float32(np.float32(np.float32(accum[0, lane]+accum[1, lane])+accum[2, lane])+accum[3, lane])+accum[4, lane])
            output[row, 1] = np.float32(np.float32(np.float32(np.float32(accum[5, lane]+accum[6, lane])+accum[7, lane])+accum[8, lane])+accum[9, lane])
            for k in range(5, 10):
                output[row, k - 3] = accum[k, lane]   # a5..a9 -> cols 2..6
    return output


# ---------------------------------------------------------------------------
# Adapter: admission mirrors the native contract's rejection set and order.
# ---------------------------------------------------------------------------

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
    shape = x.shape
    if x.ndim == 3:
        expected = (itemsize * shape[1] * shape[2], itemsize * shape[2], itemsize)
    elif x.ndim == 2:
        expected = (itemsize * shape[1], itemsize)
    else:
        expected = (itemsize,)
    if tuple(x.strides) != expected:
        raise UnsupportedInput(
            f"{name}: strides {x.strides} != C-contiguous {expected}")


def _scalar_spec(x, name):
    """'f32' | 'f64' for the admitted surface-scalar provenances."""
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
    # Mirrors the native contract: np.float32 scalar or 0-d float32 ndarray
    # (both compute float32 in numba); a Python float is NOT admitted.
    if isinstance(x, np.generic) and x.dtype == np.float32:
        return x
    if isinstance(x, np.ndarray) and x.ndim == 0 and x.dtype == np.float32:
        return x
    raise UnsupportedInput(
        f"reflection_factor: unsupported {type(x)!r} "
        f"(admitted: np.float32 scalar or 0-d float32 ndarray)")


def _ranges_overlap(a, b) -> bool:
    a0, a1 = a
    b0, b1 = b
    return a0 < b1 and b0 < a1


def _extent(x) -> tuple[int, int]:
    start = x.ctypes.data
    span = sum((d - 1) * s for d, s in zip(x.shape, x.strides)) + x.itemsize
    return (start, start + span)


def lw_primary_b(sh, vs, vb, sun, shade, solid, sine, cosine, directions,
                 gate, solar_gate, sky_down, sky_side, surface_sun,
                 surface_sh, lup, reflection_factor, rows, parallel=True,
                 out=None):
    """Run the B control on the producer's uint32 AoSoA blocks; [rows, 7] f32.

    ``sh``/``vs``/``vb`` are uint32 [G, P, W] C-contiguous (W in {4, 8},
    G = ceil(rows/W)) exactly as ``produce_blocks_aosoa`` returns them;
    ``sun``/``shade`` are bool [G, P, W] C-contiguous as
    ``classify_block_aosoa`` returns them. The remaining arguments are the
    frozen 17-argument signature's patch-level inputs unchanged. All
    rejections happen BEFORE any kernel work (UnsupportedInput, a
    TypeError), mirroring the native adapter's order: sh type/dtype/ndim,
    gang width, P bounds, rows domain, vs/vb agreement, layout, then the
    patch-level arrays, surface provenance, reflection factor, and finally
    ``out`` validity and aliasing. The zero-copy uint32->float32 views are
    made here, so the adapter cost is inside the call.
    """
    # native order: _need_array(sh) first, then the gang gate from the shape.
    _need_array(sh, "sh", np.uint32, 3)
    G, P, W = sh.shape
    if W not in (4, 8):
        raise UnsupportedInput(f"gang must be 4 or 8, got {W!r}")
    if not (1 <= P <= MAX_PATCHES):
        raise UnsupportedInput(f"P={P} outside admitted 1..{MAX_PATCHES}")
    if not isinstance(rows, (int, np.integer)) or isinstance(rows, bool):
        raise UnsupportedInput(f"rows: expected an integer, got {type(rows)!r}")
    rows = int(rows)
    if rows < 0:
        raise UnsupportedInput(f"rows={rows} negative")
    if rows == 0:
        # Mirrors the native B=0 path exactly: the zero frame returns without
        # touching patch data and without validating anything downstream
        # (including ``out``).
        return np.zeros((0, 7), dtype=np.float32)

    _need_array(vs, "vs", np.uint32, 3, sh.shape)
    _need_array(vb, "vb", np.uint32, 3, sh.shape)
    if G != -(-rows // W):
        raise UnsupportedInput(
            f"sh: {G} gangs cannot carry rows={rows} at width {W} "
            f"(expected ceil(rows/W)={-(-rows // W)})")
    for name, arr in (("sh", sh), ("vs", vs), ("vb", vb)):
        _need_c_contig(arr, name, 4)
    for name, arr in (("sun", sun), ("shade", shade)):
        _need_array(arr, name, np.bool_, 3, sh.shape)
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
    _need_array(lup, "lup", np.float32, 1, (rows,))
    _need_c_contig(lup, "lup", 4)

    spec = _scalar_spec(surface_sun, "surface_sun")
    if spec != _scalar_spec(surface_sh, "surface_sh"):
        raise UnsupportedInput(
            f"surface scalar specialization mismatch: surface_sun is "
            f"{spec}, surface_sh is {_scalar_spec(surface_sh, 'surface_sh')}")
    factor = _reflection_spec(reflection_factor)

    if out is not None:
        _need_array(out, "out", np.float32, 2, (rows, 7))
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
    else:
        out = np.zeros((rows, 7), dtype=np.float32)

    # Zero-copy reinterpretation of the producer's own bytes: same buffer,
    # same strides, no numeric conversion (0x3f800000 stays 1.0f, arbitrary
    # raw payloads keep their bits). This is the entire adapter data cost.
    kernel = lw_primary_b_parallel if parallel else lw_primary_b_serial
    return kernel(sh.view(np.float32), vs.view(np.float32),
                  vb.view(np.float32), sun, shade, solid, sine, cosine,
                  directions, gate, solar_gate, sky_down, sky_side,
                  surface_sun, surface_sh, lup, factor, rows, out)


def b_kernel_pair():
    """(row-major parallel, row-major serial) B-control kernel handles."""
    return lw_primary_b_parallel, lw_primary_b_serial
