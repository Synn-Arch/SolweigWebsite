"""Typed GVF block postprocess kernel (C6-31).

Compiles ``ground_view._postprocess_block`` (base 5e1fab46) into one
``njit(cache=True, fastmath=False)`` serial kernel over pixels. Every
arithmetic node, cast, comparison and in-place zeroing of the NumPy original
is reproduced in the original statement order; no reassociation, no FMA, no
fastmath.

Exactness contract
------------------
The kernel performs the identical IEEE-754 operations in the identical order
on identical float32 operands, so its results (including NaN payloads, signed
zeros and the flag comparisons' masking of later zeroing) are bit-for-bit the
original's. The module boundary additionally preserves the original's
observable warning/error behavior under NumPy's default error regime
(divide/over/invalid warn, underflow ignored):

* A warning-class floating-point event (overflow/invalid; divide is excluded
  by the admitted step domain) necessarily produces a nonfinite value that
  propagates to the raw (pre-clamp) ``gvf2`` or to one of the four remaining
  returned fields. The kernel reports exactly that condition. The only
  operation that can hide a nonfinite value is the ``gvf2[gvf2 > 1.0] = 1.0``
  clamp, which is why the raw pre-clamp ``gvf2`` is checked before it.
* If the flag is clean, the original NumPy sequence could not have warned,
  and the kernel result is returned directly.
* If the flag is set, the three receiver planes the original mutates in place
  (``weightsumwall``, ``weightsumLwall``, ``weightsumalbwall``) are restored
  to their exact incoming bits and the whole call is re-executed by the
  untouched ``ground_view._postprocess_block`` on the original views. That
  fallback therefore fires the identical RuntimeWarnings at the identical
  source location, and raises the identical FloatingPointError under
  ``np.errstate(all='raise')`` with the caller's arrays in the identical
  state. Under a raised error the compiled kernel's own work is discarded;
  the caller-visible state comes only from the original function.
* Scoped carve-out (underflow, review C6-60 F1): the kernel inspects result
  values, not the floating-point underflow flag, and NumPy's default regime
  ignores underflow, so this is unobservable in production. Under a
  non-default ``np.errstate(under='raise')`` the original raises
  ``FloatingPointError`` when a division result is subnormal (e.g. a
  subnormal plane sum divided by a large admitted step count) where this
  wrapper returns silently with bitwise-identical outputs; raise parity for
  overflow/invalid/divide-by-zero is unaffected (those are nonfinite-flagged
  or out-of-domain and take the fallback).
  ``test_underflow_regime_contract`` pins this boundary exactly.
* Step counts outside the admitted domain (finite integers 1 <= v < 2**24,
  the exact range the guards in ``_gvf_fused``/``_sun`` produce after
  ``np.round`` and the ``first < 1`` clamp) are also routed to the original
  untouched function, which keeps the original promotion/division semantics
  for values the kernel does not model.

Serialization choice: the wrapper default is serial (``parallel=False``),
per the task gate. The ``prange`` variant is bitwise-proven on the full
adversarial suite (per-element results are independent of the thread
schedule) and the recorded C6-31 timings show it is also the faster
configuration at the integrated shapes (block_rows=32; original NumPy is
faster than the serial kernel there), so the integration recipe passes
``parallel=True`` explicitly. Both variants share the identical fallback
contract.
"""
import math

import numpy as np
from numba import njit, prange

# Receiver-plane layout of the fused gather block, identical to the
# ``planes`` tuple order of ground_view._postprocess_block and to the
# ``output[0..15]`` plane order of the gather kernels.
_PL_SH, _PL_WALL, _PL_LUPSH, _PL_LWALL, _PL_ALBSH, _PL_ALBWALL, _PL_ALBNOSH, _PL_ALBWALLNOSH, \
    _PL_SH_FIRST, _PL_WALL_FIRST, _PL_LUPSH_FIRST, _PL_LWALL_FIRST, \
    _PL_ALBSH_FIRST, _PL_ALBWALL_FIRST, _PL_ALBNOSH_FIRST, _PL_ALBWALLNOSH_FIRST = range(16)

# Admitted step-count domain (see module docstring). first/second reach the
# postprocess as np.round(first*scale)/np.round(second*scale) with the
# first<1 clamp, i.e. finite integers >= 1, and < 2**24 keeps f32(v) and
# f32(v+1) exact under either promotion order.
_STEP_MIN = 1.0
_STEP_MAX_EXCLUSIVE = 16777216.0


def _step_scalar(value):
    """Return the admitted float64 step value, or None to use the original."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.generic, np.ndarray)):
        return None
    if isinstance(value, np.ndarray) and value.ndim != 0:
        return None
    if not np.isfinite(np.asarray(value)):
        return None
    v = float(value)
    if v < _STEP_MIN or v >= _STEP_MAX_EXCLUSIVE or v != math.floor(v):
        return None
    return v


def _reference_block(block, buildings_b, facesh_b, lup_term_b, alb_term_b, nosh_term_b, first, second):
    """The untouched NumPy postprocess, on the caller's exact views."""
    from .ground_view import _postprocess_block
    planes = tuple(block[index] for index in range(16))
    return _postprocess_block(planes, buildings_b, facesh_b, lup_term_b, alb_term_b, nosh_term_b, first, second)


def gvf_postprocess_block(block, buildings_b, facesh_b, lup_term_b, alb_term_b, nosh_term_b, first, second, parallel=False):
    """Typed replacement for ground_view._postprocess_block (G03/C6-31).

    ``block`` is the (16, block_rows, cols) float32 gather buffer; the three
    planes the original mutates in place (1, 3, 5) are mutated identically.
    Returns ``(gvf, gvfLup, gvfalb, gvfalbnosh, gvf2)`` exactly like the
    original, bitwise, on every input for which the original stays silent,
    and delegates the whole call to the original otherwise (see the module
    docstring for the warning/error contract and its underflow carve-out).

    Boundary contract (review C6-60 F4): ``block`` must be a
    ``(16, rows, cols)`` float32 array — the wrapper deliberately refuses
    (TypeError) anything else instead of silently producing float32 outputs
    for an out-of-contract dtype. The term rasters must be float32 as well:
    the typed kernel allocates float32 outputs, while the original promotes
    its returns with the rasters (a float64 ``lup_term`` — reachable in the
    fused route because ``SBC`` is outside ``_supported`` — makes the
    original's ``gvfLup`` float64). An out-of-contract raster therefore
    delegates to the original instead of downcasting; integration finding,
    C6-70 (differential ``test_float64_derived_lup_conversion_preserved``).
    """
    if block.dtype != np.dtype(np.float32) or block.ndim != 3 or block.shape[0] != 16:
        raise TypeError(
            'gvf_postprocess_block requires a (16, rows, cols) float32 block, '
            f'got dtype={block.dtype} shape={block.shape}')
    if any(np.asarray(raster).dtype != np.dtype(np.float32)
           for raster in (buildings_b, facesh_b, lup_term_b, alb_term_b, nosh_term_b)):
        return _reference_block(block, buildings_b, facesh_b, lup_term_b, alb_term_b, nosh_term_b, first, second)
    first_v = _step_scalar(first)
    second_v = _step_scalar(second)
    if first_v is None or second_v is None:
        return _reference_block(block, buildings_b, facesh_b, lup_term_b, alb_term_b, nosh_term_b, first, second)
    # Saved incoming bits of the three planes the postprocess zeroing may
    # mutate; only consumed on the fallback path so the original re-runs from
    # the exact pre-call state.
    saved_wall = block[_PL_WALL].copy()
    saved_lwall = block[_PL_LWALL].copy()
    saved_albwall = block[_PL_ALBWALL].copy()
    kernel = _postprocess_block_parallel if parallel else _postprocess_block_serial
    gvf, gvfLup, gvfalb, gvfalbnosh, gvf2, flagged = kernel(
        block, buildings_b, facesh_b, lup_term_b, alb_term_b, nosh_term_b,
        first_v, second_v)
    if flagged:
        block[_PL_WALL][:] = saved_wall
        block[_PL_LWALL][:] = saved_lwall
        block[_PL_ALBWALL][:] = saved_albwall
        return _reference_block(block, buildings_b, facesh_b, lup_term_b, alb_term_b, nosh_term_b, first, second)
    return (gvf, gvfLup, gvfalb, gvfalbnosh, gvf2)


@njit(inline='always', fastmath=False)
def _postprocess_pixel(block, buildings_b, facesh_b, lup_term_b, alb_term_b, nosh_term_b,
                       first, second, row, col, gvf, gvfLup, gvfalb, gvfalbnosh, gvf2):
    """One pixel of ground_view._postprocess_block, node by node, in order.

    Statement order, cast points and comparison order are the original's: the
    four influence comparisons precede the keep construction, the keep zeroing
    precedes the three keep==1 maskings, weightsumwall is zeroed before gvf2
    reads it, and gvf2 is clamped only after it is computed. Returns 1 when
    the raw pre-clamp gvf2 or any of the four unclamped returns is nonfinite
    (see the module docstring for why that flag is exhaustive for warning
    reproduction), else 0.
    """
    zero = np.float32(0.0)
    one = np.float32(1.0)
    half = np.float32(0.5)
    pfour = np.float32(0.4)
    pnine = np.float32(0.9)
    f_first = np.float32(first)
    d_first = np.float32(first + 1.0)
    f_second = np.float32(second)
    d_second = np.float32(second + 1.0)
    weightsumwall_first = block[_PL_WALL_FIRST, row, col]
    weightsumalbwallnosh_first = block[_PL_ALBWALLNOSH_FIRST, row, col]
    weightsumwall = block[_PL_WALL, row, col]
    weightsumalbwallnosh = block[_PL_ALBWALLNOSH, row, col]
    # wallsuninfluence_first / wallinfluence_first / wallsuninfluence_second /
    # wallinfluence_second: comparisons in the original order, before any zeroing.
    wallsuninfluence_first = weightsumwall_first > zero
    wallinfluence_first = weightsumalbwallnosh_first > zero
    wallsuninfluence_second = weightsumwall > zero
    wallinfluence_second = weightsumalbwallnosh > zero
    # keep = (weightsumwall == second).astype(np.float32) - facesh
    if weightsumwall == second:
        keep = one - facesh_b[row, col]
    else:
        keep = zero - facesh_b[row, col]
    # keep[keep == -1] = 0
    if keep == np.float32(-1.0):
        keep = zero
    # gvf1
    if wallsuninfluence_first:
        gvf1 = (weightsumwall_first + block[_PL_SH_FIRST, row, col]) / d_first * one \
            + block[_PL_SH_FIRST, row, col] / f_first * zero
    else:
        gvf1 = (weightsumwall_first + block[_PL_SH_FIRST, row, col]) / d_first * zero \
            + block[_PL_SH_FIRST, row, col] / f_first * one
    # weightsumwall[keep == 1] = 0, then gvf2 reads the zeroed plane
    if keep == one:
        block[_PL_WALL, row, col] = zero
        weightsumwall = zero
    if wallsuninfluence_second:
        gvf2raw = (weightsumwall + block[_PL_SH, row, col]) / d_second * one \
            + block[_PL_SH, row, col] / f_second * zero
    else:
        gvf2raw = (weightsumwall + block[_PL_SH, row, col]) / d_second * zero \
            + block[_PL_SH, row, col] / f_second * one
    # gvf2[gvf2 > 1.0] = 1.0 -- only after the raw value is computed and
    # after the raw value has been checked: the clamp is the one operation
    # that can hide an Inf, so the flag must observe the pre-clamp value.
    flagged = 0
    if np.isnan(gvf2raw) or np.isinf(gvf2raw):
        flagged = 1
    if gvf2raw > one:
        gvf2raw = one
    gvf2[row, col] = gvf2raw
    # gvfLup1
    if wallsuninfluence_first:
        gvfLup1 = (block[_PL_LWALL_FIRST, row, col] + block[_PL_LUPSH_FIRST, row, col]) / d_first * one \
            + block[_PL_LUPSH_FIRST, row, col] / f_first * zero
    else:
        gvfLup1 = (block[_PL_LWALL_FIRST, row, col] + block[_PL_LUPSH_FIRST, row, col]) / d_first * zero \
            + block[_PL_LUPSH_FIRST, row, col] / f_first * one
    # weightsumLwall[keep == 1] = 0, then gvfLup2
    weightsumLwall = block[_PL_LWALL, row, col]
    if keep == one:
        block[_PL_LWALL, row, col] = zero
        weightsumLwall = zero
    if wallsuninfluence_second:
        gvfLup2 = (weightsumLwall + block[_PL_LUPSH, row, col]) / d_second * one \
            + block[_PL_LUPSH, row, col] / f_second * zero
    else:
        gvfLup2 = (weightsumLwall + block[_PL_LUPSH, row, col]) / d_second * zero \
            + block[_PL_LUPSH, row, col] / f_second * one
    # gvfalb1
    if wallsuninfluence_first:
        gvfalb1 = (block[_PL_ALBWALL_FIRST, row, col] + block[_PL_ALBSH_FIRST, row, col]) / d_first * one \
            + block[_PL_ALBSH_FIRST, row, col] / f_first * zero
    else:
        gvfalb1 = (block[_PL_ALBWALL_FIRST, row, col] + block[_PL_ALBSH_FIRST, row, col]) / d_first * zero \
            + block[_PL_ALBSH_FIRST, row, col] / f_first * one
    # weightsumalbwall[keep == 1] = 0, then gvfalb2
    weightsumalbwall = block[_PL_ALBWALL, row, col]
    if keep == one:
        block[_PL_ALBWALL, row, col] = zero
        weightsumalbwall = zero
    if wallsuninfluence_second:
        gvfalb2 = (weightsumalbwall + block[_PL_ALBSH, row, col]) / d_second * one \
            + block[_PL_ALBSH, row, col] / f_second * zero
    else:
        gvfalb2 = (weightsumalbwall + block[_PL_ALBSH, row, col]) / d_second * zero \
            + block[_PL_ALBSH, row, col] / f_second * one
    # gvfalbnosh1 / gvfalbnosh2 (second denominator without the +1)
    if wallinfluence_first:
        gvfalbnosh1 = (block[_PL_ALBWALLNOSH_FIRST, row, col] + block[_PL_ALBNOSH_FIRST, row, col]) / d_first * one \
            + block[_PL_ALBNOSH_FIRST, row, col] / f_first * zero
    else:
        gvfalbnosh1 = (block[_PL_ALBWALLNOSH_FIRST, row, col] + block[_PL_ALBNOSH_FIRST, row, col]) / d_first * zero \
            + block[_PL_ALBNOSH_FIRST, row, col] / f_first * one
    if wallinfluence_second:
        gvfalbnosh2 = (weightsumalbwallnosh + block[_PL_ALBNOSH, row, col]) / f_second * one \
            + block[_PL_ALBNOSH, row, col] / f_second * zero
    else:
        gvfalbnosh2 = (weightsumalbwallnosh + block[_PL_ALBNOSH, row, col]) / f_second * zero \
            + block[_PL_ALBNOSH, row, col] / f_second * one
    # Return-path combinations: (gvfX1*0.5 + gvfX2*0.4)/0.9, then the
    # per-field additive terms in the original order.
    gvf_out = (gvf1 * half + gvf2raw * pfour) / pnine
    gvfLup_out = (gvfLup1 * half + gvfLup2 * pfour) / pnine + lup_term_b[row, col]
    gvfalb_out = (gvfalb1 * half + gvfalb2 * pfour) / pnine + alb_term_b[row, col]
    gvfalbnosh_out = (gvfalbnosh1 * half + gvfalbnosh2 * pfour) / pnine * buildings_b[row, col] \
        + nosh_term_b[row, col]
    if np.isnan(gvf_out) or np.isinf(gvf_out):
        flagged = 1
    if np.isnan(gvfLup_out) or np.isinf(gvfLup_out):
        flagged = 1
    if np.isnan(gvfalb_out) or np.isinf(gvfalb_out):
        flagged = 1
    if np.isnan(gvfalbnosh_out) or np.isinf(gvfalbnosh_out):
        flagged = 1
    gvf[row, col] = gvf_out
    gvfLup[row, col] = gvfLup_out
    gvfalb[row, col] = gvfalb_out
    gvfalbnosh[row, col] = gvfalbnosh_out
    return flagged


@njit(inline='always', fastmath=False)
def _alloc_outputs(rows, cols):
    return (np.empty((rows, cols), dtype=np.float32),
            np.empty((rows, cols), dtype=np.float32),
            np.empty((rows, cols), dtype=np.float32),
            np.empty((rows, cols), dtype=np.float32),
            np.empty((rows, cols), dtype=np.float32))


@njit(cache=True, fastmath=False)
def _postprocess_block_serial(block, buildings_b, facesh_b, lup_term_b, alb_term_b, nosh_term_b, first, second):
    rows = block.shape[1]
    cols = block.shape[2]
    gvf, gvfLup, gvfalb, gvfalbnosh, gvf2 = _alloc_outputs(rows, cols)
    flagged = 0
    for row in range(rows):
        for col in range(cols):
            flagged += _postprocess_pixel(block, buildings_b, facesh_b, lup_term_b, alb_term_b, nosh_term_b,
                                          first, second, row, col, gvf, gvfLup, gvfalb, gvfalbnosh, gvf2)
    return gvf, gvfLup, gvfalb, gvfalbnosh, gvf2, flagged


@njit(cache=True, fastmath=False, parallel=True)
def _postprocess_block_parallel(block, buildings_b, facesh_b, lup_term_b, alb_term_b, nosh_term_b, first, second):
    rows = block.shape[1]
    cols = block.shape[2]
    gvf, gvfLup, gvfalb, gvfalbnosh, gvf2 = _alloc_outputs(rows, cols)
    # Every output element and every in-place plane write depends only on the
    # same (row, col) inputs, so the parallel schedule is order-identical per
    # element; flagged accumulates disjoint per-row contributions by sum.
    flagged = 0
    for row in prange(rows):
        for col in range(cols):
            flagged += _postprocess_pixel(block, buildings_b, facesh_b, lup_term_b, alb_term_b, nosh_term_b,
                                          first, second, row, col, gvf, gvfLup, gvfalb, gvfalbnosh, gvf2)
    return gvf, gvfLup, gvfalb, gvfalbnosh, gvf2, flagged


# Debug node export, used only by the C6-31 parity tests and evidence harness
# (never by the integrated path). Extra-plane layout: one float32 plane per
# exported intermediate node, in the original statement order.
DEBUG_NODES = (
    'wallsuninfluence_first',   # 0: weightsumwall_first > 0
    'wallinfluence_first',      # 1: weightsumalbwallnosh_first > 0
    'wallsuninfluence_second',  # 2: weightsumwall > 0 (pre-zeroing)
    'wallinfluence_second',     # 3: weightsumalbwallnosh > 0
    'eq_second',                # 4: (weightsumwall == second) as float32
    'keep',                     # 5: after the keep == -1 zeroing
    'gvf1',                     # 6: gvf1 precedes the keep == 1 zeroing
    'keep_mask',                # 7: keep == 1 (the three zeroing masks)
    'gvf2_raw',                 # 8: pre-clamp gvf2
    'gvf2_clamp_mask',          # 9: gvf2_raw > 1.0
    'gvfLup1',                  # 10
    'gvfLup2',                  # 11
    'gvfalb1',                  # 12
    'gvfalb2',                  # 13
    'gvfalbnosh1',              # 14
    'gvfalbnosh2',              # 15
)

_DEBUG_EQ_SECOND = 4
_DEBUG_KEEP = 5
_DEBUG_GVF2_RAW = 8
_DEBUG_GVF2_CLAMP_MASK = 9


@njit(inline='always', fastmath=False)
def _postprocess_pixel_debug(block, buildings_b, facesh_b, lup_term_b, alb_term_b, nosh_term_b,
                             first, second, row, col, gvf, gvfLup, gvfalb, gvfalbnosh, gvf2, nodes):
    """_postprocess_pixel with every intermediate comparison/node exported."""
    zero = np.float32(0.0)
    one = np.float32(1.0)
    half = np.float32(0.5)
    pfour = np.float32(0.4)
    pnine = np.float32(0.9)
    f_first = np.float32(first)
    d_first = np.float32(first + 1.0)
    f_second = np.float32(second)
    d_second = np.float32(second + 1.0)
    weightsumwall_first = block[_PL_WALL_FIRST, row, col]
    weightsumalbwallnosh_first = block[_PL_ALBWALLNOSH_FIRST, row, col]
    weightsumwall = block[_PL_WALL, row, col]
    weightsumalbwallnosh = block[_PL_ALBWALLNOSH, row, col]
    wallsuninfluence_first = weightsumwall_first > zero
    wallinfluence_first = weightsumalbwallnosh_first > zero
    wallsuninfluence_second = weightsumwall > zero
    wallinfluence_second = weightsumalbwallnosh > zero
    nodes[0, row, col] = np.float32(1.0) if wallsuninfluence_first else zero
    nodes[1, row, col] = np.float32(1.0) if wallinfluence_first else zero
    nodes[2, row, col] = np.float32(1.0) if wallsuninfluence_second else zero
    nodes[3, row, col] = np.float32(1.0) if wallinfluence_second else zero
    if weightsumwall == second:
        keep = one - facesh_b[row, col]
        nodes[_DEBUG_EQ_SECOND, row, col] = one
    else:
        keep = zero - facesh_b[row, col]
        nodes[_DEBUG_EQ_SECOND, row, col] = zero
    if keep == np.float32(-1.0):
        keep = zero
    nodes[_DEBUG_KEEP, row, col] = keep
    if wallsuninfluence_first:
        gvf1 = (weightsumwall_first + block[_PL_SH_FIRST, row, col]) / d_first * one \
            + block[_PL_SH_FIRST, row, col] / f_first * zero
    else:
        gvf1 = (weightsumwall_first + block[_PL_SH_FIRST, row, col]) / d_first * zero \
            + block[_PL_SH_FIRST, row, col] / f_first * one
    nodes[6, row, col] = gvf1
    keep_mask = keep == one
    nodes[7, row, col] = np.float32(1.0) if keep_mask else zero
    if keep_mask:
        block[_PL_WALL, row, col] = zero
        weightsumwall = zero
    if wallsuninfluence_second:
        gvf2raw = (weightsumwall + block[_PL_SH, row, col]) / d_second * one \
            + block[_PL_SH, row, col] / f_second * zero
    else:
        gvf2raw = (weightsumwall + block[_PL_SH, row, col]) / d_second * zero \
            + block[_PL_SH, row, col] / f_second * one
    clamp = gvf2raw > one
    nodes[_DEBUG_GVF2_RAW, row, col] = gvf2raw
    nodes[_DEBUG_GVF2_CLAMP_MASK, row, col] = np.float32(1.0) if clamp else zero
    flagged = 0
    if np.isnan(gvf2raw) or np.isinf(gvf2raw):
        flagged = 1
    if clamp:
        gvf2raw = one
    gvf2[row, col] = gvf2raw
    if wallsuninfluence_first:
        gvfLup1 = (block[_PL_LWALL_FIRST, row, col] + block[_PL_LUPSH_FIRST, row, col]) / d_first * one \
            + block[_PL_LUPSH_FIRST, row, col] / f_first * zero
    else:
        gvfLup1 = (block[_PL_LWALL_FIRST, row, col] + block[_PL_LUPSH_FIRST, row, col]) / d_first * zero \
            + block[_PL_LUPSH_FIRST, row, col] / f_first * one
    nodes[10, row, col] = gvfLup1
    if keep_mask:
        block[_PL_LWALL, row, col] = zero
        weightsumLwall = zero
    else:
        weightsumLwall = block[_PL_LWALL, row, col]
    if wallsuninfluence_second:
        gvfLup2 = (weightsumLwall + block[_PL_LUPSH, row, col]) / d_second * one \
            + block[_PL_LUPSH, row, col] / f_second * zero
    else:
        gvfLup2 = (weightsumLwall + block[_PL_LUPSH, row, col]) / d_second * zero \
            + block[_PL_LUPSH, row, col] / f_second * one
    nodes[11, row, col] = gvfLup2
    if wallsuninfluence_first:
        gvfalb1 = (block[_PL_ALBWALL_FIRST, row, col] + block[_PL_ALBSH_FIRST, row, col]) / d_first * one \
            + block[_PL_ALBSH_FIRST, row, col] / f_first * zero
    else:
        gvfalb1 = (block[_PL_ALBWALL_FIRST, row, col] + block[_PL_ALBSH_FIRST, row, col]) / d_first * zero \
            + block[_PL_ALBSH_FIRST, row, col] / f_first * one
    nodes[12, row, col] = gvfalb1
    if keep_mask:
        block[_PL_ALBWALL, row, col] = zero
        weightsumalbwall = zero
    else:
        weightsumalbwall = block[_PL_ALBWALL, row, col]
    if wallsuninfluence_second:
        gvfalb2 = (weightsumalbwall + block[_PL_ALBSH, row, col]) / d_second * one \
            + block[_PL_ALBSH, row, col] / f_second * zero
    else:
        gvfalb2 = (weightsumalbwall + block[_PL_ALBSH, row, col]) / d_second * zero \
            + block[_PL_ALBSH, row, col] / f_second * one
    nodes[13, row, col] = gvfalb2
    if wallinfluence_first:
        gvfalbnosh1 = (block[_PL_ALBWALLNOSH_FIRST, row, col] + block[_PL_ALBNOSH_FIRST, row, col]) / d_first * one \
            + block[_PL_ALBNOSH_FIRST, row, col] / f_first * zero
    else:
        gvfalbnosh1 = (block[_PL_ALBWALLNOSH_FIRST, row, col] + block[_PL_ALBNOSH_FIRST, row, col]) / d_first * zero \
            + block[_PL_ALBNOSH_FIRST, row, col] / f_first * one
    nodes[14, row, col] = gvfalbnosh1
    if wallinfluence_second:
        gvfalbnosh2 = (weightsumalbwallnosh + block[_PL_ALBNOSH, row, col]) / f_second * one \
            + block[_PL_ALBNOSH, row, col] / f_second * zero
    else:
        gvfalbnosh2 = (weightsumalbwallnosh + block[_PL_ALBNOSH, row, col]) / f_second * zero \
            + block[_PL_ALBNOSH, row, col] / f_second * one
    nodes[15, row, col] = gvfalbnosh2
    gvf_out = (gvf1 * half + gvf2raw * pfour) / pnine
    gvfLup_out = (gvfLup1 * half + gvfLup2 * pfour) / pnine + lup_term_b[row, col]
    gvfalb_out = (gvfalb1 * half + gvfalb2 * pfour) / pnine + alb_term_b[row, col]
    gvfalbnosh_out = (gvfalbnosh1 * half + gvfalbnosh2 * pfour) / pnine * buildings_b[row, col] \
        + nosh_term_b[row, col]
    if np.isnan(gvf_out) or np.isinf(gvf_out):
        flagged = 1
    if np.isnan(gvfLup_out) or np.isinf(gvfLup_out):
        flagged = 1
    if np.isnan(gvfalb_out) or np.isinf(gvfalb_out):
        flagged = 1
    if np.isnan(gvfalbnosh_out) or np.isinf(gvfalbnosh_out):
        flagged = 1
    gvf[row, col] = gvf_out
    gvfLup[row, col] = gvfLup_out
    gvfalb[row, col] = gvfalb_out
    gvfalbnosh[row, col] = gvfalbnosh_out
    return flagged


@njit(cache=True, fastmath=False)
def _postprocess_block_debug(block, buildings_b, facesh_b, lup_term_b, alb_term_b, nosh_term_b, first, second):
    """Serial kernel with every intermediate node exported (debug/test only).

    Returns (gvf, gvfLup, gvfalb, gvfalbnosh, gvf2, flagged, nodes) where
    ``nodes`` has one float32 plane per DEBUG_NODES entry, in order.
    """
    rows = block.shape[1]
    cols = block.shape[2]
    gvf, gvfLup, gvfalb, gvfalbnosh, gvf2 = _alloc_outputs(rows, cols)
    nodes = np.empty((len(DEBUG_NODES), rows, cols), dtype=np.float32)
    flagged = 0
    for row in range(rows):
        for col in range(cols):
            flagged += _postprocess_pixel_debug(block, buildings_b, facesh_b, lup_term_b, alb_term_b, nosh_term_b,
                                                first, second, row, col,
                                                gvf, gvfLup, gvfalb, gvfalbnosh, gvf2, nodes)
    return gvf, gvfLup, gvfalb, gvfalbnosh, gvf2, flagged, nodes
