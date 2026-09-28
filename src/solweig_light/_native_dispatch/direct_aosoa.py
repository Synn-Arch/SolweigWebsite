#SOLWEIG-GPU: GPU-accelerated SOLWEIG model for urban thermal comfort simulation
#Copyright (C) 2022–2025 Harsh Kamath and Naveen Sudharsan

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
"""N8-11 direct AoSoA producer (dossier 02_direct_aosoa, TASKS N8-11).

Emits the consumer lane layout DIRECTLY from the packed visibility streams:
no intermediate dense [B,P] decode and no post-decode transpose. For lane
width W and G=ceil(B/W) (B=stop-start), a channel is stored as uint32
[G,P,W] C order, native element index ((g*P+p)*W+lane). For every valid
pixel x=g*W+lane the stored element equals the original V[x,p] IEEE-754
bit pattern exactly: binary/ternary codes map to the codebook uint32
constants (0x00000000/0x3f800000/0x40000000), raw payloads use the same
little-endian byte assembly _decode performs, and nothing is ever rounded
through a float conversion of arbitrary raw values.

Padding lanes (x >= B in the final gang) are never written by the producer
and never read from payloads; consumers must not read them. Original
visibility is never cast to bool.

Two producer loop orders exist and are both measured (see
measure_direct_aosoa.py; patchmajor wins on every mix/width -- the blocked
order's mandatory preflight re-reads the whole packed stream, and its
sequential-write advantage does not pay for it):

* ``patchmajor`` (default, measured winner) iterates patch-outer with
  (gang, lane) nested inner loops -- _decode's pixel order with the
  row/width division hoisted so the store index stays affine -- so the
  reserved-code-3 IndexError surfaces at the identical (patch, pixel)
  position by construction, with no preflight pass;
* ``blocked`` iterates gang/patch/lane for fully sequential output writes,
  which changes the error discovery order, so a pre-launch validation pass
  (_preflight_packed, _preflight_flat's exact order over zero-copy
  per-patch views) runs first and its cost is part of the producer.

Admission mirrors decode_block/_packed_leaves in spirit: only the exact
PackedVisibility type and MappedVisibility instances are admitted, with
the owner lease (stable-id lock order + _check_open) held through the
kernel and decode_block's range contract restated per leaf; everything
else returns None so the caller keeps the legacy route. Descriptors reuse
visibility_compiled._descriptor: zero-copy readonly per-patch uint8 views
plus mode bytes, cached per channel -- no full-payload flatten/copy. If a
caller opts into the _fused_descriptor flat copy instead, that copy must
be counted in producer timing and memory (see measure_direct_aosoa.py).

Classification (dossier stage 2): classify_block_aosoa produces the
sun/shade Boolean masks directly in AoSoA with the SAME arithmetic the
accepted exact tables run -- coefficients from patch_radiation's
_class_coefficients (R04+G06 exact tables reused via import, not
reimplemented), heights from _math_profile.tan32, per-element SLEEF
atan_fma core, float32 coefficient cast and strict < / > comparisons.
Both bits false at equality/NaN boundaries is preserved (shade is not
always not-sun).

lw_primary_aosoa_serial is a TEST/proof consumer only: _longwave_primary_serial
from the N8-04 frozen contract, verbatim, with AoSoA reads. It exists to
prove end-to-end layout correctness against the frozen reducer graph; it
is not offered as a backend win.
"""
from contextlib import ExitStack

import numpy as np
from numba import njit

from solweig_light.geometry.visibility import LazyDiffVisibility, PackedVisibility
from solweig_light.geometry.visibility_compiled import _descriptor
from solweig_light.geometry.visibility_native import MappedVisibility
from solweig_light.radiation._math_profile import tan32
from solweig_light.radiation._sleef_classifier import atan_fma
from solweig_light.radiation.patch_radiation import _class_coefficients

WIDTHS = (8, 4)  # shortlist: current W=8 plus the other native gang width


def _admitted_leaf(leaf):
    """Exact-type admission; duck types/subclasses/lazy leaves keep legacy."""
    return type(leaf) is PackedVisibility or isinstance(leaf, MappedVisibility)


@njit(cache=True, fastmath=False)
def _produce_patchmajor(payloads, modes, start, stop, patches, out, width):
    """Decode straight into [G,P,W] in _decode's patch-major, pixel-inner order.

    The reserved-code check therefore fires at the same (patch, pixel) as
    _decode with no preflight pass. Pixels iterate as (gang, lane) nested
    loops -- the same ascending row order with the row/width division
    hoisted out of the inner loop, keeping the store index affine so the
    raw-mode stream stays vectorizable like _decode's. Padding lanes are
    never visited because the tail gang's lane loop is bounded.

    N9 F1D mode specialization: the mode branch is hoisted out of the pixel
    loops to once per patch iteration, and the packed arms use literal
    shifts/masks -- binary ``(data[pixel >> 3] >> (pixel & 7)) & 1`` and
    ternary ``(data[pixel >> 2] >> ((pixel & 3) << 1)) & 3`` -- which are
    exactly ``pixel//8, pixel%8`` and ``pixel//4, pixel%4`` for the
    non-negative pixels the range contract guarantees. Codebook constants,
    the reserved-code check at the identical position, tails, padding-lane
    non-writes and the raw/fallback arm (the original generic body,
    verbatim) are all unchanged, so every observable -- bits, first-error
    (patch, pixel), error kind -- is identical for every mode value.
    """
    rows = stop - start
    full = rows // width
    tail = rows - full * width
    for patch in range(patches):
        data = payloads[patch]
        mode = modes[patch]
        if mode == 1:
            for gang in range(full):
                pixel = start + gang * width
                for lane in range(width):
                    code = (data[pixel >> 3] >> (pixel & 7)) & 1
                    if code == 3:
                        raise IndexError('Reserved visibility code')
                    value = np.uint32(0) if code == 0 else np.uint32(0x3f800000) if code == 1 else np.uint32(0x40000000)
                    out[gang, patch, lane] = value
                    pixel += 1
            if tail:
                pixel = start + full * width
                for lane in range(tail):
                    code = (data[pixel >> 3] >> (pixel & 7)) & 1
                    if code == 3:
                        raise IndexError('Reserved visibility code')
                    value = np.uint32(0) if code == 0 else np.uint32(0x3f800000) if code == 1 else np.uint32(0x40000000)
                    out[full, patch, lane] = value
                    pixel += 1
        elif mode == 2:
            for gang in range(full):
                pixel = start + gang * width
                for lane in range(width):
                    code = (data[pixel >> 2] >> ((pixel & 3) << 1)) & 3
                    if code == 3:
                        raise IndexError('Reserved visibility code')
                    value = np.uint32(0) if code == 0 else np.uint32(0x3f800000) if code == 1 else np.uint32(0x40000000)
                    out[gang, patch, lane] = value
                    pixel += 1
            if tail:
                pixel = start + full * width
                for lane in range(tail):
                    code = (data[pixel >> 2] >> ((pixel & 3) << 1)) & 3
                    if code == 3:
                        raise IndexError('Reserved visibility code')
                    value = np.uint32(0) if code == 0 else np.uint32(0x3f800000) if code == 1 else np.uint32(0x40000000)
                    out[full, patch, lane] = value
                    pixel += 1
        else:
            for gang in range(full):
                pixel = start + gang * width
                for lane in range(width):
                    if mode == 4:
                        offset = pixel * 4
                        value = (np.uint32(data[offset]) | (np.uint32(data[offset+1]) << 8)
                                 | (np.uint32(data[offset+2]) << 16) | (np.uint32(data[offset+3]) << 24))
                    else:
                        code = (data[pixel // (8 // mode)] >> ((pixel % (8 // mode))*mode)) & ((1 << mode)-1)
                        if code == 3:
                            raise IndexError('Reserved visibility code')
                        value = np.uint32(0) if code == 0 else np.uint32(0x3f800000) if code == 1 else np.uint32(0x40000000)
                    out[gang, patch, lane] = value
                    pixel += 1
            if tail:
                pixel = start + full * width
                for lane in range(tail):
                    if mode == 4:
                        offset = pixel * 4
                        value = (np.uint32(data[offset]) | (np.uint32(data[offset+1]) << 8)
                                 | (np.uint32(data[offset+2]) << 16) | (np.uint32(data[offset+3]) << 24))
                    else:
                        code = (data[pixel // (8 // mode)] >> ((pixel % (8 // mode))*mode)) & ((1 << mode)-1)
                        if code == 3:
                            raise IndexError('Reserved visibility code')
                        value = np.uint32(0) if code == 0 else np.uint32(0x3f800000) if code == 1 else np.uint32(0x40000000)
                    out[full, patch, lane] = value
                    pixel += 1


@njit(cache=True, fastmath=False)
def _preflight_packed(payloads, modes, start, stop, patches):
    """Reject reserved codes in _decode's exact patch-major, pixel-inner order.

    _preflight_flat's order and arithmetic over the zero-copy per-patch views
    (no _fused_descriptor flat copy); required before the blocked producer
    because its discovery order differs. N9 F1D: the mode branch is hoisted
    to once per patch and the packed arms use the same literal shifts/masks
    as the producer (the reserved check stays live in the ternary arm and
    provably dead in the binary arm); the original generic body is kept
    verbatim as the fallback arm.
    """
    for patch in range(patches):
        mode = modes[patch]
        if mode == 4:
            continue
        data = payloads[patch]
        if mode == 1:
            for row in range(stop-start):
                pixel = start + row
                code = (data[pixel >> 3] >> (pixel & 7)) & 1
                if code == 3:
                    raise IndexError('Reserved visibility code')
        elif mode == 2:
            for row in range(stop-start):
                pixel = start + row
                code = (data[pixel >> 2] >> ((pixel & 3) << 1)) & 3
                if code == 3:
                    raise IndexError('Reserved visibility code')
        else:
            for row in range(stop-start):
                pixel = start + row
                code = (data[pixel // (8 // mode)] >> ((pixel % (8 // mode))*mode)) & ((1 << mode)-1)
                if code == 3:
                    raise IndexError('Reserved visibility code')


@njit(cache=True, fastmath=False)
def _produce_blocked(payloads, modes, start, stop, patches, out, width):
    """Decode straight into [G,P,W] gang-major: fully sequential output writes.

    Must be preceded by _preflight_packed so reserved codes are observable in
    the original order; the in-kernel raise is retained as a backstop (a code
    3 reaching this kernel is a caller contract violation, never a silent
    0x40000000 write). The tail gang's lane loop is bounded, so padding lanes
    are neither written nor read. N9 F1D: the mode branch sits once per
    patch iteration (this order's inner loop) and the packed arms use the
    same literal shifts/masks as the patchmajor producer; the raw/fallback
    arm keeps the original generic body verbatim.
    """
    rows = stop - start
    full = rows // width
    tail = rows - full * width
    for gang in range(full):
        base = start + gang * width
        for patch in range(patches):
            data = payloads[patch]
            mode = modes[patch]
            if mode == 1:
                pixel = base
                for lane in range(width):
                    code = (data[pixel >> 3] >> (pixel & 7)) & 1
                    if code == 3:
                        raise IndexError('Reserved visibility code')
                    value = np.uint32(0) if code == 0 else np.uint32(0x3f800000) if code == 1 else np.uint32(0x40000000)
                    out[gang, patch, lane] = value
                    pixel += 1
            elif mode == 2:
                pixel = base
                for lane in range(width):
                    code = (data[pixel >> 2] >> ((pixel & 3) << 1)) & 3
                    if code == 3:
                        raise IndexError('Reserved visibility code')
                    value = np.uint32(0) if code == 0 else np.uint32(0x3f800000) if code == 1 else np.uint32(0x40000000)
                    out[gang, patch, lane] = value
                    pixel += 1
            else:
                pixel = base
                for lane in range(width):
                    if mode == 4:
                        offset = pixel * 4
                        value = (np.uint32(data[offset]) | (np.uint32(data[offset+1]) << 8)
                                 | (np.uint32(data[offset+2]) << 16) | (np.uint32(data[offset+3]) << 24))
                    else:
                        code = (data[pixel // (8 // mode)] >> ((pixel % (8 // mode))*mode)) & ((1 << mode)-1)
                        if code == 3:
                            raise IndexError('Reserved visibility code')
                        value = np.uint32(0) if code == 0 else np.uint32(0x3f800000) if code == 1 else np.uint32(0x40000000)
                    out[gang, patch, lane] = value
                    pixel += 1
    if tail:
        base = start + full * width
        for patch in range(patches):
            data = payloads[patch]
            mode = modes[patch]
            if mode == 1:
                pixel = base
                for lane in range(tail):
                    code = (data[pixel >> 3] >> (pixel & 7)) & 1
                    if code == 3:
                        raise IndexError('Reserved visibility code')
                    value = np.uint32(0) if code == 0 else np.uint32(0x3f800000) if code == 1 else np.uint32(0x40000000)
                    out[full, patch, lane] = value
                    pixel += 1
            elif mode == 2:
                pixel = base
                for lane in range(tail):
                    code = (data[pixel >> 2] >> ((pixel & 3) << 1)) & 3
                    if code == 3:
                        raise IndexError('Reserved visibility code')
                    value = np.uint32(0) if code == 0 else np.uint32(0x3f800000) if code == 1 else np.uint32(0x40000000)
                    out[full, patch, lane] = value
                    pixel += 1
            else:
                pixel = base
                for lane in range(tail):
                    if mode == 4:
                        offset = pixel * 4
                        value = (np.uint32(data[offset]) | (np.uint32(data[offset+1]) << 8)
                                 | (np.uint32(data[offset+2]) << 16) | (np.uint32(data[offset+3]) << 24))
                    else:
                        code = (data[pixel // (8 // mode)] >> ((pixel % (8 // mode))*mode)) & ((1 << mode)-1)
                        if code == 3:
                            raise IndexError('Reserved visibility code')
                        value = np.uint32(0) if code == 0 else np.uint32(0x3f800000) if code == 1 else np.uint32(0x40000000)
                    out[full, patch, lane] = value
                    pixel += 1


def _leased(leaves, start, stop, patches):
    """decode_block's lock order and range contract; returns the ExitStack."""
    stack = ExitStack()
    try:
        owners = sorted({id(leaf): leaf for leaf in leaves if hasattr(leaf, '_lock')}.values(), key=id)
        for owner in owners:
            stack.enter_context(owner._lock)
            owner._check_open()
        for leaf in leaves:
            if not 0 <= start <= stop <= leaf.shape[0]*leaf.shape[1] or not 0 <= patches <= leaf.shape[2]:
                raise IndexError('Visibility block interval out of range')
    except BaseException:
        stack.close()
        raise
    return stack


def _run_order(channel, start, stop, patches, out, order):
    payloads, modes = _descriptor(channel)
    if order == 'patchmajor':
        _produce_patchmajor(payloads, modes, start, stop, patches, out, out.shape[2])
    elif order == 'blocked':
        _preflight_packed(payloads, modes, start, stop, patches)
        _produce_blocked(payloads, modes, start, stop, patches, out, out.shape[2])
    else:
        raise ValueError('producer order must be patchmajor or blocked')


def produce_block_aosoa(channel, start, stop, patches, *, width=8, order='patchmajor', out=None):
    """One channel's block decoded directly into the AoSoA lane layout.

    Returns uint32 [G,P,W] (G=ceil((stop-start)/width)) whose valid pixels
    carry _decode's exact bits, or None when the channel is not an admitted
    packed leaf (caller keeps the legacy route). ``out`` may supply a
    preallocated/poison-filled C-contiguous uint32 [G,P,W] buffer; padding
    lanes are never written. The float32 view is out.view(np.float32).
    """
    if not _admitted_leaf(channel) or isinstance(channel, LazyDiffVisibility):
        return None
    with _leased((channel,), start, stop, patches):
        gangs = -(-(stop - start) // width)
        if out is None:
            out = np.empty((gangs, patches, width), dtype=np.uint32)
        elif out.dtype != np.uint32 or out.shape != (gangs, patches, width) or not out.flags.c_contiguous:
            raise ValueError('out must be C-contiguous uint32 [ceil(rows/width), patches, width]')
        _run_order(channel, start, stop, patches, out, order)
        return out


def produce_blocks_aosoa(shadow, vegetation, vegetation_building, start, stop, patches,
                         *, width=8, order='patchmajor'):
    """Three-channel single entry (sh/vs/vb): one lease, one validation, three kernels."""
    channels = (shadow, vegetation, vegetation_building)
    if not all(_admitted_leaf(channel) and not isinstance(channel, LazyDiffVisibility)
               for channel in channels):
        return None
    gangs = -(-(stop - start) // width)
    with _leased(channels, start, stop, patches):
        blocks = tuple(np.empty((gangs, patches, width), dtype=np.uint32) for _ in channels)
        for channel, block in zip(channels, blocks):
            _run_order(channel, start, stop, patches, block, order)
        return blocks


@njit(cache=True, fastmath=False, error_model='numpy')
def _classes_table_aosoa(heights, coefficients, radians_to_degrees, altitudes, sun, shade, rows):
    """_classes_table's exact elementwise arithmetic with AoSoA destinations.

    Per element: float32 add of the tan32 height and the per-state
    coefficient, the SLEEF scalar arctangent core atan_array loops over,
    the float32 degree multiply, and the two strict comparisons. The
    arithmetic is character-identical to patch_radiation._classes_table;
    only the destination indexing (and the gang/column/lane write order,
    which cannot change any element's value) differs. Inactive columns and
    padding lanes stay at the zero/poison initialization.
    """
    for gang in range(sun.shape[0]):
        for column in range(coefficients.shape[0]):
            altitude = altitudes[column]
            coefficient = coefficients[column]
            for lane in range(sun.shape[2]):
                row = gang * sun.shape[2] + lane
                if row >= rows:
                    break
                delta = np.float32(heights[row] + coefficient)
                degrees = np.float32(atan_fma(delta) * radians_to_degrees)
                sun[gang, column, lane] = degrees < altitude
                shade[gang, column, lane] = degrees > altitude


@njit(cache=True, fastmath=False, error_model='numpy')
def _classes_table_masked_aosoa(heights, coefficients, radians_to_degrees, altitudes,
                                indices, sun, shade, rows):
    """Masked-column form, mirroring _classes_table_masked's indirection."""
    for gang in range(sun.shape[0]):
        for column in range(coefficients.shape[0]):
            altitude = altitudes[column]
            coefficient = coefficients[column]
            target = indices[column]
            for lane in range(sun.shape[2]):
                row = gang * sun.shape[2] + lane
                if row >= rows:
                    break
                delta = np.float32(heights[row] + coefficient)
                degrees = np.float32(atan_fma(delta) * radians_to_degrees)
                sun[gang, target, lane] = degrees < altitude
                shade[gang, target, lane] = degrees > altitude


def _extent(arr):
    """Exact byte extent of THIS view (the B-control adapter's formula), so
    genuinely disjoint regions of a shared arena are not false positives."""
    addr = arr.ctypes.data
    span = sum((d - 1) * s for d, s in zip(arr.shape, arr.strides)) + arr.itemsize
    return addr, addr + span


def _spans_overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def _validated_out(mask, name, shape):
    """Supplied scratch must be a writeable C-contiguous bool [G,P,W]."""
    if not isinstance(mask, np.ndarray):
        raise ValueError(f'{name} must be a bool ndarray, got {type(mask)!r}')
    if (mask.dtype != np.bool_ or mask.shape != shape
            or not mask.flags.c_contiguous or not mask.flags.writeable):
        raise ValueError(
            f'{name} must be a writeable C-contiguous bool array of shape '
            f'{shape}, got dtype={mask.dtype} shape={mask.shape} '
            f'c_contiguous={mask.flags.c_contiguous} '
            f'writeable={mask.flags.writeable}')


def classify_block_aosoa(altitude, azimuth, geometry, asvf, start, stop, active=None,
                         prepared=None, *, width=8, sun_out=None, shade_out=None):
    """Sun/shade Boolean masks produced directly in the AoSoA lane layout.

    Reuses patch_radiation's _class_coefficients (R04+G06 exact tables,
    coefficient cast and rad2deg provenance) and _math_profile's tan32 field
    pass; the per-element classification arithmetic is _classes_table's.
    Returns (sun, shade) as bool [G,P,W], or None when the retained route
    would fall back to the per-patch python loop (caller uses _classes).
    ``prepared`` accepts the same precomputed tuple _classes takes.

    N9 F1M scratch-reuse contract: supplied ``sun_out``/``shade_out`` are
    validated (writeable C-contiguous bool of the exact shape) and checked
    for overlap against each other and against the classification's input
    storage BEFORE any write; then the valid extent of both masks (every
    lane with row < rows, all columns) is cleared, so the masked kernel's
    inactive columns carry False instead of stale scratch data. Padding
    lanes (row >= rows in the tail gang) stay untouched. The full-write
    invariant therefore holds for every valid lane regardless of the active
    set, and ``shade != not sun`` at equality/NaN is preserved (the
    arithmetic is unchanged).
    """
    field = np.asarray(asvf).reshape(-1)[start:stop].reshape(-1, 1)
    rows = stop - start
    gangs = -(-rows // width)
    if prepared is None:
        prepared = _class_coefficients(altitude, azimuth, geometry, asvf, active)
    if prepared is None:
        return None
    indices, coefficients, rad2deg = prepared
    patches = geometry.altitude.size
    shape = (gangs, patches, width)
    if sun_out is None:
        sun = np.zeros(shape, dtype=np.bool_)
    else:
        _validated_out(sun_out, 'sun_out', shape)
        sun = sun_out
    if shade_out is None:
        shade = np.zeros(shape, dtype=np.bool_)
    else:
        _validated_out(shade_out, 'shade_out', shape)
        shade = shade_out
    if sun_out is not None or shade_out is not None:
        # np.asarray(asvf), not the possibly reshape-copied field view, so the
        # caller's own storage is what the span check sees.
        inputs = (np.asarray(asvf), geometry.altitude, geometry.azimuth,
                  indices, coefficients)
        spans = [(name, _extent(arr)) for name, arr in
                 (('sun_out', sun), ('shade_out', shade), *_reusable_inputs(inputs))
                 if arr.size]
        for i, (name_a, span_a) in enumerate(spans):
            for name_b, span_b in spans[i + 1:]:
                if _spans_overlap(span_a, span_b):
                    raise ValueError(
                        f'{name_a} overlaps {name_b}; rejecting before any write')
        # Clear the valid extent (all columns, lanes with row < rows); padding
        # lanes of the tail gang keep their poison. Allocated masks are
        # already zero, so only supplied scratch pays the memset.
        full = rows // width
        tail = rows - full * width
        for mask in (sun, shade):
            mask[:full] = False
            if tail:
                mask[full, :, :tail] = False
    if indices.size:
        heights = np.ascontiguousarray(tan32(field)).reshape(-1)
        factor = np.asarray(rad2deg, dtype=field.dtype)[()]
        altitudes = geometry.altitude[indices]
        if indices[0] == 0 and indices.size == indices[-1] + 1:
            _classes_table_aosoa(heights, coefficients, factor, altitudes, sun, shade, rows)
        else:
            _classes_table_masked_aosoa(heights, coefficients, factor, altitudes,
                                        indices, sun, shade, rows)
    return sun, shade


def _reusable_inputs(inputs):
    """(name, array) pairs for the overlap check; non-arrays skipped."""
    names = ('asvf field', 'geometry.altitude', 'geometry.azimuth',
             'prepared indices', 'prepared coefficients')
    for name, arr in zip(names, inputs):
        if isinstance(arr, np.ndarray) and arr.size:
            yield name, arr


def pack_masks_aosoa(sun, shade, *, width=8):
    """Stage-1 accounting helper: pack retained [B,P] masks into [G,P,W].

    This is the packing cost a stage-1 (original-layout classification)
    pipeline would pay to feed an AoSoA consumer; measured against direct
    production in measure_direct_aosoa.py. The transpose-copy is the same
    adapter cost the rejected post-decode transform paid.
    """
    rows, patches = sun.shape
    gangs = -(-rows // width)

    def pack(mask):
        if rows != gangs * width:
            padded = np.zeros((gangs * width, patches), dtype=np.bool_)
            padded[:rows] = mask
            mask = padded
        return mask.reshape(gangs, width, patches).transpose(0, 2, 1).copy()

    return pack(sun), pack(shade)


@njit(cache=True, fastmath=False)
def lw_primary_aosoa_serial(sh, vs, vb, sun, shade, solid, sine, cosine, directions,
                            gate, solar_gate, sky_down, sky_side, surface_sun,
                            surface_sh, lup, reflection_factor, rows):
    """PROOF CONSUMER (test/measurement only, not a backend claim).

    _longwave_primary_serial from the N8-04 frozen contract reproduced
    verbatim; the only changes are the AoSoA reads sh/vs/vb/sun/shade
    [G,P,W] at (gang, patch, lane) and the derived row/gang/lane indices.
    Bitwise parity with the frozen kernel/oracle therefore proves the
    producer layout end-to-end.
    """
    gangs, patches, width = sh.shape
    output = np.zeros((rows, 7), dtype=np.float32)
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
