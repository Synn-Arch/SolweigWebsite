"""Serial bounded codec decode; descriptors borrow encoded bytes, never copy them.

Owner-local descriptors retain O(patches) readonly uint8 views and mode bytes.
Construction costs O(patches), with zero encoded-payload copies. Decode allocates
only the requested pixels x patches uint32/float32 output. Mapped owners hold
one lock from descriptor acquisition through completion of the native read.
"""
from contextlib import ExitStack

import numpy as np
from numba import njit, types
from numba.typed import List

from .visibility import PackedVisibility, LazyDiffVisibility


@njit(cache=True, fastmath=False)
def _decode(payloads, modes, start, stop, patches):
    bits = np.empty((stop-start, patches), dtype=np.uint32)
    for patch in range(patches):
        data = payloads[patch]
        mode = modes[patch]
        for row in range(stop-start):
            pixel = start + row
            if mode == 4:
                offset = pixel * 4
                value = (np.uint32(data[offset]) | (np.uint32(data[offset+1]) << 8)
                         | (np.uint32(data[offset+2]) << 16) | (np.uint32(data[offset+3]) << 24))
            else:
                code = (data[pixel // (8 // mode)] >> ((pixel % (8 // mode))*mode)) & ((1 << mode)-1)
                if code == 3:
                    raise IndexError('Reserved visibility code')
                value = np.uint32(0) if code == 0 else np.uint32(0x3f800000) if code == 1 else np.uint32(0x40000000)
            bits[row, patch] = value
    return bits.view(np.float32)


@njit(cache=True, fastmath=False)
def _preflight_flat(flat, offsets, modes, start, stop, patches):
    """Reject reserved codes in _decode's exact patch-major, pixel-inner order."""
    for patch in range(patches):
        mode = modes[patch]
        if mode == 4:
            continue
        base = offsets[patch]
        for row in range(stop-start):
            pixel = start + row
            code = (flat[base + pixel // (8 // mode)]
                    >> ((pixel % (8 // mode))*mode)) & ((1 << mode)-1)
            if code == 3:
                raise IndexError('Reserved visibility code')


@njit(cache=True, fastmath=False, inline='always')
def _decode_at(flat, offsets, modes, patch, pixel):
    """Decode one (patch, pixel) value with _decode's exact bits.

    Binary/ternary codes map to the codebook float patterns (0.0, 1.0, 2.0
    carry the exact codebook bit patterns); raw mode bitcasts the stored
    little-endian uint32. Reserved codes are handled by _preflight_flat only.
    """
    mode = modes[patch]
    base = offsets[patch]
    if mode == 4:
        offset = base + pixel*4
        bits = (np.uint32(flat[offset]) | (np.uint32(flat[offset+1]) << 8)
                | (np.uint32(flat[offset+2]) << 16) | (np.uint32(flat[offset+3]) << 24))
        return np.uint32(bits).view(np.float32)
    code = (flat[base + pixel // (8 // mode)]
            >> ((pixel % (8 // mode))*mode)) & ((1 << mode)-1)
    return np.float32(code)


@njit(cache=True, fastmath=False)
def _diff(shadow, vegetation):
    for row in range(shadow.shape[0]):
        for patch in range(shadow.shape[1]):
            difference = np.float32(np.float32(1)-vegetation[row, patch])
            product = np.float32(difference*np.float32(1-.03))
            shadow[row, patch] = np.float32(shadow[row, patch]-product)
    return shadow


@njit(cache=True, fastmath=False)
def _decode_diff(sh_payloads, sh_modes, veg_payloads, veg_modes, start, stop, patches):
    return _diff(_decode(sh_payloads, sh_modes, start, stop, patches),
                 _decode(veg_payloads, veg_modes, start, stop, patches))


def _descriptor(channel):
    descriptor = getattr(channel, '_block_descriptor', None)
    if descriptor is None:
        payloads = List.empty_list(types.Array(types.uint8, 1, 'C', readonly=True))
        for patch in channel._patches:
            view = np.frombuffer(patch.payload, dtype=np.uint8)
            view.setflags(write=False)
            payloads.append(view)
        modes = np.array([{'binary': 1, 'ternary': 2, 'raw': 4}[patch.mode]
                          for patch in channel._patches], dtype=np.uint8)
        descriptor = payloads, modes
        object.__setattr__(channel, '_block_descriptor', descriptor)
    return descriptor


def _fused_descriptor(channel):
    """Flat payload layout for the fused kernels, cached per channel.

    (flat uint8 payload, int64 offsets[patches+1], uint8 modes). The flat copy
    lets the kernels index plain arrays instead of per-patch typed-list boxes,
    and identity of the cached descriptor doubles as payload-identity, so the
    dispatcher can skip re-decoding a lazy leaf that shares its base channel.
    """
    descriptor = getattr(channel, '_fused_block_descriptor', None)
    if descriptor is None:
        payloads, modes = _descriptor(channel)
        offsets = np.zeros(len(payloads)+1, dtype=np.int64)
        for patch in range(len(payloads)):
            offsets[patch+1] = offsets[patch] + payloads[patch].shape[0]
        flat = np.empty(int(offsets[-1]), dtype=np.uint8)
        for patch in range(len(payloads)):
            flat[offsets[patch]:offsets[patch+1]] = payloads[patch]
        descriptor = flat, offsets, modes
        object.__setattr__(channel, '_fused_block_descriptor', descriptor)
    return descriptor


def decode_block(channel, start, stop, patches):
    """Return native block, or None for duck types and unsupported lazy leaves."""
    leaves = (channel.shadow, channel.vegetation) if isinstance(channel, LazyDiffVisibility) else (channel,)
    if not all(isinstance(leaf, PackedVisibility) for leaf in leaves):
        return None
    # Stable lock order also covers two mapped owners shared by reverse pairs.
    owners = sorted({id(leaf): leaf for leaf in leaves if hasattr(leaf, '_lock')}.values(), key=id)
    with ExitStack() as stack:
        for owner in owners:
            stack.enter_context(owner._lock)
            owner._check_open()
        for leaf in leaves:
            if not 0 <= start <= stop <= leaf.shape[0]*leaf.shape[1] or not 0 <= patches <= leaf.shape[2]:
                raise IndexError('Visibility block interval out of range')
        if len(leaves) == 1:
            return _decode(*_descriptor(leaves[0]), start, stop, patches)
        return _decode_diff(*_descriptor(leaves[0]), *_descriptor(leaves[1]), start, stop, patches)


def diff_from_shared_decoded(shadow, vegetation, diffuse, decoded_shadow,
                             decoded_vegetation, start, stop, patches):
    """Reuse decoded leaves after reproducing the lazy read's validation point.

    ``None`` is an exact fallback signal. Admission is deliberately restricted
    to the two known immutable implementations, never arbitrary subclasses.
    The caller must perform the original shadow, vegetation, and intervening
    vegetation/building reads before calling this function.
    """
    from .visibility_native import MappedVisibility
    immutable = (PackedVisibility, MappedVisibility)
    if (type(diffuse) is not LazyDiffVisibility
            or diffuse.shadow is not shadow or diffuse.vegetation is not vegetation
            or type(shadow) not in immutable or type(vegetation) not in immutable):
        return None
    leaves = (shadow, vegetation)
    owners = sorted({id(leaf): leaf for leaf in leaves if hasattr(leaf, '_lock')}.values(), key=id)
    with ExitStack() as stack:
        for owner in owners:
            stack.enter_context(owner._lock)
            owner._check_open()
        for leaf in leaves:
            if not 0 <= start <= stop <= leaf.shape[0]*leaf.shape[1] or not 0 <= patches <= leaf.shape[2]:
                raise IndexError('Visibility block interval out of range')
        # Keep locks through the unchanged arithmetic, matching the original
        # lazy decode's owner lifetime. Copy because _diff mutates its first arg.
        return _diff(decoded_shadow.copy(), decoded_vegetation)
