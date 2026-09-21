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


@njit(fastmath=False)
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


@njit(fastmath=False)
def _diff(shadow, vegetation):
    for row in range(shadow.shape[0]):
        for patch in range(shadow.shape[1]):
            difference = np.float32(np.float32(1)-vegetation[row, patch])
            product = np.float32(difference*np.float32(1-.03))
            shadow[row, patch] = np.float32(shadow[row, patch]-product)
    return shadow


@njit(fastmath=False)
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
