"""PRIVATE prepared multi-channel visibility decode (v6 dossier 09/D06).

The accepted radiation path decodes each demanded channel with a separate
``decode_block`` entry: per-channel owner-lock passes, range checks, descriptor
fetches and native dispatches, plus a full decoded-shadow copy inside the lazy
diffuse reuse. This module prepares all channels demanded by one consumer
shortwave (shadow, vegetation, vegetation_building, diffuse) or longwave
(shadow, vegetation, vegetation_building) and decodes them in ONE native entry
into caller-owned or exactly-sized buffers.

Exactness contract (validated under tests/optimization_v6/decoder/):

- Admission never raises and never fires an original error checkpoint early.
  It mirrors the accepted path's own predicates: base channels must satisfy
  ``decode_block``'s direct packed admission (every leaf isinstance
  ``PackedVisibility`` after lazy unwrapping, single leaf); the optional
  diffuse slot admits exactly what the accepted consumer admits: the shared
  ``LazyDiffVisibility`` pair under ``diff_from_shared_decoded``'s exact-type
  identity rule (reuse), an independent lazy pair over packed leaves, or a
  direct packed channel. Anything else yields ``None`` so the caller falls
  back COMPLETELY to the original path.

- Decode re-checks owner state at the original checkpoints, in the original
  observable order: per demanded channel, mapped-owner open checks (stable
  sorted-by-id owner order), then per-leaf interval checks, then the
  reserved-code scan of that channel's fresh decode streams in _decode's
  patch-major, pixel-inner order. Cross-channel precedence (an earlier
  channel's reserved discovery before a later channel's open/range error) is
  therefore preserved, unlike a scheme that hoists every check in front of
  one kernel.

- Kernel arithmetic is copied verbatim from visibility_compiled._decode and
  _diff: same bit classification (raw little-endian uint32 bitcast; binary/
  ternary codebook patterns 0x00000000/0x3f800000/0x40000000), same reserved
  code error type/message, same float32 cast points, ``fastmath=False``, no
  reassociation, serial order.

- No hidden whole-payload copy: descriptors borrow per-patch readonly views
  through the shared ``_block_descriptor`` cache (identical object identity
  as the accepted path); no flat payload copy exists here. Outputs go into
  caller buffers or exactly-sized (stop-start, patches) allocations. The
  independent lazy pair uses one exactly-sized uint32 scratch, and the reuse
  diffuse needs no decoded-shadow copy.

Lock lifetime: decode holds every involved mapped-owner lock, acquired once
in globally sorted-by-id order (the established _fused_guard convention),
from before the first open check until the native read completes. A
concurrent close therefore blocks until the combined decode finishes instead
of interleaving between channel decodes; serial callers observe identical
results and identical RuntimeError('Native visibility is closed') timing.
Prepared objects are reusable across start/stop calls and hold strong owner
references; owners may still be closed by anyone at any time, which the next
decode reports exactly as the original path would.

This module is PRIVATE to the v6 radiation demand. Exporters, the geometry
service verifier and every public codec entry are unaffected; implicit dense
conversion does not exist here.
"""
import os
from contextlib import ExitStack
from dataclasses import dataclass

import numpy as np
from numba import njit, types
from numba.typed import List

from .visibility import LazyDiffVisibility, PackedVisibility
from .visibility_compiled import _descriptor
from .visibility_native import MappedVisibility

# diff_from_shared_decoded's exact immutable-owner tuple, mirrored verbatim.
_IMMUTABLE_OWNERS = (PackedVisibility, MappedVisibility)

# Diffuse-slot kinds consumed by the combined kernel.
_DIFF_NONE = 0      # longwave demand: no diffuse channel
_DIFF_DIRECT = 1    # packed diffuse channel decoded from its own stream
_DIFF_REUSE = 2     # shared lazy pair: diff arithmetic over decoded channels
_DIFF_LAZY_PAIR = 3  # independent lazy pair: decode both leaves, then subtract


@njit(cache=True, fastmath=False)
def _preflight(payloads, modes, start, stop, patches):
    """Reject reserved codes in _decode's exact patch-major, pixel-inner order."""
    for patch in range(patches):
        mode = modes[patch]
        if mode == 4:
            continue
        data = payloads[patch]
        for row in range(stop-start):
            pixel = start + row
            code = (data[pixel // (8 // mode)]
                    >> ((pixel % (8 // mode))*mode)) & ((1 << mode)-1)
            if code == 3:
                raise IndexError('Reserved visibility code')


@njit(cache=True, fastmath=False)
def _decode_into(payloads, modes, start, stop, patches, out):
    """visibility_compiled._decode's verbatim loop into a caller uint32 buffer."""
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
            out[row, patch] = value


@njit(cache=True, fastmath=False)
def _diff_into(shadow, vegetation, out):
    """visibility_compiled._diff's verbatim arithmetic with a separate target.

    ``out`` may alias ``shadow`` (elementwise same-location read-then-write);
    the reuse diffuse writes the shared pair's subtraction without copying the
    decoded shadow plane, matching diff_from_shared_decoded's values bitwise.
    """
    for row in range(shadow.shape[0]):
        for patch in range(shadow.shape[1]):
            difference = np.float32(np.float32(1)-vegetation[row, patch])
            product = np.float32(difference*np.float32(1-.03))
            out[row, patch] = np.float32(shadow[row, patch]-product)
    return out


@njit(cache=True, fastmath=False)
def _decode_channels(sh_payloads, sh_modes, vs_payloads, vs_modes,
                     vb_payloads, vb_modes, dsh_payloads, dsh_modes,
                     dveg_payloads, dveg_modes, diff_kind, start, stop, patches,
                     out_sh32, out_vs32, out_vb32, out_diff32,
                     out_sh_f, out_vs_f, out_diff_f, scratch32, scratch_f):
    """Decode every demanded channel in one entry, in the original order.

    Stream arguments for slots a kind does not use are never read; callers
    pass typed empty lists and zero-sized arrays of the same dtypes so one
    compiled specialization serves every demand shape. ``scratch32``/
    ``scratch_f`` alias one buffer and back _DIFF_LAZY_PAIR only.
    """
    _decode_into(sh_payloads, sh_modes, start, stop, patches, out_sh32)
    _decode_into(vs_payloads, vs_modes, start, stop, patches, out_vs32)
    _decode_into(vb_payloads, vb_modes, start, stop, patches, out_vb32)
    if diff_kind == 1:
        _decode_into(dsh_payloads, dsh_modes, start, stop, patches, out_diff32)
    elif diff_kind == 2:
        _diff_into(out_sh_f, out_vs_f, out_diff_f)
    elif diff_kind == 3:
        # _decode_diff order: shadow leaf first, vegetation leaf second, then
        # the subtraction mutating the shadow decode in place.
        _decode_into(dsh_payloads, dsh_modes, start, stop, patches, out_diff32)
        _decode_into(dveg_payloads, dveg_modes, start, stop, patches, scratch32)
        _diff_into(out_diff_f, scratch_f, out_diff_f)


@dataclass(frozen=True, eq=False)
class _Slot:
    """One demanded channel: borrowed descriptors plus original checkpoints."""
    leaves: tuple
    owners: tuple
    streams: tuple  # (payloads, modes) pairs in original decode order; empty for reuse
    kind: int       # 1 direct, 0 reuse (checks only), 3 lazy pair


@dataclass(frozen=True, eq=False)
class PreparedVisibility:
    """PRIVATE stage-preparable decode handle (dossier 09 interface).

    Holds recognized immutable owners and borrowed layout descriptors only;
    no payload bytes are copied and no dense cube exists here. Reusable
    across blocks for one fixed demand; complete fallback stays with the
    caller whenever preparation returned None.
    """
    slots: tuple
    owners: tuple  # every mapped owner, globally sorted by id, deduplicated
    channels: int
    diff_kind: int

    def decode(self, start, stop, patches, buffers=None):
        """Decode all demanded channels in one native entry.

        Returns the channel arrays in demand order, or raises exactly what
        the original per-channel path raises at the same checkpoint. With
        ``buffers``, each must be a C-contiguous writeable float32 ndarray
        shaped (stop-start, patches); the returned arrays are those objects.
        Without buffers, exactly-sized arrays are allocated.
        """
        if buffers is not None:
            buffers = tuple(buffers)
            if len(buffers) != self.channels:
                raise ValueError('Prepared decode requires one buffer per demanded channel')
            for buffer in buffers:
                if (not isinstance(buffer, np.ndarray) or buffer.dtype != np.float32
                        or buffer.ndim != 2 or not buffer.flags.c_contiguous
                        or not buffer.flags.writeable):
                    raise ValueError('Prepared decode buffers must be C-contiguous writeable float32 matrices')
        with ExitStack() as stack:
            for owner in self.owners:
                stack.enter_context(owner._lock)
            # Original checkpoints, original observable order: per demanded
            # channel, owner open checks, per-leaf interval checks, then that
            # channel's fresh decode streams' reserved-code scan.
            for slot in self.slots:
                for owner in slot.owners:
                    owner._check_open()
                for leaf in slot.leaves:
                    if not 0 <= start <= stop <= leaf.shape[0]*leaf.shape[1] or not 0 <= patches <= leaf.shape[2]:
                        raise IndexError('Visibility block interval out of range')
                for payloads, modes in slot.streams:
                    _preflight(payloads, modes, start, stop, patches)
            rows = stop - start
            outputs = []
            if buffers is None:
                outputs = [np.empty((rows, patches), dtype=np.float32)
                           for _ in range(self.channels)]
            else:
                for buffer in buffers:
                    if buffer.shape != (rows, patches):
                        raise ValueError('Prepared decode buffer shape differs from the block interval')
                    outputs.append(buffer)
            out32 = [output.view(np.uint32) for output in outputs]
            empty_payloads = List.empty_list(types.Array(types.uint8, 1, 'C', readonly=True))
            empty_modes = np.empty(0, dtype=np.uint8)
            zero32 = np.empty((0, 0), dtype=np.uint32)
            zero_f = zero32.view(np.float32)
            direct = [slot.streams[0] for slot in self.slots[:3]]
            if self.diff_kind == _DIFF_DIRECT or self.diff_kind == _DIFF_LAZY_PAIR:
                dsh_payloads, dsh_modes = self.slots[3].streams[0]
            else:
                dsh_payloads, dsh_modes = empty_payloads, empty_modes
            if self.diff_kind == _DIFF_LAZY_PAIR:
                dveg_payloads, dveg_modes = self.slots[3].streams[1]
                scratch32 = np.empty((rows, patches), dtype=np.uint32)
                scratch_f = scratch32.view(np.float32)
            else:
                dveg_payloads, dveg_modes = empty_payloads, empty_modes
                scratch32, scratch_f = zero32, zero_f
            diff32 = out32[3] if self.channels == 4 else zero32
            diff_f = outputs[3] if self.channels == 4 else zero_f
            _decode_channels(direct[0][0], direct[0][1],
                             direct[1][0], direct[1][1],
                             direct[2][0], direct[2][1],
                             dsh_payloads, dsh_modes,
                             dveg_payloads, dveg_modes,
                             self.diff_kind, start, stop, patches,
                             out32[0], out32[1], out32[2], diff32,
                             outputs[0], outputs[1], diff_f,
                             scratch32, scratch_f)
        return tuple(outputs)


def _owners(leaves):
    """decode_block's exact stable owner order for one channel's leaves."""
    return tuple(sorted({id(leaf): leaf for leaf in leaves if hasattr(leaf, '_lock')}.values(), key=id))


def _build_slot(channel):
    """Borrow the shared descriptor for a direct packed channel under its locks.

    Owners invalidate their borrowed descriptor at close (MappedVisibility
    itself nulls ``_block_descriptor`` and drops the mapping), so the bytes of
    a closed owner can never be borrowed here: preparation declines and the
    original path reports the closed owner at its own open checkpoint. The
    ``closed`` read is belt-and-suspenders for owners that stop exposing open
    payload without invalidating the cache attribute.
    """
    cached = getattr(channel, '_block_descriptor', None)
    with ExitStack() as stack:
        owners = _owners((channel,))
        for owner in owners:
            stack.enter_context(owner._lock)
            if cached is None and getattr(owner, 'closed', False):
                return None
        if cached is not None:
            streams = (cached,)
        else:
            streams = (_descriptor(channel),)
        return _Slot((channel,), owners, streams, 1)


def _diffuse_slot(shadow, vegetation, diffuse):
    """Admit the diffuse slot exactly as the accepted consumer sequence does."""
    if (type(diffuse) is LazyDiffVisibility and diffuse.shadow is shadow
            and diffuse.vegetation is vegetation
            and type(shadow) in _IMMUTABLE_OWNERS and type(vegetation) in _IMMUTABLE_OWNERS):
        # Reuse: the accepted path re-validates the shared leaves and applies
        # the subtraction to already-decoded channels; no fresh stream exists.
        return _Slot((shadow, vegetation), _owners((shadow, vegetation)), (), _DIFF_REUSE)
    if isinstance(diffuse, LazyDiffVisibility):
        leaves = (diffuse.shadow, diffuse.vegetation)
        if not all(isinstance(leaf, PackedVisibility) for leaf in leaves):
            return None
        slots = [_build_slot(leaf) for leaf in leaves]
        if any(slot is None for slot in slots):
            return None
        return _Slot(leaves, _owners(leaves),
                     tuple(slot.streams[0] for slot in slots), _DIFF_LAZY_PAIR)
    if isinstance(diffuse, PackedVisibility):
        slot = _build_slot(diffuse)
        return slot
    return None


def _prepared_enabled():
    """Prepared route is opt-in: SOLWEIG_LIGHT_PREPARED_VIS=1, default OFF.

    C6-80 portfolio re-measurement (evidence/portfolio/decoder_probe/) showed
    the one-entry prepared decode at +21-24% per full-frame sweep versus the
    per-channel entries at both timing shapes and both block strides -- the
    preflight scan does not amortize with size, so the dispatch hooks keep
    the original per-channel decode armed unless the environment explicitly
    requests the prepared path (exactness/memory-profile adoption case only).
    """
    return os.environ.get('SOLWEIG_LIGHT_PREPARED_VIS') == '1'


def prepare_channels(shadow, vegetation, vegetation_building, diffuse=None):
    """Prepare the demanded channels for one-pass decode, or return None.

    Pure admission: never raises, never fires an original error checkpoint,
    and never copies payload bytes. ``None`` means the caller must run the
    original per-channel decode path in full. The prepared route itself is
    opt-in (``_prepared_enabled``); declined admission is indistinguishable
    from unsupported inputs and the caller falls back completely.
    """
    if not _prepared_enabled():
        return None
    base = []
    for channel in (shadow, vegetation, vegetation_building):
        # Base channels must satisfy decode_block's direct single-leaf packed
        # admission; a lazy base channel keeps the original path (the pipeline
        # never produces one, and the fallback is exact).
        if isinstance(channel, LazyDiffVisibility) or not isinstance(channel, PackedVisibility):
            return None
        base.append(channel)
    slots = []
    for channel in base:
        slot = _build_slot(channel)
        if slot is None:
            return None
        slots.append(slot)
    diff_kind = _DIFF_NONE
    if diffuse is not None:
        slot = _diffuse_slot(shadow, vegetation, diffuse)
        if slot is None:
            return None
        diff_kind = slot.kind
        slots.append(slot)
    owners = {}
    for slot in slots:
        for owner in slot.owners:
            owners[id(owner)] = owner
    return PreparedVisibility(tuple(slots), tuple(sorted(owners.values(), key=id)),
                              len(slots), diff_kind)


def decode_shortwave_block(shadow, vegetation, vegetation_building, diffuse,
                           start, stop, patches, buffers=None):
    """Drop-in shortwave demand: (shadow, vegetation, building, diffuse) or None."""
    prepared = prepare_channels(shadow, vegetation, vegetation_building, diffuse)
    if prepared is None:
        return None
    return prepared.decode(start, stop, patches, buffers)


def decode_longwave_block(shadow, vegetation, vegetation_building,
                          start, stop, patches, buffers=None):
    """Drop-in longwave demand: (shadow, vegetation, building) or None."""
    prepared = prepare_channels(shadow, vegetation, vegetation_building)
    if prepared is None:
        return None
    return prepared.decode(start, stop, patches, buffers)
