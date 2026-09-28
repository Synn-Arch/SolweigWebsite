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
"""N8-14 consumers: A/B/C plugged into the region common call boundary.

These wrappers exist to prove the boundary is backend-agnostic: each one
adapts a DIFFERENT reviewed arm to the same ``mode`` / ``produce`` /
``consume`` contract the region executor accepts, and none of them is
named inside the executor. The arms themselves are imported READ-ONLY
(never modified):

* ``DenseKernelConsumer`` -- any 17-argument dense [rows, P] float32
  reducer. In the test suite the frozen N8-04 oracle
  (tests/optimization_v8/reference/lw_reference_oracle.py) is plugged in
  as the kernel (arm A); the N8-10 workflow-owned native handle's
  ``NativeHandle.execute`` plugs in the same way (arm C, BLOCK_FANOUT:
  the ISPC entry is single-thread per call, so parallelism comes from
  the owner at block level).
* ``AosoaBConsumer`` -- the N8-12 Numba B control ``lw_primary_b``
  (SELF_PARALLEL: the kernel's prange owns the granted budget; the
  owner dispatches nothing concurrently).

Per-block semantics mirror the production driver loop
(``cylinder_longwave.define_patch_characteristics_primary``):
produce slices the block's rows, consume runs the kernel into a
scratch-frame buffer and scatters ``output[:, start:stop] = frame.T``.
All rejection/admission behavior stays inside the plugged-in arm -- a
per-block ``UnsupportedInput`` propagates out of the region unchanged.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from .region_pool import ExecutionMode

__all__ = ['DenseKernelConsumer', 'AosoaBConsumer', 'dense_to_aosoa']

#: The frozen 17-argument signature order (N8-04 contract).
_ORDERED = ('sh', 'vs', 'vb', 'sun', 'shade', 'solid', 'sine', 'cosine',
            'directions', 'gate', 'solar_gate', 'sky_down', 'sky_side',
            'surface_sun', 'surface_sh', 'lup', 'reflection_factor')

#: Names whose per-block slice is the block's rows (axis 0 of a dense
#: [total_rows, P] array); everything else is patch-level and shared.
_ROW_SLICED = ('sh', 'vs', 'vb', 'sun', 'shade', 'lup')


class DenseKernelConsumer:
    """BLOCK_FANOUT consumer over dense [total_rows, P] float32 inputs.

    ``reduce_block(payload, frame)`` must fill or return a float32
    [rows, 7] frame for the payload's block rows. The scratch frame is
    leased from the slot (allocated once per slot, reused across
    blocks) and the result is scattered into the caller's output span --
    the driver's ``output[:, start:stop] = reduced.T`` exactly.
    """

    mode = ExecutionMode.BLOCK_FANOUT

    def __init__(self, reduce_block: Callable[[Mapping, np.ndarray], Any],
                 args: Mapping[str, Any]):
        self._reduce_block = reduce_block
        self._args = dict(args)

    def produce(self, ctx) -> dict:
        """Slice this block's rows; patch-level inputs pass through."""
        start, stop = ctx.start, ctx.stop
        payload = {}
        for name in _ORDERED:
            value = self._args[name]
            payload[name] = (value[start:stop] if name in _ROW_SLICED
                             else value)
        return payload

    def consume(self, payload, ctx) -> None:
        rows = ctx.stop - ctx.start
        # Full-capacity lease + [:rows] view: fixed-shape arenas even for
        # the tail block (see ScratchShapeError).
        frame = ctx.slot.buffer('frame', (ctx.block_capacity, 7),
                                np.float32)[:rows]
        with np.errstate(all='ignore'):  # NaN/Inf are VALUES on this graph
            result = self._reduce_block(payload, frame)
        if result is not None:
            frame = np.asarray(result, dtype=np.float32)
        ctx.output[:, ctx.start:ctx.stop] = frame.T


def dense_to_aosoa(dense: np.ndarray, width: int = 8,
                   poison: float = float('nan')) -> np.ndarray:
    """Repack dense [rows, P] values into the N8-11 lane layout [G, P, W].

    ``aosoa[g, p, lane] = dense[g*W + lane, p]`` for live lanes; padding
    lanes of the last gang (rows % W != 0) are filled with ``poison`` to
    prove consumers never read them. This is a test/layout helper -- the
    production producer (N8-11 ``produce_blocks_aosoa``) writes the same
    layout directly from packed leaves.
    """
    rows, patches = dense.shape
    gangs = -(-rows // width)
    out = np.full((gangs, patches, width),
                  poison, dtype=dense.dtype)
    for g in range(gangs):
        live = min(width, rows - g * width)
        if live:
            out[g, :, :live] = dense[g * width:g * width + live, :].T
    return out


class AosoaBConsumer:
    """SELF_PARALLEL consumer over the N8-11 AoSoA lane layout (arm B).

    Global ``[G_all, P, W]`` blocks are built once (test helper
    ``dense_to_aosoa`` or the real N8-11 producer); per-block produce
    slices whole gangs -- valid because block edges are lane-aligned
    (block_pixels % W == 0, or the final partial block ending at
    total_rows). Misaligned block edges make the B arm's own admission
    reject the slice (gang count mismatch), which is exactly the serial
    behavior; the executor adds nothing.

    The kernel is ``lw_primary_b`` with ``parallel=True``: its prange
    owns the granted budget, so this consumer declares SELF_PARALLEL and
    the owner stays quiescent (in-flight == 1).
    """

    mode = ExecutionMode.SELF_PARALLEL

    def __init__(self, args: Mapping[str, Any], width: int = 8):
        # N8-41 vendoring: the N8-12 consumer ships in this package (the
        # maintainer-tree copy resolved the bare experiment name).
        from solweig_light._native_dispatch.lw_b_control import lw_primary_b
        self._lw_primary_b = lw_primary_b
        self._args = dict(args)
        self._width = width

    def produce(self, ctx) -> dict:
        w = self._width
        start, stop = ctx.start, ctx.stop
        args = self._args
        payload = {}
        for name in _ORDERED:
            value = args[name]
            if name in ('sh', 'vs', 'vb', 'sun', 'shade'):
                g0, g1 = start // w, -(-stop // w)
                payload[name] = value[g0:g1]
            elif name == 'lup':
                payload[name] = value[start:stop]
            else:
                payload[name] = value
        payload['rows'] = stop - start
        return payload

    def consume(self, payload, ctx) -> None:
        rows = payload['rows']
        frame = ctx.slot.buffer('frame', (ctx.block_capacity, 7),
                                np.float32)[:rows]
        self._lw_primary_b(
            *(payload[name] for name in _ORDERED),
            rows, parallel=True, out=frame)
        ctx.output[:, ctx.start:ctx.stop] = frame.T
