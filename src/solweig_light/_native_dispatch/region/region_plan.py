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
"""N8-14 region planner: coarse dispatch regions over the frozen block grid.

Two scales (dossier 03, "Two scales"):

* ``block_pixels`` ``b`` is the cache-sized EXECUTION microblock. It keeps
  the public default (128) and its original meaning; the planner never
  changes it and never silently increases it.
* ``region_blocks`` ``R`` is the DISPATCH granularity: how many whole
  microblocks one region submission covers (``M = R*b`` pixels, plus the
  tail). R is decoupled from b -- it is a scheduler policy knob, not a
  physical domain size.

ALIGNMENT INVARIANT (the load-bearing property for bitwise composition):
block spans are ALWAYS the spans of the original whole-tile loop
``for start in range(0, total_rows, b)``, and every region boundary is a
block-grid boundary (regions are unions of consecutive whole blocks). A
region therefore never splits, merges, or re-cuts a microblock: per-block
admission, decode/produce, kernel calls, tail handling and error behavior
are exactly the serial driver loop's, and parallel region execution
composes the same per-block results as serial composition does. The
planner expresses this structurally -- regions are ranges of block
indices, not pixel ranges -- so no runtime check is needed to keep it.

Region-count policy: with no explicit granularity the planner emits ONE
region covering every block (the current single driver loop); pass
``region_blocks=`` or ``region_pixels=`` to subdivide for dispatch.
``region_pixels`` is a convenience target that snaps DOWN to whole blocks
(``R = max(1, region_pixels // b)``) rather than cutting a microblock.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ['RegionPlan', 'plan_regions']


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{name} must be a positive integer; got {value!r}')
    return int(value)


def _non_negative_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer; got {value!r}')
    return int(value)


@dataclass(frozen=True)
class RegionPlan:
    """Immutable dispatch plan: the block grid plus its region grouping."""

    total_rows: int
    block_pixels: int
    region_blocks: int

    def __post_init__(self) -> None:
        _non_negative_int(self.total_rows, 'total_rows')
        _positive_int(self.block_pixels, 'block_pixels')
        _positive_int(self.region_blocks, 'region_blocks')

    # -- block grid (the frozen public semantics) -------------------------

    @property
    def total_blocks(self) -> int:
        return (self.total_rows + self.block_pixels - 1) // self.block_pixels

    def block_spans(self) -> tuple[tuple[int, int], ...]:
        """The exact spans of ``range(0, total_rows, block_pixels)``.

        Concatenating these spans across regions reproduces the original
        serial block loop pixel-for-pixel; this method is the single
        source of truth both for region construction and for the
        composition-equality tests.
        """
        b = self.block_pixels
        n = self.total_rows
        return tuple((start, min(start + b, n)) for start in range(0, n, b))

    # -- regions (dispatch units; unions of whole blocks) -----------------

    @property
    def region_count(self) -> int:
        t = self.total_blocks
        return (t + self.region_blocks - 1) // self.region_blocks

    def region_block_ranges(self) -> tuple[range, ...]:
        """Ascending, contiguous, non-overlapping partition of the blocks."""
        r = self.region_blocks
        t = self.total_blocks
        return tuple(
            range(k * r, min((k + 1) * r, t)) for k in range(self.region_count)
        )

    def region_spans(self) -> tuple[tuple[int, int], ...]:
        """Pixel extent (first-block start, last-block stop) per region."""
        spans = self.block_spans()
        return tuple(
            (spans[rng.start][0], spans[rng.stop - 1][1])
            for rng in self.region_block_ranges()
        )

    def describe(self) -> dict:
        """Diagnostics dictionary (no IO, safe for counters/evidence)."""
        return {
            'total_rows': self.total_rows,
            'block_pixels': self.block_pixels,
            'region_blocks': self.region_blocks,
            'total_blocks': self.total_blocks,
            'region_count': self.region_count,
            'block_spans': list(self.block_spans()),
            'region_spans': list(self.region_spans()),
        }


def plan_regions(total_rows: int, *, block_pixels: int = 128,
                 region_blocks: int | None = None,
                 region_pixels: int | None = None) -> RegionPlan:
    """Build a plan.

    ``region_blocks`` wins when both granularities are supplied
    (``region_pixels`` is only a convenience for "about M pixels per
    dispatch"); neither supplied means one region over all blocks.
    """
    _non_negative_int(total_rows, 'total_rows')
    _positive_int(block_pixels, 'block_pixels')
    if region_blocks is not None and region_pixels is not None:
        raise ValueError(
            'pass region_blocks or region_pixels, not both '
            f'(got {region_blocks!r} and {region_pixels!r})')
    if region_blocks is None and region_pixels is not None:
        _positive_int(region_pixels, 'region_pixels')
        region_blocks = max(1, region_pixels // block_pixels)
    elif region_blocks is not None:
        _positive_int(region_blocks, 'region_blocks')
    else:
        # Default dispatch policy: the whole extent in ONE region -- the
        # existing driver loop shape. Subdivision is an explicit request.
        region_blocks = max(
            1, (total_rows + block_pixels - 1) // block_pixels)
    return RegionPlan(total_rows=total_rows, block_pixels=block_pixels,
                      region_blocks=region_blocks)
