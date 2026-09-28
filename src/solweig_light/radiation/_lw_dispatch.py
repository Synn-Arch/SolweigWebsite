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
"""N9 default route for the primary-output longwave reduction.

The bounded stream (``_native_dispatch.lw_stream``) IS the shipped default
for admitted non-all-raw invocations: there is no selection policy, no
qualification registry and no per-call env read anywhere in this path
(N9 F4 -- the N8-22 qualification machinery moved out of the installed
runtime; research copies live in the repo-only archive).

Route decision, all structural, all PRE-LAUNCH (``None`` -> the caller's
trusted legacy loop):

* empty/degenerate total or a lane-misaligned block size,
* non-admitted channels (the producer's exact admission predicate),
* the all-raw payload class (``lw_stream._all_patches_raw`` -- the one
  N9-F3 measured stream loss, 0.86-0.93x per pair, direction-consistent),
* an explicit expert request ``SOLWEIG_LIGHT_LW_BACKEND=native|ispc``
  (intercepted in the caller seam ``cylinder_longwave._lw_region_route``
  so the B7-32 dev-build route keeps serving it -- established explicit
  expert compatibility, unchanged).

Decline semantics are unchanged from N8-40: a decline BEFORE launch
returns ``None`` and the caller runs the trusted Numba loop -- fallback
OUTSIDE the admitted domain. A failure AFTER the first launch propagates
loudly; it is never converted into a fallback.

This module reads NO environment variables: the expert-env intercept
lives in the caller seam, keeping the package's env-read surface exactly
as frozen (DX parity gate).
"""
import numpy as np

_LANE_WIDTH = 8


def region_route(values, geometry, solar_gate, prepared, total, block_pixels,
                 factor, sun_surface, shade_surface):
    """Run the bounded Numba stream; ``None`` routes the caller's legacy
    loop.

    ``values``/``geometry``/``solar_gate``/``prepared`` are the driver's
    own locals (``define_patch_characteristics_primary``); ``factor``,
    ``sun_surface`` and ``shade_surface`` are the already-computed scalars
    the legacy loop passes to the kernel. The caller has already stood down
    for an explicit expert-env request. Never raises for a decline:
    structural admission misses return ``None``; only a post-launch
    machinery failure is loud.
    """
    if total < 1 or block_pixels < _LANE_WIDTH \
            or block_pixels % _LANE_WIDTH:
        return None  # empty/degenerate or lane-misaligned: trusted legacy
    return _execute_row(values, geometry, solar_gate,
                        prepared, total, block_pixels, factor, sun_surface,
                        shade_surface)


def _execute_row(values, geometry, solar_gate, prepared, total,
                 block_pixels, factor, sun_surface, shade_surface):
    """Run the bounded stream (N9-F1S): plan, slots, regions, join.

    The InvocationPlan is minted ONCE per call -- channel admission,
    classification admission, the all-raw guard and the full range
    contract are validated PRE-LAUNCH; any decline returns None and the
    caller keeps its trusted legacy loop. Everything after the first
    launch is loud. The leaf lease held by the plan is released after the
    region join.
    """
    from solweig_light._native_dispatch import lw_stream
    stream = lw_stream.plan_invocation(
        'B', values, geometry, solar_gate, prepared, total, block_pixels,
        factor, sun_surface, shade_surface, width=_LANE_WIDTH)
    if stream is None:
        return None  # non-admitted channels / table / all-raw / misaligned
    try:
        output = np.empty((7, total), dtype=np.float32)
        consumer = lw_stream.AosoaBStreamConsumer(stream)
        from solweig_light._native_dispatch.region.region_plan import plan_regions
        from solweig_light._native_dispatch.region.region_pool import execute_regions
        # The owner is sized by the PLAN's pinned budget (1): the shipped
        # route is the measured B1 arm, so the shared pool is keyed at
        # (pid, 1) with zero background threads under SELF_PARALLEL --
        # never the wider runtime default (N9 F4).
        execute_regions(plan_regions(total, block_pixels=block_pixels),
                        consumer, output, budget=stream.thread_budget)
    finally:
        stream.close()
    return output
