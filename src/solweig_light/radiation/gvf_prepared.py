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
"""C6-30: per-step GVF source-expression preparation (dossier 05 G-A).

_gvf_fused evaluates several direction-independent expressions once per
direction (18x per call) although their inputs are fixed by the water
pre/post ordering: the gather Lup and the postprocessed lup_term re-evaluate
the same Lup expression, and Lwall, albshadow, aspect and the rounded search
distances never change across directions either. This module prepares all
proven-invariant expressions ONCE per gvf call ("one step": one timestep, one
owner) and then runs the identical fused row-block loop over private
snapshots. It is NOT a static radiance cache: nothing survives the call, and
no persistent state is keyed by caller array identity.

Exactness contract (bitwise against _gvf_fused on the admitted domain):

- Preparation mirrors the baseline's first-occurrence order relative to the
  water mutation: Lup is evaluated pre-mutation (serves direction 1), the
  Tg[lc_grid == 3] scatter is applied once, and Lup/Lwall/sky terms are
  evaluated post-mutation (serve directions 2..18 and the post-loop sums).
  Baseline directions 2..18 re-read the same post-mutation memory, so one
  post-mutation evaluation is bitwise identical to each of them.
- If no water mutation lands (landcover != 1 or no lc_grid == 3 cell), the
  pre-mutation Lup bitwise equals the post-mutation one and only one
  evaluation is made.
- Aliases through any prepared input defeat invariance and keep the fused
  route. Inherited from _gvf_fused: Tg may not share memory with walls,
  scale, ewall, albedo_b, landcover (frozen copies would diverge) nor with
  buildings, shadow, alb_grid (per-direction conversion is load-bearing).
  New for this stronger hoist: lc_grid (a Tg alias lets the mutation move
  the mask, so post-mutation memory is not direction-stable), dirwalls
  (aspect is read pre-mutation every direction) and Twater (the scatter
  value is re-read every direction). Tgwall, emis_grid, Ta, SBC, first,
  second and wallsun need NO gate: the preparation reads them at the same
  pre/post-mutation points where the baseline's per-direction evaluations
  read them, so an alias through those stays exact (tested).
- Warning/np.seterr contract: every hoisted expression fires its numpy
  warnings (e.g. float32 overflow in the ^4 powers) ONCE per call, at
  preparation, under the ambient errstate, instead of once per direction
  (or five times post-loop for the sky-emission term). The first warning is
  raised at the same input state as before, so errstate(raise) aborts both
  routes on the same first evaluation; only the repeat count shrinks.
  Expressions outside this closure are untouched - in particular the
  divide-by-zero-in-log warning at engine.py:1370 (svfalfaE = arcsin(exp(
  log(1 - svfE)/2)) in Lside_veg_v2022a) keeps firing once per Lside call,
  i.e. per timestep, exactly as before.
- sunwall's walls-zero divide warning is already once-per-call in the fused
  route and is unchanged.

Fallback: any input outside the admitted domain (dtype profile, nonfinite
steps, nonpositive second*scale, any gated Tg alias) delegates to
_gvf_funed's exact route via ``delegate``; no caller-visible memory is
touched before the guards pass, so failing/unsupported calls behave
identically. The compiled kernels, ray schedules and _postprocess_block are
reused verbatim from ground_view; this module adds no numba compilation and
imports only the existing math profile.
"""
import os

import numpy as np

_ENV_TOGGLE = 'SOLWEIG_LIGHT_GVF_PREPARE'


def prepared_enabled():
    """Runtime kill switch: the fallback route is the exact fused baseline."""
    return os.environ.get(_ENV_TOGGLE, '1') not in ('0', 'false', 'False')


class PreparedGVFStep(object):
    """Read-only per-call snapshot: proven-invariant expression results plus
    the private source copies the fused loop consumes. Owner is one gvf
    call; the object is never registered anywhere and dies with the call."""

    __slots__ = (
        'azimuth_degrees', 'azimuths', 'aspect', 'wallbol', 'sunwall',
        'buildings', 'shadow', 'alb_grid', 'scale', 'albedo_b',
        'lup_first', 'lup_rest', 'lup_first_snap', 'lup_rest_snap',
        'lwall', 'albshadow', 'water_mask',
        'first_steps', 'second_steps', 'lup_term', 'alb_term', 'nosh_term',
        'sky_emis', 'lup_evaluations', 'mutations', 'stats',
    )


def _admits(wallsun, walls, buildings, scale, shadow, first, second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, rows, cols, Twater, lc_grid, landcover):
    """The _gvf_fused guard union plus the stronger-hoist alias exclusions.

    Everything rejected here is handed to _gvf_fused, whose own guards then
    reproduce the exact baseline behavior for that input.
    """
    from .engine import _operate
    from .ground_view import _supported
    if not _supported((wallsun, walls, buildings, shadow, dirwalls, Tg, emis_grid, alb_grid)) or (np.asarray(Tgwall).ndim > 0 and np.asarray(Tgwall).dtype != np.float32):
        return False
    if not np.isfinite(np.asarray(first)).all() or not np.isfinite(np.asarray(second)).all() or float(np.round(_operate(np.multiply, second, scale))) <= 0:
        return False
    for value in (walls, scale, ewall, albedo_b, landcover, buildings, shadow, alb_grid):
        # First five: _gvf_fused copies them once per call, so a Tg alias
        # already delegates there. Last three: the fused route keeps
        # per-direction conversion for them; preparation snapshots harder.
        if np.may_share_memory(Tg, np.asarray(value)):
            return False
    for value in (lc_grid, dirwalls, Twater):
        # New exclusions proved necessary for the once-per-call snapshots:
        # the water mask, the aspect source and the scatter constant are all
        # re-read live per direction in the baseline (see module docstring).
        if np.may_share_memory(Tg, np.asarray(value)):
            return False
    return True


def prepare_gvf_step(wallsun, walls, buildings, scale, shadow, first, second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, rows, cols, Twater, lc_grid, landcover):
    """Build the step snapshot in the baseline's first-occurrence order, or
    return None when the call must stay on the fused route."""
    from .engine import _array, _operate, _divide
    from .ground_view import _angular_constant, _lup_expression
    if not _admits(wallsun, walls, buildings, scale, shadow, first, second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, rows, cols, Twater, lc_grid, landcover):
        return None
    step = PreparedGVFStep()
    step.stats = {}
    # Loop-external copies exactly as _gvf_fused makes them, before the loop
    # and therefore before any water mutation.
    step.scale = np.copy(_array(scale))
    ewall = np.copy(_array(ewall))
    step.albedo_b = np.copy(_array(albedo_b))
    landcover = np.copy(_array(landcover))
    step.wallbol = (walls > 0).astype(np.float32)
    step.sunwall = np.array((_operate(np.multiply, _divide(wallsun, walls), buildings) == 1).astype(np.float32), dtype=np.float32, copy=True)
    step.buildings = np.array(buildings, dtype=np.float32, copy=True)
    step.shadow = np.array(shadow, dtype=np.float32, copy=True)
    step.alb_grid = np.array(alb_grid, dtype=np.float32, copy=True)
    step.azimuth_degrees = np.arange(5, 359, 20, dtype=np.float32)
    step.azimuths = [_operate(np.multiply, step.azimuth_degrees[j], _angular_constant(step.azimuth_degrees, np.pi / 180)) for j in range(len(step.azimuth_degrees))]
    step.aspect = _divide(_operate(np.multiply, dirwalls, np.pi), 180)
    step.stats['aspect'] = 1
    # Direction 1 reads Lup pre-mutation.
    step.lup_first = _lup_expression(SBC, emis_grid, Tg, shadow, Ta)
    step.lup_evaluations = 1
    step.mutations = 0
    if landcover == 1:
        # The baseline scatters at this point in every direction; with the
        # mask and value provably fixed (gates) the repeat writes are no-ops
        # and one scatter leaves Tg bitwise identical.
        step.water_mask = lc_grid == 3
        Tg[step.water_mask] = _operate(np.subtract, Twater, Ta).astype(np.float32)
        step.mutations = 1
    else:
        step.water_mask = None
    if step.water_mask is None or not step.water_mask.any():
        step.lup_rest = step.lup_first
    else:
        # Directions 2..18 and the postprocessed lup_term read the mutated
        # state; one evaluation serves them all.
        step.lup_rest = _lup_expression(SBC, emis_grid, Tg, shadow, Ta)
        step.lup_evaluations += 1
    step.stats['lup_evaluations'] = step.lup_evaluations
    # Gather snapshots mirror _direction_snapshot's float32 conversion once
    # per state (value-changing when SBC is float64). lup_term keeps the raw
    # evaluation exactly as the baseline's inline expression does.
    step.lup_first_snap = np.array(step.lup_first, dtype=np.float32, copy=True)
    if step.lup_rest is step.lup_first:
        step.lup_rest_snap = step.lup_first_snap
    else:
        step.lup_rest_snap = np.array(step.lup_rest, dtype=np.float32, copy=True)
    # Lwall is evaluated post-mutation in every direction (it reads Tgwall
    # live), so one post-mutation evaluation is exact even for a Tgwall
    # alias; the broadcast conversion mirrors _direction_snapshot.
    lwall = _operate(np.subtract, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, _operate(np.add, Tgwall, Ta), 273.15), 4)), _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)))
    step.lwall = np.array(np.broadcast_to(lwall, step.buildings.shape), dtype=np.float32, copy=True)
    step.stats['lwall'] = 1
    albshadow = _operate(np.multiply, step.alb_grid, step.shadow)
    step.albshadow = np.array(albshadow, dtype=np.float32, copy=True)
    step.stats['albshadow'] = 1
    # first/second are read post-mutation per direction; one post-mutation
    # evaluation over the pristine arrays (no alias) is exact.
    step.first_steps = np.round(_operate(np.multiply, first, step.scale))
    if step.first_steps < 1:
        step.first_steps = 1
    step.second_steps = np.round(_operate(np.multiply, second, step.scale))
    step.stats['search_steps'] = 1
    one_minus_buildings = _operate(np.add, _operate(np.multiply, step.buildings, -1), 1)
    step.lup_term = _operate(np.multiply, step.lup_rest, one_minus_buildings)
    step.alb_term = _operate(np.multiply, _operate(np.multiply, step.alb_grid, one_minus_buildings), step.shadow)
    step.nosh_term = _operate(np.multiply, step.alb_grid, one_minus_buildings)
    step.stats['postprocess_terms'] = 1
    # The post-loop sky-emission term is evaluated five times by the
    # baseline (gvfLup + four cardinals); one evaluation is bitwise
    # identical because its inputs are post-mutation stable.
    step.sky_emis = _operate(np.multiply, _operate(np.multiply, SBC, emis_grid), _operate(np.power, _operate(np.add, Ta, 273.15), 4))
    step.stats['sky_emis'] = 1
    return step


def run_prepared_gvf_step(step, rows, cols, parallel=True, block_rows=32):
    """The _gvf_fused direction loop consuming the prepared snapshot. Every
    arithmetic node, cast, comparison and accumulation is copied 1:1 from
    _gvf_fused; only the once-per-call expressions are replaced by their
    prepared values."""
    from .engine import _zeros, _operate, _divide
    from .ground_view import _angular_constant, ray_schedule, _gather_block_parallel, _gather_block_serial, _postprocess_block
    block_rows = max(1, int(block_rows))
    gather = _gather_block_parallel if parallel else _gather_block_serial
    gvfLup = _zeros((rows, cols))
    gvfalb = _zeros((rows, cols))
    gvfalbnosh = _zeros((rows, cols))
    gvfLupE = _zeros((rows, cols))
    gvfLupS = _zeros((rows, cols))
    gvfLupW = _zeros((rows, cols))
    gvfLupN = _zeros((rows, cols))
    gvfalbE = _zeros((rows, cols))
    gvfalbS = _zeros((rows, cols))
    gvfalbW = _zeros((rows, cols))
    gvfalbN = _zeros((rows, cols))
    gvfalbnoshE = _zeros((rows, cols))
    gvfalbnoshS = _zeros((rows, cols))
    gvfalbnoshW = _zeros((rows, cols))
    gvfalbnoshN = _zeros((rows, cols))
    gvfSum = _zeros((rows, cols))
    kernel_first = np.float32(step.first_steps)
    for j in range(len(step.azimuth_degrees)):
        azimuth = step.azimuths[j]
        azilow = _operate(np.subtract, azimuth, _angular_constant(azimuth, np.pi / 2))
        azihigh = _operate(np.add, azimuth, _angular_constant(azimuth, np.pi / 2))
        if azilow >= 0 and azihigh < _angular_constant(azimuth, 2 * np.pi):
            facesh = _operate(np.add, _operate(np.subtract, np.logical_or(step.aspect < azilow, step.aspect >= azihigh).astype(np.float32), step.wallbol), 1)
        elif azilow < 0 and azihigh <= _angular_constant(azimuth, 2 * np.pi):
            azilow = _operate(np.add, azilow, _angular_constant(azimuth, 2 * np.pi))
            facesh = _operate(np.add, _operate(np.multiply, np.logical_or(step.aspect > azilow, step.aspect <= azihigh).astype(np.float32), -1), 1)
        elif azilow > 0 and azihigh >= _angular_constant(azimuth, 2 * np.pi):
            azihigh = _operate(np.subtract, azihigh, _angular_constant(azimuth, 2 * np.pi))
            facesh = _operate(np.add, _operate(np.multiply, np.logical_or(step.aspect > azilow, step.aspect <= azihigh).astype(np.float32), -1), 1)
        lup_snap = step.lup_first_snap if j == 0 else step.lup_rest_snap
        bounds = ray_schedule(step.buildings.shape, azimuth, step.scale, step.first_steps, step.second_steps)
        for row0 in range(0, step.buildings.shape[0], block_rows):
            row1 = min(row0 + block_rows, step.buildings.shape[0])
            block = np.empty((16, row1 - row0, step.buildings.shape[1]), dtype=np.float32)
            gather(row0, row1, step.buildings, step.shadow, step.sunwall, lup_snap, step.albshadow, step.alb_grid, step.lwall, np.float32(step.albedo_b), bounds, kernel_first, block)
            gvf_b, gvfLup_b, gvfalb_b, gvfalbnosh_b, gvf2_b = _postprocess_block(
                tuple(block[index] for index in range(16)), step.buildings[row0:row1],
                facesh[row0:row1], step.lup_term[row0:row1], step.alb_term[row0:row1],
                step.nosh_term[row0:row1], step.first_steps, step.second_steps)
            gvfLup[row0:row1] += gvfLup_b
            gvfalb[row0:row1] += gvfalb_b
            gvfalbnosh[row0:row1] += gvfalbnosh_b
            gvfSum[row0:row1] += gvf2_b
            if 0 <= step.azimuth_degrees[j] < 180:
                gvfLupE[row0:row1] += gvfLup_b
                gvfalbE[row0:row1] += gvfalb_b
                gvfalbnoshE[row0:row1] += gvfalbnosh_b
            if 90 <= step.azimuth_degrees[j] < 270:
                gvfLupS[row0:row1] += gvfLup_b
                gvfalbS[row0:row1] += gvfalb_b
                gvfalbnoshS[row0:row1] += gvfalbnosh_b
            if 180 <= step.azimuth_degrees[j] < 360:
                gvfLupW[row0:row1] += gvfLup_b
                gvfalbW[row0:row1] += gvfalb_b
                gvfalbnoshW[row0:row1] += gvfalbnosh_b
            if 270 <= step.azimuth_degrees[j] or step.azimuth_degrees[j] < 90:
                gvfLupN[row0:row1] += gvfLup_b
                gvfalbN[row0:row1] += gvfalb_b
                gvfalbnoshN[row0:row1] += gvfalbnosh_b
    sky = step.sky_emis
    gvfLup = _operate(np.add, _divide(gvfLup, len(step.azimuth_degrees)), sky)
    gvfalb = _divide(gvfalb, len(step.azimuth_degrees))
    gvfalbnosh = _divide(gvfalbnosh, len(step.azimuth_degrees))
    gvfLupE = _operate(np.add, _divide(gvfLupE, _divide(len(step.azimuth_degrees), 2)), sky)
    gvfLupS = _operate(np.add, _divide(gvfLupS, _divide(len(step.azimuth_degrees), 2)), sky)
    gvfLupW = _operate(np.add, _divide(gvfLupW, _divide(len(step.azimuth_degrees), 2)), sky)
    gvfLupN = _operate(np.add, _divide(gvfLupN, _divide(len(step.azimuth_degrees), 2)), sky)
    gvfalbE = _divide(gvfalbE, _divide(len(step.azimuth_degrees), 2))
    gvfalbS = _divide(gvfalbS, _divide(len(step.azimuth_degrees), 2))
    gvfalbW = _divide(gvfalbW, _divide(len(step.azimuth_degrees), 2))
    gvfalbN = _divide(gvfalbN, _divide(len(step.azimuth_degrees), 2))
    gvfalbnoshE = _divide(gvfalbnoshE, _divide(len(step.azimuth_degrees), 2))
    gvfalbnoshS = _divide(gvfalbnoshS, _divide(len(step.azimuth_degrees), 2))
    gvfalbnoshW = _divide(gvfalbnoshW, _divide(len(step.azimuth_degrees), 2))
    gvfalbnoshN = _divide(gvfalbnoshN, _divide(len(step.azimuth_degrees), 2))
    gvfNorm = _divide(gvfSum, len(step.azimuth_degrees))
    gvfNorm[step.buildings == 0] = 1
    return (gvfLup, gvfalb, gvfalbnosh, gvfLupE, gvfalbE, gvfalbnoshE, gvfLupS, gvfalbS, gvfalbnoshS, gvfLupW, gvfalbW, gvfalbnoshW, gvfLupN, gvfalbN, gvfalbnoshN, gvfSum, gvfNorm)


def prepared_gvf_step(wallsun, walls, buildings, scale, shadow, first, second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, rows, cols, Twater, lc_grid, landcover, parallel=True, block_rows=32, delegate=None):
    """Drop-in consumer of the fused route: prepare the step snapshot and run
    the identical loop, or delegate the call unchanged to _gvf_fused."""
    from .ground_view import _gvf_fused
    if delegate is None:
        delegate = _gvf_fused
    if not prepared_enabled():
        return delegate(wallsun, walls, buildings, scale, shadow, first, second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, rows, cols, Twater, lc_grid, landcover, parallel=parallel, block_rows=block_rows)
    step = prepare_gvf_step(wallsun, walls, buildings, scale, shadow, first, second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, rows, cols, Twater, lc_grid, landcover)
    if step is None:
        return delegate(wallsun, walls, buildings, scale, shadow, first, second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, rows, cols, Twater, lc_grid, landcover, parallel=parallel, block_rows=block_rows)
    return run_prepared_gvf_step(step, rows, cols, parallel=parallel, block_rows=block_rows)
