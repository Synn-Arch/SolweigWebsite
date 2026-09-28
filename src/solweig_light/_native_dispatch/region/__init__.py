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
"""N8-14 region dispatch + bounded multicore owner (dossier 03).

Public surface re-exported from the two implementation modules; see
``region_pool.py``'s module docstring for the thread-budget rule, the
fork policy and the registry-growth bound (the three decisions this
task owns).
"""
from .region_plan import RegionPlan, plan_regions
from .region_pool import (
    ExecutionMode,
    RegionExecutorError,
    BlockContext,
    RegionReport,
    RegionPool,
    ForkedRegionPool,
    ClosedRegionPool,
    PoolSessionConflict,
    PoolRegistryBound,
    ScratchShapeError,
    ScratchSlot,
    execute_regions,
    execute_serial,
    shared_pool,
    resolve_budget,
    shutdown_all_pools,
    reset_pools_for_tests,
    MAX_POOL_BUDGET_SLOTS,
)

__all__ = [
    'RegionPlan', 'plan_regions',
    'ExecutionMode', 'RegionExecutorError', 'BlockContext', 'RegionReport', 'RegionPool',
    'ForkedRegionPool', 'ClosedRegionPool', 'PoolSessionConflict',
    'PoolRegistryBound', 'ScratchShapeError', 'ScratchSlot',
    'execute_regions', 'execute_serial', 'shared_pool', 'resolve_budget',
    'shutdown_all_pools', 'reset_pools_for_tests', 'MAX_POOL_BUDGET_SLOTS',
]
