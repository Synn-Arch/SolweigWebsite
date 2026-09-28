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
"""Private home of the shipped bounded Numba stream machinery (N9).

* ``direct_aosoa`` -- the mode-specialized packed-visibility producer and
  the exact AoSoA classifier (N8-11 producer lineage, N9 F1D/F1M).
* ``lw_b_control`` -- the Numba AoSoA primary leaf ``lw_primary_b``
  (N8-12 lineage; the equally optimized Numba control's kernel).
* ``lw_stream`` -- the bounded InvocationPlan/BorrowedVisibility/BlockSlot
  stream and its SELF_PARALLEL consumer (N9 F1S): the shipped default
  route for admitted non-all-raw invocations.
* ``region`` -- the region owner (plan/pool/consumers) executing the
  stream's blocks with canonical first-error cancellation.

* ``aplus_decode`` -- the N9 A-plus mode-specialized packed decode
  (``_decode_at_plus``, exact bits, generic fallback arm) wired as the
  fused route's decode in ``radiation.patch_radiation`` and
  ``radiation.cylinder_longwave``; the transcribed parity kernels stay
  pinned by the n9_producer suite.

The N8 native row and qualification machinery (``lw_default_policy``,
``qualification_registry.json``, ``lw_native_aosoa``, ``installed_loader``,
``native_handle``, ``build_native``) is NOT part of the installed runtime:
it lives in the repo-only archive
``experiments/optimization_v8/native_dispatch/``
(N9 F4 closed_cpu_only disposition; N9-F3 terminal NATIVE_LOSS record).
Nothing here is re-exported from :mod:`solweig_light`, nothing here is a
public or DX-contract surface, and ``import solweig_light`` never imports
any of it (pinned by ``tests/optimization_v8/policy/test_legacy_env_parity.py``).

An explicit expert request (``SOLWEIG_LIGHT_LW_BACKEND=native|ispc``) is
served by the legacy B7-32 dev-build route in
``solweig_light.backends.native_lw`` -- established explicit expert
compatibility, independent of this package.
"""
