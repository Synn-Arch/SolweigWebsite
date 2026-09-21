# Cura P8 execution assessment

Host `analytica` exposes 36 physical/72 logical Intel Xeon w9-3475X CPUs,
187 GiB RAM, and four idle 49,140 MiB NVIDIA RTX A6000 GPUs (driver
550.144.03, compute capability 8.6). At assessment time load average was
0.13/0.06/0.01 and `nvidia-smi` reported no compute processes. The root/home
filesystem had only 47 GiB free at 95% utilization, so large duplicate matrices
require a separate size preflight. No batch scheduler was found.

The isolated task root is
`/home/ssynn3/workspace/solweig-light-codex-p8-20260919T1215Z-a7c3`. It was
absent before creation, is owned by `ssynn3` with mode 0700, and contains a
21 MiB targeted source/reference snapshot plus a fresh 856 MiB Conda environment
and result artifacts. No pre-existing remote environment or project was modified.

The source snapshot contains 387 files: `src`, the six queued scientific test
files, the exact state-sequence and small original-CPU scene references, the
pinned official UTCI source, the pinned UMEP component sources, and
`pyproject.toml`. The downloaded `source_sha256.txt` verifies all 387 local
source files. `results/result_sha256.txt` verifies every downloaded result file.

The initial combined execution produced 28 passes and 7 failures. A GDAL/PROJ
environment failure in the directional-wind case was isolated: Conda installed
`proj.db`, but explicit `PROJ_DATA` and `GDAL_DATA` were required in the nonactivated
SSH process. With those paths set, the unchanged directional test passed (1 test,
one GDAL future warning).

The remaining six failures are scientific-test findings:

- The two radiation tests stop at an invalid authored precondition: the genuine
  night fixture has scalar `landcover=1`, while all 1,120 `lc_grid` cells are
  class 7 and there are no water cells. Direct diagnostic execution confirms all
  ten asserted shortwave outputs are exactly zero. The returned night `Lup`
  agrees with the stated emissivity/Stefan-Boltzmann expression to a maximum
  absolute difference of `2.0703205052541307e-05 W m^-2`; therefore the fixture is
  a valid non-water analytic case despite the scalar land-cover switch.
- The two full UMEP ground-view cases fail only output field 15 under their
  per-field policy: 12/99 cells exceed `1e-6`, with maximum absolute difference
  `1.7106533043431682e-6`. Candidate serial and parallel outputs are bitwise equal
  for all 17 fields. The unchanged UMEP oracle returns float64 while the candidate
  compiled path intentionally accumulates float32. Other albedo fields remain
  below `1e-6`; longwave fields have larger raw differences but pass their
  predeclared `rtol=1e-5, atol=0.05` policy.
- The two open-scene SVF failures are caused by an incorrect unity expectation
  for directional vegetation/combined fields, not by the known zenith defect.
  Both vegetation visibility channels are exactly one at zenith; only building
  visibility contains zeros there. The inherited directional integration returns
  the established maxima `0.9752265215`, `0.9484643340`, and `0.8594677448`, also
  present in the original-upstream `reports/initial_artifact_snapshot.json`.
  The total vegetation field is exactly one. All 15 SVF fields and `svftotal`
  passed the independently frozen finite `[0,1]` bound.

The frozen `benchmark_v1.json` cannot directly authorize a Cura release matrix:
it names Apple M1 Pro/macOS ARM64, records CUDA as unavailable, and its current
reference runner explicitly rejects visible CUDA. Existing P7 pipeline tooling
also forces `CUDA_VISIBLE_DEVICES=''`. A Cura CPU matrix needs a separately frozen
dedicated-host amendment that preserves the existing fixtures/work/gates and
records the Xeon configuration. A CUDA matrix additionally needs an isolated
upstream CUDA environment and a CUDA-capable reference runner; neither currently
exists in the transferred bounded snapshot. No GPU benchmark was run or claimed.

The exact commands, package lock, JUnit XML, logs, diagnostics, source inventory,
and checksums are in `results/`. The remote workspace remains present for bounded
follow-up. Cleanup has not started.

## Radiation snapshot revision

After the initial run, the root task revised only the radiation test's fixture
precondition: scalar `landcover` may be 0 or 1, while the fixture must contain no
`lc_grid == 3` water cells. The original transferred test remains identified by
SHA-256 `2ffacba515e8b44f154510e1c9f5e289baad660ccfca6538213fcc46e8fdf76c`;
the explicit replacement is
`cc8e5a7a04e608b309f971f29bb260f7b66f693fa514c05244363051968ac030`.
`radiation_test_snapshot_revision.json` records the scope and hashes.

The two revised tests were executed unchanged on Cura with the isolated
environment and explicit data paths. Result: **2 passed in 2.21 seconds**.
`p8_radiation_analytic_revision_tests.xml` and its log were downloaded and
verified against the refreshed result manifest. The original combined-run
failure remains preserved as historical evidence.
