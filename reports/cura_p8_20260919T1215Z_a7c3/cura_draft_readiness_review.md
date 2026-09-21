# Cura draft protocol readiness review

Status: **not ready to freeze or execute as a release benchmark**. This review
does not change a workload, gate, runner, or protocol.

## Evidence now available

- The host inventory is fixed for this draft: Xeon w9-3475X, 36 physical cores,
  187 GiB RAM, and four RTX A6000 devices. GPU 0 is bound by UUID
  `GPU-bd3de2c4-3d7c-daef-54c6-3efc2b88d628`.
- Deterministic input manifests now bind repeated-block 1024, 2048, and 3600;
  dense-urban and vegetation-rich 1024 and 2048; and three native-resolution
  256×256 real windows. The real windows retain 2 m resolution and the pinned
  Zenodo archive SHA-256. Input preparation is not numerical admission.
- Original upstream CPU and candidate CPU passed the unchanged output gates and
  all 18 geometry-export fields for the genuine small 24-hour case and
  repeated-block 1024. The raw artifacts are now local and verified: 52 files,
  2,100,067,854 bytes. Their duplicate remote output/SVF trees were removed.
- Exact Torch `2.14.0+cu126` executes on one selected A6000. The original
  upstream CUDA `run_utci_tiles` geometry-warm small case passed all 13 output
  artifacts against both original CPU and candidate CPU, plus both 18-field
  geometry checks, under the 12 GiB summed-RSS limit.
- Original upstream CUDA `thermal_comfort` first-use/cold is a retained resource
  failure. Its 32-process wall/aspect pool crosses the unchanged summed-RSS cap.
  A minimal task-count worker bound exists only as an unapplied, unexecuted
  review patch. It requires an explicit patched-reference policy before use.
- The 1024 and 2048 conservative disk-only preflights fit the earlier host
  state. The 3600 all-output estimate does not fit while retaining 10 GiB free.
  Current remote free space is about 35 GiB after CUDA environment creation and
  raw-output cleanup, so 3600 remains unavailable without additional owned-space
  cleanup or a different approved host.

## Freeze blockers

1. **Correctness admission is incomplete.** Dense/vegetation 1024, all declared
   2048 cases, repeated 3600, the native real windows, default/all-off plans,
   one-step behavior, and the 72-step carried-state sequence do not yet have the
   required Cura upstream/candidate admission records. A completed local P7
   matrix is supporting evidence but must be hash-bound into this draft before
   it can satisfy a Cura protocol prerequisite.
2. **CUDA first-use policy is unresolved.** The unmodified public upstream path
   fails the frozen memory gate. Geometry-warm CUDA is admitted separately and
   cannot substitute for it. The proposed worker-bound patch must remain
   unapplied until patched-oracle policy is explicitly accepted.
3. **Resource estimates are incomplete.** Actual per-workload output, geometry,
   JIT, evidence, CUDA-library, and temporary-copy footprints must be measured or
   conservatively frozen for every scheduled cell. The 12 GiB process-tree gate
   also needs admission evidence for 2048 and larger cases; disk admission alone
   is insufficient.
4. **Execution environments are not release-frozen.** The CPU and CUDA oracle
   locks are recorded, including the isolated Numba repair and CUDA install
   failure/retry. The candidate still needs the draft's immutable installed-wheel
   provenance rather than only a source snapshot. Every final source, wheel,
   runner, comparator, environment, fixture, and kwargs manifest needs one
   protocol-bound hash inventory.
5. **Release orchestration remains proposed.** No reviewed runner currently
   schedules all declared systems, regimes, repetitions, randomized pairing,
   process-tree/GPU sampling, artifact validation, streamed download, and
   failure retention. The diagnostic CUDA runners are evidence tools, not a
   frozen benchmark harness.
6. **Hardware controls remain to be frozen.** CPU affinity/NUMA placement,
   governor and frequency reporting, GPU clocks/power/temperature sampling,
   competing-load checks, timeout policy, and OS-page-cache labeling need exact
   commands and completion criteria.
7. **The complete mandatory matrices are unexecuted.** No release timing trials
   or aggregates exist for upstream default/tuned CPU, candidate 1/36, or CUDA.
   Five/three repetitions, first-use/cold/warm separation, default/all-off/all-on
   plans, real-scene coverage, and the 72-step state matrix remain required.

## Next review gates

The next protocol review should decide the CUDA patched-reference policy and a
resource disposition for 3600 before freezing anything. After that decision, a
new validation-only orchestrator pass should bind every artifact hash, expand
the full cell schedule, prove disk and RSS admission per cell, and emit
`validated_not_executed`. Numerical admission for the remaining workloads must
precede any timed trial. No current evidence supports a speedup claim.

Primary evidence:

- `fixture_prep/fixture_preparation_manifest.json`
- `admission/admission_manifest.json`
- `admission/raw_admission_verification.json`
- `cuda_admission/cuda_feasibility_manifest.json`
- `cuda_admission/cuda_startup_memory_diagnostic.json`
- `cuda_admission/cuda_geometry_warm_admission.json`
- `cuda_admission/walls_aspect_bound_workers_review.patch` (unapplied)
