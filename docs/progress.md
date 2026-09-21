# Progress

## P0 checkpoint — initial original-upstream CPU execution

Candidate state: initial uncommitted worktree; no candidate package exists.
P0 is **in progress**, P1–P8 remain pending. No tolerance or speed target is frozen.

Completed evidence:

- Pinned checkout and all 113 tracked-file hashes in
  `reports/source_manifest.json`; seven function signatures, 27 CLI declarations,
  and two upstream console scripts in `reports/contract_snapshot.json`.
- Isolated Python 3.11 CPU oracle, installed dependency lock in
  `reports/oracle-requirements.txt`; installed source is checked against checkout.
- Deterministic 32×35 projected, 24-hour input fixture with buildings and trees;
  `reports/initial_fixture_manifest.json` records hashes and generation procedure.
- Original declared-dependency run failed on undeclared Numba. Full logs,
  environment and raw measurements are preserved under
  `reports/characterization/initial_cpu/`.
- Installing Numba (no upstream source patch) enabled the full public workflow.
  Two successful runs are recorded under
  `reports/characterization/{dependencies_cpu,repeat_cpu}/`.
- Ten runtime TIFFs have exact expected names, 24 float32 bands, original CRS and
  transform, no NoData tags, correct timestamps and exact UTCI roof NaN locations.
  Fifteen ZIP fields and three `(32,35,153)` float32 visibility fields passed
  schema checks. See `reports/initial_artifact_snapshot.json`.
- Nineteen standalone TIFFs (including inputs/intermediates) matched exactly
  across runs, including NaNs and checked metadata:
  `reports/initial_repeatability.json`. This is same-host repeatability only.
- Source audits completed for wrappers, preprocessing, walls/aspect, SVF shadows,
  chronological driver, comfort, solar and materials: `docs/audit/`.

Commands after the initial environment setup in `docs/provenance.md`:

```sh
uv pip install --python .venv-oracle/bin/python numba
uv pip freeze --python .venv-oracle/bin/python > reports/oracle-requirements.txt
.venv-oracle/bin/python tools/run_reference.py --fixture tests/fixtures/generated/own_met_small --kwargs reports/initial_kwargs.json --run reports/runs/dependencies_cpu
.venv-oracle/bin/python tools/run_reference.py --fixture tests/fixtures/generated/own_met_small --kwargs reports/initial_kwargs.json --run reports/runs/repeat_cpu
.venv-oracle/bin/python tools/snapshot_artifacts.py
.venv-oracle/bin/python tools/check_reference_repeatability.py reports/runs/dependencies_cpu/scene reports/runs/repeat_cpu/scene --report reports/initial_repeatability.json
```

The artifact verifier was also checked against a temporary copied scene with a
corrupted second TMRT timestamp; it correctly returned failure. Success is not
based merely on the public function's `None` return or process exit code.

Initial dependency-complete run: 14.5166 s including imports; sampled summed
process-tree RSS 488,816,640 bytes. This is one small unoptimized upstream run,
not a release benchmark. Raw sampling caveats are in each measurement file.
The later source-hash guard additionally covers the land-cover text table;
earlier run manifests explicitly list their narrower Python-file hash coverage.

Unavailable: CUDA on this macOS ARM64 host. No CUDA pass is claimed.
Unverified: candidate parity, cross-platform variation, intermediate state,
optional-path execution, real-scene validation, robust cache/restart behavior,
release performance targets and the rest of the Section 13 completion gate.

Next dependency-ready work remains P0: complete radiation, wind and input-builder
source audits; construct intermediate/state oracle capture and edge fixtures;
characterize dtypes and visibility categories; snapshot CLI execution; execute
matched default/tuned CPU trials; freeze justified fixture, numerical and
benchmark policies before performance-oriented implementation.

## P0 checkpoint — intermediate oracle and thread characterization

This turn is **progress**, not a wait or a completion claim. Candidate state is
still the uncommitted initial worktree without a runtime package. P0 stays in
progress; later milestones are pending.

Completed since the first checkpoint:

- Remaining runtime-family source audits: `docs/audit/radiation.md` and
  `docs/audit/optional_inputs_wind.md`. The machine-readable static graph covers
  all 14 package Python files, 202 functions, 3 classes and 3,776 call sites,
  including default/decorator calls. Independent AST counts match. Static graph
  limitations are explicit; this is not runtime coverage of optional paths.
- Executed CLI help/version/errors, accepted boolean strings and argument
  forwarding: `reports/cli_execution.json`. Spy-based forwarding is labeled
  separately from actual workflow execution.
- Read-only original CPU capture: 222 component events, 24 main-call input/output
  boundaries, 23 checked state handoffs. Six captured fields match their TIFF
  bands exactly. `reports/boundary_capture_verification.json` records the schema
  and dtype evidence; `reports/capture_noninterference.json` proves the nineteen
  standalone TIFFs match the ordinary oracle run.
- Geometry oracle: all 126 function cases executed; 18/20 independent flat-scene
  checks passed and two exact-zenith checks failed. Combined vegetation visibility
  reached 2. Preserve these failures/categories; do not Boolean-pack that channel.
- Offline original references packaged under `tests/reference/small_original_cpu`
  (255 hashed files) and `tests/reference/geometry_original_cpu` (252 NPZ files
  plus provenance). None were produced by candidate code. Packaged diagnostic
  capture includes exact copies of its harness.
- Fifteen randomized CPU-thread trials (five each default, one, four threads),
  every trial passing exact standalone-TIFF comparison. Raw trials, environments,
  logs and harness snapshots are in `reports/characterization/tuning_small_frozen`;
  summary/protocol in `reports/upstream_cpu_tuning.json`. Medians: default 9.006 s,
  one thread 9.252 s, four threads 8.739 s. Overlap precludes a reliable tuning
  benefit claim on this small scene; see `docs/performance.md`.

Reproducible commands (use fresh run/report paths for generators that reject
overwriting evidence):

```sh
python3 tools/source_call_graph.py
python3 tools/snapshot_cli_execution.py
.venv-oracle/bin/python tools/characterize_geometry.py --run reports/runs/geometry_characterization/reproduction --report reports/geometry_reproduction.json
.venv-oracle/bin/python tools/benchmark_upstream.py --run reports/runs/tuning_small_frozen --report reports/upstream_cpu_tuning.json
.venv-oracle/bin/python tools/run_reference.py --fixture tests/fixtures/generated/own_met_small --kwargs reports/initial_kwargs.json --run reports/runs/boundaries_cpu_provenance --capture
.venv-oracle/bin/python tools/verify_boundary_capture.py reports/runs/boundaries_cpu_provenance --report reports/boundary_capture_verification.json
.venv-oracle/bin/python tools/check_reference_repeatability.py reports/runs/dependencies_cpu/scene reports/runs/boundaries_cpu_provenance/scene --report reports/capture_noninterference.json
.venv-oracle/bin/python tools/package_reference.py reports/runs/boundaries_cpu_provenance tests/reference/small_original_cpu
```

Verification also deliberately corrupted temporary copies of a capture file,
a carried state field (with a recomputed artifact hash), and a timestep event.
All three were rejected by the verifier. A separate nested-function/default/
decorator AST probe verified names and call-site completeness. Tools compile;
upstream tracked checkout remains unchanged.

Invalidated evidence retained: the first CPU-tuning batch briefly used a changed
harness for two trials, adding source-copy I/O. Its timing report is explicitly
invalidated and preserved as `reports/upstream_cpu_tuning_invalidated.json`;
the entire predeclared experiment was rerun, not selectively filtered. The
replacement guards runner SHA256 before and after each trial.

Remaining P0 work: expanded end-to-end fixtures (land cover, wind/UHI, save-flag
and cache variants), stage/geometry-warm baselines and larger priority-workload
characterization, field-specific tolerance calibration and benchmark policy
freeze. `docs/numerical_contract.md` remains a proposal, not permission to optimize.
CUDA is unavailable; exact-zenith scientific failures are unresolved and visible.
The full Section 13 release gate is not satisfied.

## P0 gate closure and P1 chronological integration (2026-09-18)

Supersedes the remaining-P0 list above. P0 is complete for the correctness-first
P1 transition, following independent lead review. Frozen comparison and benchmark
proposals are in `benchmarks/protocols/{comparison_v1,benchmark_v1}.json`.
Additional original evidence includes seven optional local cases
(`reports/optional_local_characterization.json`), fourteen artifact-mode runs
(`reports/artifact_modes.json`), a 256-square cold run, and small cold/warm stage
profiles (`reports/characterization/`). Larger release benchmark matrices,
scientific deviations, and unavailable CUDA remain outstanding; P0 closure is
not a performance or release claim.

P1 now has serial NumPy geometry/radiation, solar/material/comfort functions, and
an internal chronological driver with single-owner streaming GeoTIFF writes.
A first prepared-TIFF 24-step run in `.venv-light` (no Torch installed) completed.
All ten runtime fields fell within the frozen budgets: largest TMRT error
0.00006103515625 C, UTCI 0.00005340576171875 C, WBGT 0.000057220458984375 C;
all masks matched. This preliminary comparison is being converted into a
cold/warm regression gate. Outputs: `reports/runs/p1_pipeline_warm/`.

The radiation replay decoder was corrected to preserve NumPy float64 solar/time
scalars supplied by the actual upstream driver. The corrected replay and carried
state tests pass (25 tests); no numerical-engine or tolerance changes were needed.
Exact commands:

```sh
PYTHONPATH=src .venv-light/bin/python -m pytest tests/differential/test_radiation_reference.py -q
PYTHONPATH=src .venv-light/bin/python -m pytest tests/differential/test_pipeline_reference.py -q
```

P1 remains in progress: raw-input preprocessing/public API integration, full
metadata and numerical regression checks, installed-package and CI verification
are still required. No speedup or repository-completion claim is made.

P1 integration verification update: the prepared-TIFF regression now passes all
8 cases (small cold/warm; base, UHI, landcover, explicitly normalized landcover,
directional wind, and legacy-wrapper wind). It checks ten output fields, all
24 bands, exact metadata and special-value masks, and frozen numerical budgets.
Per-case field maxima and worst coordinates are preserved in
`reports/p1_pipeline_comparison.json`. The earlier complete differential suite
passed 179 tests before this six-case expansion; the expanded pipeline suite
passed 8 tests. Divide-by-zero/invalid NumPy warnings remain visible; their output
masks and values pass the reference checks. Own-met preprocessing separately
passed seven tests, including exact source-extracted original tiling oracles.

The public `thermal_comfort` raw-TIFF path now also passes the complete 24-band,
ten-field reference comparison without Torch (`test_pipeline_reference.py -k raw`,
1 passed). This regenerates tiling, walls/aspects and SVF before simulation.
The test uses an absolute meteorological path: upstream interprets an explicit
relative own-met filename against the working directory, not base_path.
Public raw workflow precomputes SVF, so its output folder correctly omits the
cold-driver-only SVF duplicate. P1 packaging/compatibility and CI remain pending.

## P1 installed-package gate (2026-09-18)

Implemented five local public workflows, the 27-flag CLI, the main executable,
and the separate opt-in compatibility distribution. Seven public surfaces remain
inventoried; `build_inputs` and `build_wind_ext_coeff` are explicitly deferred to
P5. The companion has installed-wheel forwarding and synthetic upstream-conflict
tests, with installation-order limits documented in `docs/compatibility.md`.
The pipeline now uses six live ownership dataclasses and complete carried state.

Independent P1 review found and resolved two concrete gaps: DEM median now
propagates any NaN as the original Torch median does; written visibility NPZ,
ZIP TIFFs and standalone SVF now have explicit schema and value comparisons.
The isolated original median probe checked all six NaN positions. This preserves
an upstream behavior rather than introducing a scientific correction.

Final local gate: **245 passed, zero skipped**, including nine genuine 24-step
TIFF pipeline cases, installed main/companion wheels, CLI/public signatures,
state ownership and all numerical-family differential tests. Two numerical
warnings remain visible; special-value masks and numerical budgets pass.
The main runtime was loaded from `.venv-wheel/site-packages`, with Torch,
xarray, cdsapi and wrf absent. Wheel contents match both installed files and
current source bytewise. Evidence: `reports/p1_installed_tests.xml` and
`reports/p1_installed_verification.json` (source/wheel/reference hashes,
environment, worst field coordinates and exact command).

```sh
.venv-light/bin/python -m build --wheel --outdir dist
uv pip install --python .venv-wheel/bin/python -r requirements/macos-arm64-py311.txt
uv pip install --python .venv-wheel/bin/python --no-deps --reinstall dist/solweig_light-0.1.0.dev0-py3-none-any.whl
.venv-wheel/bin/python -m pytest tests/differential tests/unit -q --junitxml=reports/p1_installed_tests.xml
.venv-wheel/bin/python tools/record_p1_verification.py
```

`.github/workflows/cpu-reference.yml` configures a real installed-wheel own-met
PR job. Remote CI has **not executed**; local tests are the current evidence.
Candidate remains an uncommitted worktree, identified by hashes. No benchmark
claim was made. Full-float visibility, existence-based legacy cache reuse,
optional integrations, restart/concurrency and wider scientific/platform coverage
remain explicit later milestones. Final independent P1 gate recheck is pending.

Independent final review accepted the P1 correctness-first gate after checking
the three fixes, JUnit counts/hash, and recorded source hashes against current
files. `TASKS.yaml` marks P1 complete. Dependency-ready work is P2 (compiled wall,
aspect and UTCI kernels) and P3 (shadows/SVF/lossless visibility); begin with P2.
Remaining scalar/angular calibration must precede optimization of those fields.
This milestone acceptance does not close any P5–P8 release requirement.

## P2 implementation and oracle expansion (2026-09-18, in progress)

P2 lead review identified float64 wall subtraction, NaN/signed-zero maximum
semantics, exact aspect ties and float32 side reductions as mandatory invariants.
UTCI requires ordered-expression provenance and explicit scalar promotions.

Original, unmodified wall/aspect fixtures were generated in `.venv-oracle`:
`tests/reference/walls_original_cpu` (21 cases) and
`tests/reference/walls_expanded_original_cpu` (31 cases plus 900 captured rotated
filters). The expanded generator reads the exact upstream filter locals through
a read-only trace. Manifests retain input/output, source, generator and environment
provenance; neither fixture family contains candidate-generated goldens.

Compiled serial and pixel-owned parallel wall/aspect paths passed 358 targeted
checks, including all original cases at one/four/ten threads, immutable bounded
filter-cache checks, signed-zero combinations, exported geometry and nine complete
chronological pipeline cases. Side-height reductions and gradient fallback retain
NumPy behavior. A negative-threshold probe exposed a Numba signed-zero maximum
difference hidden by the normal three-meter threshold; the explicit maximum
rule and regression now preserve the pinned NumPy behavior. Diagnostics are in
`reports/p2/walls_parallel_diagnostics.txt`.

```sh
.venv-oracle/bin/python tools/characterize_walls.py
PYTHONPATH=src .venv-light/bin/python -m pytest tests/differential/test_walls_compiled.py tests/differential/test_geometry_reference.py tests/differential/test_pipeline_reference.py -q
```

`benchmarks/protocols/p2_kernels_v1.json` freezes local equal-work kernel timing
cases before candidate measurements. Harnesses separate first-call compilation/
filter construction from five warm calls, and label process-lifetime RSS versus
kernel timing. Timings are not end-to-end evidence. UTCI work and P2 integration,
measurements and final review are still in progress; P2 is not complete.

P2 integration and measurement checkpoint:

- The 211 ordered UTCI terms were mechanically extracted and reconstructed with
  checksums. Original fixtures cover 10,790 float32 values and fifteen mixed/
  float64 profiles (the latter reproduce original assignment errors). Compiled
  nominal maximum error is 0.017578125 C, within the unchanged 0.02 C gate but
  with a relatively narrow margin. Scalar pressure/polynomial diagnostics and
  original failures remain visible in `reports/utci_compiled_characterization.json`.
- Integrated the serial uniform-forcing UTCI path; pressure is computed once per
  timestep. All nine original TIFF cases pass. Unsupported compiled-profile
  values retain full NumPy evaluation, not clipping or approximation.
- All 28 frozen kernel configurations completed without failure, and their
  input hashes and outputs passed comparison. Raw trials, logs, outputs, source
  snapshots, protocol and harnesses are in
  `reports/characterization/p2_kernels_v1`. Summary:
  `reports/p2_kernel_comparison.json` and `reports/p2_kernel_measurements.md`.
  Compilation makes UTCI first calls slower than upstream despite lower warm
  kernel times. No end-to-end speedup or optimal CPU-budget claim is made.
- Sixteen additional warmed calls balanced NRT allocations/frees after output
  release. `reports/p2_allocation_audit.json` distinguishes visible traced bytes,
  output bytes and native event counts from complete application peak memory.
- Final rebuilt installed wheel: **515 passed, no skips**. The wheel matches
  installed/current source bytewise; Torch and acquisition packages are absent.
  `reports/p2_installed_tests.xml` and `reports/p2_installed_verification.json`
  contain the gate evidence and all nine TIFF worst-error reports. Full-wheel
  tests include written geometry schemas and the compatibility distribution.

```sh
.venv-light/bin/python tools/run_p2_benchmarks.py --run reports/runs/p2_kernels_v1
.venv-light/bin/python tools/check_p2_benchmarks.py reports/characterization/p2_kernels_v1 --report reports/p2_kernel_comparison.json
NUMBA_NRT_STATS=1 PYTHONPATH=src .venv-light/bin/python tools/audit_p2_allocations.py
.venv-light/bin/python -m build --wheel --outdir dist
uv pip install --python .venv-wheel/bin/python --no-deps --reinstall dist/solweig_light-0.1.0.dev0-py3-none-any.whl
.venv-wheel/bin/python -m pytest tests/differential tests/unit -q --junitxml=reports/p2_installed_tests.xml
.venv-wheel/bin/python tools/record_p1_verification.py --milestone p2
```

Final P2 independent gate review is pending. Remote CI is configured but
unexecuted; CUDA unavailable; candidate is an uncommitted worktree identified by
source hashes. Later cache/restart, optional workflows, scientific/platform and
full performance requirements remain open.

Independent final review accepted P2 after verifying installed-suite counts and
hashes, all 28 kernel comparisons, all 16 allocation accounting results and their
source hashes. `TASKS.yaml` marks P2 complete. The next dependency-ready milestone
is P3: shadows, SVF and lossless compact visibility. P4 remains dependent on P3;
P5–P8 are not complete.

## P3 — compiled shadows, SVF and lossless visibility

Candidate remains an uncommitted worktree, identified by source hashes in
`reports/p3_installed_verification.json` and archived benchmark snapshots.

- Both shadow families now have serial and parallel nopython kernels with
  unchanged ray support. Sky pixel ownership is used where the global bush
  condition is inactive; otherwise the original step recurrence is retained.
  Wall shadows compile the characterized float32 profile and retain the array
  fallback for other dtypes. Thread counts 1, 4 and 10 pass frozen comparisons.
- Original full-SVF fixtures cover four scenes and every option. Options 1–3
  match all fields and visibility; option 4 preserves the original annulus
  TypeError, explicitly recorded as an upstream failure rather than successful
  SVF computation. Normal workflows encode patches immediately; the legacy
  low-level dense-return function preserves its return contract.
- Binary/ternary/raw visibility is lossless, including noncanonical float bits.
  Radiation decodes one patch and forms diffsh lazily. Legacy exports use
  bounded numeric buffers and unique atomic temporary paths. Concurrent
  distinct-tile SVF exports pass exact arrays and metadata checks.
- Native storage uses encoded uint8 NPY with a checksummed atomic manifest,
  readonly mmap and explicit close ownership. `reports/p3_visibility_memory.json`
  verifies no whole-payload copy, exact 459-plane export and bounded visible
  allocations. It excludes mapped resident pages and untraced native allocations.
  Dependency-valid automatic reuse, cleanup and restart policy remain P6.
- All 43 frozen geometry benchmark configurations passed numerical comparison.
  Raw trials/source/protocol/harness are in
  `reports/characterization/p3_geometry_v1`; interpretation and all measurements
  are in `reports/p3_geometry_measurements.md`. Legacy compact loading and
  positive-bush serial kernels have measured runtime regressions. No end-to-end
  speedup or general release performance threshold is claimed.
- Rebuilt installed wheel: **2,038 passed, zero skips**, including nine real
  TIFF cases and compact/dense chronological intermediate comparisons. Three
  warnings arise in retained numerical recurrences. Torch and acquisition
  packages are absent. Source, wheel and installed files match bytewise.
  Evidence: `reports/p3_installed_tests.xml`,
  `reports/p3_installed_verification.json`.

```sh
.venv-light/bin/python tools/run_p3_benchmarks.py --run reports/runs/p3_geometry_v1
.venv-light/bin/python tools/check_p3_benchmarks.py reports/characterization/p3_geometry_v1 --report reports/p3_geometry_comparison.json
PYTHONPATH=src .venv-light/bin/python tools/audit_p3_visibility_memory.py
.venv-light/bin/python -m build --wheel --outdir dist
uv pip install --python .venv-wheel/bin/python --no-deps --reinstall dist/solweig_light-0.1.0.dev0-py3-none-any.whl
.venv-wheel/bin/python -m pytest tests/differential tests/unit -q --junitxml=reports/p3_installed_tests.xml
.venv-wheel/bin/python tools/record_p1_verification.py --milestone p3
```

Independent final review accepted P3 after checking installed-gate counts,
source/JUnit hashes, native mmap audit and all 43 benchmark comparisons.
`TASKS.yaml` marks P3 complete. CUDA remains unavailable and remote CI
unexecuted. P4 is dependency-ready; P5–P8 remain incomplete.

## P4 — active implementation checkpoint

P3 dependencies are complete; P4 is in progress. Lead review identified the
original GVF outside-slice scratch history, mutable sunwall and water-temperature
ordering as explicit compatibility invariants. Ground-view and patch-radiation
workers own separate new modules; engine dispatch is not yet changed. A separate
original-only sequence collector is investigating cross-day reset evidence.

- `PYTHONPATH=src .venv-light/bin/python -m pytest tests/differential/test_state_resume.py -q`: **5 passed**. Complete engine state is serialized/restored at five cuts of the original 24-step fixture; subsequent components are exact against uninterrupted candidate execution and pass original-reference gates. This is test-only numerical handoff, not production checkpoint/recovery support. Cross-day resets remain to be verified.
- `.venv-oracle/bin/python tools/calibrate_p4_gvf.py`: unperturbed original replay has zero TMRT difference; coherent positive/negative and checkerboard 1e-6 perturbations to all dimensionless GVF outputs produce maximum TMRT difference 0.00006103515625 C. `reports/p4_gvf_sensitivity.json` records script/source/reference hashes and per-step results. Perturbed runs are sensitivity experiments, never upstream goldens. Evidence is limited to one scene and does not establish a universal error bound.
- `models.GeometryCache` documentation/type now reflect compact lazy visibility; no model calculation changed in this checkpoint.

P4 kernel implementation, targeted mutation/edge fixtures, integration,
cross-day numerical resumption and final installed-wheel review remain open.

P4 additional checkpoint: forty original-only delay threshold/irregular-step
cases captured with `.venv-oracle/bin/python tools/characterize_delay.py`;
`PYTHONPATH=src .venv-light/bin/python -m pytest tests/differential/test_delay_reference.py -q`
passed all 40. The 48-hour unmodified original driver packet is complete under
`tests/reference/state_sequence_original_cpu`, with 96 main boundaries and
423 carried-state links. Its seven candidate resume-cut cases passed across
336 main-engine comparisons (39 fields each). Benchmark protocol
`benchmarks/protocols/p4_radiation_v1.json` is frozen before measurement;
component implementations and final integration remain in progress.

P4 integration checkpoint: compiled ground-view dispatch passed 71 component,
TIFF and state-handoff checks. Shortwave dispatch then passed all 50 original
component-boundary cases. After longwave dispatch, the diagnostic observer was
moved to the module-local nested patch sweep; all 50 boundary cases passed with
every original field retained. The other 23 cases (nine TIFF workflows and
12 explicit resume cases plus carried-state cases) passed in the initial run;
that run's 48 failures were missing diagnostic observations, not numerical
mismatches. A clean combined installed gate remains required.

Worker subsystem evidence: ground-view 154 passed, patch radiation 562 passed.
Parallel ground diagnostics are retained under
`reports/characterization/ground_view_compiled`. The 29-configuration benchmark
harness is prepared but has not run. Independent source review identified a
remaining plan §5.5 gap: cache fixed altitude memberships, cardinal masks and
longwave direction factors, including avoiding repeated `unique` inside the
per-step emissivity path. The worker is completing that change while keeping
forcing-dependent coefficients dynamic. No P4 completion or performance claim
is made.

P4 cache review now accepts the fixed metadata cache and dynamic coefficient
separation. A rebuilt installed-wheel suite is running, but an additional root
exception audit found a compatibility gap not covered by its existing tests:
three sampled original failure frames succeed in the candidate. The original
box shortwave call rejects a Python-bool `torch.where` condition; longwave can
reject a Python-float argument to `torch.tan`. The packet had recorded these
failures without executing equivalent candidate failure checks. A bounded fix
and tests for all recorded failure frames are in progress. Benchmarks remain
unexecuted; a final wheel rebuild/regression must include this correction.

The pre-exception-correction installed wheel completed **2,811 tests, zero
failures/errors/skips**, with three retained numerical warnings (246.68 s).
Its wheel, JUnit and result record are archived under
`reports/characterization/p4_pre_exception`; this is explicitly not the final
P4 gate. Root added reference-order guards for the first box conditional and
longwave tangent helper; worker tests now execute original failed frames and
nighttime counterexamples. The benchmark checker also records bias, MAE, RMSE,
p95/p99, maximum relative error on nonzero references, worst coordinates and
exact dtype/mask checks. Original outputs compared with themselves verified
zero-error statistics for all 30 component fields.

Exception compatibility is now source-reviewed and tested: 68 targeted checks
passed, including every original failed frame through module and engine
entrypoints. `reports/p4_scalar_counterexamples.json`, reproduced by
`.venv-oracle/bin/python tools/verify_patch_scalar_counterexamples.py`, records
eight successful unmodified-original counterexamples and fourteen bit-exact
shortwave counterpart fields. These original results are separate from
candidate-reference routing tests. The corrected source was rebuilt/reinstalled;
its complete installed suite is running before benchmark execution.

## P4 final verification evidence

- Corrected installed wheel: **2,879 passed, zero failures/errors/skips**,
  three retained numerical warnings, 218.03 s. Source, wheel and installed bytes
  match. All nine TIFF cases pass; Torch and acquisition dependencies are absent.
  Evidence: `reports/p4_installed_tests.xml` and
  `reports/p4_installed_verification.json`.
- All **29 frozen component configurations** completed successfully and passed
  equal-input, output, dtype, mask and mutation checks. Raw trials, logs,
  fixtures, environments, guarded source snapshots and harness are in
  `reports/characterization/p4_radiation_v1`; comparisons and full interpretation
  are `reports/p4_radiation_comparison.json` and
  `reports/p4_radiation_measurements.md`.
- Ground-view warm calls improve, but compact patch radiation has substantial
  regressions at the frozen 128-pixel block (approximately 11.1x shortwave and
  3.4x longwave runtime versus the best measured upstream warm configurations).
  Compilation also penalizes first calls. These are component measurements,
  not end-to-end claims. Profiling and execution-block/native-decode tuning
  remain required in P6/P7; the original matrix must remain visible.
- The numerical architecture and remaining scope are summarized in
  `docs/p4_radiation.md`. Source review accepted cache and exception compatibility
  fixes; final milestone acceptance awaits review of the timing evidence.

```sh
.venv-light/bin/python -m build --wheel --outdir dist
uv pip install --python .venv-wheel/bin/python --no-deps --reinstall dist/solweig_light-0.1.0.dev0-py3-none-any.whl
.venv-wheel/bin/python -m pytest tests/differential tests/unit -q --junitxml=reports/p4_installed_tests.xml
.venv-wheel/bin/python tools/record_p1_verification.py --milestone p4
.venv-light/bin/python tools/run_p4_benchmarks.py --run reports/runs/p4_radiation_v1
.venv-light/bin/python tools/check_p4_benchmarks.py reports/characterization/p4_radiation_v1 --report reports/p4_radiation_comparison.json
```

Candidate remains an uncommitted worktree with file-hash provenance. CUDA is
unavailable; remote CI is configured but unexecuted. P5 work packets are prepared
in `docs/p5_work_packets.md`, but P5–P8 are not complete.

Independent final review accepted P4 within its numerical/integration scope
after checking the installed/source evidence and all 29 benchmark results.
`TASKS.yaml` marks P4 complete. The measured radiation regressions remain
required P6/P7 performance work and preclude a speedup claim. P5 is now
dependency-ready; repository/release completion remains outstanding.

## P5 — optional workflows in progress

Implemented and integrated all seven public exports, with exact frozen argument
semantics. Legacy public/module forwarding now includes input construction,
wind generation and forcing helpers. Optional dependencies are declared as
`forcing`, `wind` and `inputs` extras and remain lazy. The separate Torch-free
`.venv-optional` environment exercises local optional workflows; the core wheel
environment remains free of acquisition/forcing dependencies.

- Component tests moved to `tests/optional` so core dependency isolation and
  required optional verification have separate explicit test scopes. Combined
  optional gate: **50 passed**, with real file processing and clearly labeled
  acquisition mocks. The known netCDF/NumPy extension warning is being diagnosed.
- Wind: all twelve directional outputs match original arrays/masks/metadata at
  worker counts 1/2/4. First-time roughness, precedence and failures preserved.
- ERA5/UHI: exact local NetCDF/metfile comparisons include inclusive time
  selection, DST, masks and input-coordinate variants. Original failures remain
  recorded. WRF uses only reviewed timestamp repair v1, with separate patch/hash
  and patched references; duplicate-domain shape failure remains unchanged.
- Input construction: seven targeted checks cover original unchanged extracted
  helper references, verified artifact/harness hashes, real raster/vector
  processing and offline acquisition responses. No live-service claim is made.
- Public signatures and installed companion checks: **15 passed** before the
  final module-forwarding additions; those additions are being rechecked.
  Fourteen core signature/preprocessing checks passed after optional dispatch.
- CI now has a separate optional-local wheel job, still unexecuted remotely.

Root is completing public optional preprocessing integration, final installed
main/companion gates, dependency reports and documentation. P5 is not complete.

## P5 — final installed verification

All seven public functions are operational. Final wheel/source identity and
executed gates are recorded in `reports/p5_installed_verification.json`,
`reports/p5_optional_installed_verification.json`, and
`reports/p5_forcing_extra_verification.json`. The candidate remains an
uncommitted worktree; reports preserve source, wheel and reference hashes.

- Core: 2,881 passed, zero skips, 220.53 s; nine real raw/prepared chronological
  TIFF comparisons. Torch, xarray, cdsapi and wrf are absent.
- Optional: 58 passed, zero skips, 16.69 s. Public ERA5/UHI preprocessing and the
  explicitly patched WRF path execute real files. Roughness-driven thermal
  integration generates all twelve directions and consumes them through all
  24 wind bands, with exact upstream wind outputs and frozen UTCI tolerance.
- Clean forcing-only extra: 34 passed, zero skips, 4.75 s, after review found
  and fixed its missing Shapely dependency. Other optional extras cannot mask
  dependency omissions. The three jobs are configured in CI, not run remotely.
- Input-builder coverage uses original unchanged AST-extracted helpers and
  real local transformations with offline acquisition mocks. Authenticated
  service compatibility is not established. WRF original and residual failures
  remain separate from the versioned patched reference.
- Installed companion forwarding, all seven signatures, legacy executable and
  collision checks pass within the core suite. Known warnings remain visible.

Exact commands and evidence boundaries: `docs/p5_optional.md`. No P5 speedup
claim is made. CUDA remains unavailable. Lead final review accepted the P5 gate after verifying matching hashes and
JUnit records. P6 is now in progress under `docs/p6_work_packets.md`; P6–P8
remain incomplete.

## P6 — contracts and bounded implementation started

After P5 acceptance, three workers own separate cache, persistence/output and
runtime/planner modules under `docs/p6_work_packets.md`. Root retains shared
API/pipeline/model/radiation integration. New
`tests/integration/test_runtime_pipeline.py` specifies real 24-hour original
comparisons for block/thread changes and interruption after a committed step.
These new P6 tests have not run; their runtime/persistence interfaces are still
being implemented. No P6 gate is satisfied by the P5 reports.

## P6 — integrated cache, state recovery and resource controls

The native cache, standalone export validator, transactional writer and scoped
runtime are integrated into the chronological pipeline. Existing logical tiles,
153-patch workload and numerical tolerances remain unchanged. Public tile calls
now launch workers with native settings established before numerical imports.

Executed development checks:

- `PYTHONPATH=src .venv-light/bin/python -m pytest tests/integration/test_runtime_pipeline.py tests/differential/test_pipeline_reference.py -q --junitxml=reports/p6_pipeline_checkpoint_tests.xml`
  passed 16 checks, including all nine existing raw/prepared original TIFF
  cases, thread/block variants, two tile workers, restart and 96-hour ownership.
- Additional committed/uncommitted, cold/warm restart cases passed. A later
  integration/identity run passed 11 checks; two controlled source-mutation
  checks also passed, preventing wrong-identity cache/output publication.
- Cache/native/legacy and original geometry checks passed; standalone service
  now has 30 passing tests, including shared destination locks, input changes,
  stale/corrupt exports, partial publication and early memory admission.
- Persistence/model checks passed 42 tests. State generations remain bounded
  on disk, and interrupted publication/recovery retains complete model state.
- Runtime tests cover cancellation with live children, admission and resource
  discovery; nested cgroup and child-exception propagation follow-ups are still
  being finalized before source freeze.

A provisional two-generation raster ownership assertion initially found three:
current/previous outputs plus an unreachable first engine frame. Cyclic
collection released the latter. The pipeline now explicitly releases completed
outputs before the next step, and the original two-generation bound passes over
96 hours, with all observed output arrays released after collection. No
numerical or frozen performance threshold was relaxed.

Lead review exposed and drove fixes for cancellation cleanup, cross-entrypoint
export ownership, mutable source identity, resource-limit discovery and geometry
admission before allocation. `docs/runtime.md`, `docs/cache_policy.md` and
`docs/p6_memory_inventory.md` describe implementation boundaries. The resource
protocol/harness is authored but unexecuted; memory measurements, broader stage
resource coverage and the final installed-wheel gate remain outstanding. P6 is
not complete.

## P6 — core resource characterization checkpoint

Frozen core characterization completed all eight cases with no source-hash
changes. `reports/p6_runtime_measurements.md` summarizes raw artifacts under
`reports/characterization/p6_runtime_v1_retry2_psutil`: one-worker peaks
374.4–386.1 MiB across 24/96/288 hours, two-worker peak 599.1 MiB, all below the
4 GiB configured budget. Per-tile output schemas, completion records and
artifact hashes passed. Single trials and a tiny scene limit interpretation;
no speedup or universal memory-bound claim is made. Failed dependency/harness
setup attempts remain preserved and excluded from passed cases.

The source freeze is released. The wind worker is implementing staged memory
admission, bounded directional work/results and shared output ownership without
changing any numerical function. Full P6 installed-wheel/optional gates and
final lead review remain outstanding; P6 stays in progress.

## P6 — storage recheck and container admission review

The current source passed 115 cache/service/persistence checks with
`PYTHONPATH=src .venv-light/bin/python -m pytest tests/unit/test_geometry_cache.py tests/unit/test_geometry_service.py tests/unit/test_persistence.py -q --junitxml=reports/p6_storage_tests.xml`.
The runtime/identity subset passed 18 checks with
`PYTHONPATH=src .venv-light/bin/python -m pytest tests/unit/test_identities.py tests/unit/test_runtime.py -q`.
These are source checks, not the final installed-wheel gate.

Independent review identified two remaining resource-discovery defects:
cgroup memory consumption must be subtracted from finite ancestor limits, and
hybrid/split-controller layouts must select the actual controlling hierarchy.
Targeted fixes and fixtures are in progress. The installed-wheel CI definition
now includes cache, restart and runtime integration checks; remote CI remains
unexecuted. P6 verification recorders now accept distinct P6 report paths,
preserving the accepted P5 evidence.

Both container findings were fixed and accepted by the bounded independent
source review. A follow-up zero-limit boundary fix treats literal zero limits
as exhausted and clamps over-limit headroom to zero. The runtime suite passed
19 tests with `PYTHONPATH=src .venv-light/bin/python -m pytest tests/unit/test_runtime.py -q`.
Explicit caller budgets retain their existing semantics. Default headroom is
a changing snapshot, not a hard operating-system enforcement mechanism.

## P6 — final installed gates and wind characterization

Built the wheel with `.venv-light/bin/python -m build --wheel --outdir dist`
and reinstalled it without dependencies in the minimal, optional and forcing
environments. Final installed checks passed with zero skips:

- 3,029 core/differential/integration checks, including nine original TIFF
  cases, in 308.85 s; `reports/p6_installed_verification.json`.
- 78 optional checks in 38.33 s;
  `reports/p6_optional_installed_verification.json`.
- 34 forcing-only checks in 5.13 s;
  `reports/p6_forcing_extra_verification.json`.

Each report records its exact command, dependencies, wheel/source identity,
reference hashes and JUnit hash. Known model/native/deprecation warnings
remain visible. CUDA is unavailable and remote CI is unexecuted.

The frozen wind resource protocol passed all 1/2/4-worker cases, including exact
original field and metadata checks. Sampled process-tree peaks were
152,797,184 / 154,992,640 / 157,073,408 bytes, below the explicit 4 GiB budget.
See `reports/p6_wind_measurements.md` and its raw artifact links. No speedup or
universal memory-bound claim is made. Final P6 independent gate review remains
pending; milestone status has not changed.

Independent final review accepted P6 after checking current source, matching
wheel/JUnit hashes and the executed gates above. `TASKS.yaml` now marks P6
complete. Core RSS evidence retains its historical source identity; the final
wheel is covered by installed tests, while wind RSS covers final source. Wind
publication rollback covers caught failures, not crash-atomic replacement of
the complete file set. Small sampled workloads do not establish universal
memory bounds. P7 is the next dependency-ready milestone; P4 patch-radiation
regressions and all remaining P8 release requirements are unresolved.

## P7 — baseline preserved and bottleneck measured

P7 is in progress. Preserved the accepted P6 source/wheel and verification
reports with hashes under `reports/characterization/p7_p6_baseline` before
any numerical changes. Warm cProfile observations on the frozen P4 compact
inputs locate most time in Python visibility decoding/block assembly, followed
by classification. Raw evidence and qualifications are linked from
`docs/p7_experiments.md`. No optimization or speedup is yet claimed.

Lead review defined numerical/lifetime invariants and full-workload promotion
requirements. Expanded synthetic and licensed-real fixture provenance is the
next prerequisite from the frozen benchmark protocol; an independent bounded
worker is auditing/preparing these inputs while the baseline remains unchanged.

## P7 — fixtures frozen and expanded references started

Seven input inventories and paired protocols are frozen under
`benchmarks/protocols/p7_pairs`; preparation/provenance is recorded in
`docs/p7_fixture_audit.md`. Published input definitions resolve intended model
semantics without asserting unknown acquisition history or vertical datum.
Real sample windows retain their native 2 m inputs and original metadata.

Original CPU dense-urban and vegetation-rich synthetic runs completed via
`tools/run_reference.py --threads 1`, with outputs under
`reports/runs/p7_dense_original_cpu` and `p7_vegetation_original_cpu`.
`tools/check_p7_expanded_fixture.py` checks the preserved P6 source against
these original outputs using unchanged field rules. Real-window references are
being generated with the same isolated runner. No candidate golden is used.

The paired harness now has 13 passing comparator tests, including NaN NoData
and bounded-window reads. Native block decoding is the first implementation
experiment; component timing waits for a quiet measurement window. No P7
promotion or speed claim has been made.

The decoder experiment passed 699 combined numerical/codec tests and nine
original TIFF cases, then an isolated component comparison found approximately
2.76× lower compact warm time with bitwise-equal outputs. This is not a full
pipeline claim. Classification experiment 2 passed 635 radiation checks,
66 classification/block checks and nine original TIFF cases; its full-workload
performance is unmeasured.

Expanded original comparison found a pre-existing P6 numerical mismatch on
the publisher's dense-urban window. Geometry exports pass, but several daytime
radiation/comfort fields exceed frozen limits. Diagnosis is active; no fixture
or tolerance has been removed or weakened. Performance promotion is paused.
The paired instrument's independent review also led to baseline-manifest,
warm-cache, host identity and global claim-eligibility checks; its 19 focused
tests pass, and no full performance matrix has run.

## P7 — angular compatibility defect repaired; broad gates running

Diagnosis isolated float32 scalar angular promotion in ground-view wall-facing
tests. The repair changes only local angular constants in `ground_view.py` and
the retained `engine.py` fallback; it does not change global promotion or model
physics. Original-derived masks/fields now pass 186 ground-view checks, all 17
first-daytime GVF fields are bitwise equal, and the full diagnosed 24-hour real
scene passes frozen TIFF gates with unchanged-source verification. Radiation,
state and delay checks passed 682 cases; resume checks passed five.

`tools/derive_p7_repaired_baseline.py` created the separately labeled source at
`reports/characterization/p7_repaired_p6_baseline`, with a two-file repair diff
and parent/source hashes. It excludes both P7 optimizations and leaves original
P6 untouched. Independent lead review requires this corrected baseline and
the optimized candidate each to pass all seven original fixture gates before
paired timing. Those checks are running via `tools/check_p7_expanded_fixture.py`
and `tools/check_p7_geometry_exports.py`. A new installed-wheel gate is also
running. No full-workload optimization promotion or upstream speed claim has
been made.

The expanded-check helper initially hardcoded a preserved-P6 evidence label
even for explicit `--source src` runs. Reporting now records the candidate
variant and source path. Five affected experiment/repair reports were
corrected from their recorded invocation, with initial reports retained beside
them; numerical results were unchanged.

The repaired optimized repeated-block check passed all numerical comparisons,
but its source guard detected wheel-build metadata changing
(`src/solweig_light.egg-info/SOURCES.txt`). No numerical source file changed.
The failed-guard run remains retained; a fresh run is required and has started
under `reports/runs/p7_repaired_optimized_repeated_block_256_retry` after the
build. The source guard is not being weakened or retroactively counted as a
pass.

The repeated-block retry passed numerical comparisons, unchanged-source checks
and all 18 geometry export fields. The installed P7 wheel passed 3,146 core,
78 optional and 34 forcing-only tests with zero failures or skips; reports are
`reports/p7_installed_verification.json`,
`reports/p7_optional_installed_verification.json` and
`reports/p7_forcing_extra_verification.json`. The repair-only baseline also
passed the nine original TIFF cases and 32 angular cases (41 total), recorded
in `reports/p7_repaired_baseline_regressions.xml`.

The expanded candidate nevertheless fails the unchanged Kdown gate at hour 15
on two published windows (vegetation-rich and sparse), with approximately
0.060 W/m² maximum error; all other saved fields pass. Full evidence remains
in their `p7_repaired_optimized_real_*` run directories. Diagnosis continues,
and neither the amended performance protocol nor P7 completion is approved.
The newly strengthened benchmark harness has a separate focused test gate;
its new tests were added after the installed-wheel run above.

## P7 — expanded correctness gate passed; paired timing started

The remaining Kdown failures were repaired through original scalar annulus
arithmetic. No tolerance or fixture was changed. The separate three-file
repair-only P6 baseline and optimized candidate both passed all seven full
scene gates, 18 geometry export fields per scene, the original nine TIFF
regressions, angular and annulus boundary tests, and a fresh bitwise replay of
17 GVF fields. Source inventories and evidence hashes are bound in
`reports/characterization/p7_repair_v2_correctness/gate.json`.

Commands and raw runs are recorded by `tools/run_p7_expanded_matrix.py` in
`reports/runs/p7_repair_v2_{baseline,optimized}/matrix.json`.
`reports/p7_repair_v2_installed_verification.json` records 3,195 installed-wheel
tests; optional and forcing-only reports with the same prefix record 78 and
34. All passed with zero skips. Six of 180 annulus weights still differ from
original by one ULP on this host; full-scene comparisons meet unchanged gates.

`tools/run_p4_benchmarks.py --run reports/characterization/p7_repaired_v2_component_matrix`
completed all 29 frozen component configurations; `tools/check_p4_benchmarks.py`
passed their comparisons. This remains component-only performance evidence.

`tools/amend_p7_pair_protocols.py` created v3 protocols retaining every v2
workload field and binding the reviewed three-file repair and correctness gate.
The five-pair, two-budget, cold/warm full-pipeline matrix has started. P7 remains
in progress until its measured benefit/regression gate is assessed; P8 remains
pending. CUDA and remote CI remain unavailable/unexecuted respectively.

An independent instrument review found missing validation of the linked
geometry reports. Root stopped the partial v3 small-scene timing run and
retained its raw trials and interruption reason. Geometry hashes, passed status
and all 18 distinct fields are now required; instrument/parent-protocol hashes
are checked at startup. The separate hardened instrument gate passed 54 tests
(`reports/p7_geometry_evidence_harness_tests.xml`); no numerical source changed.
The same matrix restarted with explicit v4 provenance using
`.venv-light/bin/python tools/run_p7_pairs.py --protocols benchmarks/protocols/p7_pairs_v4 --output reports/characterization/p7_pairs_v4`.

The CPU PR workflow now also selects the 32 angular-boundary cases, four
annulus arithmetic/receiver cases, and 16 independent geometry cases. Their
52 passing results were verified in the installed-wheel JUnit report before
adding the workflow step. This closes a CI selection gap; execution on the
remote runner remains unverified and is not counted as a pass.

Required top-level numerical, performance, peak-memory and optimization report
indices now exist. They link source-bound evidence and explicitly retain
incomplete release status. Root review corrected historical variant labels
against the actual five failed reports and separated the 16 analytic geometry
tests from original-reference comparisons. The memory index identifies its
historical P6 source limitations; it does not attribute those samples to the
current wheel. `tools/record_release_measurement_indices.py` reproduces the
performance/memory indices without running measurements.

The v4 small-scene benchmark completed all 20 pairs with final source/evidence,
hardware, numerical and cache guards passed. `reports/p7_pair_measurements.md`
records all four groups, including the mixed four-thread geometry-warm result
(median paired ratio 0.987, observed range 0.913–1.079). It is not being counted
as an established improvement. The other three group medians are 1.065–1.098;
all ratios compare against repair-only P6, not original upstream. The remaining
six scenes and the full promotion/regression review are still outstanding.

Three additional independent checks were authored during the quiet timing
window: night shortwave absence, Stefan–Boltzmann upward longwave under stated
night/isothermal assumptions, and explicit directional-wind sector selection
through a genuine 24-step public workflow. They are in
`tests/scientific/test_radiation_analytic.py` and
`tests/scientific/test_directional_wind_analytic.py`, with preconditions and
pre-execution tolerances in the corresponding `reports/p8_*_design.md` files.
They remain unexecuted and are not included in any passing count. Numerical
execution is deferred until it cannot interfere with paired timing.

Official UTCI source and three UMEP components have now been independently
retrieved and hashed under `reports/characterization/p8_utci_official_source`
and `reports/characterization/p8_umep_source`. The UTCI tests compare official
coefficient/exponent data, independently evaluated pressure-polynomial values,
and RH conversion; static review corrected encoding and invalid-domain cases
before execution. The UMEP delay test advances independent reference and
candidate state histories across four timestep lengths and two dtypes.
Designs and deferred commands are in `reports/p8_utci_independent_design.md`
and `reports/p8_umep_delay_design.md`. Both test files remain **unexecuted**;
syntax inspection is not numerical validation. These checks do not establish
whole-model scientific correctness, and acquired UMEP radiation/ground-view
files still require matching and executable comparisons.

The v4 repeated-block 256 scene completed all 20 pairs with numerical/cache
checks passed, frozen inputs unchanged and final global evidence eligibility
true. Median paired ratios against repair-only P6 are 3.547/3.497 for cold
1/4-thread groups and 4.399/4.337 for geometry-warm groups. Raw ranges and
qualifications are recorded in `reports/p7_pair_measurements.md`. The same
orchestrator continued to dense_urban_256; five scenes remain pending.

### P8 queued unmocked UMEP ground-view comparison

Pinned the unchanged UMEP `sunonsurface_2018a.py` at the existing reference commit; hash and size are recorded in `reports/characterization/p8_umep_source/manifest.json`. Static matching and scope limits are in `reports/p8_umep_component_matching.md`. Authored eight unmocked direct/full-ground-view cases in `tests/scientific/test_umep_ground_view_independent.py`, comparing independent reference/serial/parallel copies and mutations under existing frozen tolerances. Standard-library AST syntax inspection passed; numerical execution remains deferred during the P7 quiet timing window. These cases are not included in any passing-test count. Next: execute after the complete paired matrix, investigate any disagreement without relaxing gates.

### P7 paired matrix: dense-urban scene complete

The continuing `run_p7_pairs.py --protocols benchmarks/protocols/p7_pairs_v4 --output reports/characterization/p7_pairs_v4` run completed dense_urban_256 with exit 0. All 20 pairs pass numerical/artifact/cache checks, all four groups have five valid pairs and zero failures, and final frozen-input/global-evidence guards pass. Raw evidence: `reports/characterization/p7_pairs_v4/dense_urban_256/summary.json`, `pairs.json`, and `trials.json`. Qualified ratios are recorded in `reports/p7_pair_measurements.md`; comparison remains against repaired P6, not upstream. Source remains frozen during the running vegetation-rich scene. Four scenes and P7 promotion review remain; no milestone status changed.

### P7 full-worker profiler prepared, execution deferred

Authored `tools/profile_p7_pipeline.py` and `reports/p7_full_pipeline_profile_design.md` to profile the genuine 24-hour chronological runtime worker, with all outputs and isolated copied inputs/JIT/geometry caches. Review added source/fixture immutability checks, rejection of destinations inside input/source trees, and preservation of the virtualenv interpreter symlink path. AST validation passed; no profile or numerical test was executed during paired timing. The deferred command and acceptance checks are in the design report. P7 still requires actual profile execution and inspection after the matrix finishes.

### P7 paired matrix: vegetation-rich scene complete

The continuing `run_p7_pairs.py --protocols benchmarks/protocols/p7_pairs_v4 --output reports/characterization/p7_pairs_v4` run completed vegetation_rich_256 with exit 0. All 20 pairs pass numerical/artifact/cache checks, all four groups have five valid pairs and zero failures, and final frozen-input/global-evidence guards pass. Raw evidence: `reports/characterization/p7_pairs_v4/vegetation_rich_256/summary.json`, `pairs.json`, and `trials.json`. Qualified ratios are recorded in `reports/p7_pair_measurements.md`; comparison remains against repaired P6, not upstream. The orchestrator started real_dense_urban. Three scenes and P7 promotion review remain; source remains frozen and no milestone status changed.

### P8 Cura scientific execution and unresolved findings

User-authorized isolated execution on Cura is recorded in `reports/cura_p8_20260919T1215Z_a7c3/assessment.md`, with commands, environment lock, JUnit and hashes in its `results/` directory. Initial six-file run: 35 tests, 28 passed and 7 failed. The unchanged directional-wind test passes separately after setting PROJ_DATA/GDAL_DATA in the nonactivated Conda environment. The radiation fixture-precondition repair is authored; its rerun is pending. Two directional SVF unity failures and two UMEP ground-view comparisons remain unresolved under their original gates. No tolerance was relaxed and no production source changed. Cura exposes four RTX A6000 GPUs, but no CUDA reference or performance run has been executed; historical M1 CUDA-unavailable records remain historical. A separate Cura protocol draft is being prepared. The owned remote workspace is retained for follow-up and tracked in `remote_cleanup_ledger.json`; cleanup is not yet complete. P7 local timings continue, and neither P7 nor P8 is marked complete.

### P7 Cura chronological profile executed

The unchanged `tools/profile_p7_pipeline.py` completed the genuine small 24-hour/all-output/153-patch, one-thread source-snapshot run on Cura. `reports/cura_p8_20260919T1215Z_a7c3/profile_small24_t1/outcome.json` records success and ten timestamped 32×35×24 outputs; raw pstats, full function tables, source/fixture guards and commands are retained alongside it. Interpretation: `profile_small24_t1_report.md`. Compilation dominates this first-use diagnostic; cumulative call-site times overlap and cannot rank warmed native kernels or establish speedup. This is Xeon source-run attribution, not M1 timing or installed-wheel validation. Local P7 pair matrix and larger-fixture preparation remain in progress.

### P8 Cura radiation rerun verified

Root inspected `reports/cura_p8_20260919T1215Z_a7c3/results/p8_radiation_analytic_revision_tests.xml`: two tests pass, zero failures/errors/skips. This supersedes the pending radiation rerun above. The test-only fixture precondition revision is recorded in `radiation_test_snapshot_revision.json`; no production behavior or numerical tolerance changed. The separate directional-wind retry records one pass. These reruns are not additive to the original 35-test scope: the original report retains seven failures, three of which now have passing rerun evidence. Four independent scientific failures remain unresolved (two SVF, two UMEP ground-view). The original upstream CPU diagnostic establishes compatibility for all 17 ground-view fields in both land-cover modes, including exact float32 `gvfSum`, but does not turn the independent float64 UMEP gate into a pass. See `upstream_cpu_ground_view_diagnostic.md` and `docs/model_deviations.md`. P7/P8 status remains unchanged.

### P7 paired matrix: real dense-urban scene complete

The continuing v4 runner completed real_dense_urban with exit 0. Root inspected all 20 pair comparisons and cache checks, four groups of five valid/zero failed pairs, and final frozen-input/global-evidence guards: all passed. Measurements and qualifications are in `reports/p7_pair_measurements.md`; raw evidence is under `reports/characterization/p7_pairs_v4/real_dense_urban/`. The runner started real_vegetation_rich. Two scenes and final P7 promotion review remain; no numerical source or milestone status changed.

### P7 full paired matrix terminal and verified

Luna monitored the existing runner to completion. Root independently inspected `reports/characterization/p7_pairs_v4/execution.json` (seven exit-zero scenes), all seven summaries and 140 pair records: numerical/artifact/cache comparisons pass, all subgroups have five valid pairs and no failures, and final frozen-input/global-evidence guards pass. Last two scene measurements are recorded in `reports/p7_pair_measurements.md`. The local quiet timing window has ended. P7 promotion/regression review, original-upstream performance matrix and P8 gates remain incomplete; the small four-thread warm mixed result is retained.

### P7 warmed profiling: executed, residual compilation retained

The opt-in `tools/profile_p7_pipeline.py --warmup` path runs a complete unprofiled worker, preserves JIT/geometry caches, and moves completed-output transaction metadata aside before a fresh non-resuming profiled worker. Root inspected `reports/p7_profile_{small,repeated256}_warmed_v2/outcome.json`: both complete with ten 24-band outputs and unchanged source/fixture guards. These are attribution diagnostics, not timing claims. The first v1 warm attempts produced no new TIFFs and were rejected; their apparent import-only times are not simulation results. The harness now rejects an explicit worker failure artifact as well as nonzero exit. Residual compilation (~6.2 seconds) remains under investigation, and durable warmup/output comparisons are being finalized before accepting the detailed interpretation.


### Persistent-JIT paired experiment completed; integration review pending

The independently reviewed harness completed 40/40 valid pairs, eight groups of five, with no failed pairs and all source/fixture/harness guards passing. Command: `.venv-light/bin/python tools/run_p7_serial_variant.py --protocol benchmarks/protocols/p7_persistent_jit_serial_variant/protocol.json --output reports/characterization/p7_persistent_jit_serial_variant/pairs_v1 --execute`. Raw pairs, trials, frozen inputs and summary are retained in that output directory. This compares immutable current optimized source against the isolated distinct-function cache variant, not original upstream. Warm geometry/JIT median baseline-to-variant ratios are 1.5180/1.7055 (small, one/four threads) and 1.1156/1.1601 (256-square); first-use medians are 1.0085/0.9884 and 1.0004/0.9987 respectively, retaining mixed cold results. Main source remains unchanged pending integration review; P7/P8 are not complete.

Cura additional CPU admissions are terminal: vegetation1024 passes all 13 artifact records and 18 geometry fields; dense1024 passes geometry but fails TMRT in 12 bands, worst absolute error 0.04949951171875 versus the unchanged 0.01 gate. Both executions remained within the 12 GiB sampled process-tree RSS cap. Evidence: `reports/cura_p8_20260919T1215Z_a7c3/admission/p8_1024/batch_manifest.json`. The agent verified 52 raw files locally before removing eight remote duplicate output/SVF trees. Remote environments and fixtures remain owned and tracked for ongoing diagnosis; whole-workspace cleanup is outstanding. Dense-scene diagnosis is active; neither successful execution nor passing vegetation admission establishes full correctness.


### Persistent-JIT integration verified

Independent review inspected all 80 trial records and 40 paired comparisons (maximum recorded output difference zero), and approved only the two-file cache change. Main now matches the frozen candidate for `geometry/visibility_compiled.py` and `radiation/patch_radiation.py`; distinct serial definitions avoid dispatcher cache collisions. Permanent AST and fresh-process cache-order regressions were added. The requested source suite passed 831 tests; the final cache-only rerun passed four. An isolated installed wheel (SHA-256 `099352d77d8e09a4560f8aaab83c6d97eb7980ff86334bba090e1d9f8abd8f40`) contains the exact approved files and imports without Torch or optional acquisition packages. Fresh installed workers with empty then populated private JIT caches produce exactly equal ten-field/24-band TIFFs, also equal to the frozen candidate. Root verified current source and wheel hashes against the report. Exact commands, initial incorrect invocation failure, warnings, dependency freeze and raw comparisons: `reports/characterization/p7_persistent_jit_promotion/report.json`.

Benefit is limited to measured compatible-cache warm execution; first-use results remain mixed. Static serial body duplication is guarded by AST equivalence and costs maintenance; the small full-run private cache occupies approximately 2.1 MiB. Frozen benchmark evidence remains unchanged and refers to its snapshots. Dense1024 TMRT diagnosis, four independent scientific failures, full original-upstream timing matrix and final release/cleanup gates remain outstanding. No milestone marked complete.


### Dense1024 failure localized to ASVF preparation

The original/candidate instrumented replay and minimal fixture localize the failure to platform-dependent acos rounding before strict patch classification, not serial-versus-compiled accumulation. See the new ASVF entry in `docs/model_deviations.md` and its source-provenanced evidence. M1 characterization rejects float64 acos as a universal parity repair. Isolated backend characterization/prototyping continues without changing production arithmetic or gates. Earlier diagnostic duplicate-output deletion occurred after its verification script failed; that cleanup failure remains explicitly recorded, while compact probes and original admission outputs were independently hash-verified. Subsequent diagnostic cleanup used successful hash checks before deletion and verified absence.


### Isolated native MKL compatibility repair passes full Cura cases

`reports/cura_p8_20260919T1215Z_a7c3/admission/p8_1024/mkl_pipeline_experiment/result_summary.json` records an isolated snapshot experiment, not a production promotion. Dense and vegetation1024 each pass all ten output fields ×24 bands and 18 geometry fields against original CPU under unchanged gates; peak sampled process-tree RSS is 2,377,129,984 and 2,372,767,744 bytes. Dense TMRT maximum error is now 9.1552734375e-05 °C and vegetation 0.0001220703125 °C. These are correctness runs, not paired performance evidence.

The exact runtime profile uses pinned standalone oneMKL2024.2 VML HA/FTZDAZ-off/ignored-errors, without libtorch fallback, and participates in geometry/simulation identities. Actual per-patch original invocation preflights pass for serial/prepared classification at one and36 threads, with zero ASVF bit mismatches over189,220 inputs. An earlier vectorized classifier oracle had incorrect scalar promotion and is retained as a rejected preflight; the successful preflight follows actual scalar float64 cos/tan plus raster float32 tan/atan. Initial full-run PROJ lookup failure and explicit-environment retry are also retained. Complete run trees were downloaded/hash-verified before deletion; the main owned remote workspace and standalone runtime remain ledgered.

Production arithmetic/dependencies are unchanged. User input is pending on the material design choice of a pinned Linux x86-64 MKL dependency versus continued portable replacement research. Its download footprint and unresolved redistribution compatibility remain explicit, not grounds to weaken numerical gates. Independent M1 actual-invocation characterization continues. P7/P8 remain incomplete.


### P7 single-translation horizon experiment rejected

`tools/experiments/p7_exact_horizon.py` constructs a frozen 7×7 counterexample and deterministic 32×35 border/tall-obstruction family. At30°, the actual sample offsets begin (-1,+1), (-2,+1), (-3,+2); translating the first offset repeatedly visits the wrong support. The experiment produces six changed shadow pixels in the minimal case, nine in the larger30° case, and24 at73°. Sampled cardinal/diagonal masks agree, but reassociated float32 horizon values differ, so reusable horizon equality is not established there either. No production kernel changed.

Authoritative evidence is `reports/characterization/p7_horizon_experiment/result_v2_reference_bound.json`: explicit sampled-horizon masks are asserted equal to the actual `shadow_numpy` sunlight kernel for every fixture. That existing kernel has frozen original-CPU reference coverage; no upstream golden was regenerated from the experiment. The first unbound hand-replica result is retained and labeled exploratory. Root reran `.venv-light/bin/python tools/experiments/p7_exact_horizon.py --output /tmp/solweig-root-horizon-verification.json`, exit0. Construction/storage/query numbers are Python mechanism diagnostics, not a full-workload speedup. This rejects only the tested single-translation recurrence; phase-indexed schemes and conservative hierarchical skipping remain separate untested strategies.


### P7 conservative hierarchy experiment: correct bound, rejected implementation

`tools/experiments/p7_hierarchical_skip.py` retains exact ray samples and their order, and skips a four-sample segment only when a rectangle-maximum height minus its first valid float32 decrement cannot exceed the receiver horizon. All15 sparse/dense/border+tall cases produce bit-identical building horizons to exhaustive enumeration and masks equal to the characterized `shadow_numpy` kernel. Evidence: `reports/characterization/p7_hierarchical_skip_experiment/report.md`, manifest and raw results. Root reran the tool with `--output /tmp/solweig-root-hierarchy-verification.json`, exit0.

Despite skipping41.6–99.4% of raw samples, Python quadtree traversal is about7.2–7.4× slower in these mechanism diagnostics and has a measured219,508-byte lower bound of Python storage per32×35 hierarchy. Construction/storage and node-visit counts are retained. This implementation is rejected for promotion; a packed compiled structure remains untested. The proof covers only opaque building maxima, not vegetation ordering or any wall/vegetation outputs. No production change or full-workload speed claim follows.


### M1 isolated math-profile pipeline verified

`reports/characterization/p8_asvf_parity_review/m1_pipeline_experiment/final_review.json` records an unpromoted SLEEF-derived profile:750 source tests and nine installed-wheel chronological tests pass with no skips;18 complete ten-field/24-band comparisons pass unchanged gates. Actual upstream per-patch masks match for all24 frozen solar timesteps,774 boundary plus249 local-frozen SVFs,153 patches, blocks17/128/1023 and contiguous/strided layouts. Root verified every indexed evidence hash and read both JUnit counts. Installed snapshot hashes match, Torch/acquisition packages are absent, and the SLEEF license is included.

The reviewable patch remains isolated. Float64 NumPy scalar coefficients retain bit-level differences despite passing tested masks/outputs; no universal intermediate parity claim is made. The bounded float32 tan helper covers physicalASVF inputs, not arbitrary trigonometry. New kernels' startup/cache behavior is not a measured optimization. A cross-platform production profile design and the pending LinuxMKLdependency choice remain separate from these correctness results; P7/P8 are not complete.


### P7 full-coefficient UTCI Horner experiment rejected

`tools/experiments/p7_utci_horner.py` binds all211 inventory terms, the actual checked-in explicit evaluator, and hash-verified originalCPU fixtures. On10,762 upstream-accepted frozen inputs, Horner maximum error is0.031005859375°C against original, exceeding unchanged0.02°C; the existing explicit evaluator remains within gate at0.0078125°C. On8,015 cancellation-neighbour inputs, Horner differs from the actual explicit evaluator by up to0.02960205078125°C. These are executable acceptance/stress cases, not a claim about the published UTCI applicability domain. UniformTa/Pa specialization is bitwise equal to genericHorner but inherits its failed behavior.

Root reran `.venv-light/bin/python tools/experiments/p7_utci_horner.py --output /tmp/solweig-root-horner-verification`; expected exit2 records evidence-backed rejection. Raw trials, construction/storage, exact coefficients and fixture provenance remain in `reports/characterization/p7_utci_horner_experiment/`. No production change or full-workload speedup is claimed.

Independent review approved the bounded six-moment reflected-longwave experiment in `reports/characterization/p7_angular_moment_design.json`, retaining dynamic first-sweep radiation and strict classifications. Execution is delegated. The same review defers a Cython/C++ backend: existing retained profiles do not establish a Numba limitation justifying another maintained implementation. This is a conditional design decision, not a measured rejection of an unimplemented extension.


### P7 angular-moment full-workload experiment rejected

The isolated immutable six-moment bundle now has explicit tile-lifetime ownership, complete static identity, bounded packed decoding and exact fallback for unsupported/nonfinite/overflow-risk inputs. Independent review required and verified wrong-tile, reversible-readonly, ratio-direction and failed-subgroup regressions;12 numerical/lifecycle tests,18 original component cases, nine source-pipeline and nine installed-wheel cases passed with durable commands/logs/JUnit/wheel provenance. None of these changes entered production.

The first frozen40-pair attempt in `reports/characterization/p7_angular_moment_experiment/pairs_v1` failed before simulation because child import roots were incorrect; it contains no timing evidence. The revised harness was checked with real subprocess smoke/full-pair rehearsals, preserving failed rehearsals and v1. Reviewed protocolv3 keeps workload, ordering and quantitative gates unchanged. Command: `.venv-light/bin/python tools/run_p7_angular_moment.py --protocol reports/characterization/p7_angular_moment_experiment/paired_protocol_v3.json --output reports/characterization/p7_angular_moment_experiment/pairs_v2 --execute`.

Luna monitored the terminal run; root inspected the summary. All40pairs pass numerical/artifact/source/fixture/harness gates, five valid in each of eight groups. The frozen performance acceptance fails: candidate/baseline median ratios (cold1/cold4/warm1/warm4) are1.0444/1.1920/1.0408/1.0281 for small and1.2735/1.3121/1.4467/1.4494 for256-square. Every median is slower; the required3% warm256 benefit is absent. Exit1 denotes failed promotion criteria, not failed simulation. Reject this implementation for promotion, retaining current production radiation. Raw pairs/trials/RSS and limitations remain under `pairs_v2`. This is candidate-to-candidate evidence, not an original-upstream speedup.


### Release CI wheel-isolation audit

The independent P7 checkpoint audit in `reports/p7_checkpoint_review.json` retains P7 in progress: completed optimization evidence does not cure the current-main Cura numerical failure. Independent CI work corrected a real checkout-source leak in persistent-JIT tests and added per-pytest installed-wheel origin checks. The JIT tests now read/probe the imported distribution and run in wheel CI; forcing also checks installed origin. Native GDAL formula content is pinned by commit/SHA while retaining3.13.3 runtime enforcement; transitive Homebrew dependencies are still resolved externally. The runner label now targets documented arm64 macos15 with an explicit architecture guard.

`reports/p8_ci_audit.json` and `reports/characterization/p8_ci_audit/manifest.json` retain exact commands, wheel, stdout/stderr, JUnit, deliberate source-injection failure, and the pinned-formula validation/cleanup script. Four installed-wheel JIT tests and39CLI/publicAPI tests pass; the injected checkout source fails before collection as intended. Root inspected the workflow/guard/test changes and durable JUnit counts. Actionlint/py_compile and local formula parsing passed. Hosted GitHub jobs, hosted native GDAL installation, and full-platform matrix were not executed and are not claimed passed. No numerical source, dependency lock or milestone status changed.


### User decisions: patched CUDA oracle authorized; portable runtime retained

The user explicitly authorized the prepared worker-limit patch only in a separate, labeled upstream reference on Cura. Its scope is worker count bounded by tile count; original references/failures and the12GiB gate remain unchanged. Execution is delegated to the remote lifecycle owner. This is not approval to change upstream physics or overwrite original evidence.

The user chose continued portable replacement research instead of integrating IntelMKL as a Linux runtime dependency. Production dependencies remain unchanged. The exact standaloneMKL results remain diagnostic/reference evidence, not a selected production architecture. Source-provenanced portable math research continues with originalCPU classification and full-field gates intact.


### Approved patched-upstream CUDA cold admission passed

The user-approved worker-limit patch was applied only in an isolated upstreamcheckout/environment; SHA-256 `596ad33c05f38f860964374a515e8f0b06d970be00bd7307de16abf1fa6b2940`. Genuine publicthermal_comfort small24-hour/153-patch/all-ten-output execution passed the unchanged12GiB summedprocess-treeRSS cap at1,586,630,656bytes. All ten outputs and18geometry fields pass against both originalCPU and candidate under unchanged gates. This is explicitly patched-reference correctness, not an original-upstream result or a speed measurement.

`reports/cura_p8_20260919T1215Z_a7c3/cuda_admission/patched_workerlimit/manifest.json` retains source/env/patchprovenance and fullcomparison evidence. The original unpatchedCUDA memoryfailure remains distinct. Setup attempts rejected for sourceprovenance/build effects and stalePROJenvironment remain recorded; finalretry only changedexplicitPROJ_DATA. Fullrun trees were downloaded/hashverified before remote deletion; isolatedpatchedcheckout/env remain ledgered for followup. Release performance/memory indices now link patchedcoldpass separately from originalcoldfailure and originalwarmpass, and index the completedJIT/angular experiments.


### Portable math research: tested alternatives do not pass boundary gate

Following the user's decision to avoid a productionMKLdependency, source-provenanced CORE-MATH(MIT) and IntelIGC(MIT) profiles were tested in isolation. Their actualperpatch classification mismatches are8sun/14shade and11sun/27shade respectively. A retained exactoriginalCPU intermediate trace localizes four CORE-MATH residual failures with oracleASVF to one-ULP tangent differences at a strict66° equality. OracleASVF is diagnostic input only, not a production remedy. Evidence: `reports/characterization/p8_portable_math_review/review.json` and the linked originaltrace.

Native licensed SLEEFu35 scalar/AVX512 and BSD SVMLLA tangent trials onCura also fail:19sun/9shade and9sun/20shade respectively using originalASVF; generatedASVF combinations fail as well. Fixing selected equality cells while introducing failures elsewhere is not acceptance. `reports/characterization/p8_portable_math_review/cura_native_tangent/manifest.json` retains48 verified artifacts; remote diagnostic cleanup followed successful checksums and absence verification. Restrictively licensed HA sources were excluded. No per-input switching, lookup, threshold adjustment, MKLproductiondependency, or mainnumericalchange was introduced. These failed primitive gates prevent fullpipeline/performance claims for those variants. Further bounded source-available library research is underway.


### Portable profile final bounded-library trials

The additional pinned source-native AOCL and OpenLibm profiles both fail the unchanged actual774×153 classification gate. AOCL has24,670 finiteASVF bit mismatches over189,220 inputs and60sun/40shade complete-profile mask mismatches; OpenLibm has9,693 finiteASVF and9sun/20shade mask mismatches. OriginalASVF-assisted results are diagnostic only and do not qualify either completeprofile. Symbol origins were checked; all30 evidence entries and source/licenses/build/raw artifacts were downloaded/hashverified before the ownedremote diagnostic was removed and absence verified. Evidence: `reports/characterization/p8_portable_math_review/cura_remaining_native_profiles/manifest.json`. No productiondependency/source changes or fullpipeline runs followed failingprimitive gates.

The evaluated portable alternatives have not reproduced the frozen Cura/MKL boundary behavior. This finite search does not prove that a portable implementation is impossible. MKL production integration remains declined; frozen numerical gates remain unchanged. Further progress on this compatibility branch requires a justified new algorithm or an explicitly approved reference-policy decision, not more unstructured coefficient/rounding tuning.


### Current-source installed artifact/contract checkpoint

`reports/characterization/p8_release_contract_checkpoint/release_artifact_manifest.json` links and hashes all ten Section13 required reports without asserting their completion. Retained base wheel matches all44currentpackage files, and newlybuilt opt-in compatibility wheel matches all14compatibilityfiles. New installed49cache/restart/runtime and eightcompatibility/collision checks pass; root read their durableJUnit counts. Exact-source831test/JITpromotion and4+39CIaudit/genuine24-bandTIFF evidence is reused explicitly rather than rerun. Allsevenworkflow imports resolve installed and noTorch/CUDA/acquisition modules load. Commands, wheels, environment and earlier installed-guard staging failure are retained. Task-createdtemp staging was removed and absence checked.

No productionnumericalchange or policyapproval is inferred. Current-main dense1024 failure, inherited scientific-check disposition, declinedMKLproductiondependency, originalCUDAfailure versus approvedpatchedpass, missingfullreleasebenchmarks and unrunhostedCI remain visible. The pending scientific/reference-policy questions now block the numerical release path; artifact existence and this localcontractcheckpoint do not completeP7/P8.


### Remote workspace removed; policy decisions pending

Final Cura cleanup is complete: the exact task-owned14GBroot was removed only after process/ownership checks and localverification of1,060 preserved source/evidence/patch/manifest/lock/generator files. Existing8.5GB localadmission evidence is retained; reproducibleenvs/caches were not duplicated. Same-process and independentSSHchecks confirm rootabsence and zero taskprocesses. Root verified archive, selected-hash-manifest and whole-tree-inventory hashes against `reports/cura_p8_20260919T1215Z_a7c3/final_remote_cleanup/final_cleanup_report.json`. Earlier cleanupverification failure remains explicitlyrecorded. No outsidepath was touched.

The implementation remains incomplete. Further numericalintegration depends on the pending user decisions about inheritedscientificfailures and a separatelyversionedportableprofile versus unresolvedexactCura compatibility. Testedportableprofiles fail; MKLproductionintegration was declined. Independent installedcontract/CIaudit and boundedP7experiments are complete within their recordedscopes. No livebenchmark or remotejob remains towaiton. Resume from the retained decisionmemos and source-bound reports; do not inferapproval from automaticgoalcontinuation, loosenfrozengates, or rerun speculativebenchmark work before resolvingpolicy.


### User-approved inherited-science release exception recorded

The user selected retention of upstream behavior with a documented exception. `reports/scientific_release_exceptions.json` binds exactly four named SVF/UMEP failed checks to retained evidence hashes and records their approved known-limitation release disposition. `docs/model_deviations.md`, `docs/compatibility.md` and the decision memo now reflect that approval. No numerical source, tests, fixtures, thresholds or milestone status changed; no scientific assertion was relabeled passed. Other scientific deviations and dense1024 incompatibility are excluded. The portable-reference policy question remains pending, and release remains incomplete.


### User authorized separately versioned portable math profile

The user explicitly selected “Develop versioned portable math profile” against the originalSLEEFreference, retaining measured Cura/MKL deviations. This authorizes a separatelyversioned reference-policy design and implementation; it does not change historical failures to passes. Scope and predeclared gates are being authored before production integration. The previously removedCura workspace remains absent; any new remote validation needs a newlyowned, ledgered workspace and finalverifiedcleanup.


### Portable profile integrated after both-platform admission

The authorized `solweig-portable-sleef-5a1d179d-v1` profile is integrated from the exact reviewed snapshot. OriginalSLEEF captures cover45,172 classifierinputs×24timesteps×153patches and both1024scenes; M1 andLinux pass exactcategories,14,400 layout/threadcases each, installedcomponent/cache/API suites, all10×24largeoutputs and18geometry fields. DirectLinux/M1 comparisons and newprofile-versus-originalCura final-field comparisons pass; priorfailures remain recorded.

Rootreview discovered wide-domain NumPy fallback divergence in Linux's compiledscalar path. The isolated repair moved fallback to the actual PythonNumPy ufunc, retaining fullshape/layout/tail behavior and unchangedphysical SLEEFbranch; bothplatform source/wheel wide-domain regression and smallchronology checks pass. Priorlarge runs explicitly remain pre-fallback-repair evidence. Root checked exactoldmain hashes, copied onlynine reviewedsource/package-data files, added permanentprofiletests and CIcoverage, and launched post-copy tests. Detailed hashes and verification: `reports/characterization/p8_portable_profile_v1/integration/`. Newperformance claims and finalrelease gates still require the finalsource; noP7/P8completion is asserted. All newCuraownedroots were checksum-preserved and removed.

Post-copy verification completed: 27 tests passed, zero failures/errors/skips, covering profile/domain/cache/checkpoint, persistent dispatcher caching and genuine chronological TIFF comparisons. JUnit/log hashes are bound in the promotion record.

### Checkpoint committed; local CPU optimization resumed

The user authorized committing the current work and maximizing local CPU
performance while preserving identical mathematical results. Initial commit
`8ca23d4` preserves the implementation, reference policy, scientific exceptions,
verification records and reviewed optimization strategy. Large generated
artifacts remain local under the documented evidence-storage policy; no
historical evidence was deleted. P7 remains `in_progress` and P8 remains
`pending`.

The new experiment is recorded in
[`local_cpu_optimization_v1/experiment_scope.md`](../reports/characterization/local_cpu_optimization_v1/experiment_scope.md).
E0 requalified the final installed baseline on the local M1 Pro: both
1024-square scenes passed 240 bands and 18 geometry fields against the original
SLEEF references. Observed summed RSS was 1.410 GiB for dense1024 and 1.469 GiB
for vegetation1024. Independent review verified the 48 source-file bindings,
reference provenance, field-specific gates and output hashes; no speedup is
inferred from these admission runs. Warmed current-source attribution again
identified GVF gathering, longwave reduction, visibility decoding and residual
compilation as useful targets. Independent isolated candidates cover existing GVF
parallel dispatch, exact shared visibility decoding, and internal block size.
Candidate arithmetic must preserve finite bits and signed zero, special masks,
all carried state and artifact contracts. Existing original-reference gates
are unchanged. Initial E1 component admission passed 216 tests; E2 v2 source
review is accepted for executable qualification. Neither candidate is yet
promoted or established as faster. Luna owns long numerical runs; other
agents perform source-only work during the host reservation. The new evidence
harness and correctness-trace unit checks passed 11 tests; candidate and real
workflow qualification continue before performance selection.

Executable candidate qualification now passes: E2 v2 has 653 tests with zero
failures/errors/skips and all seven outputs match in each of three shortwave
input modes. Five genuine small-scene runs cover the baseline at one and ten
threads, E1, E2, and 4096-pixel blocks; their 72 chronological trace events
match exactly, as do geometry, final state and published artifacts. The smoke
matrix actually used the 256-square fixture and passed first-use, compiled
geometry-cold and geometry-warm modes. Earlier relative-interpreter launch
and comparison-caller shape errors remain retained as harness failures.

Root verified admission/evidence hashes, raw trace equality and the 653-test
JUnit, then froze the eight-cell, 24-pair diagnostic development protocol at
SHA-256 `d6e4ba1bdc5da6612d8a90fb13d86bfa96f4ae56551f08b4a901f521d0308587`.
Luna owns its execution and reports only completion or a blocker. No
production optimization is promoted from this admission alone.

An independent source review accepted the persistent-cache candidate's
arithmetic and content-bound namespace design, but found defects in its
unexecuted tests and example invocation paths. Those are being repaired in
a separate packet before execution. The paired-statistics method correctly
enumerates all 3,125 five-pair bootstrap resamples; review also requested
stronger frozen-schedule/protocol-byte verification and deterministic joint
budget/block selection. These analysis-tool repairs do not change the running
measurement harness or protocol. Detailed local promotion requirements are
recorded in `local_cpu_optimization_v1/promotion_method.md`.

The first development matrix completed all 24 pairs with exact artifacts and
state. Independent review recomputed the paired ratios from raw measurements:
E1 at four threads has median baseline/candidate 1.235264 (absolute medians
20.994899 s and 17.017141 s); block1024 at ten threads has paired median
1.284231 (23.609702 s and 18.360547 s). These three-pair diagnostic results
select experiments, not release speedups. E2 qualifies narrowly at ten threads
(1.033104); its one-thread result (1.018263) is below the diagnostic 3% target.
The 4096 setting is excluded from the bounded combined search because 1024 is
within 3% in both tested budgets with a quarter of decode workspace. The
analysis also records a failed relative-RSS guard for one-thread block4096;
all trials remained within the absolute 12 GiB cap.

E4v3 passed six source proofs, nine fresh-process cache/order/invalidation
checks and 300 origin-bound subsystem tests for each isolated source. Both
real chronological traces match all 72 baseline events and exact artifacts.
Root read the retained stdout counts and verified the 74 development evidence
hashes. The expanded statistics tests independently passed 21 cases with
durable JUnit `statistics_qualification_v2.junit.xml`.

The mechanically combined source has 49 files, eight changed/new paths, and
inventory `dc6e151cfcd6c8bd476921e850666d22f6dfcd2237eb8b0afa649736002af0c6`.
Independent review verified every file against its declared origin. Combined
chronological/subsystem qualification precedes its six-setting development
comparison (1/4/10 threads and 128/1024-pixel blocks). No production source or
public runtime default has changed yet. Final promotion also requires actual
unchanged-default guards, not only tests of the selected block setting.

Combined admission passed 1,131 origin-bound tests with zero failures and
three exact 72-event chronological traces at 1/128, 4/1024 and 10/1024
thread/block settings. All 18 development pairs passed exact output,
artifact and final-state checks. The predeclared minimum candidate-median
rule selected four threads and a 1024-pixel block: separate baseline and
candidate medians were 21.517622 s and 12.624037 s, with a median paired
baseline/candidate ratio of 1.646599. These remain development results.

Independent review accepted the ten-cell, 50-pair promotion protocol for
execution only after permanent regressions, installed-wheel qualification
and both final-source 1024-square admissions pass. The frozen protocol hash
is `67725d57eb18c49535f7c32e11cc47b04a846e14e3ffbd5406f3cf9c6f8602af`.
It retains the primary benefit and bootstrap gates, scene and startup guards,
unchanged-default guards, exact numerical checks and absolute/relative RSS
limits. Luna exclusively owns numerical execution; no optimized source has
been integrated and neither P7 nor P8 is complete.

Final installed admission now passes the reviewed prerequisites. All 49
package files match the combined source and wheel. Both 1024-square scenes
pass all 240 original-reference bands, exact baseline outputs/artifacts and
final state, and the 18-field geometry gate through exact baseline geometry.
Diagnostic candidate durations were 400.827 s (dense) and 341.365 s
(vegetation), with sampled process-tree peaks of 1.518 and 1.485 GiB. These
single admission runs do not establish a speedup; no large per-timestep trace
was captured.

The installed core attempt passed 3,320 cases and failed one harness test
because it named checkout source after an installed-wheel guard had already
imported the wheel. A test-only repair derives the active package path and
preserves the required `AttributeError` assertion. All 18 tests in that file
then passed with both installed guards, and the repaired case also passed
against baseline source. Their union covers all original 3,321 cases without
skips. Prior 3,303-case reruns excluded the entire file and are not presented
as complete coverage. Original failures and the supplemental repair record
are retained. Independent review accepted these prerequisites in
`local_cpu_optimization_v1/reviews/final_qualification_review.json`.
The frozen 50-pair promotion matrix is now delegated to Luna; production
integration still awaits its complete results and independent review.

### Local CPU optimization integrated with exact results

The frozen promotion matrix completed all 50 pairs across ten configurations.
Root and independent review recomputed the raw ratios, enumerated all 3,125
bootstrap resamples per cell, verified 152 evidence hashes and checked all
cache/source/cleanup, exactness and RSS gates. The primary geometry-warm
256-square, 24-step case at four threads has separate medians 20.963946 s
and 12.886704 s, with paired median baseline/candidate 1.626789 and geometric
mean bootstrap 95% interval [1.593443, 1.648973]. All scene, one-thread,
unchanged-default, geometry-cold and first-use guards passed. No startup
tradeoff rule was triggered. Maximum observed summed RSS across the 100
timed trials was 604,553,216 bytes; sampling and shared-page limitations
remain explicit. These are local candidate-to-candidate measurements.

After approval, root copied exactly eight source files and four permanent
test/probe files from the admitted snapshot. They implement existing GVF
pixel parallelism, exact duplicate visibility decode removal and content-bound
persistent JIT caches. Explicit four-thread/1024-pixel execution is documented;
public defaults and numerical policy remain unchanged. The installed-wheel CI
configuration now includes the permanent regressions; hosted CI has not run.

Final post-copy verification passed 97 tests with zero failures or skips,
including genuine TIFF execution, the moved permanent tests and source-guard
repair. All 49 repository, staged, wheel and installed files match the
qualified source before and after tests. Requested and effective thread
counts are 1, 4 and 10; Torch remained unloaded. Root verified the JUnit and
source/wheel hashes. Commands and raw evidence are retained in
`local_cpu_optimization_v1/integration/postcopy_v1/postcopy_report.json`.
The measured scope and remaining gates are summarized in
[`local_cpu_optimization.md`](../reports/local_cpu_optimization.md).
P7 remains in progress and P8 remains pending; this local optimization does
not establish the full upstream matrix, larger/default-3600 feasibility,
multi-day coverage, final-source Linux qualification or release completion.
