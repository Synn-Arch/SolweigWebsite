# P7 experiment register

P6 is accepted; P7 is in progress. No P7 optimization has been promoted.
The accepted source and wheel are preserved under
`reports/characterization/p7_p6_baseline`, with a SHA256 manifest and installed
verification reports. Retain this baseline for equal-work chronological tests.

## Baseline profile: bounded patch preparation

Profiled the frozen P4 component inputs with compact visibility, serial
reductions and the unchanged 128-pixel block. Each function was warmed once
before one cProfile call. Raw `.prof` files, sorted text and source hashes are
in `reports/characterization/p7_baseline_profile`.

The shortwave profiled call took 1.200 s: `_block` accumulated 0.783 s,
`_classes` 0.402 s and the compiled reduction 0.008 s. The longwave call took
0.686 s: `_block` accumulated 0.455 s, `_classes` 0.198 s and the reduction
0.030 s. Profiling adds overhead; these are attribution observations, not
benchmark speed or uncertainty estimates. The P4 measurements remain intact.

Reproduction is available as
`PYTHONPATH=src .venv-light/bin/python tools/profile_p7_patch.py --output reports/characterization/p7_baseline_profile_reproducible`.
That separate run records the exact invocation and harness/source hashes and
does not overwrite the initial observations.

The first hypothesis is that native bounded decoding can remove repeated
Python dispatch without changing visibility values. Classification is a
separate second experiment. Lead review requires unchanged patch order,
float32 rounding and grouping, both longwave sweeps, scalar exception ordering,
inactive-patch behavior, signed-zero/raw bit patterns and categorical value 2.
Mapped storage must retain its lifetime/close protection. No full dense
visibility cube or forcing-dependent cache is permitted.

Promotion requires the unchanged reference and counterexample checks, mapped
and raw decoding, block/thread invariance, original chronological TIFF gates,
the original P4 matrix at block 128, and a frozen paired chronological benchmark
including I/O, startup and process-tree memory. Block sweeps are separate
experiments. Expanded fixture provenance required by `benchmark_v1.json` is
established before optimization; synthetic scenes do not stand in for
licensed real scenes. `benchmarks/protocols/p7_pairs` freezes seven fixture
inventories, five pairs per regime/budget, native budgets 1/4, all requested
outputs, and cold/geometry-warm entrypoints. The paired harness validated the
small protocol without executing performance trials.

## Input admission and decoding experiment

The additional synthetic families and three publisher-sample windows are
documented in `p7_fixture_audit.md`. Publisher documentation identifies intended
terrain/building/vegetation input semantics. Original acquisition lineage and
vertical datum remain unknown; these fixtures supply compatibility/performance
coverage, not independent observational validation. The source windows retain
their 2 m resolution, zero tree NoData metadata and complete declared domains.
The analytic meteorology remains explicitly synthetic.

The first dense-urban original-upstream run completed, and further original
runs and preserved-P6 differential checks are in progress. These single runs
are correctness characterization, with concurrent development activity; their
timings are not paired performance evidence. The bounded native-decoding worker
is implementing the first experiment while retaining classification and
reduction arithmetic. No default optimization is accepted until all promotion
gates pass.

All five expanded original CPU runs completed; provenance and artifact hashes
are in `reports/p7_expanded_original_manifest.json`. The preserved-P6 dense
urban comparison passed unchanged TIFF gates. Its first comparison attempt
exposed a missing `SkyViewFactor` filename mapping in the instrument, after
successful model execution. `p7_pairs_v2` retains every workload and tolerance
from the first protocol set and maps that legacy name to the existing
`SVF_all_15_fields` rule. The initial protocols remain preserved. Recomparison
reused the successful P6 output only after validating the preserved baseline
manifest and exact recorded invocation; no numerical run or failing field was
discarded and no tolerance was relaxed.

Native decoding passed 699 combined codec/radiation/block checks, including
the unchanged 635 radiation cases, and all nine original chronological TIFF
cases. Raw results are under
`reports/characterization/p7_native_decode_correctness`. The implementation
uses owner-local O(patches) borrowed descriptors, no encoded payload copy, and
bounded float32 blocks. Mapped reads retain the owner lock through native
decoding. Component measurements and expanded/full-pipeline gates remain
separate prerequisites to promotion.

The isolated decoder comparison completed with five warm calls per fresh
process, serial reductions and block size 128. Compact shortwave medians were
0.865707 s (P6) and 0.312951 s (decoder); compact longwave 0.493311 s and
0.178941 s. Dense warm medians were essentially unchanged. All six calls per
configuration passed frozen outputs/mutations, and cross-version results were
bitwise equal. Compact longwave's first call increased by 0.088 s in one
observation, and its process-lifetime peak RSS increased from 254.9 to 272.9
MiB. Startup and memory regressions remain visible; no full-workload promotion
is implied. Full raw evidence and guarded commands are under
`reports/characterization/p7_native_decode_component`.

The decoder-only source is preserved in `p7_decode_snapshot` in the same
characterization directory. Experiment 2 now targets classification, retaining
exact scalar promotion/exception ordering and strict boundary masks. A cheap
dense-path guard may avoid importing the native decoder where it cannot apply.
No reduction arithmetic or frozen field gates may change.

## Expanded-fixture failure: performance promotion paused

The published dense-urban window exposes a mismatch in the preserved P6
baseline itself, before either P7 experiment: daytime Kup, Lup, Ldown, TMRT and
UTCI exceed frozen gates. Maximum listed daytime errors include approximately
2.65 W/m² in Lup, 0.25 °C in TMRT and 0.063 °C in UTCI. Kdown, Shadow, Ta,
Wind, WBGT, output metadata and artifact inventories pass. Full details are in
`reports/runs/p7_real_dense_urban_p6_check/verification.json`.

All 18 SVF ZIP and visibility NPZ export fields pass their existing gates;
see `reports/characterization/p7_real_diagnosis/geometry_exports.json`.
An isolated first-divergence diagnosis is in progress. The fixture remains
required and tolerances remain unchanged. No performance promotion is allowed
until this correctness issue is understood and repaired against original
upstream behavior. Existing P6 evidence remains evidence for its tested scope,
not proof of parity on this newly characterized scene.

The first material divergence is in `gvf_2018a`, with matching prepared inputs
and gathered sums. Tensor-origin float32 angular scalars were promoted to
float64 in the candidate; original Torch wraps Python angular constants to the
scalar's dtype. This flips strict wall-facing boundaries at 65°, 145°, 245°
and 325° (65° changes masks without material output change in this case).
`angular_boundary_diagnosis.json` records the reproducer and original replay.
The retained fallback reproduces the same defect and is included in the
narrow repair, without changing global promotion helpers.

Independent lead review permits a separately labeled **P6 plus angular
compatibility repair** baseline only after original-derived boundary and
17-field ground-view checks, all seven full fixture comparisons and the
original nine TIFF regressions pass for both repaired versions. Preserve the
original P6 failure evidence. The repair-only diff must exclude P7 decoding
and classification. A new protocol version must explicitly identify this
correctness-driven baseline amendment; all workloads, tolerances, repetitions,
native budgets, cache rules and ordering remain unchanged. Performance claims
would describe benefit over repaired P6, never unchanged P6 or upstream.

The expanded angular-repaired candidate still fails Kdown at band 15 on the
published vegetation-rich and sparse windows: maximum absolute errors are
0.0597076416015625 and 0.05998992919921875 W/m², respectively. All other saved
TIFF fields pass, and both source guards pass. The reports are
`reports/runs/p7_repaired_optimized_real_vegetation_rich/verification.json` and
`reports/runs/p7_repaired_optimized_real_sparse/verification.json`. These are
remaining correctness failures, not grounds to amend the frozen tolerance.
Performance timing remains deferred while their first divergence is traced.

The remaining failure was subsequently traced to annulus scalar arithmetic,
not either optimization. A third-file compatibility repair in
`geometry/svf.py` restores reciprocal-first division and native float32 sine.
It preserves NumPy zero/nonfinite semantics; six of 180 original weights retain
one-ULP differences, explicitly characterized. Both problematic receiver sums
now match original exactly. The angular-only snapshots and failures remain.

The new repair-only baseline is
`reports/characterization/p7_repaired_p6_baseline_v2`; its exact three-file
patch derives from immutable accepted P6 and excludes both P7 experiments.
Both variants passed all seven full original-reference scenes, all 18 geometry
export fields per scene, nine original TIFF cases, angular boundaries, annulus
receiver regressions and a fresh 17-field GVF replay. The aggregate is
`reports/characterization/p7_repair_v2_correctness/gate.json`. The rebuilt wheel
passed 3,195 core/scientific, 78 optional and 34 forcing-only tests, zero skips.

`benchmarks/protocols/p7_pairs_v3` explicitly amends only baseline/provenance
framing after correctness discovery. All workloads, tolerances, five paired
repetitions, one/four native-thread budgets, cache regimes and ordering remain
unchanged from v2. The full paired matrix is now running under
`reports/characterization/p7_pairs_v3`; no full-workload benefit is yet claimed.
The unchanged 29-configuration component matrix also passed, with raw trials
and numerical comparisons under `p7_repaired_v2_component_matrix`.

## Completed v4 matrix and full 256-square attribution

All seven v4 scenes completed with 140 valid paired comparisons and no failures; root verified cache and final source/evidence guards. `reports/p7_completed_matrix_review.json` and `reports/p7_pair_measurements.md` retain ranges, memory and the small four-thread geometry-warm median ratio below one. This can support an optimization decision against repaired P6, not a release speedup against original upstream. Promotion remains under review.

The full one-thread 256-square/24-hour candidate profile completed under `reports/characterization/p7_full_profile_repeated256_v2`. The first attempt produced outputs but failed the profiler's hardcoded 32-by-35 validation; it is retained at the unsuffixed path. Validation now derives spatial dimensions from the prepared DSM. No model code changed. The successful cProfile trace attributes 16.118 s cumulative to compilation locks, 6.288 s to ground-view, 4.637 s self to native serial gathering, 4.136 s cumulative to bounded decoding, and 3.476 s to SVF. Cumulative times overlap and include compilation; these are diagnostic attribution, not benchmark speedups. Native gathering is a concrete remaining cost; this trace alone does not justify a horizon shortcut or algebraic radiation reassociation.

## Persistent-JIT snapshot experiment (not promoted)

Added `cache=True` only to five decoder/patch-radiation helpers in a separate source snapshot at `reports/characterization/p7_persistent_jit_experiment/src`. Main source remains unchanged. Root inspected the persisted JUnit: 92 targeted tests passed, no failures/errors/skips. `evidence_integrity.json` records the import-isolation assertion, source provenance, 20 per-file matching output hashes across small and 256-square scenes, and each 1,819,136-byte JIT directory. Full 24-step warmed profiles remain diagnostic; roughly five seconds of compilation persists, so no speedup or fully compiled regime is inferred. A separate wall-shadow-cache variant is being tested to attribute that remaining cost. First variant and its evidence remain immutable.

### Persistent-JIT promotion stopped: dispatcher cache collision

The serial-cache prototype is rejected for promotion in its current form. Root identified that serial and parallel radiation dispatchers wrap the same Python function, and `reports/characterization/p7_cache_collision_diagnostic/report.json` records both fresh-process execution orders loading the first dispatcher's cache artifact into the second. Numba's inspected cache key omits the parallel target option. Equal values on the tiny diagnostic do not prove correct scheduling; loading parallel code into the nominal serial path can violate CPU ownership. Main source has not acquired these caching changes. All prototype profiles remain diagnostic and cannot justify promotion. A separate snapshot repair with distinct static function identities is authorized for testing, preserving arithmetic with an AST equivalence check; timing remains deferred.


### Persistent-JIT paired experiment completed; integration review pending

The independently reviewed harness completed 40/40 valid pairs, eight groups of five, with no failed pairs and all source/fixture/harness guards passing. Command: `.venv-light/bin/python tools/run_p7_serial_variant.py --protocol benchmarks/protocols/p7_persistent_jit_serial_variant/protocol.json --output reports/characterization/p7_persistent_jit_serial_variant/pairs_v1 --execute`. Raw pairs, trials, frozen inputs and summary are retained in that output directory. This compares immutable current optimized source against the isolated distinct-function cache variant, not original upstream. Warm geometry/JIT median baseline-to-variant ratios are 1.5180/1.7055 (small, one/four threads) and 1.1156/1.1601 (256-square); first-use medians are 1.0085/0.9884 and 1.0004/0.9987 respectively, retaining mixed cold results. Main source remains unchanged pending integration review; P7/P8 are not complete.

Cura additional CPU admissions are terminal: vegetation1024 passes all 13 artifact records and 18 geometry fields; dense1024 passes geometry but fails TMRT in 12 bands, worst absolute error 0.04949951171875 versus the unchanged 0.01 gate. Both executions remained within the 12 GiB sampled process-tree RSS cap. Evidence: `reports/cura_p8_20260919T1215Z_a7c3/admission/p8_1024/batch_manifest.json`. The agent verified 52 raw files locally before removing eight remote duplicate output/SVF trees. Remote environments and fixtures remain owned and tracked for ongoing diagnosis; whole-workspace cleanup is outstanding. Dense-scene diagnosis is active; neither successful execution nor passing vegetation admission establishes full correctness.


### Persistent-JIT integration verified

Independent review inspected all 80 trial records and 40 paired comparisons (maximum recorded output difference zero), and approved only the two-file cache change. Main now matches the frozen candidate for `geometry/visibility_compiled.py` and `radiation/patch_radiation.py`; distinct serial definitions avoid dispatcher cache collisions. Permanent AST and fresh-process cache-order regressions were added. The requested source suite passed 831 tests; the final cache-only rerun passed four. An isolated installed wheel (SHA-256 `099352d77d8e09a4560f8aaab83c6d97eb7980ff86334bba090e1d9f8abd8f40`) contains the exact approved files and imports without Torch or optional acquisition packages. Fresh installed workers with empty then populated private JIT caches produce exactly equal ten-field/24-band TIFFs, also equal to the frozen candidate. Root verified current source and wheel hashes against the report. Exact commands, initial incorrect invocation failure, warnings, dependency freeze and raw comparisons: `reports/characterization/p7_persistent_jit_promotion/report.json`.

Benefit is limited to measured compatible-cache warm execution; first-use results remain mixed. Static serial body duplication is guarded by AST equivalence and costs maintenance; the small full-run private cache occupies approximately 2.1 MiB. Frozen benchmark evidence remains unchanged and refers to its snapshots. Dense1024 TMRT diagnosis, four independent scientific failures, full original-upstream timing matrix and final release/cleanup gates remain outstanding. No milestone marked complete.


### P7 single-translation horizon experiment rejected

`tools/experiments/p7_exact_horizon.py` constructs a frozen 7×7 counterexample and deterministic 32×35 border/tall-obstruction family. At30°, the actual sample offsets begin (-1,+1), (-2,+1), (-3,+2); translating the first offset repeatedly visits the wrong support. The experiment produces six changed shadow pixels in the minimal case, nine in the larger30° case, and24 at73°. Sampled cardinal/diagonal masks agree, but reassociated float32 horizon values differ, so reusable horizon equality is not established there either. No production kernel changed.

Authoritative evidence is `reports/characterization/p7_horizon_experiment/result_v2_reference_bound.json`: explicit sampled-horizon masks are asserted equal to the actual `shadow_numpy` sunlight kernel for every fixture. That existing kernel has frozen original-CPU reference coverage; no upstream golden was regenerated from the experiment. The first unbound hand-replica result is retained and labeled exploratory. Root reran `.venv-light/bin/python tools/experiments/p7_exact_horizon.py --output /tmp/solweig-root-horizon-verification.json`, exit0. Construction/storage/query numbers are Python mechanism diagnostics, not a full-workload speedup. This rejects only the tested single-translation recurrence; phase-indexed schemes and conservative hierarchical skipping remain separate untested strategies.


### P7 conservative hierarchy experiment: correct bound, rejected implementation

`tools/experiments/p7_hierarchical_skip.py` retains exact ray samples and their order, and skips a four-sample segment only when a rectangle-maximum height minus its first valid float32 decrement cannot exceed the receiver horizon. All15 sparse/dense/border+tall cases produce bit-identical building horizons to exhaustive enumeration and masks equal to the characterized `shadow_numpy` kernel. Evidence: `reports/characterization/p7_hierarchical_skip_experiment/report.md`, manifest and raw results. Root reran the tool with `--output /tmp/solweig-root-hierarchy-verification.json`, exit0.

Despite skipping41.6–99.4% of raw samples, Python quadtree traversal is about7.2–7.4× slower in these mechanism diagnostics and has a measured219,508-byte lower bound of Python storage per32×35 hierarchy. Construction/storage and node-visit counts are retained. This implementation is rejected for promotion; a packed compiled structure remains untested. The proof covers only opaque building maxima, not vegetation ordering or any wall/vegetation outputs. No production change or full-workload speed claim follows.


### P7 full-coefficient UTCI Horner experiment rejected

`tools/experiments/p7_utci_horner.py` binds all211 inventory terms, the actual checked-in explicit evaluator, and hash-verified originalCPU fixtures. On10,762 upstream-accepted frozen inputs, Horner maximum error is0.031005859375°C against original, exceeding unchanged0.02°C; the existing explicit evaluator remains within gate at0.0078125°C. On8,015 cancellation-neighbour inputs, Horner differs from the actual explicit evaluator by up to0.02960205078125°C. These are executable acceptance/stress cases, not a claim about the published UTCI applicability domain. UniformTa/Pa specialization is bitwise equal to genericHorner but inherits its failed behavior.

Root reran `.venv-light/bin/python tools/experiments/p7_utci_horner.py --output /tmp/solweig-root-horner-verification`; expected exit2 records evidence-backed rejection. Raw trials, construction/storage, exact coefficients and fixture provenance remain in `reports/characterization/p7_utci_horner_experiment/`. No production change or full-workload speedup is claimed.

Independent review approved the bounded six-moment reflected-longwave experiment in `reports/characterization/p7_angular_moment_design.json`, retaining dynamic first-sweep radiation and strict classifications. Execution is delegated. The same review defers a Cython/C++ backend: existing retained profiles do not establish a Numba limitation justifying another maintained implementation. This is a conditional design decision, not a measured rejection of an unimplemented extension.


### P7 angular-moment full-workload experiment rejected

The isolated immutable six-moment bundle now has explicit tile-lifetime ownership, complete static identity, bounded packed decoding and exact fallback for unsupported/nonfinite/overflow-risk inputs. Independent review required and verified wrong-tile, reversible-readonly, ratio-direction and failed-subgroup regressions;12 numerical/lifecycle tests,18 original component cases, nine source-pipeline and nine installed-wheel cases passed with durable commands/logs/JUnit/wheel provenance. None of these changes entered production.

The first frozen40-pair attempt in `reports/characterization/p7_angular_moment_experiment/pairs_v1` failed before simulation because child import roots were incorrect; it contains no timing evidence. The revised harness was checked with real subprocess smoke/full-pair rehearsals, preserving failed rehearsals and v1. Reviewed protocolv3 keeps workload, ordering and quantitative gates unchanged. Command: `.venv-light/bin/python tools/run_p7_angular_moment.py --protocol reports/characterization/p7_angular_moment_experiment/paired_protocol_v3.json --output reports/characterization/p7_angular_moment_experiment/pairs_v2 --execute`.

Luna monitored the terminal run; root inspected the summary. All40pairs pass numerical/artifact/source/fixture/harness gates, five valid in each of eight groups. The frozen performance acceptance fails: candidate/baseline median ratios (cold1/cold4/warm1/warm4) are1.0444/1.1920/1.0408/1.0281 for small and1.2735/1.3121/1.4467/1.4494 for256-square. Every median is slower; the required3% warm256 benefit is absent. Exit1 denotes failed promotion criteria, not failed simulation. Reject this implementation for promotion, retaining current production radiation. Raw pairs/trials/RSS and limitations remain under `pairs_v2`. This is candidate-to-candidate evidence, not an original-upstream speedup.
