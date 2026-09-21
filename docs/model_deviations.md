# Model behavior and deviations

No candidate model or scientific correction has been implemented. All numerical
and performance budgets remain provisional. Source observations below are not
proof of numerical equivalence.

## Executed failure: undeclared Numba dependency

- Affected reference: pinned upstream commit, CPU, installed declared test/runtime
  dependencies, own-met projected TIFF fixture, all outputs enabled.
- Reproducer: initial characterization command in `docs/provenance.md`.
- Outcome: preprocessing, walls/aspect and standalone SVF finish; importing
  `utci_process` imports `calculate_wbgt`, which requires `numba.vectorize`.
  It raises `ModuleNotFoundError: No module named 'numba'` before simulation.
- Evidence: `reports/runs/initial_cpu/{stderr.log,outcome.json,measurement.json}`.
- Treatment: add Numba to the isolated reference environment and rerun unmodified
  source. This is an environment repair, not a patched numerical reference.

## Driver observations awaiting targeted executable characterization

Anchors refer to pinned `solweig_gpu/utci_process.py`.

- Lines 421–440: CPU NumPy land-cover view is mutated during normalization;
  CUDA host copies may behave differently. Preserve CPU behavior and separately
  characterize device disagreement when hardware becomes available.
- Lines 482–485: positive median terrain altitude becomes 3 m.
- Lines 583–593: WBGT wet bulb uses temperature plus UHI; radiation receives base
  temperature later. Lines 696–713 add UHI only to comfort-stage temperature and
  select WBGT branches using `shadow < 0.1`. Units require independent checks.
- Lines 622–645: cache eligibility controls whether SVF is recomputed; mere file
  presence does not prove geometry identity.
- Lines 655–665 and 709 onward: raster histories are retained for output, including
  unrequested radiation fields. Candidate must stream while preserving physics.

The remaining behavior-register items from plan Section 9 are still pending
source audit and reproducing tests; this initial register is not complete.

## Executed geometry observations: nonbinary visibility and exact zenith

Reference: original pinned upstream CPU, `shadow.shadow`, deterministic 16×19
float32 and float64 scenes. Reproducer:

```sh
.venv-oracle/bin/python tools/characterize_geometry.py --run reports/runs/geometry_characterization/reproduction --report reports/geometry_reproduction.json
```

`reports/geometry_characterization.json` records 126 successful function
executions (four patch tables and 122 shadow cases), source/environment hashes,
input/output NPZ hashes and the independent checks. Successful execution is not
the same as a passed scientific expectation.

- Short-ray trunk fixtures produce `vbshvegsh=2`, in both input dtypes. This
  disproves Boolean representation for that channel. Preserve exact values;
  use categorical encoding with a lossless fallback until the full value domain
  is characterized. The observed sets are fixture-specific, not universal.
- On flat zero-elevation ground with no obstructions, exact zenith produces some
  `sh=0` values. Both dtype fixtures fail the independently expected full
  visibility. Eighteen other flat positive-altitude cases pass. The cause has
  not yet been isolated; no scientific correction or oracle patch is applied.
- Negative-elevation fixtures include both a nonnegative caller-supplied ray
  bound and the signed bound used in a separate case. The former is a direct
  function characterization, not a claim that the public driver chooses that
  bound. Do not substitute a relief bound in the candidate default model.

## Executed chronological dtype behavior

The complete 24-step original CPU capture contains float32 raster outputs, but
`CI`, `I0`, `radI`, and `radD` do not have one fixed dtype across day/night.
`reports/boundary_capture_verification.json` records the observed sets. Preserve
the executed promotions in the correctness-first port; treating every scalar as
float64 is not established equivalent. The read-only capture left all nineteen
standalone TIFFs unchanged and passed 23 consecutive state handoff checks.

## P2 low-level UTCI dtype characterization

The pinned original `utci_calculator` allocates a float32 result. All fifteen
characterized combinations containing one or more float64 input arrays fail at
masked assignment with a dtype mismatch. The P1 NumPy port had inadvertently
accepted these profiles by casting the result. P2 restores the observed failure;
this is a compatibility repair, not a change to the polynomial or meteorology.
The direct `utci_polynomial` function has a different contract and retains its
characterized float64/mixed evaluation. Evidence and original exceptions are in
`tests/reference/utci_original_cpu/manifest.json` and
`reports/utci_compiled_characterization.json`.

The compiled calculator currently optimizes the characterized finite float32
profile and sentinel positions. Other values retain the full NumPy evaluator;
there is no clipping, replacement model, or reduced polynomial. An exploratory
1e-6 direct-polynomial test was rejected as an unjustified, non-frozen threshold;
the existing 0.02 degree-C UTCI gate remains unchanged. Expression/coefficient
checksums prove transcription, not identical floating-point evaluation.

## Executed full SVF option contracts

P3 captured full original SVF execution on open, building/vegetation, bush and
negative-elevation scenes for options 1–4. Options 1–3 execute; all four option-4
cases raise the original TypeError because floating annulus bounds are passed
to an integer range. The option-4 patch table remains valid, but full SVF option
4 is an upstream-failing computation. The candidate preserves that failure;
no bound cast, repaired oracle or successful option-4 claim is introduced.
`tests/reference/svf_original_cpu/manifest.json` records outputs and exceptions.

Compiled wall-height shadows currently target the float32 raster path. Other
low-level raster dtypes retain the characterized NumPy recurrence; they are not
silently narrowed or described as universally compiled. The separate sky-shadow
kernels retain float32 scratch arrays with float64 DSM state when supplied.

## P4 ground-view traversal and mutation contracts

Original source and `tests/reference/ground_view_original_cpu/manifest.json`
show that ground-view scratch fields retain values outside the slice updated by
each ray step. The compiled gather preserves that history. Building inputs are
not mutated by the minimum recurrence. `sunwall` normalization is in place;
water-cell `Tg` is changed after the initial directional `Lup` calculation, so
later directions can see different source temperatures. Before/after argument
snapshots test these effects explicitly. No physical correction was applied.

The original raises `RuntimeError` on the characterized beyond-raster traversal
cases and `UnboundLocalError` when the rounded second distance is zero. These
failures remain compatibility outcomes, not successful radiation calculations.
Float64 raster profiles retain the characterized NumPy path; compiled coverage
is not claimed for every accepted dtype.

### P4 scalar argument failure compatibility

The original patch-radiation packet records two type-sensitive failures that
were initially accepted by the NumPy port: box shortwave rejects a native-bool
condition at its first tensor `where`, and an active longwave sun-facing patch
rejects a native scalar solar altitude at the helper's tensor tangent call.
Compatibility checks now preserve the corresponding `TypeError` classes at
those execution points. Tensor-origin zero-dimensional arrays remain valid.
Nighttime/no-active-patch longwave calls must not be rejected merely because
solar altitude is native scalar; their original branch does not call the helper.
Exact Torch diagnostic wording is not reproduced. The original packet and
counterexample tests distinguish failure behavior from a scientific correction.

## WRF timestamp repair v1

The pinned `extract_datetime_strict` returns `(datetime, domain)`, but the original
WRF time-window filter compares that tuple directly with datetimes and raises
`TypeError` before reading meteorological variables. Under implementation-plan
§10.1's minimal-repair policy, `wrf_timestamp_v1` extracts element zero only in
that comparison. The candidate WRF adapter adopts this versioned repair.

`docs/upstream_patches/wrf_timestamp_v1.patch` is the complete one-line patch;
`tests/reference/wrf_patched_cpu` records its hash and separately labeled
patched-reference outputs. The unchanged original failure remains in
`tests/reference/forcing_original_cpu`. Neither the upstream checkout nor its
installed package was modified. Independent source review accepted this scope.

Tuple sorting, domain tie order, inclusive endpoints, accepted filenames,
rotation formulas, units and synthetic hourly coordinates remain unchanged.
Missing-hour or duplicate-domain handling is not repaired. The observed
multiple-domains/one-hour shape failure remains recorded and tested. Successful
WRF comparisons therefore establish parity with the explicitly patched
reference, not with an executable unmodified WRF branch.

The input builder follows executed upstream behavior: it constructs static
rasters. Its original docstring's additional promise to generate meteorological
and wind files is not implemented by that upstream entrypoint. Separate forcing,
ERA5 acquisition and wind helpers remain available; documentation describes
actual behavior rather than silently adding those operations to the builder.

## Candidate ground-view angular precision defect (P7 diagnosis)

Expanded publisher-sample inputs exposed a candidate defect, not an upstream
scientific correction. Preserved P6 promoted tensor-origin float32 scalar
angles to float64 when multiplying/subtracting Python angular constants.
Upstream retains float32 through these operations. Strict wall-aspect boundary
tests therefore selected different masks at 65°, 145°, 245° and 325°; the latter
three produced material ground-view radiation differences in the diagnosed
case. Inputs and all 18 geometry-export fields passed their original gates.

The preserved P6 and P7 optimized TIFF outputs were exactly equal on this
failing scene, locating the defect before either P7 optimization. Evidence is
under `reports/characterization/p7_real_diagnosis`. The narrow repair now covers
both compiled and retained fallback paths. It passed 186 ground-view checks,
682 radiation/state/delay checks and five resume checks; all 17 first-daytime
GVF fields match original bitwise, and the full diagnosed 24-hour scene passes
the frozen output gates. The expanded seven-fixture and installed-wheel gates
remain in progress. No reference, fixture or tolerance was changed. Original
P6 failure evidence remains retained.

## Candidate SVF annulus arithmetic defect (P7 expanded gate)

The angular-only repair left Kdown about 0.060 W/m² away from original at
hour 15 on the published vegetation-rich and sparse windows. Both repaired
P6 and the optimized candidate failed identically. A one-ULP vegetation SVF
difference changed a near-zero `svf + svfveg - 1` value into zero; its derived
angle amplified the difference in shortwave reflection. Passing the existing
intermediate SVF tolerance did not imply passing the final radiation gate.

`annulus_weight` now preserves the original reciprocal-then-multiply scalar
division and uses native float32 sine in a serial Numba kernel with
`fastmath=False` and NumPy division-error semantics. It introduces no clamp,
runtime weight table, or physical-model change. Of 180 characterized original
weights, 174 match bitwise and six differ by one ULP on this host. Both boundary
receiver sums and angles match original exactly. Zero/nonfinite scalar cases
retain their original masks and signed-zero behavior. Array-valued low-level
inputs retain the earlier NumPy path.

Evidence: `reports/characterization/p7_real_diagnosis/svf_weights`, with
original source/environment provenance in
`tests/reference/p7_svf_annulus_original_cpu/manifest.json`. The targeted
gate passed 37 tests. Both final repaired variants subsequently passed all
seven complete scene gates and 18 geometry export fields per scene; the
source-bound aggregate is
`reports/characterization/p7_repair_v2_correctness/gate.json`. This resolves
the characterized failures without loosening any existing tolerance; it does
not establish bitwise equivalence for every possible scene or platform.
Geometry and checkpoint identities include the changed source files, so
pre-repair caches are not silently reused.

## Independent UMEP ground-view precision discrepancy (Cura P8)

On the two 9×11 synthetic land-cover cases, unchanged UMEP returns float64 `gvfSum`, while pinned original SOLWEIG-GPU CPU and candidate return identical float32 sums. The independent UMEP comparison exceeds its predeclared 1e-6 absolute gate in 12/99 cells, maximum 1.7106533043431682e-6. The normalized field passes. All 17 candidate fields pass their unchanged comparison against original Torch; sums and normalized values are exact. This is an inherited precision difference, not permission to relax the independent gate or silently change compatibility arithmetic. The independent tests remain failed; their narrowly scoped compatibility-release exception is now user-approved (see below). Evidence and isolated environment provenance: `reports/cura_p8_20260919T1215Z_a7c3/upstream_cpu_ground_view_diagnostic.md` and its linked raw artifacts.

### Inherited option-2 SVF azimuth initialization truncation (Cura P8)

The executed initializer-only diagnostic at pinned upstream commit
`0d7fe742abeeddd890dd58fc76ed7f78bd47faec` and the current candidate both
declare 153 azimuth entries but initialize 149. Declared band counts are
`[31,30,28,24,19,13,7,1]`; actual writes are
`[31,29,27,24,19,12,6,1]`. Float32 reciprocal arithmetic followed by integer
truncation shortens four initializer loops. The consumer still reads declared
band lengths, crossing initialization-band boundaries and reading four trailing
zero slots. Full upstream/candidate azimuth arrays are identical.

Evidence: `reports/cura_p8_20260919T1215Z_a7c3/admission/admission_manifest.json`,
key `svf_initializer_execution`, SHA-256
`9e639c5ef62a83b12e676e62cd7a073c70672023ff68301e857e71c75ab2c6fd`.
The manifest records source hashes, actual Torch/NumPy versions and commands.
The earlier source-only emulation predicting 150 upstream entries is superseded
by this executed evidence. The directional open-scene deficits are therefore
not adequately explained as ordinary sector quadrature.

Compatibility retains this inherited behavior. The two independent open-scene
SVF unity checks remain failed; neither their tolerances nor the model were
changed. A corrected azimuth scheme would require a separate versioned
scientific policy and full downstream validation. Initializer agreement does
not establish physical correctness.


## ASVF transcendental rounding changes strict patch classification (Cura P8)

Dense1024 original-CPU admission fails the frozen TMRT gate: maximum absolute difference 0.04949951171875 °C versus 0.01 °C. The first material difference appears at output band 8. At cell [36,185], float32 SVF 0.749999940395355 produces ASVF bits 1057360531 from the Cura candidate NumPy expression and 1057360530 from original Torch CPU. The one-ULP difference moves patches across a strict shade boundary; upstream equality belongs to neither sun nor shade. Four building patches each add 0.0809573158621788 W/m² reflected shade shortwave in the candidate. The minimal fixture reproduces the reflected-energy difference. Serial and compiled candidate radiation agree given the same ASVF bits, so this is input transcendental parity, not evidence of a compiled radiation accumulation defect.

Source-provenanced fixture and reproducer: `reports/cura_p8_20260919T1215Z_a7c3/admission/p8_1024/dense_kref_diagnostic/rootcause_report.json` (SHA-256 `abd67a4d42ffdf0b7f5e4ce7277d6883a20a6a6d3a46d5eb83b11d589bb87fa2`). Independent M1 characterization in `reports/characterization/p8_asvf_parity_review/review.json` covers 411,685 inputs: float32 sqrt matches original, current acos has 5,108 finite one-ULP mismatches, and float64 acos cast to float32 has 6,903. At the Cura hotspot input, all tested M1 expressions already match original. Thus increased precision is not a universal compatibility repair. Backend characterization is ongoing. No threshold shift, input clipping, blanket nextafter, or relaxed tolerance is authorized by this diagnosis; no ASVF repair has been applied.


### Isolated repair evidence (not production)

The subsequent standalone-MKL snapshot experiment passes unchanged dense/vegetation1024 full-field gates; exact ASVF and actual per-patch classifier preflights are recorded in `reports/cura_p8_20260919T1215Z_a7c3/admission/p8_1024/mkl_pipeline_experiment/result_summary.json`. Both sqrt and acos, then classifier raster/scalar trig, require the original backend and promotion order. The earlier vectorized mask oracle is rejected because its scalar dtypes did not match actual invocation. No scientific threshold or source behavior was corrected; this is a candidate compatibility repair. Production integration awaits the dependency design decision, and M1 validation remains separate.


## Approved compatibility-release exception: inherited SVF and UMEP failures

The user explicitly selected “Retain upstream behavior; document exception.” Policy `inherited-svf-umep-compatibility-v1` is recorded in [scientific_release_exceptions.json](../reports/scientific_release_exceptions.json). It covers only the two open-scene dense/compact SVF unity tests and the two land-cover variants of the independent UMEP ground-view test named there. Their outcomes remain **failed**, with an approved known-limitation disposition for compatibility-release consideration. Tests, assertions, fixtures, tolerances and model arithmetic are unchanged.

Original Torch and candidate dense/compact SVF agree bitwise across all19outputs on the exact9×13fixture; the inherited directional-unity deficits are preserved. Original and candidate float32 `gvfSum` also agree, while12/99cells differ from UMEP's float64 sum beyond the independent1e-6gate (maximum1.7106533043431682e-6). The evidence-bound [disposition memo](../reports/characterization/p8_portable_math_review/scientific_failure_disposition.md) records exact values and assertions.

The compatibility release must not claim physically correct unobstructed directional vegetation SVF or full UMEP agreement at that tolerance. Changed or worsened behavior requires fresh investigation; this exception does not automatically accept future failures. It does not cover dense1024 TMRT incompatibility, portable-math policy, other scientific deviations, or remaining release gates. No scientific correction or overall release approval is implied.
