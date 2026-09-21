# Independent scientific checks: preliminary evidence audit

## Current execution status: Cura follow-up

The historical discovery and queued-work sections below describe their original
checkpoints. They are superseded for execution status by the following inspected
JUnit evidence in `reports/cura_p8_20260919T1215Z_a7c3/results/`:

| Check | Executed result | Evidence |
|---|---|---|
| Official UTCI coefficients and independent evaluation | 11 passed | `p8_queued_science_tests.xml` |
| Matched UMEP thermal delay | 8 passed | `p8_queued_science_tests.xml` |
| Matched UMEP ground view | 6 passed, 2 failed | `p8_queued_science_tests.xml` |
| SVF analytic checks | 3 passed, 2 failed | `p8_queued_science_tests.xml` |
| Directional wind | 1 passed after environment-path repair | `p8_directional_wind_retry.xml` |
| Night shortwave and upward longwave | 2 passed after test precondition repair | `p8_radiation_analytic_revision_tests.xml` |

The initial six-file run has 35 tests, 28 passes and seven failures. Separate
reruns resolve three initial failures; they must not be added as new independent
tests or presented as a single clean suite execution. No skips or errors occur
in these reports. These are Cura source-snapshot results, not additions to the
earlier installed-wheel count. Commands, environment locks and hashes accompany
the reports; the radiation revision is explicitly recorded in
`../radiation_test_snapshot_revision.json` relative to the results directory.

The four remaining failures are not waived. Two SVF flat-scene directional-unity
expectations require source-grounded scientific review. Two UMEP aggregate
ground-view comparisons exceed their authored absolute allowance. The separate
`upstream_cpu_ground_view_diagnostic.md` establishes that the candidate preserves
the pinned original Torch CPU result for those fields; it does not establish
independent UMEP agreement. No production arithmetic or frozen tolerance was
changed. Whole-model validation and the final P8 gate remain incomplete.

Scope: implementation plan §10.5, inspected during P7 correctness work. This
is a gap inventory, not an executed P8 gate. No numerical source was changed.
At the initial inspection, `tests/scientific` did not exist. Subsequent analytic
checks and their execution status are recorded below. Differential tests provide
compatibility evidence but do not establish independent correctness.

| Required check | Evidence located | Remaining gate |
|---|---|---|
| Four-neighbor wall heights | `tests/differential/test_walls_compiled.py` compares original captures and native reduction behavior | Independently constructed stencil expectations with stated threshold and border assumptions |
| Isolated-block shadow direction and length | Original shadow captures and wall-shadow differential tests | Analytic geometry expectation under a fixed domain and nonboundary solar angles |
| No-obstacle visibility | `reports/geometry_characterization.json`; `tools/characterize_geometry.py`; deviation register | Eighteen original flat cases pass; two exact-zenith cases fail. Execute candidate expectations and retain the inherited failure explicitly |
| Bounded SVF | Original geometry captures and expanded export comparisons | Independent valid-domain bound tests, with applicable roundoff policy stated |
| No direct solar flux at night | Original day/night sequence differential tests | Explicit independent night-flux assertion on genuine pipeline quantities |
| Stefan–Boltzmann subexpressions | Radiation differential fixtures | Independently evaluated scalar physical relation and units |
| Directional wind selection | Original UTCI/preprocessor fixtures | Independently constructed direction-to-component expectations |
| Independently evaluated UTCI coefficients | `test_utci_compiled.py` verifies reconstruction from pinned fork AST and original numerical captures | Separate coefficient source and independent evaluator; shared reconstruction is not independent validation |
| Matched UMEP components | No separately pinned UMEP reference located in `.upstream` or reference manifests during this inspection | Pin, inspect and justify matched versions/configurations before comparing selected components |
| Metamorphic checks | Block/thread/ownership and repeated-input differential tests already exist | Inventory exact coverage; add isolated-obstacle monotonicity only with valid receiver/domain preconditions |
| Stress categories near thresholds | No category-change report identified in this inspection | Determine downstream category usage and document applicable threshold comparisons without extending the API implicitly |

Inspected: plan §10.5 and §13; `TASKS.yaml` P7/P8; deviation register's geometry
observations; UTCI and wall differential test source; targeted text searches in
`tests/unit`, `tests/differential` and `tools`; reference manifest filenames.
No tests were executed for this audit. “Not located” is limited to this search,
not a claim that no relevant evidence can exist elsewhere.

Smallest next steps: author analytic component tests separately from numerical
implementation; execute them and record expected inherited failures without
changing compatibility behavior; then select and inspect an independent UMEP
revision and UTCI coefficient source. These checks do not replace the P7 original-reference comparisons or its
full-workload performance gate. The later repair-v2 correctness gate records
passing original-reference comparisons for both variants on all seven scenes;
the full-workload performance matrix remains in progress.

## Executed follow-up: elementary geometry

`tests/scientific/test_geometry_analytic.py` now tests hand-constructed exposed
wall faces, cardinal shadows on level ground at 45° elevation, monotone
occlusion for increasing isolated obstacle heights, and no-obstacle visibility
away from exact zenith. Half-integer obstacle heights avoid grazing ambiguity;
receivers and obstacles are interior to the fixed domain. Expected arrays are
constructed from geometric distances, not upstream captures or candidate
helper outputs.

Command: `PYTHONPATH=src .venv-light/bin/python -m pytest tests/scientific/test_geometry_analytic.py -q --junitxml=reports/p8_geometry_analytic_tests.xml`.
Result: **16 passed**, zero skips, in the development environment. The inherited
exact-zenith observation remains unresolved and is not included among these
passes. This follow-up covers the first three table rows only within the stated
simple geometries, plus isolated-obstacle monotonicity; other gaps remain.

## Primary-source discovery during the quiet benchmark window

The [official UTCI calculator](https://utci.org/utci_calc.php) links its
[Fortran source archive](https://utci.org/resources/UTCI%20Program%20Code.zip)
and states a 0.5–17 m/s wind domain at 10 m. The archive was subsequently downloaded from the official link and pinned in
`reports/characterization/p8_utci_official_source/source_manifest.json`: 252,552
bytes, SHA-256 `83ea34dc2428093c8b0f4e299bfb0a24752ff49ec1e11741927a75dafae283f1`.
It contains the October 2009 a0.002 Fortran source and accompanying README,
which were extracted and separately hashed. No bundled executable was run.
Independent evaluation and coefficient comparison remain unexecuted.

The [UMEP Python component directory](https://github.com/UMEP-dev/UMEP/tree/master/SOLWEIG/SOLWEIGpython)
contains the named 2018a ground-view and 2022a radiation components. Revision `3fcc0c3dca67d1d5644a6d34b9148d7a365743ba` is now pinned in
`reports/characterization/p8_umep_source/manifest.json`, with separately hashed
ground-view, thermal-delay and radiation source files. Source/configuration
matching and executable comparisons remain pending; retrieval alone is not
validation. The newer
[Rust SOLWEIG repository](https://github.com/UMEP-dev/solweig) identifies itself
as experimental and distinguishes itself from the UMEP reference. Its
[change notes](https://github.com/UMEP-dev/solweig/blob/main/CHANGES.md) describe
an accepted shadow-algorithm difference in SVF. It must not be substituted
uncritically for a matched Python component oracle.

During timed trials, activity was limited to remote documentation reads and
small reference-file acquisition/inspection; no reference installation or
scientific workload ran concurrently.

## Authored follow-up awaiting execution

`tests/scientific/test_radiation_analytic.py` adds night shortwave and
Stefan–Boltzmann assertions using genuine pipeline inputs. Its assumptions and
preselected tolerances are recorded in `reports/p8_radiation_analytic_design.md`.
`tests/scientific/test_directional_wind_analytic.py` adds a genuine 24-timestep
public-workflow check with independently specified directional coefficients,
boundaries, wraparound and fallback expectations; see
`reports/p8_directional_wind_design.md`.

These three tests are authored but **unexecuted**. Execution is deferred until
the P7 timed-trial quiet window ends. They are not included in the 16 analytic
geometry passes or the previously recorded installed-suite counts. Bounded SVF,
independent UTCI evaluation, matched UMEP components and stress-category checks
remain open.

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

## Unmocked ground-view comparison queued

The same pinned UMEP revision now includes `sunonsurface_2018a.py`, recorded
with its hash and byte count in the source manifest; it imports only NumPy.
`reports/p8_umep_component_matching.md` records static semantic matching,
dtype and angular-boundary differences, mutation behavior and explicit limits.
Eight cases in `tests/scientific/test_umep_ground_view_independent.py` compare
the unchanged reference with candidate serial and parallel paths, including
all five direct and 17 aggregated outputs and observable mutations. No numerical
dependency is mocked. These cases remain authored and unexecuted during the
benchmark quiet window; they do not close the matched-component gate yet.

## Bounded-SVF checks queued

`tests/scientific/test_svf_analytic.py` now contains four full-default-patch
scene/representation cases and one preselected-roundoff arithmetic check.
Both dense and compact results must expose all 153 patches. Physical bounds
cover all 15 SVF fields and total SVF; flat-scene unity is limited to the
vegetation channels because the inherited building zenith issue remains.
`reports/p8_svf_analytic_design.md` records assumptions and the pre-execution
allowance derivation. AST syntax inspection passed; numerical execution and
the bounded-SVF scientific gate remain pending.

## Stress-category applicability audit

`reports/p8_stress_category_design.md` records targeted source evidence that
both candidate and pinned upstream emit continuous UTCI, with no runtime
category output or decision branch. Plan section 10.5 makes downstream
category checks conditional; no such consumer was identified in these paths.
The report proposes an optional independent diagnostic with all mismatches
retained, but its threshold authority still requires primary-document pinning.
No classification API, category accuracy result, or executed diagnostic is
claimed. Continuous UTCI verification remains required.
