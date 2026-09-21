The P7 paired instrument compares the accepted P6 source tree with a candidate source tree. Its evidence class is **P6 versus candidate**, never an upstream golden. It does not create or freeze a workload itself. The lead must supply a reviewed, frozen JSON protocol and input inventory before execution. No timed benchmark matrix has been executed during harness construction.

Run validation first (fixture/source hashing only), then add `--execute` for the authorized matrix:

```sh
.venv-light/bin/python tools/benchmark_p7_pipeline.py \
  --protocol /absolute/path/to/frozen_p7_protocol.json \
  --baseline-source reports/characterization/p7_p6_baseline/src \
  --candidate-source src --run /absolute/path/to/new_report_directory
```

`--python` can select another isolated interpreter containing the frozen dependencies. The report directory must not exist. Each child receives only its selected source root on `PYTHONPATH`; it checks the imported package belongs there. Native thread environment variables are set before imports, and runtime settings explicitly use one worker with a total native budget of one or four. Imports, JIT, model execution and output I/O are included in elapsed process lifetime; fixture copying and geometry-warm setup are excluded and recorded separately.

The protocol requires these keys:

| Key | Meaning |
| --- | --- |
| `repetitions` | Exactly `5` |
| `seed` | Exactly `20260918` |
| `native_budgets` | `[1]`, `[4]`, or `[1, 4]` |
| `fixture` | Absolute frozen fixture directory |
| `fixture_hashes` | Exact relative-file-to-SHA256 inventory of the fixture |
| `kwargs_manifest` | Absolute path to the frozen public `thermal_comfort` kwargs JSON |
| `regimes` | Nonempty subset of `cold`, `geometry_warm` |
| `runtime_options` | Explicit common `RuntimeOptions` kwargs; use `{scene}` in cache paths |
| `preprocess_relative` | Prepared geometry directory relative to each scene |
| `cold_remove_relative` | Explicit scene-relative directories/files to remove before cold/setup execution |
| `warm_remove_relative` | Explicit scene-relative outputs/transactions to remove after setup; preserve geometry and promised legacy exports |
| `artifact_globs` | Scene-relative globs covering requested outputs and legacy SVF ZIP/NPZ exports |
| `tiff_field_rules` | Frozen rules keyed by TIFF stem prefix before the first underscore |

The kwargs manifest uses `{scene}` placeholders for `base_path`, `own_met_file` and other local paths. All ten `save_*` flags must explicitly be true. There is no fixture size, timestep, tiling, meteorology, geometry, or scientific option override in the harness. Warm public kwargs retain save flags, base path, date and any explicit tile keys, and use the protocol's prepared directory.

Each warm measurement first runs a complete untimed `thermal_comfort` setup using the same backend, native budget, inputs and runtime policy. Setup failures are retained and prevent that backend's warm trial. Prepared-file hashes are captured after successful setup. Native geometry cache integrity remains enforced by the backend's configured cache validation policy; the harness neither rewrites cache keys nor borrows geometry from the other backend. Fresh private JIT cache directories are used in every child, including warm trials. Geometry-warm therefore does **not** imply compiled-kernel warm. Kernels without Numba disk caching still compile in every child. OS page cache is uncontrolled.

Rules support `max_abs`, or `atol` plus `rtol`, or an `exact...` rule. Relative tolerance uses the P6 value. Every TIFF requires a rule; missing gates stop execution. Raster sizes, band counts, geotransform, CRS, dataset metadata, band metadata/descriptions, nodata, dtypes and masks must match. NaN and signed-infinity locations must match before finite comparisons. Paired NaN nodata values count as equal; None, finite sentinels and opposite infinities remain distinct. Comparisons read at most 256 × 256 cells per band window, bounding comparison scratch independently of raster size. Each band reports maximum absolute error and its worst coordinate. Artifact inventories must match and be nonempty, including cache-dependent SVF outputs. Non-TIFF ZIP/NPZ artifacts are checked for presence only; their numerical/schema round-trip gates and intermediate chronological-state gates must be verified separately before promotion.

Reports retain frozen protocol/input/model/harness hashes, exact invocation specifications, dependency versions, thread settings, stdout/stderr, failures, raw 20 ms process-tree RSS samples, elapsed times, pair order, per-pair comparisons, paired ratios, range, sample standard deviation and standard error of paired elapsed differences. Successful comparisons alone enter summaries, but failures remain explicit and invalidate claim eligibility. The final inventory recheck rejects changed inputs/source/protocol/harness. A process-tree RSS sample exceeding 12 GiB kills the child's process group and records an aborted failure. Shared pages can count twice and short peaks can fall between samples. No resource failure reduces physical work.

Full-workload promotion still requires the lead's numerical/state gates and qualified hardware interpretation. The paired summaries are instrument output, not a release claim or proof of equivalence to original upstream.

Comparator regression verification: `.venv-light/bin/python -m pytest tests/unit/test_p7_benchmark_harness.py -q` (13 tests covering nodata identities, special-value locations/masks, multi-window maximum errors, frozen numeric gates, metadata and missing artifacts/rules).

Before validation or execution, the supplied P6 source must match the accepted `reports/characterization/p7_p6_baseline/manifest.json` inventory. Every accepted snapshot artifact is checked against that manifest; every recorded Python source hash is cross-checked against `p6_installed_verification.json`. Non-Python model assets are checked against the accepted manifest (the installed verification source inventory did not record those assets). Both provenance document hashes and the result are retained.

Warm protocols must explicitly use `cache_enabled: true`, `legacy_cache_policy: "recompute"`, and `cache_dir: "{scene}/.runtime_cache"`. A separate untimed selected-backend child derives expected pipeline geometry identities from prepared tile inputs and calls the backend store's read-only integrity validation for each key. Standalone SVF cache identities are excluded from pipeline manifests. Before/after pipeline manifest hashes and the complete cache UUID-generation directory set must remain identical, and all ten chronological output TIFFs must exist for every expected tile. This is a structural reuse proof based on the current store's immutable new-generation-on-miss behavior and the pipeline's unconditional store lookup under enabled/recompute policy. It is not a direct hit-counter observation. A miss, identity mismatch, corruption, changed generation, or missing output invalidates warm evidence and cannot enter timing summaries.

CPU model, logical CPUs, physical RAM and inherited/no-pinning affinity policy are recorded before measurements. Global claim eligibility additionally requires this host to match the frozen Apple M1 Pro / 10 logical CPU / 16 GiB configuration, the final frozen guard to pass, every matrix pair's execution/comparison/cache conditions to pass, and every group to retain five valid pairs. Per-group comparison status remains separate from global evidence eligibility. A successful subgroup cannot authorize a claim after another group fails or the final guard detects mutation.

Expanded targeted verification: `.venv-light/bin/python -m pytest tests/unit/test_p7_benchmark_harness.py -q` passed 19 tests, including accepted-baseline rejection, changed/missing cache proof and global-eligibility guard failures. No full benchmark matrix was run during these changes.

A repaired baseline requires explicit protocol amendment, rather than relabeling arbitrary source as P6. New protocol keys are `baseline_manifest`, `baseline_manifest_sha256`, `baseline_label`, `repair_patch_sha256`, `correctness_report`, and `correctness_report_sha256`. These paths identify reviewed local files; hashes are SHA256 of the complete files. The unchanged accepted baseline validation remains available; a protocol-bound accepted manifest must use its exact label. New trial backend aliases are `baseline` and `candidate`, with the exact amended baseline label recorded in frozen and summary reports. Helper `compare` and `measure` signatures remain unchanged.

For the repaired path, the manifest's parent path/hash must match the immutable accepted P6 manifest. The repaired source inventory must have precisely the same file set and may change only `radiation/ground_view.py` and `radiation/engine.py`, with exact declared before/after hashes. Every other file must match its parent. The complete protocol-bound `angular_compatibility_repair.patch` is applied to a temporary copy of accepted P6 using `patch --batch -p1`; the resulting entire source inventory must equal the supplied repaired baseline. The patch hash, repaired/parent manifest hashes, source inventory and external numerical evidence references are retained. An allowed two-file diff alone cannot satisfy this gate. This amended baseline is **P6 plus angular compatibility repair**, not unchanged P6 or original upstream.

The external correctness report has this exact schema:

```json
{
  "status": "passed",
  "fixture_case_ids": ["seven unique fixture identifiers"],
  "source_hashes": {"baseline": {"relative-from-src": "sha256"}, "candidate": {"relative-from-src": "sha256"}},
  "cases": [{"case_id": "fixture identifier", "variant": "baseline", "passed": true, "evidence": {"path": "/absolute/evidence.json", "sha256": "file hash"}}],
  "original_regression": {
    "baseline": {"passed": true, "cases": 9, "evidence": {"path": "/absolute/evidence.json", "sha256": "file hash"}},
    "candidate": {"passed": true, "cases": 9, "evidence": {"path": "/absolute/evidence.json", "sha256": "file hash"}}
  },
  "angular_ground_view": {
    "baseline": {"passed": true, "field_count": 17, "boundary_passed": true, "evidence": {"path": "/absolute/evidence.json", "sha256": "file hash"}},
    "candidate": {"passed": true, "field_count": 17, "boundary_passed": true, "evidence": {"path": "/absolute/evidence.json", "sha256": "file hash"}}
  }
}
```

The example illustrates field structure; actual `fixture_case_ids` must contain seven entries and `cases` must contain exactly fourteen entries covering each identifier for both `baseline` and `candidate`. Both complete source inventories must equal current sources. Every evidence path must exist and match its hash; every listed result must pass. The independently authored report is the numerical gate authority; the instrument validates its coverage, provenance, file integrity and asserted results rather than recreating scientific comparisons. Missing, failed, stale or insufficient reports prevent execution. Final validation repeats these checks, preserving the global eligibility guard and warm-cache proof.

Repaired-baseline targeted verification: 30 tests passed, including full reviewed-patch derivation, wrong parent, disallowed changes, missing patch binding, missing/invalid reports, source mismatch and missing original/boundary/case evidence. Test reports are synthetic validation fixtures and are not scientific correctness evidence. No full timing runs were executed.

The subsequent `angular_and_svf_weights_v1` policy is a separately declared
amendment for the remaining expanded-scene failure. It permits precisely the
two angular files above plus `geometry/svf.py`; protocol and manifest must name
the same policy. Its complete patch is `compatibility_repair.patch`, still
bound by hash and applied to the original accepted P6 tree. Unknown policies,
extra files and missing numerical evidence are rejected. The original
two-file snapshot and its failed expanded reports remain unchanged.

For this policy the correctness report additionally requires `svf_weights`
entries for both `baseline` and `candidate`, each with `passed: true`,
`weight_count: 180`, `boundary_pixel_count: 2`, `maximum_weight_ulp: 1`, and a
hashed `evidence` reference. This records the original-weight and boundary
reproducer coverage, not a claim that every weight is bitwise identical.
All fourteen full-scene comparisons and earlier gates remain required.

Review subsequently found that geometry evidence links were present in the
aggregate but not validated by the instrument. The incomplete v3 timing run
was stopped and retained with `interruption.json`; it is not claim-eligible.
The validator now requires every geometry report's hash, passed status and
18 distinct artifact/field pairs. Instrument and parent-protocol hashes are
also checked at startup. Fifty-four focused tests pass, including missing,
corrupt and incomplete geometry evidence under both repair policies.

The v4 protocols change only these instrument/provenance bindings. Workloads,
numerical gates and pairing settings are unchanged. Run with
`tools/run_p7_pairs.py --protocols benchmarks/protocols/p7_pairs_v4 --output reports/characterization/p7_pairs_v4`.
