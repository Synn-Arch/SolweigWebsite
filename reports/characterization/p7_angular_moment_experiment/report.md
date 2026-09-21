# P7 angular-moment experiment

Correctness gate passed: all 18 original component cases, all 24 chronological longwave boundaries, and both source-snapshot and installed-wheel pipeline runs (9 cases each) passed unchanged gates. The wheel run imported `solweig_light` from isolated `site-packages`, had no Torch spec, and injected the unpromoted experiment through the recorded pytest hook.

Performance was not run. The current falsification kernel constructs the six moments inside every pixel/timestep call. The reviewed performance design instead requires one immutable, content-bound tile-owned `MomentBundle`, bounded packed decoding, invalidation and lifetime/concurrency checks, and one-time construction included in end-to-end timing. Timing the current kernel would evaluate a different cost model.

The preserved float32 counterexample means these are tolerance-based results, never an algebraic equivalence claim. No production source was changed and no optimization or full-workload benefit is claimed.

## Bundle implementation follow-up

The isolated immutable builder passed complete-content identity, dense/packed equality with seven-pixel bounded decoding, geometry and visibility invalidation, four concurrent tile-owner isolation, exact allocation accounting, injected-failure cleanup, lifetime release, and fail-closed ownership checks.

Pipeline integration is blocked at a concrete interface: the current tile-owned `GeometryCache` has no `MomentBundle` field and the unchanged chronological call into `Lcyl_v2022a` has no explicit bundle argument. Rebuilding or content-hashing through that per-timestep interface is O(NP) each timestep and does not implement the reviewed cost model. The paired protocol is therefore drafted for lead review but explicitly cannot be frozen or executed until an isolated snapshot adds this conduit and the lead freezes the quantitative regression rule.

## Integrated isolated snapshot

The isolated snapshot now owns one immutable bundle in `GeometryCache`, constructs and content-hashes it once before chronology, passes it explicitly through a private `Solweig_2022a_calc`/longwave conduit, and closes it through the tile `ExitStack` on success or error. `None` or failed coefficient/profile validation retains the original second-sweep path. Both serial and parallel kernels have distinct moment variants.

Post-integration validation passed 18 original component cases, nine source-snapshot pipeline cases, and nine installed-wheel pipeline cases. The lifecycle counter observed one build and close for each of eight in-process 24-step tiles; the legacy raw case executed in its separate worker and passed complete output comparison. The installed wheel imported from isolated `site-packages` with no Torch spec.

The 40-pair protocol and quantitative acceptance thresholds are frozen in `paired_protocol_frozen_for_review.json`. No timings have run; lead review is the remaining gate.

## Review-block closure (v2)

Unsafe nonfinite, overflow-risk, and zero/signed-zero pixels now execute the original ordered reflection sweep. Bundles retain the exact immutable visibility source objects and wrappers require O(1) source identity in addition to complete construction-time content identity, preventing a same-shape wrong-tile bundle. Runtime profile/source literals map to the complete v2 source manifest and the executable harness checks that mapping.

Durable evidence includes both earlier failed regression attempts, the final 11-case regression JUnit/log, 18-case component results/log, source and installed-wheel 9-case JUnit/logs, retained wheel/environment/origin/no-Torch provenance, and harness dry-run/tests. The v2 schedule has a 2/3 first-side balance in every five-pair subgroup. No timing has run.

## Final v3 review corrections

The executable ratio is now `candidate_seconds / baseline_seconds`; retained tests prove slower candidates fail, faster candidates pass, and the priority rule requires at least four of five ratios below one. Failed or incomplete warm groups return rejection without comparing a missing median. Reusable dense visibility must be immutable bytes-backed storage; owning arrays and read-only aliases of mutable bases are rejected. Packed pipeline sources retain constant-time identity authorization. Exact v3 wheel build, install, test, origin and no-Torch commands are retained.

## Child-process execution correction

The first 40-pair attempt is preserved unchanged in `pairs_v1`; every child failed before timing because its frozen source was not importable. Protocol v3 and the revised harness resolve the source and spec paths before spawning, insert the source before the child import, and propagate the absolute source through `PYTHONPATH` to the nested native runtime worker. Three bounded rehearsal failures that exposed each path layer are retained.

The final real-subprocess candidate smoke produced all valid outputs. A separate full small first-use baseline/candidate pair produced valid outputs and passed the frozen comparison; its elapsed values and ratio are diagnostics only and are excluded from benchmark evidence. The full 40-pair v2 output namespace does not exist and no new matrix timing has begun.
