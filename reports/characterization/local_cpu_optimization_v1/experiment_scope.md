# Local CPU optimization: execution scope

The user requested a checkpoint commit and active optimization for the local
CPU while retaining mathematically identical results. Commit `8ca23d4` is the
checkpoint; it does not close P7 or P8. The local host is an Apple M1 Pro with
10 CPU cores and 16 GiB RAM. The approved portable numerical profile remains
`solweig-portable-sleef-5a1d179d-v1`.

## What may change

Independent pixel scheduling, repeated decoding, internal execution-block
size, and proven redundant work may change. Logical tiles, 153 sky patches,
24 chronological timesteps, vegetation and radiation physics, output fields,
checkpoint durability, and frozen original-reference gates remain fixed.

The same-host acceptance test is stricter than the original-reference
tolerances: finite values must preserve their bits, including signed zero;
special-value masks, dtypes, shapes, categorical values, timestamps, raster
metadata and carried state must agree. The separate trace helper also records
NaN payloads and array layout. Differences confined to those additional
descriptors require classification, not silent weakening of the numerical
contract.

## Independent candidate changes

- **E1:** The engine dispatches GVF to its existing parallel implementation
  when the explicitly selected runtime thread budget exceeds one. Serial and
  parallel loops call the same `_gather_pixel` body, retain `fastmath=False`,
  write separate receiver pixels, and retain the outer direction/reduction
  order. Component admission passed 216 tests, including all 17 GVF outputs
  and changed input arrays at thread counts 1, 4 and 10. Subsequent paired
  development measurements are selection evidence, not promotion approval.
- **E2 v2:** Shortwave radiation reuses decoded shadow and vegetation leaves
  only for exact known immutable implementations and an identity-matching
  lazy diffuse expression. The original first three read/error points remain
  in order; the original lazy-read lock/open/range checks still occur after
  the third read. Diffuse arithmetic remains `_diff` on an owned shadow copy.
  Other implementations and independent diffuse fields use the original
  fourth read. Version 2 passed 653 tests and exact comparisons of all seven
  outputs in three input modes; the historical 650-test v1 result is distinct.
- **Execution blocks:** Candidate sizes 1024 and 4096 group more independent
  receiver pixels per call. Each pixel still traverses patches in its
  original order. This remains a hypothesis requiring differential and
  chronological verification, not permission to alter logical tiles.
- **Persistent compilation:** The warm profile records compilation from
  wall-shadow and portable-math entrypoints. E4v3 adds persistent caches with
  distinct serial/parallel function identities and content-bound dependency
  invalidation. Mathematical bodies, explicit FMA operations and NumPy
  fallback behavior remain unchanged. Six source proofs, nine fresh-process
  cache tests, 300 subsystem tests per isolated candidate and both full
  small-scene chronological traces passed. E4 was excluded from the first
  development matrix and included in the combined matrix. Its packet
  identifier differs from the earlier strategy document's E4 SVF-workspace
  proposal, which has not been implemented.

## Execution and evidence

E0 first requalifies the committed installed baseline on the original M1
SLEEF dense1024 and vegetation1024 oracles, then captures warm 256-scene
attribution. Luna owns the numerical host and reports completion or a real
blocker. Other agents perform source-only work while that reservation exists.
There is no new Cura workload in this experiment.

The development protocol is
[`development.json`](../../../benchmarks/protocols/local_cpu_optimization_v1/development.json).
Its 24 pairs are diagnostic selection evidence. Final promotion requires a
separate frozen protocol and independent review, including quantitative
benefit/regression/uncertainty criteria and the large-scene numerical gates.
No development result is an original-upstream speedup claim.

The combined candidate passed 1,131 subsystem tests and three exact
72-event traces at 1/128, 4/1024 and 10/1024 thread/block settings. Its
six-setting, 18-pair development matrix selected four threads and a
1024-pixel execution block under the predeclared minimum-time rule. Separate
medians were 21.517622 s for the baseline and 12.624037 s for the candidate;
the median paired ratio was 1.646599. The selected source subsequently passed
installed-wheel and both large-scene admissions, followed by the independent
[`promotion.json`](../../../benchmarks/protocols/local_cpu_optimization_v1/promotion.json)
matrix: ten cells, five pairs per cell, including unchanged-default and
first-use guards. All 50 exact pairs and all predeclared performance/memory
gates passed. Independent review approved integration; the production source
now matches the qualified candidate, and 97 final installed-wheel checks
passed without skips. Public runtime defaults remain unchanged. See
[`local_cpu_optimization.md`](../../local_cpu_optimization.md) for final
measurements, evidence links and the remaining P7/P8 gates.

The paired harness records first use, compiled geometry-cold, and geometry-warm
as distinct regimes. First-use includes startup, JIT, preprocessing, geometry,
the full chronological model and output I/O. Warm setup is measured separately;
only compatible caches are reused. Uncached kernels may still compile in a
fresh process. OS page-cache state is uncontrolled.

Every timed process tree is sampled at 20 ms with a 12 GiB RSS limit, and new
experiments preserve a 10 GiB disk reserve. Summed RSS may double-count shared
pages and sampling may miss shorter peaks. A completed launcher with lingering
owned descendants is a failed trial and triggers owned-process cleanup.
Supervisor cleanup is recorded separately from worker elapsed time.

The timing path is uninstrumented at numerical boundaries. A separate
correctness-only trace captures all engine return fields, all requested writer
inputs and complete state at every timestep. It must never be used for speed
claims. Pair checks additionally validate ten 24-band outputs, SVF/visibility
exports, complete publication records and final checkpoint payloads.

Repeated trials retain all metrics, failures, source/input bindings and exact
comparison records. Before replacing earlier successful output histories, the
harness atomically writes and synchronizes the evidence. Last successful and
failed histories remain available. Original goldens and historical evidence
are never cleanup targets.
