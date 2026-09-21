# E3: one-call ground-view invariant reuse (source-only proposal)

## Evidence and boundary

The current file, durable baseline, and E0 source snapshot are byte-identical:
`7fd7bc4f1596aeeb4ba6f1762944d46795df5d8dade4cb81369c3652fdb82d64`.
The warmed E0 profile records 14 `_gvf` calls, 252 `_sun` calls, and 252
`_gather` calls: exactly 18 directions per `_gvf`. `_gvf` is in the measured
hot family, but `_gather` owns most of its cumulative time. The profile does
not isolate the proposed setup work, so this packet makes no speed or
crossover claim. Proceed only if the lead interprets E0 as sufficient reason
for a small experiment.

This proposal reuses values only during one `_gvf` invocation. It adds no
module, geometry, or cross-timestep cache.

## Smallest coherent internal change

Add a private optional `_direction_invariants=None` argument to `_sun`; public
`sunonsurface_2018a` wrappers continue to pass `None`. `_gvf` creates one fresh
dict only when `_safe_direction_reuse(...)` admits the exact NumPy profile.
The existing `_sun` support/fallback check remains first, before any cache
read or population.

Populate each entry lazily at the expression's current first-evaluation site:

1. In `_gvf`, compute and retain `aspect =
   _divide(_operate(np.multiply, dirwalls, np.pi), 180)` while evaluating the
   first direction's arguments. Later directions reuse that exact array.
2. In the first admitted `_sun`, compute/cache `wallbol` at its existing
   location, before `sunwall` normalization.
3. Keep the first `Lup` calculation unchanged before the water assignment.
   At the existing post-water radiance expression used by the line-261
   `gvfLup` correction, name/cache only that inner `A - B` array as
   `Lup_after`. Directions 2–18 use `Lup_after` at the line-216 gather-input
   location and at the line-261 correction. Thus direction 1 still sees the
   pre-water `Lup`, while every later direction sees the post-water value.
4. In the first admitted `_sun`, compute/cache `Lwall` and `albshadow` at their
   existing locations after the water assignment. Reuse them thereafter.

Keep scalar copies, `sunwall[sunwall > 0] = 1`, azimuth arithmetic,
first/second rounding, all `_gather` calls, directional accumulations, and
final reductions unchanged. `_gather` already snapshots every reused source
before its kernel and never writes the cached arrays (current lines 144–150).

## Admission guard and fallback

The optimized path requires exact `numpy.ndarray` objects for array-valued
inputs and ordinary Python/NumPy immutable scalars for scalar inputs. Array
subclasses, duck arrays, lists, and other dispatchable objects use the current
per-direction path, preserving `__array_ufunc__`, conversion, and exception
behavior.

Place the guard after the existing outer fallback and line-317 `sunwall`
construction, immediately before the loop. It must be non-throwing: type-gate
without coercing unknown objects, and on any exception from a memory-overlap
query return false and run the existing loop.

`_sun` also normalizes `sunwall` in place on every direction. `_gvf` constructs
that float32 array as a fresh result at line 317, so it cannot alias a caller
field; the proposal nevertheless leaves every normalization write in place.
`Tg` is the caller field mutated by `_sun`. Before enabling reuse, conservatively
apply `np.may_share_memory(Tg, value)` to every other array-valued argument:
`wallsun`, `walls`, `buildings`, `scale`, `shadow`, `first`, `second`,
`dirwalls`, `Tgwall`, `Ta`, `emis_grid`, `ewall`, `alb_grid`, `SBC`,
`albedo_b`, `Twater`, `lc_grid`, and `landcover`. Any possible overlap uses
the unchanged loop. A false result proves disjoint address bounds; a true
result is only a fallback, including non-overlapping views sharing a base.
No pairwise-disjoint assumption is needed among the non-mutated fields.

The guard is needed because current `_sun` mutates `Tg[lc_grid == 3]` when
`landcover == 1`. If, for example, `dirwalls`, `walls`, `shadow`, `alb_grid`,
`Tgwall`, `Ta`, or `lc_grid` aliases `Tg`, later directions can legitimately
observe changed aspect, `wallbol`, `albshadow`, `Lwall`, or water masks.
Those calls must retain current behavior. Repeated water writes are
idempotent on the admitted path because the mask, `Twater`, and `Ta` cannot be
changed through `Tg`.

## Equivalence obligations

- First evaluation order is unchanged: aspect argument; `_sun` admission;
  scalar copies; `wallbol`; `sunwall` normalization; azimuth; pre-water
  `Lup`; water mutation; `Lwall`; `albshadow`; gather; post-water radiance.
- The post-water radiance cached in direction 1 is textually the same
  `_operate` graph used for direction 2's `Lup`. Reuse therefore preserves
  dtype, NaN payload/signed-zero bits produced by that graph, and avoids an
  algebraic rewrite.
- After direction 1, admitted dependencies are immutable and `Tg` has reached
  the state written again by later directions. Cached arrays therefore equal
  every removed reevaluation byte-for-byte.
- `_gather` copies `Lup`, `albshadow`, and `Lwall` before computation, so a
  direction cannot corrupt the retained arrays.
- The `_sun` compiled/fallback decision remains at current lines 203–204. An
  exception in the first pre-water expression occurs in direction 1 as now;
  an exception specific to the post-water state occurs at direction 1's
  existing correction expression. Unsafe types or aliases execute the exact
  original statements.

The proof assumes no concurrent external mutation during `_gvf` and excludes
resource-failure observability such as deliberately induced `MemoryError`.
Finite differential tests can falsify mistakes but cannot replace these
alias and ordering arguments.

## Adversarial experiment gate

1. Compare all 17 outputs and every post-call input by dtype, shape, raw bytes,
   NaN/Inf masks, and signed zero for serial and parallel paths. Include
   `landcover` 0/1, water masks, NaN payloads, infinities, negative zero,
   nonbinary buildings/shadows, and angular boundary fixtures.
2. Wrap the real `_gather` (capture then delegate) to prove direction 1 gets
   pre-water `Lup` and directions 2–18 get the exact post-water bytes. Check
   the final `Tg` mutation against the baseline.
3. For each dependency family, construct an overlapping or negative-stride
   view of `Tg`: at minimum `dirwalls`, `walls`, `shadow`, `alb_grid`,
   `Tgwall`, `Ta`, `Twater`, and `lc_grid`. Assert guard rejection, then compare
   outputs, mutations, warnings/exceptions, and fallback calls to baseline.
4. Use an ndarray subclass with `__array_ufunc__`, a duck array, and a
   conservative `may_share_memory` false-positive view. Assert the unchanged
   path before any cached operation.
5. Exercise float64 `_gvf` fallback, nonfinite/zero `second`, multi-element
   `landcover`, and read-only `Tg`. Match exception type/message, first failing
   direction, and mutations. Retain existing original-CPU and independent
   UMEP suites after the targeted tests.

## Source anchors

- Current/baseline `_sun`: support fallback 203–204; `wallbol` 211;
  `sunwall` mutation 212; pre-water `Lup` 216; water mutation 217–218;
  `Lwall`/`albshadow` 219–220; gather 226–230; post-water correction 261.
- Current/baseline `_gvf`: outer fallback 298–299; local `sunwall` 317;
  18-direction loop and repeated aspect conversion 318–339; ordered final
  reductions 340–357.
- Current/baseline `_gather`: defensive source snapshots 144–150.
- E0 profile: `reports/characterization/local_cpu_optimization_v1/e0/profile/
  repeated256_warmed_v1/profile/worker_cumulative.txt`, lines 15–23.
