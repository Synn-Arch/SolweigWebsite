# P7 exact building-horizon experiment

## Decision

Reject a standard single-translation max-plus scan as a general replacement for the frozen discrete building ray. At 30°, the first offsets are `(-1,+1), (-2,+1), (-3,+2), (-4,+2)`: their increments alternate. Repeating the first translation visits `(-2,+2)` and misses the required `(-2,+1)` sample.

The executable 7×7 counterexample places a 20 m blocker only at that required second sample. The exact recurrence shadows six pixels that the translated scan leaves unshadowed; the receiver `[4,3]` has exact horizon `19.59279` and scan horizon `0`.

## Evidence

The experiment reproduces the source formulas for azimuth normalization, major-axis selection, NumPy rounding, per-index vertical decrement, border termination, and `amaxvalue >= dz`. Its deterministic 32×35 fixture includes border obstructions and isolated 120 m and 70 m blockers.

At 30° and 73°, the translated scan changes 9 and 24 building-shadow classifications. Cardinal and exact-diagonal cases preserve the tested binary masks, but their stored float32 horizon values differ by up to `6.10e-05` because repeated decrement reassociates the frozen `ds * index * tas` calculation. Such a stored horizon is not exact for arbitrary later threshold queries without a separately proved arithmetic representation.

The Python construction diagnostic used 4,480 bytes for one float32 horizon. Exact enumeration took 6.1–9.5 ms, the invalid translated scan 0.48–0.54 ms, and 10,000 threshold queries 9.7–10.2 ms on this small fixture. These are mechanism diagnostics, not end-to-end performance evidence.

A more complex phase-indexed recurrence may remain viable if it reproduces every rounded sample and the original floating operation order. This experiment rejects only the ordinary one-predecessor translation recurrence. Vegetation and wall outputs were outside scope.

## Reproduction

```bash
python3 tools/experiments/p7_exact_horizon.py \
  --output reports/characterization/p7_horizon_experiment/result.json \
  --fixture reports/characterization/p7_horizon_experiment/minimal_counterexample.npz
```

The existing original-reference scope is recorded in `tests/reference/wall_shadows_original_cpu/manifest.json`; this experiment does not regenerate or replace that oracle.

## Reference binding revision

`result_v2_reference_bound.json` is the authoritative result. The earlier `result.json` is retained as an explicitly unbound exploratory result and must not be used as acceptance evidence.

Version 2 imports `shadow_numpy` directly from `src/solweig_light/geometry/shadows.py`, invokes it with float32 DSM and zero vegetation fields, and asserts that `(sunlight_output == 0)` equals the explicit horizon shadow mask for every minimal and deterministic-family case. All assertions pass. That candidate serial recurrence is already covered by the frozen original-CPU shadow reference scope in `tests/reference/wall_shadows_original_cpu/manifest.json`; this experiment does not claim a new upstream oracle.

Actual operand policy is preserved: Python angle inputs are converted to float32 before degree-to-radian multiplication, offsets begin as float32, NumPy uses ties-to-even rounding, and the vertical decrement is cast to DSM float32 at subtraction. Cardinal and diagonal observations establish sampled binary-mask agreement only. They do not establish an exact reusable horizon value or an approved specialization.
