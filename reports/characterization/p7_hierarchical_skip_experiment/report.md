# P7 conservative hierarchical skipping experiment

## Result

The building-only bound is correct, but this Python quadtree design is rejected for promotion. It preserves every exact rounded ray index and evaluates non-skipped samples in the frozen order. Across 15 fixture/azimuth combinations, its float32 horizon is bit-identical to exhaustive enumeration and its mask equals the repository's characterized `shadow_numpy` output.

For each consecutive four-sample segment, the hierarchy queries the maximum source height over a rectangle enclosing all exact sample coordinates. Because vertical decrement increases with sample index, `rectangle_max − float32(first_valid_dz)` upper-bounds every exact `source_height − float32(dz)` candidate. A segment is skipped only when that bound cannot exceed the receiver's current horizon. Cells inside the rectangle but outside the ray only loosen the bound.

## Measurements

The sparse fixture skipped 97.8–99.4% of exact sample visits; the dense fixture skipped 41.6–66.1%; the border/tall-obstruction fixture skipped 77.4–87.1%. Despite this, hierarchy queries visited a median 8.91 tree nodes per baseline exact sample. Diagnostic runtime was about 7.2–7.4× slower than direct enumeration.

Each 32×35 hierarchy contained 1,557 nodes. Its maxima alone occupy 6,228 bytes; measured Python node/list/scalar storage had a 219,508-byte lower bound. Construction took 2.6–3.9 ms. These Python measurements identify overhead and are not end-to-end speed claims. A packed compiled range-maximum structure could have different costs and would require a new experiment.

## Scope boundary

This proof covers opaque building height only. It cannot be extended to vegetation or wall outputs from the same scalar bound: canopy and trunk intervals, vegetation/building ordering, accumulated vegetation state, wall height, wall sun, and face masks can change even when the building horizon cannot. Each output would require a simultaneous conservative bound and unchanged exact-index fallback.

The experiment imports `shadow_numpy` from `src/solweig_light/geometry/shadows.py` and asserts full-mask equality. That kernel's original-CPU reference scope is recorded by `tests/reference/wall_shadows_original_cpu/manifest.json`; no oracle was regenerated.

## Reproduction

```bash
python3 tools/experiments/p7_hierarchical_skip.py \
  --output reports/characterization/p7_hierarchical_skip_experiment/result.json
```
