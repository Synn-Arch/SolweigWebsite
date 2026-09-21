# P8 UMEP component match: ground-view directional aggregation

Status: source inspection and test authoring only; no comparison executed.  The
unmocked comparison is authored in
`tests/scientific/test_umep_ground_view_independent.py`.  Its deferred command is
`pytest -q tests/scientific/test_umep_ground_view_independent.py` after the P7
benchmark quiet window ends.
The injected-probe case is a structural test with a mocked numerical dependency,
not independent full-physics numerical evidence.

## Proposed matched boundary

The next tractable independent comparison is `gvf_2018a`, excluding its ray/surface
subroutine.  The unchanged reference is
`reports/characterization/p8_umep_source/gvf_2018a.py::gvf_2018a`, acquired from
UMEP commit `3fcc0c3dca67d1d5644a6d34b9148d7a365743ba` with SHA-256
`9bfb49dd281924bff5352d57d9c28a8274d1a2c4660c4fbca532afa6de167c94` recorded in
`manifest.json`.  The candidate is
`src/solweig_light/radiation/ground_view.py::_gvf`, exposed by `gvf_2018a` and
`gvf_2018a_parallel`.

This boundary is scientifically relevant: UMEP's chronological calculation calls
`gvf_2018a` immediately before applying the already-covered temperature-wave delay
(`Solweig_2022a_calc.py`, lines 313--371).  It produces the upward longwave driver,
sunlit and unshaded reflected-shortwave factors, four directional versions of each,
and the summed/normalized ground view factor.

## Shared semantics established by source inspection

Both implementations:

- sample the 18 azimuths `5, 25, ..., 345` in that order;
- derive `sunwall` as `(wallsun / walls * buildings) == 1` and pass wall aspects in
  radians to the surface routine;
- sum outputs 2--5 from each surface call into `gvfLup`, `gvfalb`,
  `gvfalbnosh`, and `gvfSum`;
- use the same overlapping half-plane membership: east `[0,180)`, south
  `[90,270)`, west `[180,360)`, north `[270,360) union [0,90)`;
- divide whole-circle fields by 18 and directional fields by 9;
- add `SBC * emis_grid * (Ta + 273.15)**4` to the whole-circle and four
  directional longwave fields only; and
- return the same 17 fields in the same order, with `gvfNorm = gvfSum / 18` and
  `gvfNorm[buildings == 0] = 1`.

The candidate deliberately differs in execution representation.  Its optimized
path requires selected raster inputs to be `float32`, creates a `float32` azimuth
array, uses backend arithmetic helpers, and can call `_sun` with either serial or
parallel pixel execution.  UMEP uses NumPy's default integer azimuth and `float64`
zero arrays.  Therefore source identity supports a shared algorithm, but does not
justify bitwise equality for arbitrary floating inputs.  Also, this boundary test
does not validate ray traversal, surface emission/reflection, or mutation behavior
inside `sunonsurface_2018a`.

## Full surface implementation now available

The unchanged UMEP dependency is now pinned as
`reports/characterization/p8_umep_source/sunonsurface_2018a.py` from the same
commit, SHA-256
`b5abe6568ffad18781b1011e13681238c63a2ebfba21c11c61e40d986089e089`, 9,396
bytes.  Its only AST import is NumPy.  The manifest records this acquisition; the
status remains source acquisition and matching pending execution.

Static comparison with `ground_view.py::_sun` shows the same model equations and
operation sequence: temperature-dependent ground and wall longwave excess,
optional class-3 water mutation, shadowed albedo, rounded `first`/`second` ray
lengths, direction-dependent shifted slices, persistent temporary fields outside
the current slice, cumulative building minima, wall-influence latches, the three
self-shadow aspect branches, `gvf2` clipping, 0.5/0.4/0.9 weighting, and the same
five return fields.  The candidate's `_gather` is a pixel-owned implementation of
the UMEP array recurrence; in particular, it intentionally preserves UMEP's
otherwise non-obvious stale shifted values outside each new slice and the
effectively unnormalized `first` snapshots (`ind` is reset to 1 in every UMEP loop
iteration).

Compatibility differences that the executable comparison must respect are:

- UMEP allocates its temporary and accumulator arrays as NumPy `float64`; the
  candidate optimized path snapshots dynamic inputs and accumulates as `float32`.
- UMEP's azimuth is normally derived from an integer degree scalar in the outer
  function.  The candidate outer function uses `float32` azimuths and typed angular
  constants, so trigonometry and equality boundaries can differ by float32
  rounding.
- The candidate validates finite ray lengths and a positive rounded `second`, and
  routes unsupported dtypes/profiles to its characterized fallback.  UMEP has no
  corresponding validation and may instead fail later through unbound first-range
  variables, invalid loop ranges, or slice assignment.
- The candidate precomputes and caches immutable slice descriptors and explicitly
  rejects incompatible source/destination shapes.  UMEP relies on NumPy slice
  assignment behavior.
- UMEP mutates positive `sunwall` values to one and, for `landcover == 1`, mutates
  class-3 cells of `Tg` to `Twater - Ta`.  The candidate preserves both mutations
  and their ordering (the current call's `Lup` is computed before the water
  mutation).  Fresh argument copies are therefore required for each side, and
  before/after state is part of the comparison.
- Candidate serial and parallel modes share the equations but use different
  pixel-loop kernels.  Both require comparison to the single UMEP result.

## Minimal executable comparison

Add a focused test that loads the pinned UMEP file directly from its recorded path,
after verifying its SHA-256.  Give the temporary package a minimal
`sunonsurface_2018a` module so the relative import succeeds; do not copy or rewrite
the UMEP function.  After import, replace UMEP's `sunonsurface_2018a` and the
candidate module's `_sun` with adapters around one test-local deterministic probe.
The candidate adapter accepts and checks the extra `parallel` keyword.  The probe
should record all calls and return five small arrays whose values depend separately
on azimuth and pixel position.

Use a 2-by-3 fixture with mixed building/non-building cells, nonzero walls (to avoid
making divide-by-zero warning policy part of this test), nonuniform emissivity and
air temperature, and `float32` candidate raster inputs so the optimized `_gvf` path
is exercised.  Execute UMEP `gvf_2018a`, candidate `gvf_2018a`, and candidate
`gvf_2018a_parallel`.  Assert:

1. each path made exactly 18 calls in the frozen azimuth order;
2. `sunwall`, radian aspects, and all forwarded arguments agree after accounting
   for the documented scalar dtype difference;
3. all 17 candidate fields agree with the pinned UMEP fields, comparing candidate
   precision against `reference.astype(candidate.dtype)`; and
4. serial and parallel candidate results agree field by field.

Use exactly representable probe values and first run a structural case with
`SBC = 0`; this can demand exact equality after the explicit reference cast.  Add a
second zero-probe case with nonuniform `emis_grid`, `Ta`, and physical `SBC` to
isolate the longwave baseline.  Select its tolerance before execution from the
existing frozen longwave contract and a documented rounding-error argument.  Do
not calibrate acceptance to observed candidate disagreement; no tolerance is
claimed by this inspection.  Compute independent expected directional membership
sums in the test as an additional assertion, so agreement cannot be caused by
applying the same bad probe convention to both sides.

Suggested location: `tests/scientific/test_umep_gvf_aggregation_independent.py`.
The test must read the pinned UMEP file at runtime and must not store candidate
outputs as goldens.

## Unmocked full-physics proposed next step

The required UMEP sources are now present, so a second test can load the unchanged
`sunonsurface_2018a.py` first and then load unchanged `gvf_2018a.py` in a temporary
package whose relative import resolves to that real module.  Verify both hashes
against `manifest.json` before execution.  Do not inject a numerical probe in this
test.

Use a deterministic rectangular fixture (at least 9 by 11) with independent spatial
variation: binary `float32` `buildings`; finite positive `float32` `walls` to keep
the outer `wallsun / walls` operation out of divide-by-zero policy; `wallsun` equal
to either zero or the corresponding wall value; shadow values including 0,
fractional vegetation transmission, and 1; nonuniform `Tg`, `emis_grid`, and
`alb_grid`; scalar `float32` `Tgwall`; wall directions spanning the wraparound and
ordinary aspect branches; and a nonuniform integer `lc_grid`.  Use `scale = 1`,
`first = 2`, and `second = 4`, so rounded ray lengths are positive, finite, and
smaller than both raster dimensions.  Use physical finite values for `Ta`, `ewall`,
`SBC`, `albedo_b`, and `Twater`.

Run two cases from fresh deep copies: `landcover = 0`, then `landcover = 1` with at
least one class-3 cell.  For each, compare UMEP `gvf_2018a` with candidate serial
and parallel `gvf_2018a`; also compare direct UMEP `sunonsurface_2018a` with both
candidate modes at representative azimuths from all three aspect branches and both
ray-schedule branches (the frozen 5-degree sequence supplies such angles).  Assert
all return shapes, finite/nonfinite masks, all 17 outer fields or five direct
fields, and the observable `Tg`/`sunwall` before-after state.  Preserve the existing
frozen ground-view gates already used by
`tests/differential/test_ground_view_compiled.py`: longwave flux fields use
`rtol=1e-5, atol=0.05`; non-flux fields use the frozen exact comparison where dtype
permits, otherwise the existing stronger `atol=1e-6, rtol=0` angular gate.  These
limits are fixed before this UMEP comparison and must not be recalibrated after
observing results.

The fixture preconditions intentionally exclude zero/negative/nonfinite rounded ray
lengths, non-float32 optimized inputs, zero walls in the outer aggregation, and
pathological slice broadcasting.  Those cases exercise differing validation or
failure policy rather than the shared physical calculation and should be recorded
separately if tested.

This proposal does not compare the entire `Solweig_2022a_calc` workflow, establish
scientific correctness independently of UMEP, or validate other UMEP versions.
The lower-value structural test remains an unauthored mocked design.  If the
authored unmocked test passes,
it would provide pinned UMEP evidence for this ground-view component, including its
ray/surface calculation, within the explicit fixture domain and frozen gates.
