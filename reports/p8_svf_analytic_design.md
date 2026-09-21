# P8 bounded-SVF analytic checks

Status: authored, not executed.  The deferred command is
`pytest -q tests/scientific/test_svf_analytic.py` after the P7 benchmark quiet
window ends.

## Claim and scope

`tests/scientific/test_svf_analytic.py` checks a narrow physical invariant: for
finite, valid geometry, every one of the fifteen returned sky-view-factor fields
and the combined `svftotal` must be finite and lie in `[0, 1]`, subject only to a
preselected float32 roundoff allowance.  This is independent analytic evidence; it
does not compare against candidate-generated goldens or mock the shadow recurrence.

Both dense `svf_calculator` and native `svf_calculator_compact` run the complete
default option-2 sky table: all 153 patches, their original order, every annulus
degree, and full ray support.  The test uses the characterized 9-by-13 scale-1
domain, once flat and unobstructed and once with buildings, canopy, trunk, and bush.
All elevation fields and the ray bound are finite float32 values.  Canopy and trunk
heights are nonnegative absolute surfaces, the ray bound is at least their and the
DSM's maximum, and scale is finite and positive.

## Frozen numerical allowance

The option-2 annulus widths and counts are fixed in `geometry/shadows.py`.  A whole
SVF field receives

`12 * (31 + 30 + 28 + 24 + 19 + 13 + 7) + 6 * 1 = 1830`

float32 updates.  With unit roundoff `u = 2^-24`, conservatively charging one
rounding for contribution formation and one for accumulation gives
`gamma_(2n) = 2n*u / (1 - 2n*u)`, approximately `2.182e-4`.  The gate is fixed at
`2.5e-4`, rounded upward before execution.  This is a representation allowance,
not permission for materially negative or super-unity view factors.  The source
upper-clips each of the fifteen component fields at one; `svftotal` is a derived
subtraction and is therefore checked with the same symmetric allowance.

The test contains an executable arithmetic assertion tying the allowance to the
default patch count and update count.  It must not be changed in response to test
output without a separately reviewed numerical argument.

## Open-scene boundary

The flat scene has `a = vegdem = vegdem2 = bush = 0`.  For this model, the two
vegetation visibility masks are one for every patch on that scene, so the ten
returned vegetation and combined-vegetation SVF fields are expected to be unity
within the preselected accumulation allowance.  The indices are derived from the
executed return order in `geometry/svf.py`, not its historical docstring.

No unity claim is made for the five building-only SVF fields.  The preserved model
has a documented exact-zenith failure on flat zero-elevation ground: some building
visibility cells are zero for that patch.  The full default workload necessarily
contains the 90-degree patch.  Requiring those fields to equal one would silently
reinterpret that unresolved behavior.  They remain subject to the finite physical
bounds check, and this test neither repairs nor excuses the exact-zenith failure.

The three visibility matrices are excluded from the `[0, 1]` SVF assertion.  They
are categorical intermediates rather than view-factor outputs, and existing
evidence includes value `2` in `vbshvegsh` for valid short-ray fixtures.  Treating
them as Boolean would contradict the frozen visibility policy.

Option 4 is also outside this test.  Its full SVF calculation has a preserved exact
TypeError from floating annulus bounds.  The bounded-scene claim covers the normal
default option-2 public workload and does not reinterpret that known failure.
