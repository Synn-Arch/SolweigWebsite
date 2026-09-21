# P8 UTCI near-threshold stress-category diagnostic design

Status: source audit and test design only; no numerical comparison executed and no
category API authored.

## What the product and upstream actually expose

The compatibility requirement is continuous UTCI, not a categorical output.
Candidate `src/solweig_light/pipeline.py` lines 267--269 evaluates UTCI, masks
invalid recipients, and places the continuous array in the `UTCI` output.  The
writer emits that array as the `UTCI_<tile>.tif` bands.  Pinned upstream
`.upstream/SOLWEIG-GPU/solweig_gpu/utci_process.py` lines 706--710 and 739--750
does the same: it evaluates, masks, appends, and writes float32 UTCI bands.  Neither
path assigns stress categories, writes category codes, or branches model behavior
on a category.

Candidate `src/solweig_light/comfort/utci.py` and pinned upstream
`calculate_utci.py` each contain the same informal four-band interpretation only
in the `utci_calculator` docstring (`<9`, `9--26`, `26--32`, `>32`).  No executable
code consumes those bands.  That text is also coarser than the official ten-class
assessment scale.  It cannot be promoted into a compatibility contract merely
because it appears in a docstring.

Therefore plan section 10.5's phrase “where ... stress categories are used
downstream” is conditional.  No such downstream use exists in this repository or
the pinned upstream workflow.  A category comparison is useful scientific
interpretation of continuous error near decision boundaries; it is not a new
public result, raster, enum, or pass criterion that replaces the frozen UTCI error
gate.

## Primary threshold authority still required

The independently pinned official files under
`reports/characterization/p8_utci_official_source` define the polynomial and its
validity domain, but do not provide a machine-readable category table.  Before
implementing the diagnostic, acquire and pin the official UTCI assessment-scale
document from the UTCI organization, record its URL, SHA-256, retrieval date, and
page/figure location in the existing source manifest (or a sibling manifest), and
verify that hash in the test.  The official UTCI poster describes the stress scale
and is available from [the UTCI organization](https://utci.org/resources/utci_poster.pdf);
the organization also hosts the [UTCI calculator and primary resources](https://utci.org/utci_calc.php).

Subject to verification against that pinned primary artifact, the established
ten-class breakpoints are `-40, -27, -13, 0, 9, 26, 32, 38, 46 °C`, separating
extreme cold, very strong cold, strong cold, moderate cold, slight cold, no thermal
stress, moderate heat, strong heat, very strong heat, and extreme heat.  These
numbers remain a proposed fixture inventory until the primary document is locally
pinned and checked; the calculator docstring is not sufficient authority for the
five omitted boundaries.

## Proposed independent diagnostic

Extend `tests/scientific/test_utci_independent.py`, reusing its hash-verified
official Fortran parser, `_official_utci`, and `_official_saturation_hpa`.  Do not
derive expected temperatures from candidate coefficients or outputs.

For every verified official breakpoint, construct reference-condition inputs and
solve only the air-temperature input by deterministic bisection of the official
evaluator.  Hold `Tmrt = Ta`, wind at the official lower valid limit `0.5 m/s`, and
use the official reference humidity convention (50% below 29 °C and 20 hPa vapour
pressure above it).  Bracket and solve separate official UTCI targets at offsets
`-0.03, -0.01, 0, +0.01, +0.03 °C` from each breakpoint.  Require every generated
input to remain inside the official domains already asserted by the existing test:
`-50 <= Ta <= 50`, `-30 <= Tmrt-Ta <= 70`, `0.5 <= wind <= 17`, and valid vapour
pressure.  Use a fixed bisection iteration count and require the official result to
be within a predeclared small construction residual of its target; this residual
checks fixture construction and is not a candidate accuracy gate.

Evaluate the candidate float32 interface once on those independently constructed
inputs.  Preserve the frozen continuous requirement exactly:
`abs(candidate - official) <= 0.02 °C`.  Do not tighten it for category testing,
and do not promise exact classification for a point whose official UTCI is within
`0.02 °C` (plus the declared construction residual) of any breakpoint.  A candidate
can satisfy the continuous contract and legitimately cross the interpretation
boundary there.

For points farther than that guard distance from every breakpoint, category
agreement follows from the continuous gate and should be asserted as a consistency
check.  For all points, including guarded points, record rather than suppress:

- meteorological inputs and target offset;
- official and candidate UTCI;
- signed and absolute continuous error;
- nearest breakpoint and signed distance;
- official and candidate category; and
- whether the category changed and whether the point was guarded.

Attach the complete JSON list as a JUnit `record_property`, and run with an explicit
durable report path, for example:

`pytest -q tests/scientific/test_utci_independent.py --junitxml=reports/p8_utci_near_threshold.xml`

The test should fail on a continuous error over `0.02 °C`, a category mismatch
outside the guard band, an invalid generated input, or failure to serialize every
case.  Category mismatches inside the guard band remain visible rows in the report
and do not fail solely because the label changed.  This preserves all observed
mismatches without weakening the numerical contract.

## Tie convention

After the official table is pinned, resolve an exact breakpoint toward the category
with less thermal stress (toward “no thermal stress”).  The resulting intervals are
`(-inf,-40)`, `[-40,-27)`, `[-27,-13)`, `[-13,0)`, `[0,9)`, `[9,26]`,
`(26,32]`, `(32,38]`, `(38,46]`, and `(46,+inf)`, in order from extreme cold
through extreme heat.  This is consistent with the published endpoint language
“below” for extreme cold and “above” for extreme heat, makes the adjacent ranges
disjoint and total, and avoids assigning the more severe label at an otherwise
ambiguous equality.  A single `searchsorted` side is insufficient because cold and
heat severities run in opposite directions from the no-stress interval.

The convention must still be checked against the notation in the locally pinned
primary source.  It is only a diagnostic convention.  Exact-threshold candidate
category equality is not asserted because the official evaluator's construction
residual and the frozen `0.02 °C` candidate allowance can place two conforming
values on opposite sides.

WBGT categories require their own primary source and diagnostic if an actual
downstream consumer is found.  This UTCI design does not infer WBGT thresholds from
UTCI or from the model's sun/shade calculation.
