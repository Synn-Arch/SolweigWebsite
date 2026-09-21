# P7 exact-coefficient UTCI Horner experiment

Decision: **reject_horner_evaluator**. The frozen numerical gate remained `0.02 °C`.

The experiment loaded all 211 mechanically extracted coefficient/exponent terms, called the checked-in explicit evaluator, and compared against the hash-verified original-upstream CPU fixture. It is a polynomial-component diagnostic, not an end-to-end performance claim.

Frozen upstream-accepted cases: 10762; numerical cancellation-neighbour cases: 8015. These labels describe executable source-test filtering, not the published UTCI applicability domain.
Horner vs original CPU max/RMS: 0.0310058594 / 0.00115029063 °C.
Horner vs explicit on cancellation neighbours max/RMS: 0.0296020508 / 0.0033135878 °C.
Uniform Ta/Pa specialization was bitwise equal to generic Horner: `True`.

Raw construction/runtime trials and provenance are in `result.json`. Timings are diagnostics on this host and do not establish a full-workload benefit.
