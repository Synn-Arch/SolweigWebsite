# Separately pinned UMEP thermal-delay comparison

Status: authored, unexecuted during the P7 benchmark quiet window.

The reference is `TsWaveDelay_2015a.py` from UMEP revision
`3fcc0c3dca67d1d5644a6d34b9148d7a365743ba`, recorded in
`characterization/p8_umep_source/manifest.json`. The test checks its source hash
before loading the unchanged NumPy routine. No candidate-generated expected
outputs are used.

Source inspection matches the first-morning reset, 59-minute accumulated-time
branch, exponential coefficient 33.27 and returned carried-state semantics.
The test advances two independent eight-step histories for each of 10, 30, 59
and 60-minute steps, with float32 and float64 fields. This exercises accumulation,
threshold equality and above-threshold behavior with nonuniform forcing.

The array tolerance is the existing original-reference delay gate: absolute
0.01, zero relative tolerance. Returned time is compared exactly. These gates
were selected before execution; a failure requires diagnosis, not relaxation.

This is a matched component check, not independent validation of the physical
delay model or parity with the full UMEP pipeline. Shared model ancestry remains
explicit. Ground-view and radiation source matching remain outstanding.

Deferred command:
`.venv-wheel/bin/python -m pytest tests/scientific/test_umep_delay_independent.py -q --junitxml=reports/p8_umep_delay_tests.xml`
