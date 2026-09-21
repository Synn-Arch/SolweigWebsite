# Independent official-source UTCI checks — authored, unexecuted

Status: **authored but not executed** during the paired benchmark quiet window.
No test, model import, compiler, build, or numerical workload was run. No source,
benchmark, existing gate, or frozen evidence was changed.

The tests verify both extracted official text files against
`reports/characterization/p8_utci_official_source/source_manifest.json`, whose
archive SHA256 is `83ea34dc2428093c8b0f4e299bfb0a24752ff49ec1e11741927a75dafae283f1`.
The official files contain legacy non-UTF8 bytes and are decoded explicitly as
Latin-1 after hash verification. The tests parse the 211 official Fortran
polynomial terms directly, retaining each written factor sequence. Expected UTCI values are produced by that independent
parser/evaluator; candidate helpers never generate expected values.

Three claims are separated:

1. **Coefficient identity.** Every coefficient and exponent tuple in the
   candidate coefficient manifest must exactly equal the official Fortran
   source. Decimal parsing avoids treating close binary floats as equivalent.
2. **Pressure-polynomial parity.** Five predeclared points cover validity
   boundaries and mixed-sign, cancellation-sensitive combinations of air
   temperature, radiant-temperature difference, wind, and vapour pressure.
   The runtime float32 `polynomial_scalar` is compared with the independent
   official double-precision evaluator under the already frozen `0.02 °C`
   UTCI gate. The focused points do not create a tighter continuous-domain
   guarantee.
3. **RH-interface behavior.** Four valid-domain points convert RH with an
   independent implementation of the official Hardy saturation-pressure
   equation before official polynomial evaluation. The candidate public RH
   interface uses its characterized float32 conversion path. This comparison
   therefore retains the already frozen `0.02 °C` full-UTCI gate rather than
   attributing RH-to-pressure rounding to the polynomial coefficients.

The official README limits the approximation to air temperature -50..50 °C,
radiant temperature difference -30..70 °C, wind 0.5..17 m/s, and vapour
pressure at most 50 hPa (also constrained by RH <=100%). Each pressure case also asserts pressure does not exceed independently
evaluated saturation pressure at its air temperature; the pressure and RH
cases remain inside all stated bounds.

The inherited SOLWEIG workflow floors effective wind at 0.15 m/s for WBGT
compatibility before calling UTCI. That is below the official polynomial's
0.5 m/s validity boundary. A source-level boundary assertion records both
facts, but no official scientific-validation claim is made for 0.15..<0.5 m/s.
Changing that floor would be a separate compatibility/scientific policy.

Intended command after the benchmark owner releases the quiet window:

```sh
PYTHONPATH=src .venv-light/bin/python -m pytest -q \
  tests/scientific/test_utci_independent.py \
  --junitxml=reports/p8_utci_independent_tests.xml
```

Until this runs and its output is inspected, these tests are not claimed as
passing and no P8 UTCI scientific gate is complete.
