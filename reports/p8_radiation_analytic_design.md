# Independent elementary radiation checks — Cura verified

Current status: **two revised tests passed on Cura**; local execution remains deferred.
The following authoring description records the initial quiet-window preparation.
No model import, numerical command, test, or benchmark was run while writing
these checks. No candidate source, frozen evidence, or model data was changed.

`tests/scientific/test_radiation_analytic.py` uses timestep 0 from the genuine
48-hour state-sequence input fixture. It verifies the fixture hash and loads
only its input boundary; captured outputs are never read. Assertions first
establish the applicable physical preconditions: solar altitude at or below
the horizon, zero global/diffuse/direct solar forcing, no land-cover water
emission override, emissivity in [0, 1], and positive absolute temperature. A
fixture change that violates these assumptions fails rather than skipping.

The night test asserts zero for downwelling and upwelling shortwave radiation,
four cardinal shortwave components, direct and diffuse cylindrical-side
components, anisotropic diffuse contribution, and total cylindrical-side
shortwave radiation. This expectation follows from absence of incident solar
energy below the horizon under zero solar forcing. It is independent of an
upstream or candidate output capture.

The longwave test independently evaluates the scalar grey-body relation

`Lup = emissivity × sigma × (Ta + 273.15 K)^4`.

It explicitly uses the compatibility model's historical Stefan–Boltzmann
constant, `sigma = 5.67051e-8 W m^-2 K^-4`. This is a check of the supported
compatibility physics, not a silent constant correction. The non-water and
isothermal-night preconditions exclude the separate water-temperature branch.
The expected array is evaluated in float64 directly from the genuine input air
temperature and emissivity. It does not call a candidate radiation helper.

The tolerance is `rtol=2e-6, atol=2e-4 W m^-2`. The model raster path uses
float32 intermediates, so this bounds accumulated representation and fourth-
power rounding around terrestrial flux magnitudes. It is not calibrated from
candidate or oracle outputs and is much smaller than the physical/model
uncertainty relevant to SOLWEIG.

After the benchmark owner declares the quiet window complete, the intended
command is:

```sh
PYTHONPATH=src .venv-light/bin/python -m pytest -q \
  tests/scientific/test_radiation_analytic.py \
  --junitxml=reports/p8_radiation_analytic_tests.xml
```

That local command remains deferred. The separately recorded Cura execution
below verifies these two tests on its pinned Linux environment.

## Cura execution and fixture-precondition repair

The first Cura run stopped both tests at the authored `landcover == 0` assertion. The hash-verified fixture instead enables land-cover handling (`landcover=1`) and contains 1,120 class-7 cells, with no class-3 water. The test now explicitly requires no water cells and a valid land-cover switch. Input data, physical expectations, and tolerances are unchanged. Initial failures and diagnostic results are retained in `cura_p8_20260919T1215Z_a7c3/results/`. Diagnostic model execution gave exact zero for all ten shortwave fields and maximum Lup error 2.0703205052541307e-05 W/m². The revised tests required a fresh execution before being counted as passing; that execution is recorded below. AST syntax validation only was run locally during the benchmark quiet window.

Cura rerun verification: `cura_p8_20260919T1215Z_a7c3/results/p8_radiation_analytic_revision_tests.xml` records two tests, zero failures/errors/skips, 2.213 s. The snapshot revision manifest binds the current test hash. Initial failures remain retained. These two elementary checks do not establish whole-model scientific validation.
