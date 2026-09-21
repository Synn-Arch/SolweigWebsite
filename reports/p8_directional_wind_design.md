# Directional-wind independent expectation — authored, not executed

`tests/scientific/test_directional_wind_analytic.py` drives the public
`run_utci_tiles` workflow through all 24 chronological steps on the existing
small morphology/forcing fixture. It recomputes geometry and uses no numerical
pipeline mocks or original output arrays. Only the meteorological speed and
direction columns are replaced; the model still executes its physical path.

Twelve constant coefficient rasters give each named 30-degree sector a distinct
speed. An explicitly enumerated table checks every sector centre, both sides
of selected half-sector boundaries, north wraparound and missing directions.
The 0.15 m/s floor is checked as compatibility behavior, not endorsed as an
independently validated physical law. All other selected speeds are exactly
representable products of two and quarter-integer coefficients, justifying
exact float32 assertions before execution.

This tests directional selection and diagnostic wind propagation, not the
physical validity of the wind attenuation model or UTCI wind-height conversion.
It awaits execution after the quiet benchmark window; no passing result is
claimed. Command to execute:
`PYTHONPATH=src .venv-light/bin/python -m pytest tests/scientific/test_directional_wind_analytic.py -q`.
