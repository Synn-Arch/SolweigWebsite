"""Independent elementary radiation expectations on a genuine night input.

The expected values below come from physical definitions and branch
preconditions. Captured candidate or upstream outputs are never loaded.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from solweig_light.geometry.visibility import LazyDiffVisibility, PackedVisibility
from solweig_light.radiation import engine


REFERENCE = Path(__file__).parents[1] / "reference/state_sequence_original_cpu"
MANIFEST = json.loads((REFERENCE / "boundaries/manifest.json").read_text())
# Compatibility physics uses the historical SOLWEIG constant. Changing it is a
# scientific-policy decision, so this independent check states it explicitly.
STEFAN_BOLTZMANN_COMPAT = 5.67051e-8  # W m^-2 K^-4
CELSIUS_ZERO_KELVIN_COMPAT = 273.15


def _load_genuine_night_input():
    event = next(
        item for item in MANIFEST["events"]
        if item["boundary"] == "input" and item["timestep"] == 0
    )
    path = REFERENCE / "boundaries" / event["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == event["sha256"]
    values = {}
    with np.load(path, allow_pickle=False) as archive:
        for name, spec in event["fields"].items():
            if "/" in name:
                continue
            if spec["kind"] == "dict":
                values[name] = {
                    key: archive[f"{name}/{key}"].item() for key in spec["keys"]
                }
            elif spec["kind"] == "none":
                values[name] = None
            elif spec["kind"] == "list":
                assert spec["length"] == 0
                values[name] = []
            else:
                value = archive[name].copy()
                original_type = spec["original_python_type"]
                if original_type.startswith("builtins."):
                    value = value.item()
                elif original_type.startswith("numpy.") and spec["kind"] == "scalar":
                    value = value[()]
                values[name] = value
    for name in ("shmat", "vegshmat", "vbshvegshmat"):
        values[name] = PackedVisibility.from_dense(values[name])
    values["diffsh"] = LazyDiffVisibility(values["shmat"], values["vegshmat"])
    return values


def _run_genuine_night_case():
    arguments = _load_genuine_night_input()
    # These assertions define the physical domain of both expectations. A
    # changed fixture must fail here instead of silently testing daylight or a
    # water-temperature override.
    assert float(arguments["altitude"]) <= 0.0
    # Land-cover handling may be enabled without any water pixels.
    assert int(arguments["landcover"]) in (0, 1)
    assert not np.any(np.asarray(arguments["lc_grid"]) == 3)
    for name in ("radG", "radD", "radI"):
        assert float(arguments[name]) == 0.0
    assert np.all(np.asarray(arguments["emis_grid"]) >= 0.0)
    assert np.all(np.asarray(arguments["emis_grid"]) <= 1.0)
    assert float(arguments["Ta"]) + CELSIUS_ZERO_KELVIN_COMPAT > 0.0
    with np.errstate(all="ignore"):
        return arguments, engine.Solweig_2022a_calc(**arguments)


def test_no_shortwave_flux_below_horizon():
    """With the sun below the horizon and zero solar forcing, flux is zero."""
    _, result = _run_genuine_night_case()
    # Kdown, Kup, four cardinal fluxes, direct/diffuse cylindrical side
    # components, anisotropic diffuse contribution, and total Kside.
    shortwave_indices = (1, 2, 19, 20, 21, 22, 27, 36, 37, 38)
    for index in shortwave_indices:
        actual = np.asarray(result[index])
        np.testing.assert_array_equal(actual, np.zeros_like(actual))


def test_night_upward_longwave_obeys_stefan_boltzmann_law():
    """An isothermal non-water surface emits εσT⁴ in W m^-2."""
    arguments, result = _run_genuine_night_case()
    temperature_kelvin = float(arguments["Ta"]) + CELSIUS_ZERO_KELVIN_COMPAT
    expected = (
        np.asarray(arguments["emis_grid"], dtype=np.float64)
        * STEFAN_BOLTZMANN_COMPAT
        * temperature_kelvin**4
    )
    actual = np.asarray(result[4], dtype=np.float64)  # Lup, W m^-2
    # The candidate's raster path uses float32 intermediates. At terrestrial
    # flux magnitudes, rtol=2e-6 plus 2e-4 W m^-2 bounds accumulated float32
    # rounding while remaining far below observational/model uncertainty.
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-4)
