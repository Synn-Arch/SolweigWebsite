"""Compare carried thermal state with a separately pinned UMEP component."""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from solweig_light.radiation.engine import TsWaveDelay_2015a


@pytest.fixture(scope="module")
def umep_delay():
    path = (Path(__file__).resolve().parents[2] / "reports/characterization/"
            "p8_umep_source/TsWaveDelay_2015a.py")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "5310481421e5f144e7a813e1ab41873a48cdf47b7e294f481aa59bd0ae9ed034")
    spec = importlib.util.spec_from_file_location("pinned_umep_delay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TsWaveDelay_2015a


@pytest.mark.parametrize("minutes", [10, 30, 59, 60])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_umep_delay_chronological_state(umep_delay, minutes, dtype):
    # Both histories advance independently; candidate state never feeds the oracle.
    candidate_state = np.full((2, 3), 280, dtype=dtype)
    reference_state = candidate_state.copy()
    candidate_time = reference_time = 0.0
    step = minutes / 1440
    for index, level in enumerate([300, 340, 410, 380, 300, 260, 290, 360]):
        forcing = np.asarray([[level, level + 10, level - 20],
                              [level + 40, level - 30, level + 5]], dtype=dtype)
        actual = TsWaveDelay_2015a(forcing.copy(), int(index == 0),
                                   candidate_time, step, candidate_state.copy())
        expected = umep_delay(forcing.copy(), int(index == 0),
                              reference_time, step, reference_state.copy())
        # Existing original-reference delay gate, fixed before this test executes.
        for field in (0, 2):
            assert np.asarray(actual[field]).shape == expected[field].shape
            np.testing.assert_allclose(actual[field], expected[field], rtol=0, atol=.01)
        np.testing.assert_array_equal(actual[1], expected[1])
        _, candidate_time, candidate_state = actual
        _, reference_time, reference_state = expected
