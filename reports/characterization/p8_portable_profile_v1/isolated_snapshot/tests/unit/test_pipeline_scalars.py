"""Scalar edge semantics retained by the chronological driver."""
import numpy as np
import pytest
from solweig_light.pipeline import _dem_median


@pytest.mark.parametrize('values,expected', [([1, 4, 2, 3], 2), ([-5, -2, -1], -2),
                                           ([1, np.inf], 1), ([-np.inf, 2], -np.inf)])
def test_dem_median_lower_middle(values, expected):
    assert _dem_median(np.array(values, dtype=np.float32)) == expected


@pytest.mark.parametrize('position', range(6))
def test_any_dem_nan_propagates(position):
    dem = np.arange(6, dtype=np.float32).reshape(2, 3)
    dem.flat[position] = np.nan
    assert np.isnan(_dem_median(dem))
