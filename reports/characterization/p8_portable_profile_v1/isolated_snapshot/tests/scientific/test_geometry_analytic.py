"""Independent geometric expectations; no captured upstream outputs.

These intentionally avoid exact grazing rays and domain boundaries. They
verify simple geometry, not the model's empirical environmental accuracy.
"""
import numpy as np
import pytest

from solweig_light.geometry.shadows import shadow
from solweig_light.geometry.walls import findwalls


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_isolated_building_four_adjacent_wall_heights(dtype):
    # A 3 m square cell on level 20 m terrain has four 3 m exposed faces.
    # Diagonal receivers do not share an edge with the building.
    terrain = np.full((9, 9), 20, dtype=dtype)
    terrain[4, 4] = 23
    expected = np.zeros((9, 9))
    expected[3, 4] = expected[5, 4] = 3
    expected[4, 3] = expected[4, 5] = 3
    np.testing.assert_array_equal(findwalls(terrain, 1), expected)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('azimuth,delta', [(90, (0, -1)), (270, (0, 1)),
                                        (180, (-1, 0)), (0, (1, 0))])
def test_cardinal_shadow_length_and_obstacle_monotonicity(dtype, azimuth, delta):
    # At 45 degrees the shadow length in metres equals obstacle height.
    # Half-integer heights keep every sampled receiver away from grazing.
    # Rows increase southwards, columns eastwards; shadows oppose the sun.
    previous = np.ones((17, 17), dtype=np.float32)
    for height in (1.5, 2.5, 3.5):
        terrain = np.zeros((17, 17), dtype=dtype)
        terrain[8, 8] = height
        zeros = np.zeros_like(terrain)
        actual, vegetation, combined = shadow(
            dtype(height), terrain, zeros, zeros, zeros,
            float(azimuth), 45., 1.)
        expected = np.ones_like(previous)
        for distance in range(1, int(height) + 1):
            expected[8 + distance * delta[0], 8 + distance * delta[1]] = 0
        np.testing.assert_array_equal(actual, expected)
        assert np.all(actual <= previous)
        np.testing.assert_array_equal(vegetation, np.ones_like(vegetation))
        np.testing.assert_array_equal(combined, np.ones_like(combined))
        previous = actual


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('azimuth,altitude', [(35., 15.), (125., 35.), (215., 70.)])
def test_unobstructed_nonzenith_sky_is_visible(dtype, azimuth, altitude):
    # Nonnegative flat ground, no canopy, and a sun away from exact zenith.
    # The inherited exact-zenith failure is recorded separately, not skipped
    # or silently treated as a passing physical expectation here.
    zeros = np.zeros((13, 15), dtype=dtype)
    outputs = shadow(dtype(0), zeros, zeros, zeros, zeros, azimuth, altitude, 1.)
    for actual in outputs:
        np.testing.assert_array_equal(actual, np.ones_like(actual))
