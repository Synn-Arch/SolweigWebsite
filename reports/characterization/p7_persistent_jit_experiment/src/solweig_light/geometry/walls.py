#SOLWEIG-GPU: GPU-accelerated SOLWEIG model for urban thermal comfort simulation
#Copyright (C) 2022–2025 Harsh Kamath and Naveen Sudharsan

#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
import numpy as np
import math
from scipy.ndimage import rotate

def findwalls_serial(dem_array, walllimit):
    """Exact four-neighbor stencil; float64 subtraction and output."""
    from .walls_compiled import walls_serial
    dem_array = np.asarray(dem_array)
    if 0 in dem_array.shape:
        raise IndexError("wall stencil requires nonempty raster axes")
    return walls_serial(dem_array, walllimit)


def findwalls_parallel(dem_array, walllimit):
    """Pixel-owned parallel stencil using the configured Numba thread budget."""
    from .walls_compiled import walls_parallel
    dem_array = np.asarray(dem_array)
    if 0 in dem_array.shape:
        raise IndexError("wall stencil requires nonempty raster axes")
    return walls_parallel(dem_array, walllimit)


def findwalls(dem_array, walllimit):
    """Public compatibility stencil; serial promotion precedes benchmarking."""
    return findwalls_serial(dem_array, walllimit)


def cart2pol(x, y, units='deg'):
    """
    Convert Cartesian coordinates to polar coordinates.
    
    Args:
        x (np.ndarray or float): X coordinate(s)
        y (np.ndarray or float): Y coordinate(s)
        units (str): Output angle units ('deg' or 'rad'). Default: 'deg'
    
    Returns:
        tuple: (theta, radius) where theta is angle and radius is distance
    """
    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if units in ['deg', 'degs']:
        theta = theta * 180 / np.pi
    return theta, radius

def get_ders(dsm, scale):
    """
    Calculate slope derivatives (aspect and gradient) from DSM.
    
    Args:
        dsm (np.ndarray): Digital Surface Model array
        scale (float): Pixel size in meters
    
    Returns:
        tuple: (aspect, gradient) where:
            - aspect: slope orientation in radians
            - gradient: slope magnitude
    """
    dx = 1 / scale
    fy, fx = np.gradient(dsm, dx, dx)
    asp, grad = cart2pol(fy, fx, 'rad')
    grad = np.arctan(grad)
    asp = -asp
    asp[asp < 0] += 2 * np.pi
    return grad, asp

# Cache identity includes scalar type/value and the filter-generation policy.
# Arrays are backed by immutable bytes; callers cannot re-enable writing.
from dataclasses import dataclass
from functools import lru_cache
import scipy

FILTER_POLICY_VERSION = "goodwin-v1-scipy-" + scipy.__version__


@dataclass(frozen=True)
class FilterTables:
    filtersize: int
    score_pointers: np.ndarray
    score_rows: np.ndarray
    score_cols: np.ndarray
    score_coefficients: np.ndarray
    side_pointers: np.ndarray
    side_rows: np.ndarray
    side_cols: np.ndarray


def _immutable(values, dtype):
    array = np.asarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _scale_key(scale):
    if isinstance(scale, float):
        return ('float', scale.hex())
    if isinstance(scale, int):
        return ('int', scale)
    array = np.asarray(scale)
    if array.ndim != 0:
        raise TypeError("scale must be scalar")
    return ('numpy', array.dtype.str, array.tobytes())


@lru_cache(maxsize=8)
def _cached_filters(key, version):
    if key[0] == 'float':
        scale = float.fromhex(key[1])
    elif key[0] == 'int':
        scale = key[1]
    else:
        scale = np.frombuffer(key[2], dtype=key[1])[0]
    filtersize = int(np.floor((scale + 1e-10) * 9))
    if filtersize <= 2:
        filtersize = 3
    elif filtersize != 9 and filtersize % 2 == 0:
        filtersize += 1
    n = filtersize - 1
    ceil = int(np.ceil(filtersize / 2.))
    half = int(np.floor(filtersize / 2.))
    line = np.zeros((filtersize, filtersize))
    build = np.zeros((filtersize, filtersize))
    line[:, ceil - 1] = 1
    build[ceil - 1, :half] = 1
    build[ceil - 1, ceil:] = 2
    pointers = [0]
    score_rows, score_cols, coefficients = [], [], []
    side_pointers = []
    side_rows, side_cols = [], []
    for h in range(180):
        score = np.round(rotate(line, h, order=1, reshape=False, mode='nearest'))
        sides = np.round(rotate(build, h, order=0, reshape=False, mode='nearest'))
        if h in [150, 30]:
            sides[:, n] = 0
        index = 270 - h
        if index == 225:
            score[0, 0] = score[n, n] = 1
        if index == 135:
            score[0, n] = score[n, 0] = 1
        rr, cc = np.nonzero(score)
        score_rows.extend(rr - half)
        score_cols.extend(cc - half)
        coefficients.extend(score[rr, cc])
        pointers.append(len(score_rows))
        bounds = [len(side_rows)]
        for side in [1, 2]:
            rr, cc = np.nonzero(sides == side)
            side_rows.extend(rr - half)
            side_cols.extend(cc - half)
            bounds.append(len(side_rows))
        side_pointers.append(bounds)
    return FilterTables(filtersize, _immutable(pointers, np.int64),
                        _immutable(score_rows, np.int32), _immutable(score_cols, np.int32),
                        _immutable(coefficients, np.float64), _immutable(side_pointers, np.int64),
                        _immutable(side_rows, np.int32), _immutable(side_cols, np.int32))


def filter_tables(scale):
    """Bounded immutable table cache, keyed by exact scale and policy version."""
    return _cached_filters(_scale_key(scale), FILTER_POLICY_VERSION)


def _aspect(walls, scale, a, parallel):
    from .walls_compiled import angles_serial, angles_parallel
    tables = filter_tables(scale)
    walls = (walls > 0).astype(np.uint8)
    kernel = angles_parallel if parallel else angles_serial
    selected = kernel(walls, tables.filtersize // 2, tables.score_pointers,
                      tables.score_rows, tables.score_cols, tables.score_coefficients)
    y = np.zeros(a.shape)
    for row, col in zip(*np.nonzero(selected >= 0)):
        h = selected[row, col]
        start, middle, end = tables.side_pointers[h]
        # Keep original masked row-major NumPy pairwise reductions. Replacing
        # them with sequential compiled sums can reverse the selected side.
        side1 = np.sum(a[row + tables.side_rows[start:middle], col + tables.side_cols[start:middle]])
        side2 = np.sum(a[row + tables.side_rows[middle:end], col + tables.side_cols[middle:end]])
        index = 270 - h
        if side1 > side2:
            index -= 180
        if index < 0:
            index += 360
        y[row, col] = index
    # Preserve the entire expression: false-mask multiplication still
    # propagates NaNs from derivative aspect at non-wall pixels upstream.
    grad, asp = get_ders(a, scale)
    y += ((walls == 1) & (y == 0)) * (asp / (math.pi / 180.))
    return y


def filter1Goodwin_as_aspect_v3_serial(walls, scale, a):
    return _aspect(walls, scale, a, parallel=False)


def filter1Goodwin_as_aspect_v3_parallel(walls, scale, a):
    return _aspect(walls, scale, a, parallel=True)


def filter1Goodwin_as_aspect_v3(walls, scale, a):
    """Exact sparse directional scoring with original tie and side semantics."""
    return filter1Goodwin_as_aspect_v3_serial(walls, scale, a)
