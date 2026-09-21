# SOLWEIG-GPU Copyright (C) 2022–2025 Harsh Kamath and Naveen Sudharsan
# GPL version 3 or (at your option) any later version; no warranty.
"""Ownership-safe serial and parallel wall stencil/filter-score kernels."""
import numpy as np
from numba import njit, prange


@njit(inline='always', fastmath=False)
def _maximum(left, right):
    # Preserve NaN propagation and NumPy's maximum signed-zero rule on
    # the pinned platform (+0 wins over -0; all-negative zeros stay -0).
    if np.isnan(left):
        return left
    if np.isnan(right):
        return right
    if left == 0 and right == 0:
        return left + right
    if left > right:
        return left
    return right


@njit(inline='always', fastmath=False)
def _wall_pixel(dsm, row, col, walllimit):
    # Original cross selection is row-major: north, west, east, south.
    # Propagate NaNs and preserve the reduction's equal-value selection.
    maximum = _maximum(dsm[row-1,col], dsm[row,col-1])
    maximum = _maximum(maximum, dsm[row,col+1])
    maximum = _maximum(maximum, dsm[row+1,col])
    value = np.float64(maximum) - np.float64(dsm[row,col])
    if value < walllimit:
        value = 0.0
    return value


@njit(cache=True, fastmath=False)
def walls_serial(dsm, walllimit):
    rows,cols=dsm.shape
    output=np.zeros((rows,cols),dtype=np.float64)
    for row in range(1,rows-1):
        for col in range(1,cols-1):
            output[row,col]=_wall_pixel(dsm,row,col,walllimit)
    return output


@njit(cache=True, fastmath=False, parallel=True)
def walls_parallel(dsm, walllimit):
    rows,cols=dsm.shape
    output=np.zeros((rows,cols),dtype=np.float64)
    for row in prange(1,rows-1):
        for col in range(1,cols-1):
            output[row,col]=_wall_pixel(dsm,row,col,walllimit)
    return output


@njit(inline='always', fastmath=False)
def _winning_angle(walls,row,col,pointers,offset_rows,offset_cols,coefficients):
    best_score=0.0
    best_angle=-1
    for angle in range(180):
        score=0.0
        for index in range(pointers[angle],pointers[angle+1]):
            score+=walls[row+offset_rows[index],col+offset_cols[index]]*coefficients[index]
        if best_score < score:
            best_score=score
            best_angle=angle
    return best_angle


@njit(cache=True, fastmath=False)
def angles_serial(walls,half,pointers,offset_rows,offset_cols,coefficients):
    rows,cols=walls.shape
    output=np.full((rows,cols),-1,dtype=np.int16)
    # The unusual excluded two trailing rows/columns are upstream behavior.
    for row in range(half,rows-half-2):
        for col in range(half,cols-half-2):
            if walls[row,col]==1:
                output[row,col]=_winning_angle(walls,row,col,pointers,offset_rows,offset_cols,coefficients)
    return output


@njit(cache=True, fastmath=False, parallel=True)
def angles_parallel(walls,half,pointers,offset_rows,offset_cols,coefficients):
    rows,cols=walls.shape
    output=np.full((rows,cols),-1,dtype=np.int16)
    # Every worker owns one output cell; only read-only neighbor masks overlap.
    for pixel in prange(rows*cols):
        row=pixel//cols
        col=pixel%cols
        if half<=row<rows-half-2 and half<=col<cols-half-2 and walls[row,col]==1:
            output[row,col]=_winning_angle(walls,row,col,pointers,offset_rows,offset_cols,coefficients)
    return output
