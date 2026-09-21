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
"""Serial NumPy translation of pinned SOLWEIG-GPU radiation and state.

Arithmetic helpers preserve scalar/array dtype promotion and operation order.
No approximation, parallel reduction, or physical-model alteration is applied.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import numpy as np
from math import radians
from copy import deepcopy
from osgeo import gdal, osr
import datetime
import calendar
import scipy.ndimage.interpolation as sc
from scipy.ndimage import rotate
from solweig_light.geometry.shadows import create_patches
gdal.UseExceptions()

def ensure_tensor(x, device=None):
    return np.asarray(x)

def daylen(DOY, XLAT):
    """
    Calculate day length and solar declination for given day and latitude.
    
    Args:
        DOY (np.ndarray): Day of year (1-365)
        XLAT (np.ndarray): Latitude in degrees
    
    Returns:
        tuple: (DAYL, DEC, SNDN, SNUP) where:
            - DAYL: Day length in hours
            - DEC: Solar declination in degrees
            - SNDN: Time of solar noon in hours
            - SNUP: Time of sunrise in hours
    """
    RAD = _divide(np.pi, 180.0)
    DEC = _operate(np.multiply, -23.45, np.cos(_divide(_operate(np.multiply, _operate(np.multiply, 2.0, np.pi), _operate(np.add, DOY, 10.0)), 365.0)))
    SOC = _operate(np.multiply, np.tan(_operate(np.multiply, RAD, DEC)), np.tan(_operate(np.multiply, RAD, XLAT)))
    SOC = np.clip(SOC, -1.0, 1.0)
    DAYL = _operate(np.add, 12.0, _divide(_operate(np.multiply, 24.0, np.arcsin(SOC)), np.pi))
    SNUP = _operate(np.subtract, 12.0, _divide(DAYL, 2.0))
    SNDN = _operate(np.add, 12.0, _divide(DAYL, 2.0))
    return (DAYL, DEC, SNDN, SNUP)

def sunonsurface_2018a(azimuthA, scale, buildings, shadow, sunwall, first, second, aspect, walls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, Twater, lc_grid, landcover):
    """
    Calculate solar radiation on surfaces with different orientations.
    
    Determines radiation on walls and ground surfaces accounting for
    building geometry, shadows, and surface properties.
    
    Args:
        azimuthA (float): Solar azimuth angle (degrees)
        scale (float): Grid scale (pixels per meter)
        buildings (np.ndarray): Building mask array
        shadow (np.ndarray): Shadow map
        sunwall (np.ndarray): Sunlit wall mask
        first (np.ndarray): First surface type
        second (np.ndarray): Second surface type
        aspect (np.ndarray): Wall aspect angles
        walls (np.ndarray): Wall heights
        Tg (np.ndarray): Ground temperature
        Tgwall (np.ndarray): Wall temperature
        Ta (float): Air temperature
        emis_grid (np.ndarray): Ground emissivity
        ewall (float): Wall emissivity
        alb_grid (np.ndarray): Ground albedo
        SBC (float): Stefan-Boltzmann constant
        albedo_b (float): Building albedo
        Twater (float): Water temperature
        lc_grid (np.ndarray): Land cover grid
        landcover (np.ndarray): Land cover classification
    
    Returns:
        tuple: Radiation components for different surfaces
    """
    scale = np.copy(_array(scale))
    ewall = np.copy(_array(ewall))
    albedo_b = np.copy(_array(albedo_b))
    landcover = np.copy(_array(landcover))
    sizex = walls.shape[0]
    sizey = walls.shape[1]
    wallbol = (walls > 0).astype(np.float32)
    sunwall[sunwall > 0] = 1
    azimuth = _operate(np.multiply, azimuthA, _divide(np.pi, 180))
    index = 0
    f = buildings
    Lup = _operate(np.subtract, _operate(np.multiply, _operate(np.multiply, SBC, emis_grid), _operate(np.power, _operate(np.add, _operate(np.add, _operate(np.multiply, Tg, shadow), Ta), 273.15), 4)), _operate(np.multiply, _operate(np.multiply, SBC, emis_grid), _operate(np.power, _operate(np.add, Ta, 273.15), 4)))
    if landcover == 1:
        Tg[lc_grid == 3] = _operate(np.subtract, Twater, Ta).astype(np.float32)
    Lwall = _operate(np.subtract, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, _operate(np.add, Tgwall, Ta), 273.15), 4)), _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)))
    albshadow = _operate(np.multiply, alb_grid, shadow)
    alb = alb_grid
    tempsh = _zeros((sizex, sizey))
    tempbu = _zeros((sizex, sizey))
    tempbub = _zeros((sizex, sizey))
    tempbubwall = _zeros((sizex, sizey))
    tempwallsun = _zeros((sizex, sizey))
    weightsumsh = _zeros((sizex, sizey))
    weightsumwall = _zeros((sizex, sizey))
    first = np.round(_operate(np.multiply, first, scale))
    if first < 1:
        first = 1
    second = np.round(_operate(np.multiply, second, scale))
    weightsumLupsh = _zeros((sizex, sizey))
    weightsumLwall = _zeros((sizex, sizey))
    weightsumalbsh = _zeros((sizex, sizey))
    weightsumalbwall = _zeros((sizex, sizey))
    weightsumalbnosh = _zeros((sizex, sizey))
    weightsumalbwallnosh = _zeros((sizex, sizey))
    tempLupsh = _zeros((sizex, sizey))
    tempalbsh = _zeros((sizex, sizey))
    tempalbnosh = _zeros((sizex, sizey))
    pibyfour = _divide(np.pi, 4)
    threetimespibyfour = _operate(np.multiply, 3, pibyfour)
    fivetimespibyfour = _operate(np.multiply, 5, pibyfour)
    seventimespibyfour = _operate(np.multiply, 7, pibyfour)
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)
    for n in np.arange(0, second):
        if pibyfour <= azimuth and azimuth < threetimespibyfour or (fivetimespibyfour <= azimuth and azimuth < seventimespibyfour):
            dy = _operate(np.multiply, signsinazimuth, index)
            dx = _operate(np.multiply, _operate(np.multiply, -1, signcosazimuth), np.abs(np.round(_divide(index, tanazimuth))))
        else:
            dy = _operate(np.multiply, signsinazimuth, np.abs(np.round(_operate(np.multiply, index, tanazimuth))))
            dx = _operate(np.multiply, _operate(np.multiply, -1, signcosazimuth), index)
        absdx = np.abs(dx)
        absdy = np.abs(dy)
        xc1 = _divide(_operate(np.add, dx, absdx), 2).astype(np.int32)
        xc2 = _operate(np.add, sizex, _divide(_operate(np.subtract, dx, absdx), 2)).astype(np.int32)
        yc1 = _divide(_operate(np.add, dy, absdy), 2).astype(np.int32)
        yc2 = _operate(np.add, sizey, _divide(_operate(np.subtract, dy, absdy), 2)).astype(np.int32)
        xp1 = -_divide(_operate(np.subtract, dx, absdx), 2).astype(np.int32)
        xp2 = _operate(np.subtract, sizex, _divide(_operate(np.add, dx, absdx), 2)).astype(np.int32)
        yp1 = -_divide(_operate(np.subtract, dy, absdy), 2).astype(np.int32)
        yp2 = _operate(np.subtract, sizey, _divide(_operate(np.add, dy, absdy), 2)).astype(np.int32)
        tempbu[xp1:xp2, yp1:yp2] = buildings[xc1:xc2, yc1:yc2]
        tempsh[xp1:xp2, yp1:yp2] = shadow[xc1:xc2, yc1:yc2]
        tempLupsh[xp1:xp2, yp1:yp2] = Lup[xc1:xc2, yc1:yc2]
        tempalbsh[xp1:xp2, yp1:yp2] = albshadow[xc1:xc2, yc1:yc2]
        tempalbnosh[xp1:xp2, yp1:yp2] = alb[xc1:xc2, yc1:yc2]
        f = np.minimum(f, tempbu)
        shadow2 = _operate(np.multiply, tempsh, f)
        weightsumsh += shadow2
        Lupsh = _operate(np.multiply, tempLupsh, f)
        weightsumLupsh += Lupsh
        albsh = _operate(np.multiply, tempalbsh, f)
        weightsumalbsh += albsh
        albnosh = _operate(np.multiply, tempalbnosh, f)
        weightsumalbnosh += albnosh
        tempwallsun[xp1:xp2, yp1:yp2] = sunwall[xc1:xc2, yc1:yc2]
        tempb = _operate(np.multiply, tempwallsun, f)
        tempbwall = _operate(np.add, _operate(np.multiply, f, -1), 1)
        tempbub = (_operate(np.add, tempb, tempbub) > 0).astype(np.float32)
        tempbubwall = (_operate(np.add, tempbwall, tempbubwall) > 0).astype(np.float32)
        weightsumLwall += _operate(np.multiply, tempbub, Lwall)
        weightsumalbwall += _operate(np.multiply, tempbub, albedo_b)
        weightsumwall += tempbub
        weightsumalbwallnosh += _operate(np.multiply, tempbubwall, albedo_b)
        ind = 1
        if _operate(np.add, n, 1) <= first:
            weightsumwall_first = _divide(weightsumwall, ind)
            weightsumsh_first = _divide(weightsumsh, ind)
            wallsuninfluence_first = weightsumwall_first > 0
            weightsumLwall_first = _divide(weightsumLwall, ind)
            weightsumLupsh_first = _divide(weightsumLupsh, ind)
            weightsumalbwall_first = _divide(weightsumalbwall, ind)
            weightsumalbsh_first = _divide(weightsumalbsh, ind)
            weightsumalbwallnosh_first = _divide(weightsumalbwallnosh, ind)
            weightsumalbnosh_first = _divide(weightsumalbnosh, ind)
            wallinfluence_first = weightsumalbwallnosh_first > 0
            ind += 1
        index += 1
    wallsuninfluence_second = weightsumwall > 0
    wallinfluence_second = weightsumalbwallnosh > 0
    azilow = _operate(np.subtract, azimuth, _divide(np.pi, 2))
    azihigh = _operate(np.add, azimuth, _divide(np.pi, 2))
    if azilow >= 0 and azihigh < _operate(np.multiply, 2, np.pi):
        facesh = _operate(np.add, _operate(np.subtract, np.logical_or(aspect < azilow, aspect >= azihigh).astype(np.float32), wallbol), 1)
    elif azilow < 0 and azihigh <= _operate(np.multiply, 2, np.pi):
        azilow = _operate(np.add, azilow, _operate(np.multiply, 2, np.pi))
        facesh = _operate(np.add, _operate(np.multiply, np.logical_or(aspect > azilow, aspect <= azihigh).astype(np.float32), -1), 1)
    elif azilow > 0 and azihigh >= _operate(np.multiply, 2, np.pi):
        azihigh = _operate(np.subtract, azihigh, _operate(np.multiply, 2, np.pi))
        facesh = _operate(np.add, _operate(np.multiply, np.logical_or(aspect > azilow, aspect <= azihigh).astype(np.float32), -1), 1)
    keep = _operate(np.subtract, (weightsumwall == second).astype(np.float32), facesh)
    keep[keep == -1] = 0
    gvf1 = _operate(np.add, _operate(np.multiply, _divide(_operate(np.add, weightsumwall_first, weightsumsh_first), _operate(np.add, first, 1)), wallsuninfluence_first), _operate(np.multiply, _divide(weightsumsh_first, first), _operate(np.add, _operate(np.multiply, wallsuninfluence_first, -1), 1)))
    weightsumwall[keep == 1] = 0
    gvf2 = _operate(np.add, _operate(np.multiply, _divide(_operate(np.add, weightsumwall, weightsumsh), _operate(np.add, second, 1)), wallsuninfluence_second), _operate(np.multiply, _divide(weightsumsh, second), _operate(np.add, _operate(np.multiply, wallsuninfluence_second, -1), 1)))
    gvf2[gvf2 > 1.0] = 1.0
    gvfLup1 = _operate(np.add, _operate(np.multiply, _divide(_operate(np.add, weightsumLwall_first, weightsumLupsh_first), _operate(np.add, first, 1)), wallsuninfluence_first), _operate(np.multiply, _divide(weightsumLupsh_first, first), _operate(np.add, _operate(np.multiply, wallsuninfluence_first, -1), 1)))
    weightsumLwall[keep == 1] = 0
    gvfLup2 = _operate(np.add, _operate(np.multiply, _divide(_operate(np.add, weightsumLwall, weightsumLupsh), _operate(np.add, second, 1)), wallsuninfluence_second), _operate(np.multiply, _divide(weightsumLupsh, second), _operate(np.add, _operate(np.multiply, wallsuninfluence_second, -1), 1)))
    gvfalb1 = _operate(np.add, _operate(np.multiply, _divide(_operate(np.add, weightsumalbwall_first, weightsumalbsh_first), _operate(np.add, first, 1)), wallsuninfluence_first), _operate(np.multiply, _divide(weightsumalbsh_first, first), _operate(np.add, _operate(np.multiply, wallsuninfluence_first, -1), 1)))
    weightsumalbwall[keep == 1] = 0
    gvfalb2 = _operate(np.add, _operate(np.multiply, _divide(_operate(np.add, weightsumalbwall, weightsumalbsh), _operate(np.add, second, 1)), wallsuninfluence_second), _operate(np.multiply, _divide(weightsumalbsh, second), _operate(np.add, _operate(np.multiply, wallsuninfluence_second, -1), 1)))
    gvfalbnosh1 = _operate(np.add, _operate(np.multiply, _divide(_operate(np.add, weightsumalbwallnosh_first, weightsumalbnosh_first), _operate(np.add, first, 1)), wallinfluence_first), _operate(np.multiply, _divide(weightsumalbnosh_first, first), _operate(np.add, _operate(np.multiply, wallinfluence_first, -1), 1)))
    gvfalbnosh2 = _operate(np.add, _operate(np.multiply, _divide(_operate(np.add, weightsumalbwallnosh, weightsumalbnosh), second), wallinfluence_second), _operate(np.multiply, _divide(weightsumalbnosh, second), _operate(np.add, _operate(np.multiply, wallinfluence_second, -1), 1)))
    gvf = _divide(_operate(np.add, _operate(np.multiply, gvf1, 0.5), _operate(np.multiply, gvf2, 0.4)), 0.9)
    gvfLup = _divide(_operate(np.add, _operate(np.multiply, gvfLup1, 0.5), _operate(np.multiply, gvfLup2, 0.4)), 0.9)
    gvfLup = _operate(np.add, gvfLup, _operate(np.multiply, _operate(np.subtract, _operate(np.multiply, _operate(np.multiply, SBC, emis_grid), _operate(np.power, _operate(np.add, _operate(np.add, _operate(np.multiply, Tg, shadow), Ta), 273.15), 4)), _operate(np.multiply, _operate(np.multiply, SBC, emis_grid), _operate(np.power, _operate(np.add, Ta, 273.15), 4))), _operate(np.add, _operate(np.multiply, buildings, -1), 1)))
    gvfalb = _divide(_operate(np.add, _operate(np.multiply, gvfalb1, 0.5), _operate(np.multiply, gvfalb2, 0.4)), 0.9)
    gvfalb = _operate(np.add, gvfalb, _operate(np.multiply, _operate(np.multiply, alb_grid, _operate(np.add, _operate(np.multiply, buildings, -1), 1)), shadow))
    gvfalbnosh = _divide(_operate(np.add, _operate(np.multiply, gvfalbnosh1, 0.5), _operate(np.multiply, gvfalbnosh2, 0.4)), 0.9)
    gvfalbnosh = _operate(np.add, _operate(np.multiply, gvfalbnosh, buildings), _operate(np.multiply, alb_grid, _operate(np.add, _operate(np.multiply, buildings, -1), 1)))
    del tempbu, tempsh, tempLupsh, tempalbsh, tempalbnosh, shadow2, Lupsh, albsh, albnosh, tempwallsun, tempb, tempbwall, tempbub, tempbubwall
    del weightsumLupsh, weightsumLwall, weightsumalbsh, weightsumalbwall, weightsumalbnosh, weightsumalbwallnosh, weightsumsh, weightsumwall
    return (gvf, gvfLup, gvfalb, gvfalbnosh, gvf2)

def gvf_2018a(wallsun, walls, buildings, scale, shadow, first, second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, rows, cols, Twater, lc_grid, landcover):
    """
    Calculate ground view factors for radiation exchange between surfaces.
    
    Computes how much ground surfaces "see" walls and other surfaces,
    accounting for shadows and multiple reflections.
    
    Args:
        wallsun (np.ndarray): Sunlit wall indicator
        walls (np.ndarray): Wall heights
        buildings (np.ndarray): Building mask
        scale (float): Grid scale
        shadow (np.ndarray): Shadow map
        first/second (np.ndarray): Surface classification
        dirwalls (np.ndarray): Wall directions
        Tg/Tgwall/Ta (np.ndarray): Temperatures (ground/wall/air)
        emis_grid (np.ndarray): Ground emissivity
        ewall (float): Wall emissivity  
        alb_grid (np.ndarray): Ground albedo
        SBC (float): Stefan-Boltzmann constant
        albedo_b (float): Building albedo
        rows/cols (int): Grid dimensions
        Twater (float): Water temperature
        lc_grid (np.ndarray): Land cover grid
        landcover (np.ndarray): Land cover data
    
    Returns:
        tuple: View factors and albedo components for different directions
    """
    azimuthA = np.arange(5, 359, 20, dtype=np.float32)
    gvfLup = _zeros((rows, cols))
    gvfalb = _zeros((rows, cols))
    gvfalbnosh = _zeros((rows, cols))
    gvfLupE = _zeros((rows, cols))
    gvfLupS = _zeros((rows, cols))
    gvfLupW = _zeros((rows, cols))
    gvfLupN = _zeros((rows, cols))
    gvfalbE = _zeros((rows, cols))
    gvfalbS = _zeros((rows, cols))
    gvfalbW = _zeros((rows, cols))
    gvfalbN = _zeros((rows, cols))
    gvfalbnoshE = _zeros((rows, cols))
    gvfalbnoshS = _zeros((rows, cols))
    gvfalbnoshW = _zeros((rows, cols))
    gvfalbnoshN = _zeros((rows, cols))
    gvfSum = _zeros((rows, cols))
    sunwall = (_operate(np.multiply, _divide(wallsun, walls), buildings) == 1).astype(np.float32)
    for j in np.arange(0, len(azimuthA)):
        _, gvfLupi, gvfalbi, gvfalbnoshi, gvf2 = sunonsurface_2018a(azimuthA[j], scale, buildings, shadow, sunwall, first, second, _divide(_operate(np.multiply, dirwalls, np.pi), 180), walls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, Twater, lc_grid, landcover)
        gvfLup += gvfLupi
        gvfalb += gvfalbi
        gvfalbnosh += gvfalbnoshi
        gvfSum += gvf2
        if 0 <= azimuthA[j] < 180:
            gvfLupE += gvfLupi
            gvfalbE += gvfalbi
            gvfalbnoshE += gvfalbnoshi
        if 90 <= azimuthA[j] < 270:
            gvfLupS += gvfLupi
            gvfalbS += gvfalbi
            gvfalbnoshS += gvfalbnoshi
        if 180 <= azimuthA[j] < 360:
            gvfLupW += gvfLupi
            gvfalbW += gvfalbi
            gvfalbnoshW += gvfalbnoshi
        if 270 <= azimuthA[j] or azimuthA[j] < 90:
            gvfLupN += gvfLupi
            gvfalbN += gvfalbi
            gvfalbnoshN += gvfalbnoshi
    gvfLup = _operate(np.add, _divide(gvfLup, len(azimuthA)), _operate(np.multiply, _operate(np.multiply, SBC, emis_grid), _operate(np.power, _operate(np.add, Ta, 273.15), 4)))
    gvfalb = _divide(gvfalb, len(azimuthA))
    gvfalbnosh = _divide(gvfalbnosh, len(azimuthA))
    gvfLupE = _operate(np.add, _divide(gvfLupE, _divide(len(azimuthA), 2)), _operate(np.multiply, _operate(np.multiply, SBC, emis_grid), _operate(np.power, _operate(np.add, Ta, 273.15), 4)))
    gvfLupS = _operate(np.add, _divide(gvfLupS, _divide(len(azimuthA), 2)), _operate(np.multiply, _operate(np.multiply, SBC, emis_grid), _operate(np.power, _operate(np.add, Ta, 273.15), 4)))
    gvfLupW = _operate(np.add, _divide(gvfLupW, _divide(len(azimuthA), 2)), _operate(np.multiply, _operate(np.multiply, SBC, emis_grid), _operate(np.power, _operate(np.add, Ta, 273.15), 4)))
    gvfLupN = _operate(np.add, _divide(gvfLupN, _divide(len(azimuthA), 2)), _operate(np.multiply, _operate(np.multiply, SBC, emis_grid), _operate(np.power, _operate(np.add, Ta, 273.15), 4)))
    gvfalbE = _divide(gvfalbE, _divide(len(azimuthA), 2))
    gvfalbS = _divide(gvfalbS, _divide(len(azimuthA), 2))
    gvfalbW = _divide(gvfalbW, _divide(len(azimuthA), 2))
    gvfalbN = _divide(gvfalbN, _divide(len(azimuthA), 2))
    gvfalbnoshE = _divide(gvfalbnoshE, _divide(len(azimuthA), 2))
    gvfalbnoshS = _divide(gvfalbnoshS, _divide(len(azimuthA), 2))
    gvfalbnoshW = _divide(gvfalbnoshW, _divide(len(azimuthA), 2))
    gvfalbnoshN = _divide(gvfalbnoshN, _divide(len(azimuthA), 2))
    gvfNorm = _divide(gvfSum, len(azimuthA))
    gvfNorm[buildings == 0] = 1
    return (gvfLup, gvfalb, gvfalbnosh, gvfLupE, gvfalbE, gvfalbnoshE, gvfLupS, gvfalbS, gvfalbnoshS, gvfLupW, gvfalbW, gvfalbnoshW, gvfLupN, gvfalbN, gvfalbnoshN, gvfSum, gvfNorm)

def cylindric_wedge(zen, svfalfa, rows, cols):
    """
    Calculate form factors for cylindrical geometry (human body model).
    
    Args:
        zen (np.ndarray): Solar zenith angle
        svfalfa (np.ndarray): SVF alpha component
        rows, cols (int): Grid dimensions
    
    Returns:
        tuple: (Fside, Fup, Fcyl) - Form factors for cylinder sides, top, and total
    """
    np.seterr(divide='ignore', invalid='ignore')
    beta = _array(zen, dtype=np.float32)
    alfa = _operate(np.add, _zeros((rows, cols)), svfalfa)
    xa = _operate(np.subtract, 1, _divide(2.0, _operate(np.multiply, np.tan(alfa), np.tan(beta))))
    ha = _divide(2.0, _operate(np.multiply, np.tan(alfa), np.tan(beta)))
    ba = _divide(1.0, np.tan(alfa))
    hkil = _operate(np.multiply, _operate(np.multiply, 2.0, ba), ha)
    qa = _zeros((rows, cols))
    qa[xa < 0] = _divide(np.tan(beta), 2)
    Za = _zeros((rows, cols))
    Za[xa < 0] = _operate(np.power, _operate(np.subtract, _operate(np.power, ba[xa < 0], 2), _divide(_operate(np.power, qa[xa < 0], 2), 4)), 0.5)
    phi = _zeros((rows, cols))
    phi[xa < 0] = np.arctan(_divide(Za[xa < 0], qa[xa < 0]))
    A = _zeros((rows, cols))
    A[xa < 0] = _divide(_operate(np.subtract, np.sin(phi[xa < 0]), _operate(np.multiply, phi[xa < 0], np.cos(phi[xa < 0]))), _operate(np.subtract, 1, np.cos(phi[xa < 0])))
    ukil = _zeros((rows, cols))
    ukil[xa < 0] = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, 2, ba[xa < 0]), xa[xa < 0]), A[xa < 0])
    Ssurf = _operate(np.add, hkil, ukil)
    F_sh = _divide(_operate(np.subtract, _operate(np.multiply, _operate(np.multiply, 2, np.pi), ba), Ssurf), _operate(np.multiply, _operate(np.multiply, 2, np.pi), ba))
    del alfa, beta, hkil, ukil, phi, A, Ssurf, qa, Za
    return F_sh

def TsWaveDelay_2015a(gvfLup, firstdaytime, timeadd, timestepdec, Tgmap1):
    """
    Calculate surface temperature wave delay.
    
    Models thermal inertia and temperature wave propagation in surfaces.
    
    Args:
        gvfLup: Ground view factor for upward longwave
        firstdaytime: First time step flag
        timeadd: Time addition parameter
        timestepdec: Time step decimal
        Tgmap1: Previous ground temperature map
    
    Returns:
        np.ndarray: Temperature with wave delay applied
    """
    Tgmap0 = gvfLup
    if firstdaytime == 1:
        Tgmap1 = Tgmap0
    if timeadd >= _divide(59, 1440):
        weight1 = np.exp(_operate(np.multiply, -33.27, _array(timeadd)))
        Tgmap1 = _operate(np.add, _operate(np.multiply, Tgmap0, _operate(np.subtract, 1, weight1)), _operate(np.multiply, Tgmap1, weight1))
        Lup = Tgmap1
        if timestepdec > _divide(59, 1440):
            timeadd = timestepdec
        else:
            timeadd = 0
    else:
        timeadd = _operate(np.add, timeadd, timestepdec)
        weight1 = np.exp(_operate(np.multiply, -33.27, _array(timeadd)))
        Lup = _operate(np.add, _operate(np.multiply, Tgmap0, _operate(np.subtract, 1, weight1)), _operate(np.multiply, Tgmap1, weight1))
    return (Lup, timeadd, Tgmap1)

def Kup_veg_2015a(radI, radD, radG, altitude, svfbuveg, albedo_b, F_sh, gvfalb, gvfalbE, gvfalbS, gvfalbW, gvfalbN, gvfalbnosh, gvfalbnoshE, gvfalbnoshS, gvfalbnoshW, gvfalbnoshN):
    """
    Calculate upward shortwave radiation with vegetation effects.
    
    Accounts for multiple reflections between ground, walls, and vegetation.
    
    Returns:
        tuple: Upward shortwave components for different directions
    """
    albedo_b = np.copy(_array(albedo_b))
    Kup = _operate(np.add, _operate(np.multiply, _operate(np.multiply, gvfalb, radI), np.sin(_operate(np.multiply, altitude, _divide(np.pi, 180.0)))), _operate(np.multiply, _operate(np.add, _operate(np.multiply, radD, svfbuveg), _operate(np.multiply, _operate(np.multiply, albedo_b, _operate(np.subtract, 1, svfbuveg)), _operate(np.add, _operate(np.multiply, radG, _operate(np.subtract, 1, F_sh)), _operate(np.multiply, radD, F_sh)))), gvfalbnosh))
    KupE = _operate(np.add, _operate(np.multiply, _operate(np.multiply, gvfalbE, radI), np.sin(_operate(np.multiply, altitude, _divide(np.pi, 180.0)))), _operate(np.multiply, _operate(np.add, _operate(np.multiply, radD, svfbuveg), _operate(np.multiply, _operate(np.multiply, albedo_b, _operate(np.subtract, 1, svfbuveg)), _operate(np.add, _operate(np.multiply, radG, _operate(np.subtract, 1, F_sh)), _operate(np.multiply, radD, F_sh)))), gvfalbnoshE))
    KupS = _operate(np.add, _operate(np.multiply, _operate(np.multiply, gvfalbS, radI), np.sin(_operate(np.multiply, altitude, _divide(np.pi, 180.0)))), _operate(np.multiply, _operate(np.add, _operate(np.multiply, radD, svfbuveg), _operate(np.multiply, _operate(np.multiply, albedo_b, _operate(np.subtract, 1, svfbuveg)), _operate(np.add, _operate(np.multiply, radG, _operate(np.subtract, 1, F_sh)), _operate(np.multiply, radD, F_sh)))), gvfalbnoshS))
    KupW = _operate(np.add, _operate(np.multiply, _operate(np.multiply, gvfalbW, radI), np.sin(_operate(np.multiply, altitude, _divide(np.pi, 180.0)))), _operate(np.multiply, _operate(np.add, _operate(np.multiply, radD, svfbuveg), _operate(np.multiply, _operate(np.multiply, albedo_b, _operate(np.subtract, 1, svfbuveg)), _operate(np.add, _operate(np.multiply, radG, _operate(np.subtract, 1, F_sh)), _operate(np.multiply, radD, F_sh)))), gvfalbnoshW))
    KupN = _operate(np.add, _operate(np.multiply, _operate(np.multiply, gvfalbN, radI), np.sin(_operate(np.multiply, altitude, _divide(np.pi, 180.0)))), _operate(np.multiply, _operate(np.add, _operate(np.multiply, radD, svfbuveg), _operate(np.multiply, _operate(np.multiply, albedo_b, _operate(np.subtract, 1, svfbuveg)), _operate(np.add, _operate(np.multiply, radG, _operate(np.subtract, 1, F_sh)), _operate(np.multiply, radD, F_sh)))), gvfalbnoshN))
    return (Kup, KupE, KupS, KupW, KupN)

def Kvikt_veg(svf, svfveg, vikttot):
    """Calculate shortwave weight factor accounting for vegetation."""
    viktwall = _divide(_operate(np.subtract, vikttot, _operate(np.subtract, _operate(np.add, _operate(np.subtract, _operate(np.add, _operate(np.subtract, _operate(np.multiply, 63.227, _operate(np.power, svf, 6)), _operate(np.multiply, 161.51, _operate(np.power, svf, 5))), _operate(np.multiply, 156.91, _operate(np.power, svf, 4))), _operate(np.multiply, 70.424, _operate(np.power, svf, 3))), _operate(np.multiply, 16.773, _operate(np.power, svf, 2))), _operate(np.multiply, 0.4863, svf))), vikttot)
    svfvegbu = _operate(np.subtract, _operate(np.add, svfveg, svf), 1)
    viktveg = _divide(_operate(np.subtract, vikttot, _operate(np.subtract, _operate(np.add, _operate(np.subtract, _operate(np.add, _operate(np.subtract, _operate(np.multiply, 63.227, _operate(np.power, svfvegbu, 6)), _operate(np.multiply, 161.51, _operate(np.power, svfvegbu, 5))), _operate(np.multiply, 156.91, _operate(np.power, svfvegbu, 4))), _operate(np.multiply, 70.424, _operate(np.power, svfvegbu, 3))), _operate(np.multiply, 16.773, _operate(np.power, svfvegbu, 2))), _operate(np.multiply, 0.4863, svfvegbu))), vikttot)
    viktveg = _operate(np.subtract, viktveg, viktwall)
    del svfvegbu
    return (viktveg, viktwall)

def shaded_or_sunlit(solar_altitude, solar_azimuth, patch_altitude, patch_azimuth, asvf):
    """
    Determine if sky patches are shaded or sunlit.
    
    Args:
        solar_altitude (float): Solar altitude angle
        solar_azimuth (float): Solar azimuth angle
        patch_altitude (np.ndarray): Patch altitude angles
        patch_azimuth (np.ndarray): Patch azimuth angles
        asvf (np.ndarray): Anisotropic sky view factor
    
    Returns:
        np.ndarray: Binary mask (1=sunlit, 0=shaded)
    """
    patch_to_sun_azi = np.abs(_operate(np.subtract, solar_azimuth, patch_azimuth))
    deg2rad = _divide(np.pi, 180.0)
    rad2deg = _divide(180.0, np.pi)
    xi = np.cos(_operate(np.multiply, patch_to_sun_azi, deg2rad))
    yi = _operate(np.multiply, _operate(np.multiply, 2, xi), np.tan(_operate(np.multiply, solar_altitude, deg2rad)))
    hsvf = np.tan(asvf)
    yi_ = np.where(yi > 0, 0.0, yi)
    tan_delta = _operate(np.add, hsvf, yi_)
    sunlit_degrees = _operate(np.multiply, np.arctan(tan_delta), rad2deg)
    sunlit_patches = sunlit_degrees < patch_altitude
    shaded_patches = sunlit_degrees > patch_altitude
    return (sunlit_patches, shaded_patches)

def Kside_veg_v2022a(radI, radD, radG, shadow, svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg, azimuth, altitude, psi, t, albedo, F_sh, KupE, KupS, KupW, KupN, cyl, lv, anisotropic_diffuse, diffsh, rows, cols, asvf, shmat, vegshmat, vbshvegshmat):
    """
    Calculate shortwave radiation on vertical surfaces (walls) with vegetation effects.
    
    Computes direct, diffuse, and reflected shortwave radiation on walls in the
    four cardinal directions, accounting for vegetation shading and ground reflections.
    
    Args:
        radI, radD, radG (float): Direct, diffuse, and global radiation (W/m²)
        shadow (np.ndarray): Shadow map
        svfS, svfW, svfN, svfE (np.ndarray): Directional sky view factors
        svf*veg (np.ndarray): Vegetation-obstructed SVFs
        azimuth, altitude (float): Solar angles (degrees)
        psi (np.ndarray): Tilt angles
        t (float): Transmissivity factor
        albedo (np.ndarray): Surface albedo
        F_sh (np.ndarray): Form factor
        KupE, KupS, KupW, KupN (np.ndarray): Upward shortwave per direction
        cyl (np.ndarray): Cylindrical geometry factor
        lv (float): Leaf area index factor
        anisotropic_diffuse (bool): Use anisotropic diffuse model
        diffsh (np.ndarray): Diffuse shadowing
        rows, cols (int): Grid dimensions
        asvf (np.ndarray): Anisotropic SVF
        shmat, vegshmat, vbshvegshmat (np.ndarray): Shadow matrices
    
    Returns:
        tuple: (Keast, Ksouth, Kwest, Knorth, KsideI, KsideD, Kside) - 
               Shortwave radiation components for each direction
    """
    vikttot = 4.4897
    aziE = _operate(np.add, azimuth, t)
    aziS = _operate(np.add, _operate(np.subtract, azimuth, 90), t)
    aziW = _operate(np.add, _operate(np.subtract, azimuth, 180), t)
    aziN = _operate(np.add, _operate(np.subtract, azimuth, 270), t)
    deg2rad = _divide(np.pi, 180.0)
    deg2rad = _array(deg2rad)
    KsideD = _zeros((rows, cols))
    Kref_sun = _zeros((rows, cols))
    Kref_sh = _zeros((rows, cols))
    Kref_veg = _zeros((rows, cols))
    Kside = _zeros((rows, cols))
    Kref_veg_n = _zeros((rows, cols))
    Kref_veg_s = _zeros((rows, cols))
    Kref_veg_e = _zeros((rows, cols))
    Kref_veg_w = _zeros((rows, cols))
    Kref_sh_n = _zeros((rows, cols))
    Kref_sh_s = _zeros((rows, cols))
    Kref_sh_e = _zeros((rows, cols))
    Kref_sh_w = _zeros((rows, cols))
    Kref_sun_n = _zeros((rows, cols))
    Kref_sun_s = _zeros((rows, cols))
    Kref_sun_e = _zeros((rows, cols))
    Kref_sun_w = _zeros((rows, cols))
    KeastRef = _zeros((rows, cols))
    KwestRef = _zeros((rows, cols))
    KnorthRef = _zeros((rows, cols))
    KsouthRef = _zeros((rows, cols))
    diffRadE = _zeros((rows, cols))
    diffRadS = _zeros((rows, cols))
    diffRadW = _zeros((rows, cols))
    diffRadN = _zeros((rows, cols))
    altitude = _array(altitude)
    if cyl == 1:
        KsideI = _operate(np.multiply, _operate(np.multiply, shadow, radI), np.cos(_operate(np.multiply, altitude, deg2rad)))
        KeastI = _zeros((rows, cols))
        KsouthI = _zeros((rows, cols))
        KwestI = _zeros((rows, cols))
        KnorthI = _zeros((rows, cols))
    else:
        KeastI = np.where((azimuth > _operate(np.subtract, 360, t)) | (azimuth <= _operate(np.subtract, 180, t)), _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, radI, shadow), np.cos(_operate(np.multiply, altitude, deg2rad))), np.sin(_operate(np.multiply, aziE, deg2rad))), _zeros((rows, cols)))
        KsouthI = np.where((azimuth > _operate(np.subtract, 90, t)) & (azimuth <= _operate(np.subtract, 270, t)), _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, radI, shadow), np.cos(_operate(np.multiply, altitude, deg2rad))), np.sin(_operate(np.multiply, aziS, deg2rad))), _zeros((rows, cols)))
        KwestI = np.where((azimuth > _operate(np.subtract, 180, t)) & (azimuth <= _operate(np.subtract, 360, t)), _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, radI, shadow), np.cos(_operate(np.multiply, altitude, deg2rad))), np.sin(_operate(np.multiply, aziW, deg2rad))), _zeros((rows, cols)))
        KnorthI = np.where((azimuth <= _operate(np.subtract, 90, t)) | (azimuth > _operate(np.subtract, 270, t)), _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, radI, shadow), np.cos(_operate(np.multiply, altitude, deg2rad))), np.sin(_operate(np.multiply, aziN, deg2rad))), _zeros((rows, cols)))
        KsideI = _operate(np.multiply, shadow, 0)
    viktveg, viktwall = Kvikt_veg(svfE, svfEveg, vikttot)
    svfviktbuvegE = _operate(np.add, viktwall, _operate(np.multiply, viktveg, _operate(np.subtract, 1, psi)))
    viktveg, viktwall = Kvikt_veg(svfS, svfSveg, vikttot)
    svfviktbuvegS = _operate(np.add, viktwall, _operate(np.multiply, viktveg, _operate(np.subtract, 1, psi)))
    viktveg, viktwall = Kvikt_veg(svfW, svfWveg, vikttot)
    svfviktbuvegW = _operate(np.add, viktwall, _operate(np.multiply, viktveg, _operate(np.subtract, 1, psi)))
    viktveg, viktwall = Kvikt_veg(svfN, svfNveg, vikttot)
    svfviktbuvegN = _operate(np.add, viktwall, _operate(np.multiply, viktveg, _operate(np.subtract, 1, psi)))
    if anisotropic_diffuse == 1:
        anisotropic_sky = True
        patch_altitude = lv[:, 0]
        patch_azimuth = lv[:, 1]
        if anisotropic_sky:
            patch_luminance = lv[:, 2]
        else:
            patch_luminance = _divide(_ones(patch_altitude.shape[0]), patch_altitude.shape[0])
        skyalt, skyalt_c = np.unique(patch_altitude, return_counts=True)
        radTot = _zeros(1)
        steradian = _zeros(patch_altitude.shape[0])
        for i in range(patch_altitude.shape[0]):
            if skyalt_c[skyalt == patch_altitude[i]].item() > 1:
                steradian[i] = _operate(np.multiply, _operate(np.multiply, _divide(360, skyalt_c[skyalt == patch_altitude[i]].item()), deg2rad), _operate(np.subtract, np.sin(_operate(np.multiply, _operate(np.add, patch_altitude[i], patch_altitude[0]), deg2rad)), np.sin(_operate(np.multiply, _operate(np.subtract, patch_altitude[i], patch_altitude[0]), deg2rad))))
            else:
                steradian[i] = _operate(np.multiply, _operate(np.multiply, _divide(360, skyalt_c[skyalt == patch_altitude[i]].item()), deg2rad), _operate(np.subtract, np.sin(_operate(np.multiply, patch_altitude[i], deg2rad)), np.sin(_operate(np.multiply, _operate(np.add, patch_altitude[_operate(np.subtract, i, 1)], patch_altitude[0]), deg2rad))))
            radTot += _operate(np.multiply, _operate(np.multiply, patch_luminance[i], steradian[i]), np.sin(_operate(np.multiply, patch_altitude[i], deg2rad)))
        lumChi = _divide(_operate(np.multiply, patch_luminance, radD), radTot)
        if cyl == 1:
            for idx in range(patch_azimuth.shape[0]):
                anglIncC = _operate(np.multiply, np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad)), np.cos(_array(0.0)))
                KsideD += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, diffsh[:, :, idx], lumChi[idx]), anglIncC), steradian[idx])
                sunlit_surface = _divide(_operate(np.add, _operate(np.multiply, albedo, _operate(np.multiply, radI, np.cos(_operate(np.multiply, altitude, deg2rad)))), _operate(np.multiply, radD, 0.5)), np.pi)
                shaded_surface = _divide(_operate(np.multiply, _operate(np.multiply, albedo, radD), 0.5), np.pi)
                temp_vegsh = (vegshmat[:, :, idx] == 0) | (vbshvegshmat[:, :, idx] == 0)
                Kref_veg += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, temp_vegsh), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad)))
                temp_vbsh = _operate(np.multiply, _operate(np.subtract, 1, shmat[:, :, idx]), vbshvegshmat[:, :, idx])
                temp_sh = temp_vbsh == 1
                sunlit_patches, shaded_patches = shaded_or_sunlit(altitude, azimuth, patch_altitude[idx], patch_azimuth[idx], asvf)
                Kref_sun += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, sunlit_surface, sunlit_patches), temp_sh), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad)))
                Kref_sh += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, shaded_patches), temp_sh), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad)))
            Kside = _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, KsideI, KsideD), Kref_sun), Kref_sh), Kref_veg)
            Keast = _operate(np.multiply, KupE, 0.5)
            Kwest = _operate(np.multiply, KupW, 0.5)
            Knorth = _operate(np.multiply, KupN, 0.5)
            Ksouth = _operate(np.multiply, KupS, 0.5)
        else:
            for idx in range(patch_azimuth.shape[0]):
                if patch_azimuth[idx] > 360 or patch_azimuth[idx] <= 180:
                    anglIncE = _operate(np.multiply, np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad)), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 90, patch_azimuth[idx]), t), deg2rad)))
                    diffRadE += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, diffsh[:, :, idx], lumChi[idx]), anglIncE), steradian[idx])
                if patch_azimuth[idx] > 90 and patch_azimuth[idx] <= 270:
                    anglIncS = _operate(np.multiply, np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad)), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 180, patch_azimuth[idx]), t), deg2rad)))
                    diffRadS += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, diffsh[:, :, idx], lumChi[idx]), anglIncS), steradian[idx])
                if patch_azimuth[idx] > 180 and patch_azimuth[idx] <= 360:
                    anglIncW = _operate(np.multiply, np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad)), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 270, patch_azimuth[idx]), t), deg2rad)))
                    diffRadW += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, diffsh[:, :, idx], lumChi[idx]), anglIncW), steradian[idx])
                if patch_azimuth[idx] > 270 or patch_azimuth[idx] <= 90:
                    anglIncN = _operate(np.multiply, np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad)), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 0, patch_azimuth[idx]), t), deg2rad)))
                    diffRadN += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, diffsh[:, :, idx], lumChi[idx]), anglIncN), steradian[idx])
                sunlit_surface = _divide(_operate(np.add, _operate(np.multiply, albedo, _operate(np.multiply, radI, np.cos(_operate(np.multiply, altitude, deg2rad)))), _operate(np.multiply, radD, 0.5)), np.pi)
                shaded_surface = _divide(_operate(np.multiply, _operate(np.multiply, albedo, radD), 0.5), np.pi)
                temp_vegsh = (vegshmat[:, :, idx] == 0) | (vbshvegshmat[:, :, idx] == 0)
                Kref_veg += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, temp_vegsh), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad)))
                if patch_azimuth[idx] > 360 or patch_azimuth[idx] < 180:
                    Kref_veg_e += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_vegsh), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 90, patch_azimuth[idx]), t), deg2rad)))
                if patch_azimuth[idx] > 90 and patch_azimuth[idx] < 270:
                    Kref_veg_s += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_vegsh), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 180, patch_azimuth[idx]), t), deg2rad)))
                if patch_azimuth[idx] > 180 and patch_azimuth[idx] < 360:
                    Kref_veg_w += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_vegsh), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 270, patch_azimuth[idx]), t), deg2rad)))
                if patch_azimuth[idx] > 270 or patch_azimuth[idx] < 90:
                    Kref_veg_n += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_vegsh), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 0, patch_azimuth[idx]), t), deg2rad)))
                temp_vbsh = _operate(np.multiply, _operate(np.subtract, 1, shmat[:, :, idx]), vbshvegshmat[:, :, idx])
                temp_sh = temp_vbsh == 1
                azimuth_difference = np.abs(_operate(np.subtract, azimuth, patch_azimuth[idx]))
                if azimuth_difference > 90 and azimuth_difference < 270:
                    sunlit_patches, shaded_patches = shaded_or_sunlit(altitude, azimuth, patch_altitude[idx], patch_azimuth[idx], asvf)
                    Kref_sun += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, sunlit_surface, sunlit_patches), temp_sh), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad)))
                    Kref_sh += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, shaded_patches), temp_sh), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad)))
                    if patch_azimuth[idx] > 360 or patch_azimuth[idx] < 180:
                        Kref_sun_e += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, sunlit_surface, sunlit_patches), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 90, patch_azimuth[idx]), t), deg2rad)))
                        Kref_sh_e += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, shaded_patches), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 90, patch_azimuth[idx]), t), deg2rad)))
                    if patch_azimuth[idx] > 90 and patch_azimuth[idx] < 270:
                        Kref_sun_s += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, sunlit_surface, sunlit_patches), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 180, patch_azimuth[idx]), t), deg2rad)))
                        Kref_sh_s += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, shaded_patches), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 180, patch_azimuth[idx]), t), deg2rad)))
                    if patch_azimuth[idx] > 180 and patch_azimuth[idx] < 360:
                        Kref_sun_w += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, sunlit_surface, sunlit_patches), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 270, patch_azimuth[idx]), t), deg2rad)))
                        Kref_sh_w += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, shaded_patches), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 270, patch_azimuth[idx]), t), deg2rad)))
                    if patch_azimuth[idx] > 270 or patch_azimuth[idx] < 90:
                        Kref_sun_n += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, sunlit_surface, sunlit_patches), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 0, patch_azimuth[idx]), t), deg2rad)))
                        Kref_sh_n += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, shaded_patches), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 0, patch_azimuth[idx]), t), deg2rad)))
                else:
                    Kref_sh += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, temp_sh), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad)))
                    if patch_azimuth[idx] > 360 or patch_azimuth[idx] < 180:
                        Kref_sh_e += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 90, patch_azimuth[idx]), t), deg2rad)))
                    if patch_azimuth[idx] > 90 and patch_azimuth[idx] < 270:
                        Kref_sh_s += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 180, patch_azimuth[idx]), t), deg2rad)))
                    if patch_azimuth[idx] > 180 and patch_azimuth[idx] < 360:
                        Kref_sh_w += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 270, patch_azimuth[idx]), t), deg2rad)))
                    if patch_azimuth[idx] > 270 or patch_azimuth[idx] < 90:
                        Kref_sh_n += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.add, _operate(np.subtract, 0, patch_azimuth[idx]), t), deg2rad)))
            Keast = _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, KeastI, diffRadE), Kref_sun_e), Kref_sh_e), Kref_veg_e), _operate(np.multiply, KupE, 0.5))
            Kwest = _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, KwestI, diffRadW), Kref_sun_w), Kref_sh_w), Kref_veg_w), _operate(np.multiply, KupW, 0.5))
            Knorth = _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, KnorthI, diffRadN), Kref_sun_n), Kref_sh_n), Kref_veg_n), _operate(np.multiply, KupN, 0.5))
            Ksouth = _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, KsouthI, diffRadS), Kref_sun_s), Kref_sh_s), Kref_veg_s), _operate(np.multiply, KupS, 0.5))
    else:
        KeastDG = _operate(np.multiply, _operate(np.add, _operate(np.add, _operate(np.multiply, radD, _operate(np.subtract, 1, svfviktbuvegE)), _operate(np.multiply, albedo, _operate(np.multiply, svfviktbuvegE, _operate(np.add, _operate(np.multiply, radG, _operate(np.subtract, 1, F_sh)), _operate(np.multiply, radD, F_sh))))), KupE), 0.5)
        Keast = _operate(np.add, KeastI, KeastDG)
        KsouthDG = _operate(np.multiply, _operate(np.add, _operate(np.add, _operate(np.multiply, radD, _operate(np.subtract, 1, svfviktbuvegS)), _operate(np.multiply, albedo, _operate(np.multiply, svfviktbuvegS, _operate(np.add, _operate(np.multiply, radG, _operate(np.subtract, 1, F_sh)), _operate(np.multiply, radD, F_sh))))), KupS), 0.5)
        Ksouth = _operate(np.add, KsouthI, KsouthDG)
        KwestDG = _operate(np.multiply, _operate(np.add, _operate(np.add, _operate(np.multiply, radD, _operate(np.subtract, 1, svfviktbuvegW)), _operate(np.multiply, albedo, _operate(np.multiply, svfviktbuvegW, _operate(np.add, _operate(np.multiply, radG, _operate(np.subtract, 1, F_sh)), _operate(np.multiply, radD, F_sh))))), KupW), 0.5)
        Kwest = _operate(np.add, KwestI, KwestDG)
        KnorthDG = _operate(np.multiply, _operate(np.add, _operate(np.add, _operate(np.multiply, radD, _operate(np.subtract, 1, svfviktbuvegN)), _operate(np.multiply, albedo, _operate(np.multiply, svfviktbuvegN, _operate(np.add, _operate(np.multiply, radG, _operate(np.subtract, 1, F_sh)), _operate(np.multiply, radD, F_sh))))), KupN), 0.5)
        Knorth = _operate(np.add, KnorthI, KnorthDG)
    del temp_vegsh, temp_vbsh, temp_sh
    del Kref_sun, Kref_sh, Kref_veg
    del Kref_veg_n, Kref_veg_s, Kref_veg_e, Kref_veg_w
    del Kref_sh_n, Kref_sh_s, Kref_sh_e, Kref_sh_w
    del Kref_sun_n, Kref_sun_s, Kref_sun_e, Kref_sun_w
    del KeastRef, KwestRef, KnorthRef, KsouthRef, diffRadE, diffRadS, diffRadW, diffRadN
    del KeastI, KsouthI, KwestI, KnorthI, viktveg, viktwall, svfviktbuvegE, svfviktbuvegS, svfviktbuvegW, svfviktbuvegN, KupW, KupN
    return (Keast, Ksouth, Kwest, Knorth, KsideI, KsideD, Kside)

def sun_distance(jday):
    """
    Calculate Earth-Sun distance correction factor for given day.
    
    Args:
        jday (np.ndarray): Julian day of year
    
    Returns:
        np.ndarray: Distance correction factor (dimensionless)
    """
    b = _divide(_operate(np.multiply, _operate(np.multiply, 2.0, np.pi), jday), 365.0)
    D = np.sqrt(_operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, 1.00011, _operate(np.multiply, 0.034221, np.cos(b))), _operate(np.multiply, 0.00128, np.sin(b))), _operate(np.multiply, 0.000719, np.cos(_operate(np.multiply, 2.0, b)))), _operate(np.multiply, 7.7e-05, np.sin(_operate(np.multiply, 2.0, b)))))
    return D

def clearnessindex_2013b(zen, jday, Ta, RH, radG, location, P):
    """
    Calculate atmospheric clearness index.
    
    Args:
        zen (np.ndarray): Solar zenith angle (radians)
        jday (np.ndarray): Julian day
        Ta (float): Air temperature (°C)
        RH (float): Relative humidity (%)
        radG (float): Global radiation (W/m²)
        location (dict): Geographic location
        P (float): Atmospheric pressure (kPa)
    
    Returns:
        np.ndarray: Clearness index (dimensionless, 0-1)
    """
    if P == -999.0:
        p = 1013.0
    else:
        p = _operate(np.multiply, P, 10.0)
    Itoa = 1370.0
    D = sun_distance(jday)
    m = _operate(np.multiply, _operate(np.multiply, 35.0, np.cos(zen)), _operate(np.power, _operate(np.add, _operate(np.multiply, 1224.0, _operate(np.power, np.cos(zen), 2)), 1.0), _divide(-1, 2.0)))
    Trpg = _operate(np.subtract, 1.021, _operate(np.multiply, 0.084, _operate(np.power, _operate(np.multiply, m, _operate(np.add, _operate(np.multiply, 0.000949, p), 0.051)), 0.5)))
    latitude = location['latitude']
    if latitude < 10.0:
        G = [3.37, 2.85, 2.8, 2.64]
    elif 10.0 <= latitude < 20.0:
        G = [2.99, 3.02, 2.7, 2.93]
    elif 20.0 <= latitude < 30.0:
        G = [3.6, 3.0, 2.98, 2.93]
    elif 30.0 <= latitude < 40.0:
        G = [3.04, 3.11, 2.92, 2.94]
    elif 40.0 <= latitude < 50.0:
        G = [2.7, 2.95, 2.77, 2.71]
    elif 50.0 <= latitude < 60.0:
        G = [2.52, 3.07, 2.67, 2.93]
    elif 60.0 <= latitude < 70.0:
        G = [1.76, 2.69, 2.61, 2.61]
    elif 70.0 <= latitude < 80.0:
        G = [1.6, 1.67, 2.24, 2.63]
    elif 80.0 <= latitude < 90.0:
        G = [1.11, 1.44, 1.94, 2.02]
    if jday > 335 or jday <= 60:
        G = G[0]
    elif 60 < jday <= 152:
        G = G[1]
    elif 152 < jday <= 244:
        G = G[2]
    elif 244 < jday <= 335:
        G = G[3]
    a2 = _array(17.27)
    b2 = _array(237.7)
    Td = _divide(_operate(np.multiply, b2, _operate(np.add, _divide(_operate(np.multiply, a2, Ta), _operate(np.add, b2, Ta)), np.log(RH))), _operate(np.subtract, a2, _operate(np.add, _divide(_operate(np.multiply, a2, Ta), _operate(np.add, b2, Ta)), np.log(RH))))
    Td = _operate(np.add, _operate(np.multiply, Td, 1.8), 32)
    u = np.exp(_operate(np.add, _operate(np.subtract, 0.1133, np.log(_array(_operate(np.add, G, 1.0)))), _operate(np.multiply, 0.0393, Td)))
    Tw = _operate(np.subtract, 1, _operate(np.multiply, 0.077, _operate(np.power, _operate(np.multiply, u, m), 0.3)))
    Tar = _operate(np.power, 0.935, m)
    I0 = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, Itoa, np.cos(zen)), Trpg), Tw), D), Tar)
    I0 = np.where(np.abs(zen) > _divide(np.pi, 2), _array(0.0), I0)
    I0 = np.where(np.isnan(I0), _array(0.0), I0)
    corr = _operate(np.add, _operate(np.multiply, 0.1473, np.log(_operate(np.subtract, 90, _operate(np.multiply, _divide(zen, np.pi), 180)))), 0.3454)
    CIuncorr = _divide(radG, I0)
    CI = _operate(np.add, CIuncorr, _operate(np.subtract, 1, corr))
    I0et = _operate(np.multiply, _operate(np.multiply, Itoa, np.cos(zen)), D)
    Kt = _divide(radG, I0et)
    return (I0, CI, Kt, I0et, CIuncorr)

def diffusefraction(radG, altitude, Kt, Ta, RH):
    """
    Calculate fraction of diffuse radiation from global radiation.
    
    Uses empirical models to partition global radiation into direct and diffuse components.
    
    Args:
        radG (float): Global horizontal radiation (W/m²)
        altitude (np.ndarray): Solar altitude (degrees)
        Kt (np.ndarray): Clearness index
        Ta (float): Air temperature (°C)
        RH (float): Relative humidity (%)
    
    Returns:
        tuple: (radD, radI) where:
            - radD: Diffuse radiation (W/m²)
            - radI: Direct beam radiation (W/m²)
    """
    Ta = ensure_tensor(Ta)
    RH = ensure_tensor(RH)
    alfa = _operate(np.multiply, altitude, _divide(np.pi, 180.0))
    alfa = ensure_tensor(alfa)
    if Ta <= -999.0 or RH <= -999.0 or np.isnan(Ta) or np.isnan(RH):
        if Kt <= 0.3:
            radD = _operate(np.multiply, radG, _operate(np.subtract, 1.02, _operate(np.multiply, 0.248, Kt)))
        elif 0.3 < Kt < 0.78:
            radD = _operate(np.multiply, radG, _operate(np.subtract, 1.45, _operate(np.multiply, 1.67, Kt)))
        else:
            radD = _operate(np.multiply, radG, 0.147)
    else:
        RH = _divide(RH, 100)
        if Kt <= 0.3:
            radD = _operate(np.multiply, radG, _operate(np.add, _operate(np.subtract, _operate(np.add, _operate(np.subtract, 1, _operate(np.multiply, 0.232, Kt)), _operate(np.multiply, 0.0239, np.sin(alfa))), _operate(np.multiply, 0.000682, Ta)), _operate(np.multiply, 0.0195, RH)))
        elif 0.3 < Kt < 0.78:
            radD = _operate(np.multiply, radG, _operate(np.add, _operate(np.subtract, _operate(np.add, _operate(np.subtract, 1.329, _operate(np.multiply, 1.716, Kt)), _operate(np.multiply, 0.267, np.sin(alfa))), _operate(np.multiply, 0.00357, Ta)), _operate(np.multiply, 0.106, RH)))
        else:
            radD = _operate(np.multiply, radG, _operate(np.add, _operate(np.add, _operate(np.subtract, _operate(np.multiply, 0.426, Kt), _operate(np.multiply, 0.256, np.sin(alfa))), _operate(np.multiply, 0.00349, Ta)), _operate(np.multiply, 0.0734, RH)))
    radI = _divide(_operate(np.subtract, radG, radD), np.sin(alfa))
    radI = np.where(radI < 0, _array(0.0), radI)
    radI = np.where((altitude < 1) & (radI > radG), radG, radI)
    radD = np.where(radD > radG, radG, radD)
    return (radI, radD)

def shadowingfunction_wallheight_13(a, azimuth, altitude, scale, walls, aspect):
    """
    Calculate shadow patterns accounting for wall heights (method 1.3).
    
    Determines which surfaces are in shadow cast by nearby walls based on
    solar angle, wall height, and wall orientation.
    
    Returns:
        tuple: (vegsh, sh, vbshvegsh, wallsh, wallsun, wallshve, facesh, facesun)
    """
    if not walls.size:
        pass
    azimuth = np.copy(_array(_operate(np.multiply, azimuth, _divide(np.pi, 180.0))))
    altitude = np.copy(_array(_operate(np.multiply, altitude, _divide(np.pi, 180.0))))
    sizex = a.shape[0]
    sizey = a.shape[1]
    f = np.copy(_array(a))
    dx = np.copy(_array(0.0))
    dy = np.copy(_array(0.0))
    dz = np.copy(_array(0.0))
    temp = np.copy(_zeros((sizex, sizey)))
    wallbol = (walls > 0).astype(np.float32)
    amaxvalue = np.max(a)
    pibyfour = _divide(np.pi, 4)
    threetimespibyfour = _operate(np.multiply, 3, pibyfour)
    fivetimespibyfour = _operate(np.multiply, 5, pibyfour)
    seventimespibyfour = _operate(np.multiply, 7, pibyfour)
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)
    dssin = np.abs(_divide(1, sinazimuth))
    dscos = np.abs(_divide(1, cosazimuth))
    tanaltitudebyscale = _divide(np.tan(altitude), scale)
    index = 1
    while amaxvalue >= dz and np.abs(dx) < sizex and (np.abs(dy) < sizey):
        if pibyfour <= azimuth and azimuth < threetimespibyfour or (fivetimespibyfour <= azimuth and azimuth < seventimespibyfour):
            dy = _operate(np.multiply, signsinazimuth, index)
            dx = _operate(np.multiply, _operate(np.multiply, -1, signcosazimuth), np.abs(np.round(_divide(index, tanazimuth))))
            ds = dssin
        else:
            dy = _operate(np.multiply, signsinazimuth, np.abs(np.round(_operate(np.multiply, index, tanazimuth))))
            dx = _operate(np.multiply, _operate(np.multiply, -1, signcosazimuth), index)
            ds = dscos
        dz = _operate(np.multiply, _operate(np.multiply, ds, index), tanaltitudebyscale)
        temp[0:sizex, 0:sizey] = 0
        absdx = np.abs(dx)
        absdy = np.abs(dy)
        xc1 = int(_divide(_operate(np.add, dx, absdx), 2))
        xc2 = int(_operate(np.add, sizex, _divide(_operate(np.subtract, dx, absdx), 2)))
        yc1 = int(_divide(_operate(np.add, dy, absdy), 2))
        yc2 = int(_operate(np.add, sizey, _divide(_operate(np.subtract, dy, absdy), 2)))
        xp1 = int(-_divide(_operate(np.subtract, dx, absdx), 2))
        xp2 = int(_operate(np.subtract, sizex, _divide(_operate(np.add, dx, absdx), 2)))
        yp1 = int(-_divide(_operate(np.subtract, dy, absdy), 2))
        yp2 = int(_operate(np.subtract, sizey, _divide(_operate(np.add, dy, absdy), 2)))
        temp[xp1:xp2, yp1:yp2] = _operate(np.subtract, a[xc1:xc2, yc1:yc2], dz)
        f = np.maximum(f, temp)
        index = _operate(np.add, index, 1)
    azilow = _operate(np.subtract, azimuth, _divide(np.pi, 2))
    azihigh = _operate(np.add, azimuth, _divide(np.pi, 2))
    if azilow >= 0 and azihigh < _operate(np.multiply, 2, np.pi):
        facesh = _operate(np.add, _operate(np.subtract, np.logical_or(aspect < azilow, aspect >= azihigh).astype(np.float32), wallbol), 1)
    elif azilow < 0 and azihigh <= _operate(np.multiply, 2, np.pi):
        azilow = _operate(np.add, azilow, _operate(np.multiply, 2, np.pi))
        facesh = _operate(np.add, _operate(np.multiply, np.logical_or(aspect > azilow, aspect <= azihigh).astype(np.float32), -1), 1)
    elif azilow > 0 and azihigh >= _operate(np.multiply, 2, np.pi):
        azihigh = _operate(np.subtract, azihigh, _operate(np.multiply, 2, np.pi))
        facesh = _operate(np.add, _operate(np.multiply, np.logical_or(aspect > azilow, aspect <= azihigh).astype(np.float32), -1), 1)
    sh = _operate(np.subtract, f, _array(a))
    facesun = np.logical_and(_operate(np.add, facesh, wallbol) == 1, walls > 0).astype(np.float32)
    wallsun = _operate(np.subtract, walls, sh)
    wallsun[wallsun < 0] = 0
    wallsun[facesh == 1] = 0
    wallsh = _operate(np.subtract, walls, wallsun)
    sh = np.logical_not(np.logical_not(sh)).astype(np.float32)
    sh = _operate(np.add, _operate(np.multiply, sh, -1), 1)
    del temp
    return (sh, wallsh, wallsun, facesh, facesun)

def shadowingfunction_wallheight_23(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, walls, aspect):
    """
    Calculate shadow patterns with vegetation and wall heights (method 2.3).
    
    Extended shadow calculation including vegetation layers and building walls.
    
    Returns:
        tuple: Shadow components including vegetation effects
    """
    degrees = _array(_divide(np.pi, 180.0))
    azimuth = _operate(np.multiply, _array(azimuth), degrees)
    altitude = _operate(np.multiply, _array(altitude), degrees)
    sizex, sizey = a.shape
    dx = _array(0.0)
    dy = _array(0.0)
    dz = _array(0.0)
    temp = _zeros((sizex, sizey))
    tempvegdem = _zeros((sizex, sizey))
    tempvegdem2 = _zeros((sizex, sizey))
    templastfabovea = _zeros((sizex, sizey))
    templastgabovea = _zeros((sizex, sizey))
    bushplant = (bush > 1).astype(np.float32)
    sh = _zeros((sizex, sizey))
    vbshvegsh = _zeros((sizex, sizey))
    vegsh = _operate(np.add, _zeros((sizex, sizey)), bushplant)
    f = a
    shvoveg = vegdem
    wallbol = (walls > 0).astype(np.float32)
    pibyfour = _array(_divide(np.pi, 4.0))
    threetimespibyfour = _operate(np.multiply, 3, pibyfour)
    fivetimespibyfour = _operate(np.multiply, 5, pibyfour)
    seventimespibyfour = _operate(np.multiply, 7, pibyfour)
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)
    dssin = np.abs(_divide(1, sinazimuth))
    dscos = np.abs(_divide(1, cosazimuth))
    tanaltitudebyscale = _divide(np.tan(altitude), scale)
    index = 0
    dzprev = _array(0.0)
    fabovea = None
    gabovea = None
    lastfabovea = None
    lastgabovea = None
    vegsh2 = None
    while amaxvalue >= dz and np.abs(dx) < sizex and (np.abs(dy) < sizey):
        if pibyfour <= azimuth and azimuth < threetimespibyfour or (fivetimespibyfour <= azimuth and azimuth < seventimespibyfour):
            dy = _operate(np.multiply, signsinazimuth, index)
            dx = _operate(np.multiply, _operate(np.multiply, -1, signcosazimuth), np.abs(np.round(_divide(index, tanazimuth))))
            ds = dssin
        else:
            dy = _operate(np.multiply, signsinazimuth, np.abs(np.round(_operate(np.multiply, index, tanazimuth))))
            dx = _operate(np.multiply, _operate(np.multiply, -1, signcosazimuth), index)
            ds = dscos
        dz = _operate(np.multiply, _operate(np.multiply, ds, index), tanaltitudebyscale)
        tempvegdem.fill(0)
        tempvegdem2.fill(0)
        temp.fill(0)
        templastfabovea.fill(0)
        templastgabovea.fill(0)
        absdx = np.abs(dx)
        absdy = np.abs(dy)
        xc1 = int(_divide(_operate(np.add, dx, absdx), 2))
        xc2 = int(_operate(np.add, sizex, _divide(_operate(np.subtract, dx, absdx), 2)))
        yc1 = int(_divide(_operate(np.add, dy, absdy), 2))
        yc2 = int(_operate(np.add, sizey, _divide(_operate(np.subtract, dy, absdy), 2)))
        xp1 = -int(_divide(_operate(np.subtract, dx, absdx), 2))
        xp2 = int(_operate(np.subtract, sizex, _divide(_operate(np.add, dx, absdx), 2)))
        yp1 = -int(_divide(_operate(np.subtract, dy, absdy), 2))
        yp2 = int(_operate(np.subtract, sizey, _divide(_operate(np.add, dy, absdy), 2)))
        tempvegdem[xp1:xp2, yp1:yp2] = _operate(np.subtract, vegdem[xc1:xc2, yc1:yc2], dz)
        tempvegdem2[xp1:xp2, yp1:yp2] = _operate(np.subtract, vegdem2[xc1:xc2, yc1:yc2], dz)
        temp[xp1:xp2, yp1:yp2] = _operate(np.subtract, a[xc1:xc2, yc1:yc2], dz)
        f = np.maximum(f, temp)
        shvoveg = np.maximum(shvoveg, tempvegdem)
        sh = np.where(f > a, _array(1.0), _array(0.0))
        fabovea = (tempvegdem > a).astype(np.float32)
        gabovea = (tempvegdem2 > a).astype(np.float32)
        templastfabovea[xp1:xp2, yp1:yp2] = _operate(np.subtract, vegdem[xc1:xc2, yc1:yc2], dzprev)
        templastgabovea[xp1:xp2, yp1:yp2] = _operate(np.subtract, vegdem2[xc1:xc2, yc1:yc2], dzprev)
        lastfabovea = templastfabovea > a
        lastgabovea = templastgabovea > a
        dzprev = dz
        vegsh2 = _operate(np.add, _operate(np.add, _operate(np.add, fabovea, gabovea), lastfabovea.astype(np.float32)), lastgabovea.astype(np.float32))
        vegsh2 = np.where(vegsh2 == 4, _array(0.0), vegsh2)
        vegsh2 = np.where(vegsh2 > 0, _array(1.0), vegsh2)
        vegsh = np.maximum(vegsh, vegsh2)
        vegsh = np.where(_operate(np.multiply, vegsh, sh) > 0, _array(0.0), vegsh)
        vbshvegsh = _operate(np.add, vbshvegsh, vegsh)
        index += 1
    azilow = _operate(np.subtract, azimuth, _divide(np.pi, 2))
    azihigh = _operate(np.add, azimuth, _divide(np.pi, 2))
    if azilow >= 0 and azihigh < _operate(np.multiply, 2, np.pi):
        facesh = _operate(np.add, _operate(np.subtract, np.logical_or(aspect < azilow, aspect >= azihigh).astype(np.float32), wallbol), 1)
    elif azilow < 0 and azihigh <= _operate(np.multiply, 2, np.pi):
        azilow += _operate(np.multiply, 2, np.pi)
        facesh = _operate(np.add, _operate(np.multiply, np.logical_or(aspect > azilow, aspect <= azihigh).astype(np.float32), -1), 1)
    elif azilow > 0 and azihigh >= _operate(np.multiply, 2, np.pi):
        azihigh -= _operate(np.multiply, 2, np.pi)
        facesh = _operate(np.add, _operate(np.multiply, np.logical_or(aspect > azilow, aspect <= azihigh).astype(np.float32), -1), 1)
    sh = _operate(np.subtract, 1, sh)
    vbshvegsh = np.where(vbshvegsh > 0, _array(1.0), vbshvegsh)
    vbshvegsh = _operate(np.subtract, vbshvegsh, vegsh)
    vegsh = np.where(vegsh > 0, _array(1.0), vegsh)
    shvoveg = _operate(np.multiply, _operate(np.subtract, shvoveg, a), vegsh)
    vegsh = _operate(np.subtract, 1, vegsh)
    vbshvegsh = _operate(np.subtract, 1, vbshvegsh)
    shvo = _operate(np.subtract, f, a)
    facesun = np.logical_and(_operate(np.add, facesh, wallbol) == 1, walls > 0).astype(np.float32)
    wallsun = _operate(np.subtract, walls, shvo)
    wallsun = np.where(wallsun < 0, _array(0.0), wallsun)
    wallsun = np.where(facesh == 1, _array(0.0), wallsun)
    wallsh = _operate(np.subtract, walls, wallsun)
    wallshve = _operate(np.multiply, shvoveg, wallbol)
    wallshve = _operate(np.subtract, wallshve, wallsh)
    wallshve = np.where(wallshve < 0, _array(0.0), wallshve)
    wallsun = _operate(np.subtract, wallsun, wallshve)
    wallsun = np.where(wallsun < 0, _array(0.0), wallsun)
    wallshve = np.where(wallshve > walls, walls, wallshve)
    del fabovea, gabovea, lastfabovea, lastgabovea, vegsh2
    del tempvegdem, tempvegdem2, templastfabovea, templastgabovea, shvoveg, wallbol
    return (vegsh, sh, vbshvegsh, wallsh, wallsun, wallshve, facesh, facesun)

def Perez_v3(zen, azimuth, radD, radI, jday, patchchoice, patch_option):
    """
    Calculate anisotropic diffuse radiation distribution using Perez model.
    
    Implements the Perez all-weather sky model for anisotropic diffuse radiation,
    accounting for circumsolar brightening and horizon brightening.
    
    Args:
        zen (np.ndarray): Solar zenith angle (radians)
        azimuth (np.ndarray): Solar azimuth (radians)
        radD (float): Diffuse radiation (W/m²)
        radI (float): Direct beam radiation (W/m²)
        jday (np.ndarray): Julian day
        patchchoice (int): Patch selection
        patch_option (int): Sky discretization (144 or 2304)
    
    Returns:
        tuple: Patch-wise diffuse radiation distribution and anisotropic SVF
    
    Reference:
        Perez et al. (1993). All-weather model for sky luminance distribution.
        Solar Energy, 50(3), 235-245.
    """
    m_a1 = _array([1.3525, -1.2219, -1.1, -0.5484, -0.6, -1.0156, -1.0, -1.05])
    m_a2 = _array([-0.2576, -0.773, -0.2515, -0.6654, -0.3566, -0.367, 0.0211, 0.0289])
    m_a3 = _array([-0.269, 1.4148, 0.8952, -0.2672, -2.5, 1.0078, 0.5025, 0.426])
    m_a4 = _array([-1.4366, 1.1016, 0.0156, 0.7117, 2.325, 1.4051, -0.5119, 0.359])
    m_b1 = _array([-0.767, -0.2054, 0.2782, 0.7234, 0.2937, 0.2875, -0.3, -0.325])
    m_b2 = _array([0.0007, 0.0367, -0.1812, -0.6219, 0.0496, -0.5328, 0.1922, 0.1156])
    m_b3 = _array([1.2734, -3.9128, -4.5, -5.6812, -5.6812, -3.85, 0.7023, 0.7781])
    m_b4 = _array([-0.1233, 0.9156, 1.1766, 2.6297, 1.8415, 3.375, -1.6317, 0.0025])
    m_c1 = _array([2.8, 6.975, 24.7219, 33.3389, 21.0, 14.0, 19.0, 31.0625])
    m_c2 = _array([0.6004, 0.1774, -13.0812, -18.3, -4.7656, -0.9999, -5.0, -14.5])
    m_c3 = _array([1.2375, 6.4477, -37.7, -62.25, -21.5906, -7.1406, 1.2438, -46.1148])
    m_c4 = _array([1.0, -0.1239, 34.8438, 52.0781, 7.2492, 7.5469, -1.9094, 55.375])
    m_d1 = _array([1.8734, -1.5798, -5.0, -3.5, -3.5, -3.4, -4.0, -7.2312])
    m_d2 = _array([0.6297, -0.5081, 1.5218, 0.0016, -0.1554, -0.1078, 0.025, 0.405])
    m_d3 = _array([0.9738, -1.7812, 3.9229, 1.1477, 1.4062, -1.075, 0.3844, 13.35])
    m_d4 = _array([0.2809, 0.108, -2.6204, 0.1062, 0.3988, 1.5702, 0.2656, 0.6234])
    m_e1 = _array([0.0356, 0.2624, -0.0156, 0.4659, 0.0032, -0.0672, 1.0468, 1.5])
    m_e2 = _array([-0.1246, 0.0672, 0.1597, -0.3296, 0.0766, 0.4016, -0.3788, -0.6426])
    m_e3 = _array([-0.5718, -0.219, 0.4199, -0.0876, -0.0656, 0.3017, -2.4517, 1.8564])
    m_e4 = _array([0.9938, -0.4285, -0.5562, -0.0329, -0.1294, -0.4844, 1.4656, 0.5636])
    acoeff = np.stack([m_a1, m_a2, m_a3, m_a4], axis=1)
    bcoeff = np.stack([m_b1, m_b2, m_b3, m_b4], axis=1)
    ccoeff = np.stack([m_c1, m_c2, m_c3, m_c4], axis=1)
    dcoeff = np.stack([m_d1, m_d2, m_d3, m_d4], axis=1)
    ecoeff = np.stack([m_e1, m_e2, m_e3, m_e4], axis=1)
    deg2rad = np.copy(_array(_divide(np.pi, 180)))
    rad2deg = np.copy(_array(_divide(180, np.pi)))
    altitude = _operate(np.subtract, 90, zen)
    zen = _operate(np.multiply, _array(zen), deg2rad)
    azimuth = _operate(np.multiply, _array(azimuth), deg2rad)
    altitude = _operate(np.multiply, _array(altitude), deg2rad)
    Idh = radD
    Ibn = radI
    PerezClearness = _divide(_divide(_operate(np.add, Idh, Ibn), _operate(np.add, Idh, _operate(np.multiply, 1.041, np.power(zen, 3)))), _operate(np.add, 1, _operate(np.multiply, 1.041, np.power(zen, 3))))
    day_angle = _divide(_operate(np.multiply, _operate(np.multiply, jday, 2), np.pi), 365)
    I0 = _operate(np.multiply, 1367, _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, 1.00011, _operate(np.multiply, 0.034221, np.cos(_array(day_angle)))), _operate(np.multiply, 0.00128, np.sin(_array(day_angle)))), _operate(np.multiply, 0.000719, np.cos(_operate(np.multiply, 2, _array(day_angle))))), _operate(np.multiply, 7.7e-05, np.sin(_operate(np.multiply, 2, _array(day_angle))))))
    if altitude >= _operate(np.multiply, 10, deg2rad):
        AirMass = _divide(1, np.sin(altitude))
    elif altitude < 0:
        raise TypeError("Pinned upstream Perez negative-altitude branch passes an integer imaginary argument to complex().")
    else:
        AirMass = _operate(np.add, _divide(1, np.sin(altitude)), _operate(np.multiply, 0.50572, np.power(_operate(np.add, _divide(_operate(np.multiply, 180, altitude), np.pi), 6.07995), -1.6364)))
    PerezBrightness = _divide(_operate(np.multiply, AirMass, Idh), I0)
    if Idh <= 10:
        PerezBrightness = _array(0.0)
    if PerezClearness < 1.065:
        intClearness = 0
    elif PerezClearness < 1.23:
        intClearness = 1
    elif PerezClearness < 1.5:
        intClearness = 2
    elif PerezClearness < 1.95:
        intClearness = 3
    elif PerezClearness < 2.8:
        intClearness = 4
    elif PerezClearness < 4.5:
        intClearness = 5
    elif PerezClearness < 6.2:
        intClearness = 6
    else:
        intClearness = 7
    m_a = _operate(np.add, _operate(np.add, acoeff[intClearness, 0], _operate(np.multiply, acoeff[intClearness, 1], zen)), _operate(np.multiply, PerezBrightness, _operate(np.add, acoeff[intClearness, 2], _operate(np.multiply, acoeff[intClearness, 3], zen))))
    m_b = _operate(np.add, _operate(np.add, bcoeff[intClearness, 0], _operate(np.multiply, bcoeff[intClearness, 1], zen)), _operate(np.multiply, PerezBrightness, _operate(np.add, bcoeff[intClearness, 2], _operate(np.multiply, bcoeff[intClearness, 3], zen))))
    m_e = _operate(np.add, _operate(np.add, ecoeff[intClearness, 0], _operate(np.multiply, ecoeff[intClearness, 1], zen)), _operate(np.multiply, PerezBrightness, _operate(np.add, ecoeff[intClearness, 2], _operate(np.multiply, ecoeff[intClearness, 3], zen))))
    if intClearness > 0:
        m_c = _operate(np.add, _operate(np.add, ccoeff[intClearness, 0], _operate(np.multiply, ccoeff[intClearness, 1], zen)), _operate(np.multiply, PerezBrightness, _operate(np.add, ccoeff[intClearness, 2], _operate(np.multiply, ccoeff[intClearness, 3], zen))))
        m_d = _operate(np.add, _operate(np.add, dcoeff[intClearness, 0], _operate(np.multiply, dcoeff[intClearness, 1], zen)), _operate(np.multiply, PerezBrightness, _operate(np.add, dcoeff[intClearness, 2], _operate(np.multiply, dcoeff[intClearness, 3], zen))))
    else:
        m_c = _operate(np.subtract, np.exp(np.power(_operate(np.multiply, PerezBrightness, _operate(np.add, ccoeff[intClearness, 0], _operate(np.multiply, ccoeff[intClearness, 1], zen))), ccoeff[intClearness, 2])), 1)
        m_d = _operate(np.add, _operate(np.add, -np.exp(_operate(np.multiply, PerezBrightness, _operate(np.add, dcoeff[intClearness, 0], _operate(np.multiply, dcoeff[intClearness, 1], zen)))), dcoeff[intClearness, 2]), _operate(np.multiply, _operate(np.multiply, PerezBrightness, dcoeff[intClearness, 3]), PerezBrightness))
    if patchchoice == 2:
        skyvaultalt = _zeros((90, 361))
        skyvaultazi = _zeros((90, 361))
        for j in range(90):
            skyvaultalt[j, :] = _operate(np.subtract, 91, j)
            skyvaultazi[j, :] = np.arange(361)
    elif patchchoice == 1:
        skyvaultalt, skyvaultazi, _, _, _, _, _ = create_patches(patch_option)
    skyvaultzen = _operate(np.multiply, _operate(np.subtract, 90, skyvaultalt), deg2rad)
    skyvaultalt = _operate(np.multiply, skyvaultalt, deg2rad)
    skyvaultazi = _operate(np.multiply, skyvaultazi, deg2rad)
    cosSkySunAngle = _operate(np.add, _operate(np.multiply, np.sin(skyvaultalt), np.sin(altitude)), _operate(np.multiply, _operate(np.multiply, np.cos(altitude), np.cos(skyvaultalt)), np.cos(np.abs(_operate(np.subtract, skyvaultazi, azimuth)))))
    lv = _operate(np.multiply, _operate(np.add, 1, _operate(np.multiply, m_a, np.exp(_divide(m_b, np.cos(skyvaultzen))))), _operate(np.add, _operate(np.add, 1, _operate(np.multiply, m_c, np.exp(_operate(np.multiply, m_d, np.arccos(cosSkySunAngle))))), _operate(np.multiply, _operate(np.multiply, m_e, cosSkySunAngle), cosSkySunAngle)))
    lv = _divide(lv, np.sum(lv))
    if patchchoice == 1:
        x = np.swapaxes(np.expand_dims(_operate(np.multiply, skyvaultalt, rad2deg), 0), 0, 1)
        y = np.swapaxes(np.expand_dims(_operate(np.multiply, skyvaultazi, rad2deg), 0), 0, 1)
        z = np.swapaxes(np.expand_dims(lv, 0), 0, 1)
        lv = np.concatenate((x, y, z), axis=1)
    return (lv, PerezClearness, PerezBrightness)

def model1(sky_patches, esky, Ta):
    """Calculate longwave sky radiation using Model 1 (isotropic)."""
    SBC = 5.67051e-08
    deg2rad = _array(_divide(np.pi, 180))
    skyalt, skyalt_c = np.unique(sky_patches[:, 0], return_counts=True)
    skyzen = _operate(np.subtract, 90, skyalt)
    cosskyzen = np.cos(_operate(np.multiply, skyzen, deg2rad))
    a_c = 0.67
    b_c = 0.094
    ln_u_prec = _operate(np.subtract, _operate(np.subtract, _divide(esky, b_c), _divide(a_c, b_c)), 0.5)
    u_prec = np.exp(ln_u_prec)
    owp = _divide(u_prec, cosskyzen)
    log_owp = np.log(owp)
    esky_band = _operate(np.add, a_c, _operate(np.multiply, b_c, log_owp))
    p_alt = sky_patches[:, 0]
    patch_emissivity = _zeros(p_alt.shape[0])
    for idx in skyalt:
        temp_emissivity = esky_band[skyalt == idx]
        patch_emissivity[p_alt == idx] = temp_emissivity
    patch_emissivity_normalized = _divide(patch_emissivity, np.sum(patch_emissivity))
    return (patch_emissivity_normalized, esky_band)

def model2(sky_patches, esky, Ta):
    """Calculate longwave sky radiation using Model 2 (simple anisotropic)."""
    deg2rad = _array(_divide(np.pi, 180))
    skyalt, skyalt_c = np.unique(sky_patches[:, 0], return_counts=True)
    skyzen = _operate(np.subtract, 90, skyalt)
    b_c = 0.308
    esky_band = _operate(np.subtract, 1, _operate(np.multiply, _operate(np.subtract, 1, esky), np.exp(_operate(np.multiply, b_c, _operate(np.subtract, 1.7, _divide(1, np.cos(_operate(np.multiply, skyzen, deg2rad))))))))
    p_alt = sky_patches[:, 0]
    patch_emissivity = _zeros(p_alt.shape[0])
    for idx in skyalt:
        temp_emissivity = esky_band[skyalt == idx]
        patch_emissivity[p_alt == idx] = temp_emissivity
    patch_emissivity_normalized = _divide(patch_emissivity, np.sum(patch_emissivity))
    return (patch_emissivity_normalized, esky_band)

def model3(sky_patches, esky, Ta):
    """Calculate longwave sky radiation using Model 3 (advanced anisotropic)."""
    deg2rad = _array(_divide(np.pi, 180))
    skyalt, skyalt_c = np.unique(sky_patches[:, 0], return_counts=True)
    skyzen = _operate(np.subtract, 90, skyalt)
    b_c = 1.8
    esky_band = _operate(np.subtract, 1, _operate(np.power, _operate(np.subtract, 1, esky), _divide(1, _operate(np.multiply, b_c, np.cos(_operate(np.multiply, skyzen, deg2rad))))))
    p_alt = sky_patches[:, 0]
    patch_emissivity = _zeros(p_alt.shape[0])
    for idx in skyalt:
        temp_emissivity = esky_band[skyalt == idx]
        patch_emissivity[p_alt == idx] = temp_emissivity
    patch_emissivity_normalized = _divide(patch_emissivity, np.sum(patch_emissivity))
    return (patch_emissivity_normalized, esky_band)

def define_patch_characteristics(solar_altitude, solar_azimuth, patch_altitude, patch_azimuth, steradian, asvf, shmat, vegshmat, vbshvegshmat, Lsky_down, Lsky_side, Lsky, Lup, Ta, Tgwall, ewall, rows, cols):
    """
    Calculate longwave radiation from discretized sky hemisphere patches.
    
    Computes downward and sideward longwave radiation by integrating
    contributions from individual sky patches, accounting for their
    position, solid angle, shadow state, and temperature.
    
    Args:
        solar_altitude, solar_azimuth (float): Solar position (degrees)
        patch_altitude, patch_azimuth (np.ndarray): Patch positions
        steradian (np.ndarray): Solid angle of each patch
        asvf (np.ndarray): Anisotropic sky view factor
        shmat, vegshmat, vbshvegshmat (np.ndarray): Shadow matrices
        Lsky_down, Lsky_side, Lsky (np.ndarray): Sky longwave components
        Lup (np.ndarray): Upward longwave from ground
        Ta (float): Air temperature (°C)
        Tgwall (np.ndarray): Wall/ground temperature (K)
        ewall (float): Wall emissivity
        rows, cols (int): Grid dimensions
    
    Returns:
        tuple: (Ldown, Lside, Lside_sky, Lside_veg, Lside_sh, Lside_sun, 
                Lside_ref, Least, Lwest, Lnorth, Lsouth) - Longwave components
                for downward, sideward (total and directional)
    
    Notes:
        - Integrates over all sky patches using solid angle weighting
        - Accounts for patch visibility through shadow matrices
        - Distinguishes between sunlit and shaded patches
        - Computes directional components (E, S, W, N)
    """
    SBC = _array(5.67051e-08)
    deg2rad = _array(_divide(np.pi, 180))
    Ldown = _zeros((rows, cols))
    Ldown_sky = _zeros((rows, cols))
    Ldown_veg = _zeros((rows, cols))
    Ldown_sun = _zeros((rows, cols))
    Ldown_sh = _zeros((rows, cols))
    Ldown_ref = _zeros((rows, cols))
    Lside = _zeros((rows, cols))
    Lside_sky = _zeros((rows, cols))
    Lside_veg = _zeros((rows, cols))
    Lside_sun = _zeros((rows, cols))
    Lside_sh = _zeros((rows, cols))
    Lside_ref = _zeros((rows, cols))
    Least = _zeros((rows, cols))
    Lwest = _zeros((rows, cols))
    Lnorth = _zeros((rows, cols))
    Lsouth = _zeros((rows, cols))
    ewall = np.copy(_array(ewall))
    for idx in range(patch_altitude.shape[0]):
        temp_sky = (shmat[:, :, idx] == 1) & (vegshmat[:, :, idx] == 1)
        Ldown_sky += _operate(np.multiply, temp_sky, Lsky_down[idx, 2])
        Lside_sky += _operate(np.multiply, temp_sky, Lsky_side[idx, 2])
        temp_vegsh = (vegshmat[:, :, idx] == 0) | (vbshvegshmat[:, :, idx] == 0)
        vegetation_surface = _divide(_operate(np.multiply, _operate(np.multiply, ewall, SBC), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), _array(np.pi))
        Lside_veg += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, vegetation_surface, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_vegsh)
        Ldown_veg += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, vegetation_surface, steradian[idx]), np.sin(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_vegsh)
        if patch_azimuth[idx] > 360 or patch_azimuth[idx] < 180:
            Least += _operate(np.multiply, _operate(np.multiply, temp_sky, Lsky_side[idx, 2]), np.cos(_operate(np.multiply, _operate(np.subtract, 90, patch_azimuth[idx]), deg2rad)))
            Least += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, vegetation_surface, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_vegsh), np.cos(_operate(np.multiply, _operate(np.subtract, 90, patch_azimuth[idx]), deg2rad)))
        if patch_azimuth[idx] > 90 and patch_azimuth[idx] < 270:
            Lsouth += _operate(np.multiply, _operate(np.multiply, temp_sky, Lsky_side[idx, 2]), np.cos(_operate(np.multiply, _operate(np.subtract, 180, patch_azimuth[idx]), deg2rad)))
            Lsouth += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, vegetation_surface, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_vegsh), np.cos(_operate(np.multiply, _operate(np.subtract, 180, patch_azimuth[idx]), deg2rad)))
        if patch_azimuth[idx] > 180 and patch_azimuth[idx] < 360:
            Lwest += _operate(np.multiply, _operate(np.multiply, temp_sky, Lsky_side[idx, 2]), np.cos(_operate(np.multiply, _operate(np.subtract, 270, patch_azimuth[idx]), deg2rad)))
            Lwest += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, vegetation_surface, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_vegsh), np.cos(_operate(np.multiply, _operate(np.subtract, 270, patch_azimuth[idx]), deg2rad)))
        if patch_azimuth[idx] > 270 or patch_azimuth[idx] < 90:
            Lnorth += _operate(np.multiply, _operate(np.multiply, temp_sky, Lsky_side[idx, 2]), np.cos(_operate(np.multiply, _operate(np.subtract, 0, patch_azimuth[idx]), deg2rad)))
            Lnorth += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, vegetation_surface, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_vegsh), np.cos(_operate(np.multiply, _operate(np.subtract, 0, patch_azimuth[idx]), deg2rad)))
        temp_vbsh = _operate(np.multiply, _operate(np.subtract, 1, shmat[:, :, idx]), vbshvegshmat[:, :, idx])
        temp_sh = temp_vbsh == 1
        azimuth_difference = np.abs(_operate(np.subtract, solar_azimuth, patch_azimuth[idx]))
        sunlit_surface = _divide(_operate(np.multiply, _operate(np.multiply, ewall, SBC), _operate(np.power, _operate(np.add, _operate(np.add, Ta, Tgwall), 273.15), 4)), _array(np.pi))
        shaded_surface = _divide(_operate(np.multiply, _operate(np.multiply, ewall, SBC), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), _array(np.pi))
        if azimuth_difference > 90 and azimuth_difference < 270 and (solar_altitude > 0):
            sunlit_patches, shaded_patches = shaded_or_sunlit(solar_altitude, solar_azimuth, patch_altitude[idx], patch_azimuth[idx], asvf)
            Lside_sun += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, sunlit_surface, sunlit_patches), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh)
            Lside_sh += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, shaded_patches), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh)
            Ldown_sun += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, sunlit_surface, sunlit_patches), steradian[idx]), np.sin(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh)
            Ldown_sh += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, shaded_patches), steradian[idx]), np.sin(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh)
            if patch_azimuth[idx] > 360 or patch_azimuth[idx] < 180:
                Least += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, sunlit_surface, sunlit_patches), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.subtract, 90, patch_azimuth[idx]), deg2rad)))
                Least += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, shaded_patches), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.subtract, 90, patch_azimuth[idx]), deg2rad)))
            if patch_azimuth[idx] > 90 and patch_azimuth[idx] < 270:
                Lsouth += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, sunlit_surface, sunlit_patches), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.subtract, 180, patch_azimuth[idx]), deg2rad)))
                Lsouth += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, shaded_patches), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.subtract, 180, patch_azimuth[idx]), deg2rad)))
            if patch_azimuth[idx] > 180 and patch_azimuth[idx] < 360:
                Lwest += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, sunlit_surface, sunlit_patches), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.subtract, 270, patch_azimuth[idx]), deg2rad)))
                Lwest += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, shaded_patches), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.subtract, 270, patch_azimuth[idx]), deg2rad)))
            if patch_azimuth[idx] > 270 or patch_azimuth[idx] < 90:
                Lnorth += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, sunlit_surface, sunlit_patches), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.subtract, 0, patch_azimuth[idx]), deg2rad)))
                Lnorth += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, shaded_patches), steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.subtract, 0, patch_azimuth[idx]), deg2rad)))
        else:
            Lside_sh += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh)
            Ldown_sh += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, steradian[idx]), np.sin(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh)
            if patch_azimuth[idx] > 360 or patch_azimuth[idx] < 180:
                Least += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.subtract, 90, patch_azimuth[idx]), deg2rad)))
            if patch_azimuth[idx] > 90 and patch_azimuth[idx] < 270:
                Lsouth += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.subtract, 180, patch_azimuth[idx]), deg2rad)))
            if patch_azimuth[idx] > 180 and patch_azimuth[idx] < 360:
                Lwest += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.subtract, 270, patch_azimuth[idx]), deg2rad)))
            if patch_azimuth[idx] > 270 or patch_azimuth[idx] < 90:
                Lnorth += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, shaded_surface, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.subtract, 0, patch_azimuth[idx]), deg2rad)))
    reflected_on_surfaces = _divide(_operate(np.multiply, _operate(np.multiply, _operate(np.add, Ldown_sky, Lup), _operate(np.subtract, 1, ewall)), 0.5), _array(np.pi))
    for idx in range(patch_altitude.shape[0]):
        temp_sh = (shmat[:, :, idx] == 0) | (vegshmat[:, :, idx] == 0) | (vbshvegshmat[:, :, idx] == 0)
        Lside_ref += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, reflected_on_surfaces, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh)
        Ldown_ref += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, reflected_on_surfaces, steradian[idx]), np.sin(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh)
        if patch_azimuth[idx] > 360 or patch_azimuth[idx] < 180:
            Least += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, reflected_on_surfaces, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.subtract, 90, patch_azimuth[idx]), deg2rad)))
        if patch_azimuth[idx] > 90 and patch_azimuth[idx] < 270:
            Lsouth += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, reflected_on_surfaces, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.subtract, 180, patch_azimuth[idx]), deg2rad)))
        if patch_azimuth[idx] > 180 and patch_azimuth[idx] < 360:
            Lwest += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, reflected_on_surfaces, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.subtract, 270, patch_azimuth[idx]), deg2rad)))
        if patch_azimuth[idx] > 270 or patch_azimuth[idx] < 90:
            Lnorth += _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, reflected_on_surfaces, steradian[idx]), np.cos(_operate(np.multiply, patch_altitude[idx], deg2rad))), temp_sh), np.cos(_operate(np.multiply, _operate(np.subtract, 0, patch_azimuth[idx]), deg2rad)))
    Lside = _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, Lside_sky, Lside_veg), Lside_sh), Lside_sun), Lside_ref)
    Ldown = _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, Ldown_sky, Ldown_veg), Ldown_sh), Ldown_sun), Ldown_ref)
    del Ldown_sky, Ldown_veg, Ldown_sun, Ldown_sh, Ldown_ref
    del temp_vegsh, vegetation_surface, temp_vbsh, temp_sh
    return (Ldown, Lside, Lside_sky, Lside_veg, Lside_sh, Lside_sun, Lside_ref, Least, Lwest, Lnorth, Lsouth)

def Lcyl_v2022a(esky, sky_patches, Ta, Tgwall, ewall, Lup, shmat, vegshmat, vbshvegshmat, solar_altitude, solar_azimuth, rows, cols, asvf):
    """
    Calculate longwave radiation on cylindrical surface (human body model).
    
    Computes longwave radiation received by a standing person from sky,
    ground, and wall surfaces, accounting for shadows and anisotropic effects.
    
    Args:
        esky (float): Sky emissivity
        sky_patches (np.ndarray): Sky hemisphere discretization
        Ta (float): Air temperature (°C)
        Tgwall (np.ndarray): Wall/ground temperature (K)
        ewall (float): Wall emissivity
        Lup (np.ndarray): Upward longwave radiation
        shmat, vegshmat, vbshvegshmat (np.ndarray): Shadow matrices
        solar_altitude, solar_azimuth (float): Solar angles
        rows, cols (int): Grid dimensions
        asvf (np.ndarray): Anisotropic SVF
    
    Returns:
        tuple: (Lsky, Lrefl) - Sky and reflected longwave components
    """
    SBC = np.copy(_array(5.67051e-08))
    Ldown_prata = _operate(np.multiply, _operate(np.multiply, esky, SBC), _operate(np.power, _operate(np.add, Ta, 273.15), 4))
    deg2rad = np.copy(_array(_divide(np.pi, 180)))
    sky_patches_cpu = sky_patches
    skyalt, skyalt_c = np.unique(sky_patches_cpu[:, 0], return_counts=True)
    skyalt = np.copy(_array(skyalt))
    skyalt_c = np.copy(_array(skyalt_c))
    patch_altitude = sky_patches[:, 0]
    patch_azimuth = sky_patches[:, 1]
    emis_m = 2
    if emis_m == 1:
        patch_emissivity_normalized, esky_band = model1(sky_patches, esky, Ta)
    elif emis_m == 2:
        patch_emissivity_normalized, esky_band = model2(sky_patches, esky, Ta)
    elif emis_m == 3:
        patch_emissivity_normalized, esky_band = model3(sky_patches, esky, Ta)
    steradian = _zeros(patch_altitude.shape[0])
    for i in range(patch_altitude.shape[0]):
        if skyalt_c[skyalt == patch_altitude[i]].item() > 1:
            steradian[i] = _operate(np.multiply, _operate(np.multiply, _divide(360, skyalt_c[skyalt == patch_altitude[i]].item()), deg2rad), _operate(np.subtract, np.sin(_operate(np.multiply, _operate(np.add, patch_altitude[i], patch_altitude[0]), deg2rad)), np.sin(_operate(np.multiply, _operate(np.subtract, patch_altitude[i], patch_altitude[0]), deg2rad))))
        else:
            steradian[i] = _operate(np.multiply, _operate(np.multiply, _divide(360, skyalt_c[skyalt == patch_altitude[i]].item()), deg2rad), _operate(np.subtract, np.sin(_operate(np.multiply, patch_altitude[i], deg2rad)), np.sin(_operate(np.multiply, _operate(np.add, patch_altitude[_operate(np.subtract, i, 1)], patch_altitude[0]), deg2rad))))
    anisotropic_sky = True
    Ldown = _zeros(patch_altitude.shape[0])
    Lside = _zeros(patch_altitude.shape[0])
    Lnormal = _zeros(patch_altitude.shape[0])
    for altitude in skyalt:
        if anisotropic_sky:
            temp_emissivity = esky_band[skyalt == altitude]
        else:
            temp_emissivity = esky
        Ldown[patch_altitude == altitude] = _operate(np.multiply, _operate(np.multiply, _divide(_operate(np.multiply, _operate(np.multiply, temp_emissivity, SBC), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), _array(np.pi)), steradian[patch_altitude == altitude]), np.sin(_operate(np.multiply, altitude, deg2rad)))
        Lside[patch_altitude == altitude] = _operate(np.multiply, _operate(np.multiply, _divide(_operate(np.multiply, _operate(np.multiply, temp_emissivity, SBC), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), _array(np.pi)), steradian[patch_altitude == altitude]), np.cos(_operate(np.multiply, altitude, deg2rad)))
        Lnormal[patch_altitude == altitude] = _operate(np.multiply, _divide(_operate(np.multiply, _operate(np.multiply, temp_emissivity, SBC), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), _array(np.pi)), steradian[patch_altitude == altitude])
    Lsky_normal = np.copy(sky_patches)
    Lsky_down = np.copy(sky_patches)
    Lsky_side = np.copy(sky_patches)
    Lsky_normal[:, 2] = Lnormal
    Lsky_down[:, 2] = Ldown
    Lsky_side[:, 2] = Lside
    Ldown, Lside, Lside_sky, Lside_veg, Lside_sh, Lside_sun, Lside_ref, Least_, Lwest_, Lnorth_, Lsouth_ = define_patch_characteristics(solar_altitude, solar_azimuth, patch_altitude, patch_azimuth, steradian, asvf, shmat, vegshmat, vbshvegshmat, Lsky_down, Lsky_side, Lsky_normal, Lup, Ta, Tgwall, ewall, rows, cols)
    del Lnormal, Lsky_normal, Lsky_down, Lsky_side
    return (Ldown, Lside, Least_, Lwest_, Lnorth_, Lsouth_)

def Lvikt_veg(svf, svfveg, svfaveg, vikttot):
    """Calculate longwave radiation weight factors accounting for vegetation."""
    viktonlywall = _divide(_operate(np.subtract, vikttot, _operate(np.subtract, _operate(np.add, _operate(np.subtract, _operate(np.add, _operate(np.subtract, _operate(np.multiply, 63.227, _operate(np.power, svf, 6)), _operate(np.multiply, 161.51, _operate(np.power, svf, 5))), _operate(np.multiply, 156.91, _operate(np.power, svf, 4))), _operate(np.multiply, 70.424, _operate(np.power, svf, 3))), _operate(np.multiply, 16.773, _operate(np.power, svf, 2))), _operate(np.multiply, 0.4863, svf))), vikttot)
    viktaveg = _divide(_operate(np.subtract, vikttot, _operate(np.subtract, _operate(np.add, _operate(np.subtract, _operate(np.add, _operate(np.subtract, _operate(np.multiply, 63.227, _operate(np.power, svfaveg, 6)), _operate(np.multiply, 161.51, _operate(np.power, svfaveg, 5))), _operate(np.multiply, 156.91, _operate(np.power, svfaveg, 4))), _operate(np.multiply, 70.424, _operate(np.power, svfaveg, 3))), _operate(np.multiply, 16.773, _operate(np.power, svfaveg, 2))), _operate(np.multiply, 0.4863, svfaveg))), vikttot)
    viktwall = _operate(np.subtract, viktonlywall, viktaveg)
    svfvegbu = _operate(np.subtract, _operate(np.add, svfveg, svf), 1)
    viktsky = _divide(_operate(np.subtract, _operate(np.add, _operate(np.subtract, _operate(np.add, _operate(np.subtract, _operate(np.multiply, 63.227, _operate(np.power, svfvegbu, 6)), _operate(np.multiply, 161.51, _operate(np.power, svfvegbu, 5))), _operate(np.multiply, 156.91, _operate(np.power, svfvegbu, 4))), _operate(np.multiply, 70.424, _operate(np.power, svfvegbu, 3))), _operate(np.multiply, 16.773, _operate(np.power, svfvegbu, 2))), _operate(np.multiply, 0.4863, svfvegbu)), vikttot)
    viktrefl = _divide(_operate(np.subtract, vikttot, _operate(np.subtract, _operate(np.add, _operate(np.subtract, _operate(np.add, _operate(np.subtract, _operate(np.multiply, 63.227, _operate(np.power, svfvegbu, 6)), _operate(np.multiply, 161.51, _operate(np.power, svfvegbu, 5))), _operate(np.multiply, 156.91, _operate(np.power, svfvegbu, 4))), _operate(np.multiply, 70.424, _operate(np.power, svfvegbu, 3))), _operate(np.multiply, 16.773, _operate(np.power, svfvegbu, 2))), _operate(np.multiply, 0.4863, svfvegbu))), vikttot)
    viktveg = _divide(_operate(np.subtract, vikttot, _operate(np.subtract, _operate(np.add, _operate(np.subtract, _operate(np.add, _operate(np.subtract, _operate(np.multiply, 63.227, _operate(np.power, svfvegbu, 6)), _operate(np.multiply, 161.51, _operate(np.power, svfvegbu, 5))), _operate(np.multiply, 156.91, _operate(np.power, svfvegbu, 4))), _operate(np.multiply, 70.424, _operate(np.power, svfvegbu, 3))), _operate(np.multiply, 16.773, _operate(np.power, svfvegbu, 2))), _operate(np.multiply, 0.4863, svfvegbu))), vikttot)
    viktveg = _operate(np.subtract, viktveg, viktwall)
    del viktonlywall, viktaveg, svfvegbu
    return (viktveg, viktwall, viktsky, viktrefl)

def Lside_veg_v2022a(svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg, azimuth, altitude, Ta, Tw, SBC, ewall, Ldown, esky, t, F_sh, CI, LupE, LupS, LupW, LupN, anisotropic_longwave):
    """
    Calculate longwave radiation on vertical surfaces (walls) with vegetation effects.
    
    Computes longwave radiation received by walls in the four cardinal directions,
    accounting for sky emission, ground emission, wall-to-wall exchanges, and
    vegetation obstruction.
    
    Args:
        svfS, svfW, svfN, svfE (np.ndarray): Directional sky view factors
        svf*veg (np.ndarray): Vegetation-obstructed SVFs
        svf*aveg (np.ndarray): Vegetation-above SVFs
        azimuth, altitude (float): Solar angles (degrees)
        Ta (float): Air temperature (°C)
        Tw (np.ndarray): Wall temperature
        SBC (float): Stefan-Boltzmann constant
        ewall (float): Wall emissivity
        Ldown (np.ndarray): Downward longwave
        esky (float): Sky emissivity
        t (float): Time parameter
        F_sh (np.ndarray): Shadow factor
        CI (np.ndarray): Clearness index
        LupE, LupS, LupW, LupN (np.ndarray): Upward longwave per direction
        anisotropic_longwave (bool): Use anisotropic model
    
    Returns:
        tuple: (Ldown, Lside, Least, Lwest, Lnorth, Lsouth) - Longwave components
    """
    azimuth = _array(azimuth)
    altitude = _array(altitude)
    ewall = _array(ewall)
    t = _array(t)
    anisotropic_longwave = _array(anisotropic_longwave)
    svfalfaE = np.arcsin(np.exp(_divide(np.log(_operate(np.subtract, 1, svfE)), 2)))
    svfalfaS = np.arcsin(np.exp(_divide(np.log(_operate(np.subtract, 1, svfS)), 2)))
    svfalfaW = np.arcsin(np.exp(_divide(np.log(_operate(np.subtract, 1, svfW)), 2)))
    svfalfaN = np.arcsin(np.exp(_divide(np.log(_operate(np.subtract, 1, svfN)), 2)))
    vikttot = _array(4.4897)
    aziW = _operate(np.add, azimuth, t)
    aziN = _operate(np.add, _operate(np.subtract, azimuth, 90), t)
    aziE = _operate(np.add, _operate(np.subtract, azimuth, 180), t)
    aziS = _operate(np.add, _operate(np.subtract, azimuth, 270), t)
    F_sh = _operate(np.subtract, _operate(np.multiply, 2, F_sh), 1)
    c = _operate(np.subtract, 1, CI)
    Lsky_allsky = _operate(np.add, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, esky, SBC), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), _operate(np.subtract, 1, c)), _operate(np.multiply, _operate(np.multiply, c, SBC), _operate(np.power, _operate(np.add, Ta, 273.15), 4)))
    viktveg, viktwall, viktsky, viktrefl = Lvikt_veg(svfE, svfEveg, svfEaveg, vikttot)
    if altitude > 0:
        alfaB = np.arctan(svfalfaE)
        betaB = np.arctan(np.tan(_operate(np.multiply, svfalfaE, F_sh)))
        betasun = _operate(np.add, _divide(_operate(np.subtract, alfaB, betaB), 2), betaB)
        if azimuth > _operate(np.subtract, 180, t) and azimuth <= _operate(np.subtract, 360, t):
            Lwallsun = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, _operate(np.add, Ta, 273.15), _operate(np.multiply, Tw, np.sin(_operate(np.multiply, aziE, _divide(_array(np.pi), 180))))), 4)), viktwall), _operate(np.subtract, 1, F_sh)), np.cos(betasun)), 0.5)
            Lwallsh = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), viktwall), F_sh), 0.5)
        else:
            Lwallsun = _array(0)
            Lwallsh = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), viktwall), 0.5)
    else:
        Lwallsun = _array(0)
        Lwallsh = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), viktwall), 0.5)
    if anisotropic_longwave == 1:
        Lground = _operate(np.multiply, LupE, 0.5)
        Least = Lground
    else:
        Lsky = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.subtract, _operate(np.add, svfE, svfEveg), 1), Lsky_allsky), viktsky), 0.5)
        Lveg = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), viktveg), 0.5)
        Lground = _operate(np.multiply, LupE, 0.5)
        Lrefl = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.add, Ldown, LupE), viktrefl), _operate(np.subtract, 1, ewall)), 0.5)
        Least = _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, Lsky, Lwallsun), Lwallsh), Lveg), Lground), Lrefl)
    viktveg, viktwall, viktsky, viktrefl = Lvikt_veg(svfS, svfSveg, svfSaveg, vikttot)
    if altitude > 0:
        alfaB = np.arctan(svfalfaS)
        betaB = np.arctan(np.tan(_operate(np.multiply, svfalfaS, F_sh)))
        betasun = _operate(np.add, _divide(_operate(np.subtract, alfaB, betaB), 2), betaB)
        if azimuth <= _operate(np.subtract, 90, t) or azimuth > _operate(np.subtract, 270, t):
            Lwallsun = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, _operate(np.add, Ta, 273.15), _operate(np.multiply, Tw, np.sin(_operate(np.multiply, aziS, _divide(_array(np.pi), 180))))), 4)), viktwall), _operate(np.subtract, 1, F_sh)), np.cos(betasun)), 0.5)
            Lwallsh = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), viktwall), F_sh), 0.5)
        else:
            Lwallsun = _array(0)
            Lwallsh = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), viktwall), 0.5)
    else:
        Lwallsun = _array(0)
        Lwallsh = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), viktwall), 0.5)
    if anisotropic_longwave == 1:
        Lground = _operate(np.multiply, LupS, 0.5)
        Lsouth = Lground
    else:
        Lsky = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.subtract, _operate(np.add, svfS, svfSveg), 1), Lsky_allsky), viktsky), 0.5)
        Lveg = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), viktveg), 0.5)
        Lground = _operate(np.multiply, LupS, 0.5)
        Lrefl = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.add, Ldown, LupS), viktrefl), _operate(np.subtract, 1, ewall)), 0.5)
        Lsouth = _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, Lsky, Lwallsun), Lwallsh), Lveg), Lground), Lrefl)
    viktveg, viktwall, viktsky, viktrefl = Lvikt_veg(svfW, svfWveg, svfWaveg, vikttot)
    if altitude > 0:
        alfaB = np.arctan(svfalfaW)
        betaB = np.arctan(np.tan(_operate(np.multiply, svfalfaW, F_sh)))
        betasun = _operate(np.add, _divide(_operate(np.subtract, alfaB, betaB), 2), betaB)
        if azimuth > _operate(np.subtract, 360, t) or azimuth <= _operate(np.subtract, 180, t):
            Lwallsun = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, _operate(np.add, Ta, 273.15), _operate(np.multiply, Tw, np.sin(_operate(np.multiply, aziW, _divide(_array(np.pi), 180))))), 4)), viktwall), _operate(np.subtract, 1, F_sh)), np.cos(betasun)), 0.5)
            Lwallsh = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), viktwall), F_sh), 0.5)
        else:
            Lwallsun = _array(0)
            Lwallsh = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), viktwall), 0.5)
    else:
        Lwallsun = _array(0)
        Lwallsh = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), viktwall), 0.5)
    if anisotropic_longwave == 1:
        Lground = _operate(np.multiply, LupW, 0.5)
        Lwest = Lground
    else:
        Lsky = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.subtract, _operate(np.add, svfW, svfWveg), 1), Lsky_allsky), viktsky), 0.5)
        Lveg = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), viktveg), 0.5)
        Lground = _operate(np.multiply, LupW, 0.5)
        Lrefl = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.add, Ldown, LupW), viktrefl), _operate(np.subtract, 1, ewall)), 0.5)
        Lwest = _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, Lsky, Lwallsun), Lwallsh), Lveg), Lground), Lrefl)
    viktveg, viktwall, viktsky, viktrefl = Lvikt_veg(svfN, svfNveg, svfNaveg, vikttot)
    if altitude > 0:
        alfaB = np.arctan(svfalfaN)
        betaB = np.arctan(np.tan(_operate(np.multiply, svfalfaN, F_sh)))
        betasun = _operate(np.add, _divide(_operate(np.subtract, alfaB, betaB), 2), betaB)
        if azimuth > _operate(np.subtract, 90, t) and azimuth <= _operate(np.subtract, 270, t):
            Lwallsun = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, _operate(np.add, Ta, 273.15), _operate(np.multiply, Tw, np.sin(_operate(np.multiply, aziN, _divide(_array(np.pi), 180))))), 4)), viktwall), _operate(np.subtract, 1, F_sh)), np.cos(betasun)), 0.5)
            Lwallsh = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), viktwall), F_sh), 0.5)
        else:
            Lwallsun = _array(0)
            Lwallsh = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), viktwall), 0.5)
    else:
        Lwallsun = _array(0)
        Lwallsh = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), viktwall), 0.5)
    if anisotropic_longwave == 1:
        Lground = _operate(np.multiply, LupN, 0.5)
        Lnorth = Lground
    else:
        Lsky = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.subtract, _operate(np.add, svfN, svfNveg), 1), Lsky_allsky), viktsky), 0.5)
        Lveg = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), viktveg), 0.5)
        Lground = _operate(np.multiply, LupN, 0.5)
        Lrefl = _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.add, Ldown, LupN), viktrefl), _operate(np.subtract, 1, ewall)), 0.5)
        Lnorth = _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.add, Lsky, Lwallsun), Lwallsh), Lveg), Lground), Lrefl)
    del LupE, LupS, LupW, LupN, svfalfaE, svfalfaS, svfalfaW, svfalfaN
    del viktveg, viktwall, viktsky, viktrefl
    return (Least, Lsouth, Lwest, Lnorth)

def Solweig_2022a_calc(i, dsm, scale, rows, cols, svf, svfN, svfW, svfE, svfS, svfveg, svfNveg, svfEveg, svfSveg, svfWveg, svfaveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg, vegdem, vegdem2, albedo_b, absK, absL, ewall, Fside, Fup, Fcyl, altitude, azimuth, zen, jday, usevegdem, onlyglobal, buildings, location, psi, landcover, lc_grid, dectime, altmax, dirwalls, walls, cyl, elvis, Ta, RH, radG, radD, radI, P, amaxvalue, bush, Twater, TgK, Tstart, alb_grid, emis_grid, TgK_wall, Tstart_wall, TmaxLST, TmaxLST_wall, first, second, svfalfa, svfbuveg, firstdaytime, timeadd, timestepdec, Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, CI, TgOut1, diffsh, shmat, vegshmat, vbshvegshmat, anisotropic_sky, asvf, patch_option):
    """
    Main SOLWEIG 2022a calculation kernel - integrates all radiation and temperature calculations.
    
    This is the core GPU-accelerated function that computes:
    - Shortwave radiation (direct, diffuse, reflected)
    - Longwave radiation (sky, ground, wall emissions)
    - Surface energy balance
    - Ground and wall surface temperatures
    - Mean radiant temperature (Tmrt)
    
    This function is called once per time step and performs the complete
    radiation budget calculation accounting for 3D urban geometry, vegetation,
    and surface-atmosphere interactions.
    
    Args:
        i (int): Time step index
        dsm (np.ndarray): Digital Surface Model
        scale (float): Grid resolution (pixels/meter)
        rows, cols (int): Domain dimensions
        svf* (np.ndarray): Sky view factors (multiple directional variants)
        vegdem, vegdem2, bush (np.ndarray): Vegetation layers
        albedo_b, absK, absL, ewall (float): Surface optical/thermal properties
        Fside, Fup, Fcyl (np.ndarray): Form factors for different geometries
        altitude, azimuth, zen (np.ndarray): Solar geometry
        jday, dectime, altmax (np.ndarray): Temporal parameters
        usevegdem, onlyglobal (bool): Model configuration flags
        buildings (np.ndarray): Building footprint mask
        location (dict): Geographic coordinates
        psi (np.ndarray): Tilt angles
        landcover, lc_grid: Land cover classification
        dirwalls, walls, cyl (np.ndarray): Wall geometry
        elvis (np.ndarray): Elevation data
        Ta, RH, P (float): Meteorological conditions (air temp, humidity, pressure)
        radG, radD, radI (float): Incoming radiation components (global, diffuse, direct)
        amaxvalue (float): Maximum domain elevation
        Twater (float): Water surface temperature
        TgK, Tstart, TgK_wall, Tstart_wall (np.ndarray): Temperature states
        TmaxLST, TmaxLST_wall (np.ndarray): Maximum temperatures
        alb_grid, emis_grid (np.ndarray): Spatial albedo and emissivity
        first, second (np.ndarray): Surface type classifications
        svfalfa, svfbuveg (np.ndarray): Vegetation view factors
        firstdaytime, timeadd, timestepdec (float): Temporal parameters
        Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N (np.ndarray): Previous temperature maps
        CI, TgOut1 (np.ndarray): Clearness index and output temperature
        diffsh, shmat, vegshmat, vbshvegshmat (np.ndarray): Shadow matrices
        anisotropic_sky (bool): Use anisotropic sky model
        asvf (np.ndarray): Anisotropic SVF
        patch_option (int): Sky discretization option
    
    Returns:
        tuple: (KsideI, TgOut1, TgOut, radIout, radDout, Lside, Lsky_patch, CI_Tg, CI_TgG, 
                KsideD, dRad, Kside) - Comprehensive radiation and temperature outputs
    
    Notes:
        - GPU-accelerated for performance
        - Most computationally intensive function in SOLWEIG
        - Implements surface energy balance with iteration
        - Accounts for multiple reflections and anisotropic effects
    """
    t = 0.0
    altitude = np.copy(_array(altitude))
    azimuth = np.copy(_array(azimuth))
    zen = np.copy(_array(zen))
    dectime = np.copy(_array(dectime))
    altmax = np.copy(_array(altmax))
    Twater = np.copy(_array(Twater))
    SBC = np.copy(_array(5.67051e-08))
    _, _, _, SNUP = daylen(_array(jday.item()), _array(location['latitude']))
    ea = _operate(np.multiply, _operate(np.multiply, 6.107, _operate(np.power, 10, _divide(_operate(np.multiply, 7.5, Ta), _operate(np.add, 237.3, Ta)))), _divide(RH, 100.0))
    msteg = _operate(np.multiply, 46.5, _divide(ea, _operate(np.add, Ta, 273.15)))
    esky = _operate(np.add, _operate(np.subtract, 1, _operate(np.multiply, _operate(np.add, 1, msteg), np.exp(-_operate(np.power, _operate(np.add, 1.2, _operate(np.multiply, 3.0, msteg)), 0.5)))), elvis)
    if altitude > 0:
        I0, CI, Kt, I0et, CIuncorr = clearnessindex_2013b(_array(zen.item()), _array(jday.item()), _array(Ta.item()), _divide(_array(RH.item()), 100.0), _array(radG.item()), location, _array(P.item()))
        CI = min(CI, 1.0)
        if onlyglobal == 1:
            I0, CI, Kt, I0et, CIuncorr = clearnessindex_2013b(_array(zen.item()), _array(jday.item()), Ta.item(), _divide(_array(RH.item()), 100.0), _array(radG.item()), location, _array(P.item()))
            CI = min(CI, 1.0)
            radI, radD = diffusefraction(_array(radG.item()), _array(altitude.item()), Kt, _array(Ta.item()), _array(RH.item()))
        if anisotropic_sky == 1:
            patchchoice = 1
            zenDeg = _operate(np.multiply, zen, _divide(180, np.pi))
            lv, pc_, pb_ = Perez_v3(zenDeg.item(), azimuth.item(), radD, radI, jday.item(), patchchoice, patch_option)
            aniLum = _zeros((rows, cols))
            for idx in range(lv.shape[0]):
                aniLum += _operate(np.multiply, diffsh[:, :, idx], lv[idx, 2])
            dRad = _operate(np.multiply, aniLum, radD)
        else:
            dRad = _operate(np.multiply, radD, svfbuveg)
            patchchoice = 1
            lv = None
        if usevegdem == 1:
            vegsh, sh, _, wallsh, wallsun, wallshve, _, facesun = shadowingfunction_wallheight_23(dsm, vegdem, vegdem2, azimuth.item(), altitude.item(), scale, amaxvalue.item(), bush, walls, _divide(_operate(np.multiply, dirwalls, np.pi), 180.0))
            shadow = _operate(np.subtract, sh, _operate(np.multiply, _operate(np.subtract, 1, vegsh), _operate(np.subtract, 1, psi)))
        else:
            sh, wallsh, wallsun, facesh, facesun = shadowingfunction_wallheight_13(dsm, azimuth.item(), altitude.item(), scale, walls, _divide(_operate(np.multiply, dirwalls, np.pi), 180.0))
            shadow = sh
        Tgamp = _operate(np.add, _operate(np.multiply, TgK, altmax), Tstart)
        Tgampwall = _operate(np.add, _operate(np.multiply, TgK_wall, altmax), Tstart_wall)
        Tg = _operate(np.multiply, Tgamp, np.sin(_divide(_operate(np.multiply, _divide(_operate(np.subtract, _operate(np.subtract, dectime, np.floor(dectime)), _divide(SNUP, 24)), _operate(np.subtract, _divide(TmaxLST, 24), _divide(SNUP, 24))), np.pi), 2)))
        Tgwall = _operate(np.multiply, Tgampwall, np.sin(_divide(_operate(np.multiply, _divide(_operate(np.subtract, _operate(np.subtract, dectime, np.floor(dectime)), _divide(SNUP, 24)), _operate(np.subtract, _divide(TmaxLST_wall, 24), _divide(SNUP, 24))), np.pi), 2)))
        Tgwall = np.maximum(Tgwall, _array(0))
        radI0, _ = diffusefraction(I0, altitude.item(), 1.0, Ta.item(), RH.item())
        corr = _operate(np.add, _operate(np.multiply, 0.1473, np.log(_operate(np.subtract, 90, _operate(np.multiply, _divide(zen, np.pi), 180)))), 0.3454)
        CI_Tg = _operate(np.add, _divide(radG, radI0), _operate(np.subtract, 1, corr))
        CI_Tg = min(CI_Tg, 1.0)
        deg2rad = _divide(np.pi, 180)
        radG0 = _operate(np.add, _operate(np.multiply, radI0, np.sin(_operate(np.multiply, altitude, deg2rad))), _)
        CI_TgG = _operate(np.add, _divide(radG, radG0), _operate(np.subtract, 1, corr))
        CI_TgG = min(CI_TgG, 1.0)
        Tg = _operate(np.multiply, Tg, CI_TgG)
        Tgwall = _operate(np.multiply, Tgwall, CI_TgG)
        if landcover == 1:
            Tg = np.maximum(Tg, _array(0))
        gvfLup, gvfalb, gvfalbnosh, gvfLupE, gvfalbE, gvfalbnoshE, gvfLupS, gvfalbS, gvfalbnoshS, gvfLupW, gvfalbW, gvfalbnoshW, gvfLupN, gvfalbN, gvfalbnoshN, gvfSum, gvfNorm = gvf_2018a(wallsun, walls, buildings, scale, shadow, first, second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, rows, cols, Twater, lc_grid, landcover)
        Lup, timeaddnotused, Tgmap1 = TsWaveDelay_2015a(gvfLup, firstdaytime, timeadd, timestepdec, Tgmap1)
        LupE, timeaddnotused, Tgmap1E = TsWaveDelay_2015a(gvfLupE, firstdaytime, timeadd, timestepdec, Tgmap1E)
        LupS, timeaddnotused, Tgmap1S = TsWaveDelay_2015a(gvfLupS, firstdaytime, timeadd, timestepdec, Tgmap1S)
        LupW, timeaddnotused, Tgmap1W = TsWaveDelay_2015a(gvfLupW, firstdaytime, timeadd, timestepdec, Tgmap1W)
        LupN, timeaddnotused, Tgmap1N = TsWaveDelay_2015a(gvfLupN, firstdaytime, timeadd, timestepdec, Tgmap1N)
        TgTemp = _operate(np.add, _operate(np.multiply, Tg, shadow), Ta)
        TgOut, timeadd, TgOut1 = TsWaveDelay_2015a(TgTemp, firstdaytime, timeadd, timestepdec, TgOut1)
        F_sh = cylindric_wedge(zen.item(), svfalfa, rows, cols)
        F_sh[np.isnan(F_sh)] = 0.5
        Kdown = _operate(np.add, _operate(np.add, _operate(np.multiply, _operate(np.multiply, radI, shadow), np.sin(_operate(np.multiply, altitude, _divide(np.pi, 180)))), dRad), _operate(np.multiply, _operate(np.multiply, albedo_b, _operate(np.subtract, 1, svfbuveg)), _operate(np.add, _operate(np.multiply, radG, _operate(np.subtract, 1, F_sh)), _operate(np.multiply, radD, F_sh))))
        Kup, KupE, KupS, KupW, KupN = Kup_veg_2015a(radI, radD, radG, altitude, svfbuveg, albedo_b, F_sh, gvfalb, gvfalbE, gvfalbS, gvfalbW, gvfalbN, gvfalbnosh, gvfalbnoshE, gvfalbnoshS, gvfalbnoshW, gvfalbnoshN)
        Keast, Ksouth, Kwest, Knorth, KsideI, KsideD, Kside = Kside_veg_v2022a(radI, radD, radG, shadow, svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg, azimuth.item(), altitude.item(), psi, t, albedo_b, F_sh, KupE, KupS, KupW, KupN, cyl, lv, anisotropic_sky, diffsh, rows, cols, asvf, shmat, vegshmat, vbshvegshmat)
        firstdaytime = 0
    else:
        Tgwall = _array(0)
        Knight = _zeros((rows, cols))
        Kdown = _zeros((rows, cols))
        Kwest = _zeros((rows, cols))
        Kup = _zeros((rows, cols))
        Keast = _zeros((rows, cols))
        Ksouth = _zeros((rows, cols))
        Knorth = _zeros((rows, cols))
        KsideI = _zeros((rows, cols))
        KsideD = _zeros((rows, cols))
        F_sh = _zeros((rows, cols))
        Tg = _zeros((rows, cols))
        shadow = _zeros((rows, cols))
        CI_Tg = deepcopy(CI)
        CI_TgG = deepcopy(CI)
        dRad = _zeros((rows, cols))
        Kside = _zeros((rows, cols))
        Lup = _operate(np.multiply, _operate(np.multiply, SBC, emis_grid), _operate(np.power, _operate(np.add, _operate(np.add, _operate(np.add, Knight, Ta), Tg), 273.15), 4))
        if landcover == 1:
            Lup[lc_grid == 3] = _operate(np.multiply, _operate(np.multiply, SBC, 0.98), _operate(np.power, _operate(np.add, Twater, 273.15), 4)).astype(np.float32)
        LupE = Lup
        LupS = Lup
        LupW = Lup
        LupN = Lup
        TgOut = _operate(np.add, Ta, Tg)
        I0 = 0
        timeadd = 0
        firstdaytime = 1
    if anisotropic_sky == 1:
        if 'lv' not in locals():
            skyvaultalt, skyvaultazi, _, _, _, _, _ = create_patches(patch_option)
            patch_emissivities = _zeros(skyvaultalt.shape[0])
            x = np.swapaxes(np.atleast_2d(skyvaultalt), 0, 1)
            y = np.swapaxes(np.atleast_2d(skyvaultazi), 0, 1)
            z = np.swapaxes(np.atleast_2d(patch_emissivities), 0, 1)
            L_patches = np.concatenate((x, y, z), axis=1)
        else:
            L_patches = deepcopy(lv)
        if altitude < 0:
            CI = deepcopy(CI)
        if CI < 0.95:
            esky_c = _operate(np.add, _operate(np.multiply, CI, esky), _operate(np.multiply, _operate(np.subtract, 1, CI), 1.0))
            esky = esky_c
        Ldown, Lside, Least_, Lwest_, Lnorth_, Lsouth_ = Lcyl_v2022a(esky, L_patches, Ta, Tgwall, ewall, Lup, shmat, vegshmat, vbshvegshmat, altitude, azimuth, rows, cols, asvf)
    else:
        Ldown = _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.subtract, _operate(np.add, svf, svfveg), 1), esky), SBC), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.subtract, _operate(np.subtract, 2, svfveg), svfaveg), ewall), SBC), _operate(np.power, _operate(np.add, Ta, 273.15), 4))), _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.subtract, svfaveg, svf), ewall), SBC), _operate(np.power, _operate(np.add, _operate(np.add, Ta, 273.15), Tgwall), 4))), _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.subtract, _operate(np.subtract, 2, svf), svfveg), _operate(np.subtract, 1, ewall)), esky), SBC), _operate(np.power, _operate(np.add, Ta, 273.15), 4)))
        Lside = _zeros((rows, cols))
        L_patches = None
        if CI < 0.95:
            c = _operate(np.subtract, 1, CI)
            Ldown = _operate(np.add, _operate(np.multiply, Ldown, _operate(np.subtract, 1, c)), _operate(np.multiply, c, _operate(np.add, _operate(np.add, _operate(np.add, _operate(np.multiply, _operate(np.multiply, _operate(np.subtract, _operate(np.add, svf, svfveg), 1), SBC), _operate(np.power, _operate(np.add, Ta, 273.15), 4)), _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.subtract, _operate(np.subtract, 2, svfveg), svfaveg), ewall), SBC), _operate(np.power, _operate(np.add, Ta, 273.15), 4))), _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.subtract, svfaveg, svf), ewall), SBC), _operate(np.power, _operate(np.add, _operate(np.add, Ta, 273.15), Tgwall), 4))), _operate(np.multiply, _operate(np.multiply, _operate(np.multiply, _operate(np.subtract, _operate(np.subtract, 2, svf), svfveg), _operate(np.subtract, 1, ewall)), SBC), _operate(np.power, _operate(np.add, Ta, 273.15), 4)))))
    Least, Lsouth, Lwest, Lnorth = Lside_veg_v2022a(svfS, svfW, svfN, svfE, svfEveg, svfSveg, svfWveg, svfNveg, svfEaveg, svfSaveg, svfWaveg, svfNaveg, azimuth.item(), altitude.item(), Ta, Tgwall, SBC, ewall, Ldown, esky, t, F_sh, CI, LupE, LupS, LupW, LupN, anisotropic_sky)
    if cyl == 0 and anisotropic_sky == 1:
        Least += Least_
        Lwest += Lwest_
        Lnorth += Lnorth_
        Lsouth += Lsouth_
    if cyl == 1 and anisotropic_sky == 0:
        Sstr = _operate(np.add, _operate(np.multiply, absK, _operate(np.add, _operate(np.add, _operate(np.multiply, KsideI, Fcyl), _operate(np.multiply, _operate(np.add, Kdown, Kup), Fup)), _operate(np.multiply, _operate(np.add, _operate(np.add, _operate(np.add, Knorth, Keast), Ksouth), Kwest), Fside))), _operate(np.multiply, absL, _operate(np.add, _operate(np.multiply, _operate(np.add, Ldown, Lup), Fup), _operate(np.multiply, _operate(np.add, _operate(np.add, _operate(np.add, Lnorth, Least), Lsouth), Lwest), Fside))))
    elif cyl == 1 and anisotropic_sky == 1:
        Sstr = _operate(np.add, _operate(np.multiply, absK, _operate(np.add, _operate(np.add, _operate(np.multiply, Kside, Fcyl), _operate(np.multiply, _operate(np.add, Kdown, Kup), Fup)), _operate(np.multiply, _operate(np.add, _operate(np.add, _operate(np.add, Knorth, Keast), Ksouth), Kwest), Fside))), _operate(np.multiply, absL, _operate(np.add, _operate(np.add, _operate(np.multiply, _operate(np.add, Ldown, Lup), Fup), _operate(np.multiply, Lside, Fcyl)), _operate(np.multiply, _operate(np.add, _operate(np.add, _operate(np.add, Lnorth, Least), Lsouth), Lwest), Fside))))
    else:
        Sstr = _operate(np.add, _operate(np.multiply, absK, _operate(np.add, _operate(np.multiply, _operate(np.add, Kdown, Kup), Fup), _operate(np.multiply, _operate(np.add, _operate(np.add, _operate(np.add, Knorth, Keast), Ksouth), Kwest), Fside))), _operate(np.multiply, absL, _operate(np.add, _operate(np.multiply, _operate(np.add, Ldown, Lup), Fup), _operate(np.multiply, _operate(np.add, _operate(np.add, _operate(np.add, Lnorth, Least), Lsouth), Lwest), Fside))))
    Tmrt = _operate(np.subtract, np.sqrt(np.sqrt(_divide(Sstr, _operate(np.multiply, absL, SBC)))), 273.2)
    if cyl == 1 and anisotropic_sky == 1:
        Least += Least_
        Lwest += Lwest_
        Lnorth += Lnorth_
        Lsouth += Lsouth_
    return (Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, shadow, firstdaytime, timestepdec, timeadd, Tgmap1, Tgmap1E, Tgmap1S, Tgmap1W, Tgmap1N, Keast, Ksouth, Kwest, Knorth, Least, Lsouth, Lwest, Lnorth, KsideI, TgOut1, TgOut, radI, radD, Lside, L_patches, CI_Tg, CI_TgG, KsideD, dRad, Kside)

def _array(x, dtype=None):
    if dtype is None:
        dtype = None if isinstance(x, (np.ndarray, np.generic)) else np.float32 if np.asarray(x).dtype.kind == 'f' else None
    return np.array(x, dtype=dtype, copy=True)

def _zeros(shape, dtype=np.float32):
    return np.zeros(shape, dtype=dtype)

def _ones(shape, dtype=np.float32):
    return np.ones(shape, dtype=dtype)

def _divide(a, b):
    a, b = _operands(a, b)
    return np.divide(a, b)

def _operands(a,b):
    """Preserve wrapped scalar promotion and floating/integer array hierarchy.

    The source keeps float32 raster arithmetic float32 with a float64 scalar
    or integer mask. NumPy otherwise promotes those operations to float64.
    """
    aa,bb=np.asarray(a),np.asarray(b)
    if aa.dtype.kind=='f' and aa.ndim>0 and (bb.ndim==0 or bb.dtype.kind in 'biu'):
        b=np.asarray(b,dtype=aa.dtype)
    elif bb.dtype.kind=='f' and bb.ndim>0 and (aa.ndim==0 or aa.dtype.kind in 'biu'):
        a=np.asarray(a,dtype=bb.dtype)
    return a,b
def _operate(operation,a,b):
    a,b=_operands(a,b)
    return operation(a,b)


# Keep the characterized array recurrences for non-float32 low-level inputs
# and diagnostics. The core TIFF path has float32 raster fields.
shadowingfunction_wallheight_13_numpy = shadowingfunction_wallheight_13
shadowingfunction_wallheight_23_numpy = shadowingfunction_wallheight_23


def shadowingfunction_wallheight_13(a, azimuth, altitude, scale, walls, aspect):
    if any(np.asarray(value).dtype != np.float32 for value in (a, walls, aspect)):
        return shadowingfunction_wallheight_13_numpy(a, azimuth, altitude, scale, walls, aspect)
    from .wall_shadows import exact_13
    return exact_13(a, azimuth, altitude, scale, walls, aspect, parallel=False)


def shadowingfunction_wallheight_23(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, walls, aspect):
    if any(np.asarray(value).dtype != np.float32 for value in (a, vegdem, vegdem2, bush, walls, aspect)):
        return shadowingfunction_wallheight_23_numpy(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, walls, aspect)
    from .wall_shadows import exact_23
    return exact_23(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, walls, aspect, parallel=False)
