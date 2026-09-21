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
"""Pixel-owned ground-view gathers with persistent outside-slice history.

Only addressing descriptors are cached. Emitted/reflected fields are rebuilt
and snapshotted every direction. Non-float32 raster profiles retain the
characterized NumPy implementation. Directions are reduced in original order.
"""
from functools import lru_cache
import numpy as np
from numba import njit, prange

SCHEDULE_VERSION='ground-view-v1'
MAX_CACHED_SCHEDULE_BYTES=65536


def _key(value):
    array=np.asarray(value)
    return array.dtype.str,array.tobytes()


def _scalar(key):return np.frombuffer(key[1],dtype=key[0])[0]


def _build_schedule(shape,azimuth,second):
    sx,sy=shape
    sin,cos,tan=np.sin(azimuth),np.cos(azimuth),np.tan(azimuth)
    ss,sc=np.sign(sin),np.sign(cos)
    bounds=[]
    for index in range(int(second)):
        if np.pi/4<=azimuth<3*np.pi/4 or 5*np.pi/4<=azimuth<7*np.pi/4:
            dy=ss*index;dx=-sc*abs(np.round(index/tan))
        else:
            dy=ss*abs(np.round(index*tan));dx=-sc*index
        ax,ay=abs(dx),abs(dy)
        xc1,xc2=int((dx+ax)/2),int(sx+(dx-ax)/2)
        yc1,yc2=int((dy+ay)/2),int(sy+(dy-ay)/2)
        xp1,xp2=int(-(dx-ax)/2),int(sx-(dx+ax)/2)
        yp1,yp2=int(-(dy-ay)/2),int(sy-(dy+ay)/2)
        xc1,xc2,_=slice(xc1,xc2).indices(sx);yc1,yc2,_=slice(yc1,yc2).indices(sy)
        xp1,xp2,_=slice(xp1,xp2).indices(sx);yp1,yp2,_=slice(yp1,yp2).indices(sy)
        source=(max(xc2-xc1,0),max(yc2-yc1,0));destination=(max(xp2-xp1,0),max(yp2-yp1,0))
        if any(a!=b and a!=1 for a,b in zip(source,destination)):
            raise RuntimeError('original ray source/destination slices cannot be assigned')
        bounds.append((xc1,xc2,yc1,yc2,xp1,xp2,yp1,yp2))
    array=np.asarray(bounds,dtype=np.int64).reshape(-1,8)
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


@lru_cache(maxsize=36)
def _schedule_cached(shape,azimuth_key,scale_key,first_key,second_key,version):
    return _build_schedule(shape,_scalar(azimuth_key),_scalar(second_key))


def ray_schedule(shape,azimuth,scale,first,second):
    # Keys contain exact dtype/value and all normalized search parameters.
    # The size guard bounds persistent descriptor storage to <=36*64 KiB.
    if int(second)*8*8>MAX_CACHED_SCHEDULE_BYTES:
        return _build_schedule(shape,azimuth,second)
    return _schedule_cached(tuple(shape),_key(azimuth),_key(scale),_key(first),_key(second),SCHEDULE_VERSION)


def clear_schedule_cache():_schedule_cached.cache_clear()


@njit(inline='always',fastmath=False)
def _minimum(left,right):
    if np.isnan(left):return left
    if np.isnan(right):return right
    if left<right:return left
    return right


@njit(inline='always',fastmath=False)
def _gather_pixel(row,col,buildings,shadow,sunwall,lup,albshadow,alb,lwall,albedo,bounds,first,output):
    f=buildings[row,col]
    bu=sh=lu=al=an=sw=np.float32(0)
    bub=bubwall=np.float32(0)
    sumsh=sumwall=sumlu=sumlw=sumal=sumaw=suman=sumanw=np.float32(0)
    firstsh=firstwall=firstlu=firstlw=firstal=firstaw=firstan=firstanw=np.float32(0)
    for step in range(len(bounds)):
        xc1,xc2,yc1,yc2,xp1,xp2,yp1,yp2=bounds[step]
        if xp1<=row<xp2 and yp1<=col<yp2:
            sr=xc1 if xc2-xc1==1 else xc1+row-xp1
            sc=yc1 if yc2-yc1==1 else yc1+col-yp1
            bu=buildings[sr,sc];sh=shadow[sr,sc];lu=lup[sr,sc]
            al=albshadow[sr,sc];an=alb[sr,sc];sw=sunwall[sr,sc]
        # Outside a current slice the previous neighbor samples persist.
        f=_minimum(f,bu)
        sumsh=np.float32(sumsh+np.float32(sh*f))
        sumlu=np.float32(sumlu+np.float32(lu*f))
        sumal=np.float32(sumal+np.float32(al*f))
        suman=np.float32(suman+np.float32(an*f))
        tempb=np.float32(sw*f)
        tempbwall=np.float32(np.float32(f*np.float32(-1))+np.float32(1))
        bub=np.float32(np.float32(tempb+bub)>0)
        bubwall=np.float32(np.float32(tempbwall+bubwall)>0)
        sumlw=np.float32(sumlw+np.float32(bub*lwall[row,col]))
        sumaw=np.float32(sumaw+np.float32(bub*albedo))
        sumwall=np.float32(sumwall+bub)
        sumanw=np.float32(sumanw+np.float32(bubwall*albedo))
        if step+1<=first:
            # Upstream resets ind=1 each iteration; snapshots are unnormalized.
            firstsh=sumsh;firstwall=sumwall;firstlu=sumlu;firstlw=sumlw
            firstal=sumal;firstaw=sumaw;firstan=suman;firstanw=sumanw
    output[0,row,col]=sumsh;output[1,row,col]=sumwall
    output[2,row,col]=sumlu;output[3,row,col]=sumlw
    output[4,row,col]=sumal;output[5,row,col]=sumaw
    output[6,row,col]=suman;output[7,row,col]=sumanw
    output[8,row,col]=firstsh;output[9,row,col]=firstwall
    output[10,row,col]=firstlu;output[11,row,col]=firstlw
    output[12,row,col]=firstal;output[13,row,col]=firstaw
    output[14,row,col]=firstan;output[15,row,col]=firstanw


@njit(cache=True,fastmath=False)
def _gather_serial(buildings,shadow,sunwall,lup,albshadow,alb,lwall,albedo,bounds,first,output):
    for row in range(buildings.shape[0]):
        for col in range(buildings.shape[1]):
            _gather_pixel(row,col,buildings,shadow,sunwall,lup,albshadow,alb,lwall,albedo,bounds,first,output)


@njit(cache=True,fastmath=False,parallel=True)
def _gather_parallel(buildings,shadow,sunwall,lup,albshadow,alb,lwall,albedo,bounds,first,output):
    rows,cols=buildings.shape
    for pixel in prange(rows*cols):
        row=pixel//cols;col=pixel%cols
        _gather_pixel(row,col,buildings,shadow,sunwall,lup,albshadow,alb,lwall,albedo,bounds,first,output)


def _gather(azimuth,scale,first,second,buildings,shadow,sunwall,lup,albshadow,alb,lwall,albedo,parallel):
    shape=buildings.shape
    bounds=ray_schedule(shape,azimuth,scale,first,second)
    # Snapshot dynamic neighbor inputs before any worker starts. Each direction
    # owns fresh receiver fields; subsequent Tg mutation cannot rewrite Lup.
    sources=[np.array(value,dtype=np.float32,copy=True) for value in (buildings,shadow,sunwall,lup,albshadow,alb)]
    lwall=np.array(np.broadcast_to(lwall,shape),dtype=np.float32,copy=True)
    output=np.empty((16,*shape),dtype=np.float32)
    kernel=_gather_parallel if parallel else _gather_serial
    kernel(*sources,lwall,np.float32(albedo),bounds,np.float32(first),output)
    return tuple(output[index] for index in range(16))


def _supported(values):
    return all(np.asarray(value).dtype==np.float32 for value in values)


def _fallback(name,*args):
    from . import engine
    function=getattr(engine,name+'_numpy',getattr(engine,name))
    return function(*args)


def _angular_constant(angle, value):
    """Torch wraps Python constants in the floating scalar tensor dtype."""
    dtype = np.asarray(angle).dtype
    return np.asarray(value, dtype=dtype) if dtype.kind == 'f' else value


def _sun(azimuthA, scale, buildings, shadow, sunwall, first, second, aspect, walls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, Twater, lc_grid, landcover, parallel=False):
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
    from .engine import _array, _zeros, _operate, _divide
    if not _supported((buildings, shadow, sunwall, aspect, walls, Tg, emis_grid, alb_grid)) or (np.asarray(Tgwall).ndim>0 and np.asarray(Tgwall).dtype!=np.float32) or not np.isfinite(np.asarray(first)).all() or not np.isfinite(np.asarray(second)).all() or float(np.round(_operate(np.multiply, second, scale)))<=0:
        return _fallback('sunonsurface_2018a', azimuthA, scale, buildings, shadow, sunwall, first, second, aspect, walls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, Twater, lc_grid, landcover)
    scale = np.copy(_array(scale))
    ewall = np.copy(_array(ewall))
    albedo_b = np.copy(_array(albedo_b))
    landcover = np.copy(_array(landcover))
    sizex = walls.shape[0]
    sizey = walls.shape[1]
    wallbol = (walls > 0).astype(np.float32)
    sunwall[sunwall > 0] = 1
    azimuth = _operate(np.multiply, azimuthA, _angular_constant(azimuthA, np.pi / 180))
    index = 0
    f = buildings
    Lup = _operate(np.subtract, _operate(np.multiply, _operate(np.multiply, SBC, emis_grid), _operate(np.power, _operate(np.add, _operate(np.add, _operate(np.multiply, Tg, shadow), Ta), 273.15), 4)), _operate(np.multiply, _operate(np.multiply, SBC, emis_grid), _operate(np.power, _operate(np.add, Ta, 273.15), 4)))
    if landcover == 1:
        Tg[lc_grid == 3] = _operate(np.subtract, Twater, Ta).astype(np.float32)
    Lwall = _operate(np.subtract, _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, _operate(np.add, Tgwall, Ta), 273.15), 4)), _operate(np.multiply, _operate(np.multiply, SBC, ewall), _operate(np.power, _operate(np.add, Ta, 273.15), 4)))
    albshadow = _operate(np.multiply, alb_grid, shadow)
    alb = alb_grid
    first = np.round(_operate(np.multiply, first, scale))
    if first < 1:
        first = 1
    second = np.round(_operate(np.multiply, second, scale))
    (weightsumsh, weightsumwall, weightsumLupsh, weightsumLwall,
     weightsumalbsh, weightsumalbwall, weightsumalbnosh, weightsumalbwallnosh,
     weightsumsh_first, weightsumwall_first, weightsumLupsh_first, weightsumLwall_first,
     weightsumalbsh_first, weightsumalbwall_first, weightsumalbnosh_first, weightsumalbwallnosh_first) = _gather(
        azimuth, scale, first, second, buildings, shadow, sunwall, Lup, albshadow, alb, Lwall, albedo_b, parallel)
    wallsuninfluence_first = weightsumwall_first > 0
    wallinfluence_first = weightsumalbwallnosh_first > 0
    wallsuninfluence_second = weightsumwall > 0
    wallinfluence_second = weightsumalbwallnosh > 0
    azilow = _operate(np.subtract, azimuth, _angular_constant(azimuth, np.pi / 2))
    azihigh = _operate(np.add, azimuth, _angular_constant(azimuth, np.pi / 2))
    if azilow >= 0 and azihigh < _angular_constant(azimuth, 2 * np.pi):
        facesh = _operate(np.add, _operate(np.subtract, np.logical_or(aspect < azilow, aspect >= azihigh).astype(np.float32), wallbol), 1)
    elif azilow < 0 and azihigh <= _angular_constant(azimuth, 2 * np.pi):
        azilow = _operate(np.add, azilow, _angular_constant(azimuth, 2 * np.pi))
        facesh = _operate(np.add, _operate(np.multiply, np.logical_or(aspect > azilow, aspect <= azihigh).astype(np.float32), -1), 1)
    elif azilow > 0 and azihigh >= _angular_constant(azimuth, 2 * np.pi):
        azihigh = _operate(np.subtract, azihigh, _angular_constant(azimuth, 2 * np.pi))
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
    return (gvf, gvfLup, gvfalb, gvfalbnosh, gvf2)

def _gvf(wallsun, walls, buildings, scale, shadow, first, second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, rows, cols, Twater, lc_grid, landcover, parallel=False):
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
    from .engine import _zeros, _operate, _divide
    if not _supported((wallsun, walls, buildings, shadow, dirwalls, Tg, emis_grid, alb_grid)) or (np.asarray(Tgwall).ndim>0 and np.asarray(Tgwall).dtype!=np.float32):
        return _fallback('gvf_2018a', wallsun, walls, buildings, scale, shadow, first, second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, rows, cols, Twater, lc_grid, landcover)
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
        _, gvfLupi, gvfalbi, gvfalbnoshi, gvf2 = _sun(azimuthA[j], scale, buildings, shadow, sunwall, first, second, _divide(_operate(np.multiply, dirwalls, np.pi), 180), walls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, Twater, lc_grid, landcover, parallel=parallel)
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

def sunonsurface_2018a(azimuthA, scale, buildings, shadow, sunwall, first, second, aspect, walls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, Twater, lc_grid, landcover):
    return _sun(azimuthA, scale, buildings, shadow, sunwall, first, second, aspect, walls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, Twater, lc_grid, landcover, parallel=False)

def sunonsurface_2018a_parallel(azimuthA, scale, buildings, shadow, sunwall, first, second, aspect, walls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, Twater, lc_grid, landcover):
    return _sun(azimuthA, scale, buildings, shadow, sunwall, first, second, aspect, walls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, Twater, lc_grid, landcover, parallel=True)

def gvf_2018a(wallsun, walls, buildings, scale, shadow, first, second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, rows, cols, Twater, lc_grid, landcover):
    return _gvf(wallsun, walls, buildings, scale, shadow, first, second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, rows, cols, Twater, lc_grid, landcover, parallel=False)

def gvf_2018a_parallel(wallsun, walls, buildings, scale, shadow, first, second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, rows, cols, Twater, lc_grid, landcover):
    return _gvf(wallsun, walls, buildings, scale, shadow, first, second, dirwalls, Tg, Tgwall, Ta, emis_grid, ewall, alb_grid, SBC, albedo_b, rows, cols, Twater, lc_grid, landcover, parallel=True)
