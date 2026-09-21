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
"""Compiled wall-shadow recurrences with independent pixel ownership.

Ray schedules retain the two original algorithms' distinct initial indices.
The host evaluates scalar recurrences before deterministic per-pixel execution.
"""
import numpy as np
from numba import njit, prange


def _geometry(a, azimuth, altitude, scale, maximum, vegetation):
    # Lazy import permits the engine to import this module for dispatch.
    from . import engine as e
    if vegetation:
        degrees = e._array(np.pi / 180.0)
        az = e._operate(np.multiply, e._array(azimuth), degrees)
        alt = e._operate(np.multiply, e._array(altitude), degrees)
        quarter = e._array(np.pi / 4.0)
    else:
        az = e._array(azimuth * (np.pi / 180.0))
        alt = e._array(altitude * (np.pi / 180.0))
        quarter = np.pi / 4
    sin, cos, tan = np.sin(az), np.cos(az), np.tan(az)
    ss, sc = np.sign(sin), np.sign(cos)
    with np.errstate(all='ignore'):
        dss, dsc = abs(e._divide(1, sin)), abs(e._divide(1, cos))
        tas = e._divide(np.tan(alt), scale)
    dx = dy = dz = np.float32(0)
    index = 0 if vegetation else 1
    schedule = []
    while maximum >= dz and abs(dx) < a.shape[0] and abs(dy) < a.shape[1]:
        if quarter <= az < e._operate(np.multiply, 3, quarter) or e._operate(np.multiply, 5, quarter) <= az < e._operate(np.multiply, 7, quarter):
            dy = e._operate(np.multiply, ss, index)
            dx = e._operate(np.multiply, e._operate(np.multiply, -1, sc), abs(np.round(e._divide(index, tan))))
            ds = dss
        else:
            dy = e._operate(np.multiply, ss, abs(np.round(e._operate(np.multiply, index, tan))))
            dx = e._operate(np.multiply, e._operate(np.multiply, -1, sc), index)
            ds = dsc
        dz = e._operate(np.multiply, e._operate(np.multiply, ds, index), tas)
        schedule.append((int(dx), int(dy), np.float32(dz)))
        index += 1
    low = e._operate(np.subtract, az, np.pi / 2)
    high = e._operate(np.add, az, np.pi / 2)
    if low >= 0 and high < 2 * np.pi:
        mode = 0
    elif low < 0 and high <= 2 * np.pi:
        low = e._operate(np.add, low, 2 * np.pi)
        mode = 1
    elif low > 0 and high >= 2 * np.pi:
        high = e._operate(np.subtract, high, 2 * np.pi)
        mode = 1
    else:
        raise UnboundLocalError('Pinned upstream leaves facesh unbound for this azimuth')
    return np.asarray([(x,y) for x,y,z in schedule], dtype=np.int64).reshape(-1,2), np.asarray([z for x,y,z in schedule], dtype=np.float32), low, high, mode


@njit(inline='always', fastmath=False)
def _maximum(x,y):
    if np.isnan(x) or np.isnan(y):
        return np.float32(np.nan)
    return x if x >= y else y


@njit(cache=True, fastmath=False, parallel=True)
def _wall13(a, walls, aspect, offsets, heights, low, high, mode):
    sx,sy=a.shape
    out=np.empty((5,sx,sy),dtype=np.float32)
    for pixel in prange(sx*sy):
        row,col=pixel//sy,pixel%sy
        base=a[row,col]
        f=base
        for step in range(heights.size):
            x,y=row+offsets[step,0],col+offsets[step,1]
            temp=np.float32(0)
            if 0<=x<sx and 0<=y<sy:
                temp=np.float32(a[x,y]-heights[step])
            f=_maximum(f,temp)
        wallbol=np.float32(walls[row,col]>0)
        if mode==0:
            face=np.float32(np.float32(aspect[row,col]<low or aspect[row,col]>=high)-wallbol+np.float32(1))
        else:
            face=np.float32(1)-np.float32(aspect[row,col]>low or aspect[row,col]<=high)
        volume=np.float32(f-base)
        sun=np.float32(walls[row,col]-volume)
        if sun<0 or face==1:
            sun=np.float32(0)
        out[0,row,col]=np.float32(volume==0)
        out[1,row,col]=np.float32(walls[row,col]-sun)
        out[2,row,col]=sun
        out[3,row,col]=face
        out[4,row,col]=np.float32(face+wallbol==1 and walls[row,col]>0)
    return out


@njit(cache=True, fastmath=False, parallel=True)
def _wall23(a, vegdem, vegdem2, bush, walls, aspect, offsets, heights, low, high, mode):
    sx,sy=a.shape
    out=np.empty((8,sx,sy),dtype=np.float32)
    for pixel in prange(sx*sy):
        row,col=pixel//sy,pixel%sy
        base=a[row,col]
        f=base
        volumeveg=vegdem[row,col]
        vs=np.float32(bush[row,col]>1)
        building=np.float32(0)
        vb=np.float32(0)
        prev=np.float32(0)
        for step in range(heights.size):
            x,y=row+offsets[step,0],col+offsets[step,1]
            temp=tv=tv2=lastv=lastv2=np.float32(0)
            if 0<=x<sx and 0<=y<sy:
                temp=np.float32(a[x,y]-heights[step])
                tv=np.float32(vegdem[x,y]-heights[step])
                tv2=np.float32(vegdem2[x,y]-heights[step])
                lastv=np.float32(vegdem[x,y]-prev)
                lastv2=np.float32(vegdem2[x,y]-prev)
            f=_maximum(f,temp)
            volumeveg=_maximum(volumeveg,tv)
            building=np.float32(f>base)
            count=np.float32(tv>base)+np.float32(tv2>base)+np.float32(lastv>base)+np.float32(lastv2>base)
            candidate=np.float32(count>0 and count!=4)
            vs=_maximum(vs,candidate)
            if np.float32(vs*building)>0:
                vs=np.float32(0)
            vb=np.float32(vb+vs)
            prev=heights[step]
        wallbol=np.float32(walls[row,col]>0)
        if mode==0:
            face=np.float32(np.float32(aspect[row,col]<low or aspect[row,col]>=high)-wallbol+np.float32(1))
        else:
            face=np.float32(1)-np.float32(aspect[row,col]>low or aspect[row,col]<=high)
        vb=np.float32(np.float32(vb>0)-vs)
        vs=np.float32(vs>0)
        volumeveg=np.float32(np.float32(volumeveg-base)*vs)
        sun=np.float32(walls[row,col]-np.float32(f-base))
        if sun<0 or face==1:
            sun=np.float32(0)
        wallshade=np.float32(walls[row,col]-sun)
        wallveg=np.float32(np.float32(volumeveg*wallbol)-wallshade)
        if wallveg<0:
            wallveg=np.float32(0)
        sun=np.float32(sun-wallveg)
        if sun<0:
            sun=np.float32(0)
        if wallveg>walls[row,col]:
            wallveg=walls[row,col]
        out[0,row,col]=np.float32(1)-vs
        out[1,row,col]=np.float32(1)-building
        out[2,row,col]=np.float32(1)-vb
        out[3,row,col]=wallshade
        out[4,row,col]=sun
        out[5,row,col]=wallveg
        out[6,row,col]=face
        out[7,row,col]=np.float32(face+wallbol==1 and walls[row,col]>0)
    return out


def exact_13(a, azimuth, altitude, scale, walls, aspect, *, parallel=True):
    offsets,heights,low,high,mode=_geometry(a,azimuth,altitude,scale,np.max(a),False)
    low, high = np.asarray(low,dtype=aspect.dtype)[()], np.asarray(high,dtype=aspect.dtype)[()]
    kernel=_wall13 if parallel else _wall13_serial
    return tuple(kernel(a,walls,aspect,offsets,heights,low,high,mode))


def exact_23(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, walls, aspect, *, parallel=True):
    offsets,heights,low,high,mode=_geometry(a,azimuth,altitude,scale,amaxvalue,True)
    low, high = np.asarray(low,dtype=aspect.dtype)[()], np.asarray(high,dtype=aspect.dtype)[()]
    kernel=_wall23 if parallel else _wall23_serial
    return tuple(kernel(a,vegdem,vegdem2,bush,walls,aspect,offsets,heights,low,high,mode))


_wall13_serial=njit(fastmath=False)(_wall13.py_func)
_wall23_serial=njit(fastmath=False)(_wall23.py_func)
