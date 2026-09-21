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
"""Exact step-major sky shadows with pixel ownership within each ray step.

Schedule arithmetic stays NumPy, matching the diagnostic recurrence's scalar
promotion, trig and rounding. Schedules are freshly prepared, never reused
under an incomplete geometry/forcing key. No ray is truncated on shadow state.
"""
import numpy as np
from numba import njit, prange


def ray_schedule(shape, amaxvalue, azimuth, altitude, scale):
    """Return actual executed slice bounds/heights, including the first step."""
    def angle(value):
        if isinstance(value,(float,int)) or np.asarray(value).dtype.kind in 'iu':
            return np.float32(value)
        return np.asarray(value)[()]
    if azimuth==0:
        azimuth=1e-12
    azimuth=angle(azimuth)*(np.pi/180)
    altitude=angle(altitude)*(np.pi/180)
    sx,sy=shape
    sin,cos,tan=np.sin(azimuth),np.cos(azimuth),np.tan(azimuth)
    ss,sc=np.sign(sin),np.sign(cos)
    ds_s,ds_c=abs(1/sin),abs(1/cos)
    tas=np.tan(altitude)/scale
    dx=dy=dz=np.float32(0)
    index=1
    bounds=[]
    heights=[]
    while amaxvalue>=dz and abs(dx)<sx and abs(dy)<sy:
        if np.pi/4<=azimuth<3*np.pi/4 or 5*np.pi/4<=azimuth<7*np.pi/4:
            dy=ss*index
            dx=-sc*abs(np.round(index/tan))
            ds=ds_s
        else:
            dy=ss*abs(np.round(index*tan))
            dx=-sc*index
            ds=ds_c
        dz=ds*index*tas
        ax,ay=abs(dx),abs(dy)
        xc1,xc2=int((dx+ax)/2),int(sx+(dx-ax)/2)
        yc1,yc2=int((dy+ay)/2),int(sy+(dy-ay)/2)
        xp1,xp2=int(-(dx-ax)/2),int(sx-(dx+ax)/2)
        yp1,yp2=int(-(dy-ay)/2),int(sy-(dy+ay)/2)
        # Retain Python slicing even at out-of-domain/empty final steps.
        xc1,xc2,_=slice(xc1,xc2).indices(sx)
        yc1,yc2,_=slice(yc1,yc2).indices(sy)
        xp1,xp2,_=slice(xp1,xp2).indices(sx)
        yp1,yp2,_=slice(yp1,yp2).indices(sy)
        if max(xc2-xc1,0)!=max(xp2-xp1,0) or max(yc2-yc1,0)!=max(yp2-yp1,0):
            raise ValueError('ray source/destination slice shapes differ')
        bounds.append((xc1,xc2,yc1,yc2,xp1,xp2,yp1,yp2))
        heights.append(dz)
        index+=1
    return np.asarray(bounds,dtype=np.int64).reshape(-1,8),np.asarray(heights)


@njit(inline='always',fastmath=False)
def _maximum(left,right):
    if np.isnan(left):return left
    if np.isnan(right):return right
    if left>right:return left
    return right


@njit(inline='always',fastmath=False)
def _advance_pixel(row,col,a,canopy,trunk,bounds,da,dv,dt,first_height,first,f,sh,vs,vb,tv):
    xc1,xc2,yc1,yc2,xp1,xp2,yp1,yp2=bounds
    building=np.float32(0)
    vegetation=np.float32(0)
    trunk_value=np.float32(0)
    if xp1<=row<xp2 and yp1<=col<yp2:
        source_row=xc1+row-xp1
        source_col=yc1+col-yp1
        # Delta arrays have each source's dtype: subtraction executes at the
        # source precision before conversion into the original float32 scratch.
        building=np.float32(a[source_row,source_col]-da)
        vegetation=np.float32(canopy[source_row,source_col]-dv)
        trunk_value=np.float32(trunk[source_row,source_col]-dt)
    tv[row,col]=vegetation
    height=a[row,col]
    f[row,col]=_maximum(f[row,col],building)
    if f[row,col]>height:
        sh[row,col]=np.float32(1)
    elif f[row,col]<=height:
        sh[row,col]=np.float32(0)
    above=vegetation>height
    trunk_above=trunk_value>height
    vegetation_shadow=np.float32(above)-np.float32(trunk_above)
    vs[row,col]=_maximum(vs[row,col],vegetation_shadow)
    if vs[row,col]*sh[row,col]>0:
        vs[row,col]=np.float32(0)
    vb[row,col]=vb[row,col]+vs[row,col]
    if first:
        first_vegetation=np.float32(vegetation-building)
        if first_vegetation<=0:
            first_vegetation=np.float32(1000)
        if first_vegetation<first_height:
            vs[row,col]=np.float32(1)
        vs[row,col]=vs[row,col]*np.float32(trunk[row,col]>height)
        vb[row,col]=np.float32(0)


@njit(cache=True,fastmath=False)
def _advance_serial(a,canopy,trunk,bounds,da,dv,dt,first_height,first,f,sh,vs,vb,tv):
    for row in range(a.shape[0]):
        for col in range(a.shape[1]):
            _advance_pixel(row,col,a,canopy,trunk,bounds,da,dv,dt,first_height,first,f,sh,vs,vb,tv)


@njit(cache=True,fastmath=False,parallel=True)
def _advance_parallel(a,canopy,trunk,bounds,da,dv,dt,first_height,first,f,sh,vs,vb,tv):
    rows,cols=a.shape
    for pixel in prange(rows*cols):
        row=pixel//cols
        col=pixel%cols
        _advance_pixel(row,col,a,canopy,trunk,bounds,da,dv,dt,first_height,first,f,sh,vs,vb,tv)


@njit(cache=True,fastmath=False)
def _bush_active(tv,a,bush):
    # A serial deterministic global reduction preserves NaN poisoning from
    # false * infinity (and false * NaN); an any-positive shortcut would not.
    maximum=-np.inf
    for row in range(a.shape[0]):
        for col in range(a.shape[1]):
            value=np.float32(tv[row,col]>a[row,col])*bush[row,col]
            if np.isnan(value):return False
            if value>maximum:maximum=value
    return maximum>0


@njit(inline='always',fastmath=False)
def _bush_pixel(row,col,bush,bounds,delta,g,bp):
    xc1,xc2,yc1,yc2,xp1,xp2,yp1,yp2=bounds
    value=np.float32(0)
    if xp1<=row<xp2 and yp1<=col<yp2:
        value=np.float32(bush[xc1+row-xp1,yc1+col-yp1]-delta)
    g[row,col]=_maximum(g[row,col],value)*bp[row,col]


@njit(cache=True,fastmath=False)
def _bush_serial(bush,bounds,delta,g,bp):
    for row in range(bush.shape[0]):
        for col in range(bush.shape[1]):
            _bush_pixel(row,col,bush,bounds,delta,g,bp)


@njit(cache=True,fastmath=False,parallel=True)
def _bush_parallel(bush,bounds,delta,g,bp):
    rows,cols=bush.shape
    for pixel in prange(rows*cols):
        row=pixel//cols
        col=pixel%cols
        _bush_pixel(row,col,bush,bounds,delta,g,bp)


@njit(inline='always',fastmath=False)
def _finish_pixel(row,col,bush,bush_positive,bp,g,sh,vs,vb,vegetation):
    original_vs=vs[row,col]
    sh[row,col]=np.float32(1)-sh[row,col]
    combined=vb[row,col]
    if combined>0:combined=np.float32(1)
    combined=np.float32(combined-original_vs)
    vb[row,col]=np.float32(1)-combined
    value=original_vs
    if bush_positive:
        difference=g[row,col]-bush[row,col]
        if difference>0:difference=1
        if difference<0:difference=0
        value=np.float32(original_vs-bp[row,col])+difference
        if value<0:value=0
    if value>0:value=1
    vegetation[row,col]=1-value


@njit(cache=True,fastmath=False)
def _finish_serial(bush,bush_positive,bp,g,sh,vs,vb,vegetation):
    for row in range(bush.shape[0]):
        for col in range(bush.shape[1]):
            _finish_pixel(row,col,bush,bush_positive,bp,g,sh,vs,vb,vegetation)


@njit(cache=True,fastmath=False,parallel=True)
def _finish_parallel(bush,bush_positive,bp,g,sh,vs,vb,vegetation):
    rows,cols=bush.shape
    for pixel in prange(rows*cols):
        row=pixel//cols
        col=pixel%cols
        _finish_pixel(row,col,bush,bush_positive,bp,g,sh,vs,vb,vegetation)


@njit(cache=True,fastmath=False)
def _run_serial(a,canopy,trunk,bush,bounds,da,dv,dt,db,first_heights,bush_positive,f,sh,vs,vb,tv,g,bp,vegetation):
    for step in range(len(bounds)):
        _advance_serial(a,canopy,trunk,bounds[step],da[step],dv[step],dt[step],first_heights[step],step==0,f,sh,vs,vb,tv)
        if bush_positive and _bush_active(tv,a,bush):
            _bush_serial(bush,bounds[step],db[step],g,bp)
    _finish_serial(bush,bush_positive,bp,g,sh,vs,vb,vegetation)


@njit(cache=True,fastmath=False)
def _run_parallel(a,canopy,trunk,bush,bounds,da,dv,dt,db,first_heights,bush_positive,f,sh,vs,vb,tv,g,bp,vegetation):
    # The driver is serial. Native parallel callees complete at each stage,
    # providing the global bush-reduction and next-step ownership barriers.
    for step in range(len(bounds)):
        _advance_parallel(a,canopy,trunk,bounds[step],da[step],dv[step],dt[step],first_heights[step],step==0,f,sh,vs,vb,tv)
        if bush_positive and _bush_active(tv,a,bush):
            _bush_parallel(bush,bounds[step],db[step],g,bp)
    _finish_parallel(bush,bush_positive,bp,g,sh,vs,vb,vegetation)


def _shadow(amaxvalue,a,vegdem,vegdem2,bush,azimuth,altitude,scale,parallel):
    a,vegdem,vegdem2,bush=map(np.asarray,(a,vegdem,vegdem2,bush))
    if any(array.shape!=a.shape for array in (vegdem,vegdem2,bush)):
        raise ValueError('shadow input shapes must match')
    bounds,heights=ray_schedule(a.shape,amaxvalue,azimuth,altitude,scale)
    da=np.asarray(heights,dtype=a.dtype)
    dv=np.asarray(heights,dtype=vegdem.dtype)
    dt=np.asarray(heights,dtype=vegdem2.dtype)
    db=np.asarray(heights,dtype=bush.dtype)
    first_height=np.asarray(heights,dtype=np.float32)
    f=a.copy()
    sh=np.zeros(a.shape,dtype=np.float32)
    bp=(bush>1).astype(np.float32)
    vs=bp.copy()
    vb=np.zeros(a.shape,dtype=np.float32)
    tv=np.empty(a.shape,dtype=np.float32)
    g=np.zeros(a.shape,dtype=np.float32)
    bush_positive=bool(bush.max()>0)
    # The legacy bush tail promotes vegetation output to bush dtype only when
    # its global branch executes. Combined visibility always stays float32.
    vegetation=np.empty(a.shape,dtype=np.result_type(np.float32,bush.dtype) if bush_positive else np.float32)
    kernel=_run_parallel if parallel else _run_serial
    kernel(a,vegdem,vegdem2,bush,bounds,da,dv,dt,db,first_height,bush_positive,f,sh,vs,vb,tv,g,bp,vegetation)
    return sh,vegetation,vb


def shadow_serial(amaxvalue,a,vegdem,vegdem2,bush,azimuth,altitude,scale):
    return _shadow(amaxvalue,a,vegdem,vegdem2,bush,azimuth,altitude,scale,False)


def shadow_parallel(amaxvalue,a,vegdem,vegdem2,bush,azimuth,altitude,scale):
    return _shadow(amaxvalue,a,vegdem,vegdem2,bush,azimuth,altitude,scale,True)


def shadow(amaxvalue,a,vegdem,vegdem2,bush,azimuth,altitude,scale):
    """Serial default; parallel execution is explicit until benchmark review."""
    return shadow_serial(amaxvalue,a,vegdem,vegdem2,bush,azimuth,altitude,scale)


@njit(inline='always',fastmath=False)
def _trace_pixel(row,col,a,canopy,trunk,bush,bounds,da,dv,dt,first_heights):
    height=a[row,col]
    f=height
    sh=np.float32(0)
    vs=np.float32(bush[row,col]>1)
    vb=np.float32(0)
    for step in range(len(bounds)):
        xc1,xc2,yc1,yc2,xp1,xp2,yp1,yp2=bounds[step]
        building=np.float32(0)
        vegetation=np.float32(0)
        trunk_value=np.float32(0)
        if xp1<=row<xp2 and yp1<=col<yp2:
            source_row=xc1+row-xp1
            source_col=yc1+col-yp1
            building=np.float32(a[source_row,source_col]-da[step])
            vegetation=np.float32(canopy[source_row,source_col]-dv[step])
            trunk_value=np.float32(trunk[source_row,source_col]-dt[step])
        f=_maximum(f,building)
        if f>height:
            sh=np.float32(1)
        elif f<=height:
            sh=np.float32(0)
        vs=_maximum(vs,np.float32(vegetation>height)-np.float32(trunk_value>height))
        if vs*sh>0:
            vs=np.float32(0)
        vb=np.float32(vb+vs)
        if step==0:
            first_vegetation=np.float32(vegetation-building)
            if first_vegetation<=0:
                first_vegetation=np.float32(1000)
            if first_vegetation<first_heights[step]:
                vs=np.float32(1)
            vs=np.float32(vs*np.float32(trunk[row,col]>height))
            vb=np.float32(0)
    if vb>0:vb=np.float32(1)
    vb=np.float32(vb-vs)
    if vs>0:vs=np.float32(1)
    return np.float32(1)-sh,np.float32(1)-vs,np.float32(1)-vb


@njit(cache=True,fastmath=False)
def _pixel_serial(a,canopy,trunk,bush,bounds,da,dv,dt,first_heights,sh,vs,vb):
    for row in range(a.shape[0]):
        for col in range(a.shape[1]):
            sh[row,col],vs[row,col],vb[row,col]=_trace_pixel(row,col,a,canopy,trunk,bush,bounds,da,dv,dt,first_heights)


@njit(cache=True,fastmath=False,parallel=True)
def _pixel_parallel(a,canopy,trunk,bush,bounds,da,dv,dt,first_heights,sh,vs,vb):
    rows,cols=a.shape
    for pixel in prange(rows*cols):
        row=pixel//cols
        col=pixel%cols
        sh[row,col],vs[row,col],vb[row,col]=_trace_pixel(row,col,a,canopy,trunk,bush,bounds,da,dv,dt,first_heights)


def _shadow_pixel(amaxvalue,a,vegdem,vegdem2,bush,azimuth,altitude,scale,parallel):
    a,vegdem,vegdem2,bush=map(np.asarray,(a,vegdem,vegdem2,bush))
    if any(array.shape!=a.shape for array in (vegdem,vegdem2,bush)):
        raise ValueError('shadow input shapes must match')
    if bush.max()>0:
        # A positive global bush layer may update g based on whole-domain
        # visibility. Preserve the step-major implementation for that case.
        return _shadow(amaxvalue,a,vegdem,vegdem2,bush,azimuth,altitude,scale,parallel)
    # Independence proof: the static global bush predicate is false for every
    # ray, so g and the bush tail never execute. f/sh/vs/vb only read their own
    # prior value and immutable shifted input fields. Each pixel follows the
    # complete original schedule in order; no pixel/ray early termination.
    bounds,heights=ray_schedule(a.shape,amaxvalue,azimuth,altitude,scale)
    da=np.asarray(heights,dtype=a.dtype)
    dv=np.asarray(heights,dtype=vegdem.dtype)
    dt=np.asarray(heights,dtype=vegdem2.dtype)
    first_heights=np.asarray(heights,dtype=np.float32)
    outputs=tuple(np.empty(a.shape,dtype=np.float32) for _ in range(3))
    kernel=_pixel_parallel if parallel else _pixel_serial
    kernel(a,vegdem,vegdem2,bush,bounds,da,dv,dt,first_heights,*outputs)
    return outputs


def shadow_pixel_serial(amaxvalue,a,vegdem,vegdem2,bush,azimuth,altitude,scale):
    """Pixel-major when the bush predicate is false; exact step-major fallback."""
    return _shadow_pixel(amaxvalue,a,vegdem,vegdem2,bush,azimuth,altitude,scale,False)


def shadow_pixel_parallel(amaxvalue,a,vegdem,vegdem2,bush,azimuth,altitude,scale):
    """Independent pixel ownership with the same positive-bush fallback."""
    return _shadow_pixel(amaxvalue,a,vegdem,vegdem2,bush,azimuth,altitude,scale,True)
