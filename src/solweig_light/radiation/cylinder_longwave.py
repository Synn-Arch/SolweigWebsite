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
"""Private demand-specialized cylinder longwave (dossier D03 reduction R-B).

The compiled ``_longwave`` kernel family keeps two ordered sweeps and fourteen
per-pixel accumulators. Accumulators 10..13 feed only the returned cardinal
diagnostic columns 7..10; they are never read by any primary computation, and
the reflected sweep depends on accumulator 0 alone. Under the private
pipeline cylinder-anisotropic demand the engine consumes only the primary
totals (columns 0 and 1), so the four cardinal projection loops can be
skipped without touching any primary result. Both sweeps, every primary
operation, cast and grouping are retained bit-exactly; only the write-only
cardinal accumulation is omitted.

``FULL_DIAGNOSTICS`` delegates to the untouched public full path, so all
fourteen channels, both sweeps and all live state are retained whenever
diagnostics are requested. Omitted diagnostics are ``NOT_REQUESTED``, never
zeros. Unsupported profiles fall back to the original serial reference in
both profiles, in the original order of guard failures.
"""
from contextlib import contextmanager
from enum import Enum
import inspect
import os

import numpy as np
from numba import njit, prange

# The njit kernels need the packed-visibility decode helper as a module
# global; importing here is cycle-free exactly as in patch_radiation.
from .._native_dispatch.aplus_decode import _decode_at_plus as _decode_at


class CylinderLongwaveDemand(Enum):
    """Private radiation demand profile (dossier D09 interface 3)."""

    PIPELINE_CYLINDERS_ANISOTROPIC = 'pipeline_cylinders_anisotropic'
    FULL_DIAGNOSTICS = 'full_diagnostics'


class _NotRequested:
    """Singleton marker: a diagnostic omitted by contract, never a zero."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return 'NOT_REQUESTED'


NOT_REQUESTED = _NotRequested()

_demand = CylinderLongwaveDemand.FULL_DIAGNOSTICS


def current_demand():
    """Active private demand profile; defaults to the full public behavior."""
    return _demand


def set_demand(demand):
    """Set the private demand profile and return the previous value."""
    global _demand
    previous = _demand
    _demand = CylinderLongwaveDemand(demand)
    return previous


@contextmanager
def demand_scope(demand):
    """Scope the private demand profile; restores the previous value."""
    previous = set_demand(demand)
    try:
        yield current_demand()
    finally:
        set_demand(previous)


@njit(cache=True, fastmath=False, parallel=True)
def _longwave_primary(sh,vs,vb,sun,shade,solid,sine,cosine,directions,gate,solar_gate,sky_down,sky_side,surface_sun,surface_sh,lup,reflection_factor):
    pixels,patches=sh.shape
    output=np.zeros((pixels,7),dtype=np.float32)
    for pixel in prange(pixels):
        accum=np.zeros(10,dtype=np.float32)
        for patch in range(patches):
            sky=sh[pixel,patch]==1 and vs[pixel,patch]==1
            veg=vs[pixel,patch]==0 or vb[pixel,patch]==0
            building=np.float32(np.float32(1)-sh[pixel,patch])*vb[pixel,patch]==1
            accum[0]=np.float32(accum[0]+np.float32(sky*sky_down[patch]))
            accum[5]=np.float32(accum[5]+np.float32(sky*sky_side[patch]))
            vegetation_side=((surface_sh*solid[patch])*cosine[patch])*veg
            vegetation_down=((surface_sh*solid[patch])*sine[patch])*veg
            accum[6]=np.float32(accum[6]+vegetation_side)
            accum[1]=np.float32(accum[1]+vegetation_down)
            if solar_gate[patch]:
                sun_side=((((surface_sun*sun[pixel,patch])*solid[patch])*cosine[patch])*building)
                shade_side=((((surface_sh*shade[pixel,patch])*solid[patch])*cosine[patch])*building)
                sun_down=((((surface_sun*sun[pixel,patch])*solid[patch])*sine[patch])*building)
                shade_down=((((surface_sh*shade[pixel,patch])*solid[patch])*sine[patch])*building)
                accum[8]=np.float32(accum[8]+sun_side)
                accum[7]=np.float32(accum[7]+shade_side)
                accum[3]=np.float32(accum[3]+sun_down)
                accum[2]=np.float32(accum[2]+shade_down)
            else:
                shade_side=(((surface_sh*solid[patch])*cosine[patch])*building)
                shade_down=(((surface_sh*solid[patch])*sine[patch])*building)
                accum[7]=np.float32(accum[7]+shade_side)
                accum[2]=np.float32(accum[2]+shade_down)
        # The reflection field depends on the completed ordered sky sweep.
        reflected=np.float32(np.float32(np.float32(np.float32(accum[0]+lup[pixel])*reflection_factor)*np.float32(.5))/np.float32(np.pi))
        for patch in range(patches):
            mask=sh[pixel,patch]==0 or vs[pixel,patch]==0 or vb[pixel,patch]==0
            side=np.float32(np.float32(np.float32(reflected*solid[patch])*cosine[patch])*mask)
            down=np.float32(np.float32(np.float32(reflected*solid[patch])*sine[patch])*mask)
            accum[9]=np.float32(accum[9]+side)
            accum[4]=np.float32(accum[4]+down)
        output[pixel,0]=np.float32(np.float32(np.float32(np.float32(accum[0]+accum[1])+accum[2])+accum[3])+accum[4])
        output[pixel,1]=np.float32(np.float32(np.float32(np.float32(accum[5]+accum[6])+accum[7])+accum[8])+accum[9])
        output[pixel,2:7]=accum[5:10]
    return output


@njit(cache=True, fastmath=False)
def _longwave_primary_serial(sh,vs,vb,sun,shade,solid,sine,cosine,directions,gate,solar_gate,sky_down,sky_side,surface_sun,surface_sh,lup,reflection_factor):
    pixels,patches=sh.shape
    output=np.zeros((pixels,7),dtype=np.float32)
    for pixel in range(pixels):
        accum=np.zeros(10,dtype=np.float32)
        for patch in range(patches):
            sky=sh[pixel,patch]==1 and vs[pixel,patch]==1
            veg=vs[pixel,patch]==0 or vb[pixel,patch]==0
            building=np.float32(np.float32(1)-sh[pixel,patch])*vb[pixel,patch]==1
            accum[0]=np.float32(accum[0]+np.float32(sky*sky_down[patch]))
            accum[5]=np.float32(accum[5]+np.float32(sky*sky_side[patch]))
            vegetation_side=((surface_sh*solid[patch])*cosine[patch])*veg
            vegetation_down=((surface_sh*solid[patch])*sine[patch])*veg
            accum[6]=np.float32(accum[6]+vegetation_side)
            accum[1]=np.float32(accum[1]+vegetation_down)
            if solar_gate[patch]:
                sun_side=((((surface_sun*sun[pixel,patch])*solid[patch])*cosine[patch])*building)
                shade_side=((((surface_sh*shade[pixel,patch])*solid[patch])*cosine[patch])*building)
                sun_down=((((surface_sun*sun[pixel,patch])*solid[patch])*sine[patch])*building)
                shade_down=((((surface_sh*shade[pixel,patch])*solid[patch])*sine[patch])*building)
                accum[8]=np.float32(accum[8]+sun_side)
                accum[7]=np.float32(accum[7]+shade_side)
                accum[3]=np.float32(accum[3]+sun_down)
                accum[2]=np.float32(accum[2]+shade_down)
            else:
                shade_side=(((surface_sh*solid[patch])*cosine[patch])*building)
                shade_down=(((surface_sh*solid[patch])*sine[patch])*building)
                accum[7]=np.float32(accum[7]+shade_side)
                accum[2]=np.float32(accum[2]+shade_down)
        # The reflection field depends on the completed ordered sky sweep.
        reflected=np.float32(np.float32(np.float32(np.float32(accum[0]+lup[pixel])*reflection_factor)*np.float32(.5))/np.float32(np.pi))
        for patch in range(patches):
            mask=sh[pixel,patch]==0 or vs[pixel,patch]==0 or vb[pixel,patch]==0
            side=np.float32(np.float32(np.float32(reflected*solid[patch])*cosine[patch])*mask)
            down=np.float32(np.float32(np.float32(reflected*solid[patch])*sine[patch])*mask)
            accum[9]=np.float32(accum[9]+side)
            accum[4]=np.float32(accum[4]+down)
        output[pixel,0]=np.float32(np.float32(np.float32(np.float32(accum[0]+accum[1])+accum[2])+accum[3])+accum[4])
        output[pixel,1]=np.float32(np.float32(np.float32(np.float32(accum[5]+accum[6])+accum[7])+accum[8])+accum[9])
        output[pixel,2:7]=accum[5:10]
    return output


def _lw_region_route(values,geometry,solar_gate,prepared,total,block_pixels,factor,sun_surface,shade_surface):
    """N9 default route: the bounded Numba stream for admitted invocations.

    Returns a [7, total] float32 field stack, or None when the invocation
    is structurally declined (non-admitted channels, the all-raw payload
    class, a lane-misaligned block size, degenerate extent) -- the
    caller's legacy loop runs unchanged in every None case. A post-launch
    failure is never hidden. There is no selection policy and no registry
    consult anywhere in this path (N9 F4); the only env read on this seam
    remains the expert stand-down below.

    Established expert compatibility: an explicit native/ispc request is
    served by the B7-32 route in _lw_kernel, so the stream route stands
    down here (this file remains the ONLY env-read site).
    """
    if os.environ.get(_LW_BACKEND_ENV,'').strip().lower() in ('native','ispc'):
        return None
    from ._lw_dispatch import region_route
    return region_route(values,geometry,solar_gate,prepared,total,block_pixels,factor,sun_surface,shade_surface)


_LW_BACKEND_ENV='SOLWEIG_LIGHT_LW_BACKEND'


def _lw_kernel(parallel):
    """Kernel resolver for the primary-output reduction.

    Default (env unset): the unchanged baseline Numba kernels. With
    SOLWEIG_LIGHT_LW_BACKEND=native, dispatch to the optional ISPC backend
    (solweig_light.backends.native_lw, B7-32 selection); inputs outside the
    backend's reviewed admission domain fall back to the Numba kernel before
    the backend launches (its guard raises pre-launch), and a requested
    backend that cannot build fails loudly instead of silently falling back.
    """
    kernel=_longwave_primary if parallel else _longwave_primary_serial
    backend=os.environ.get(_LW_BACKEND_ENV,'').strip().lower()
    if backend not in ('native','ispc'):
        return kernel
    from ..backends.native_lw import native_longwave_primary, UnsupportedInput
    def dispatch(*args):
        try:
            return native_longwave_primary(*args)
        except UnsupportedInput:
            return kernel(*args)
    return dispatch


@njit(cache=True, fastmath=False, parallel=True)
def _longwave_fused_primary(sh_flat,sh_off,sh_modes,vs_flat,vs_off,vs_modes,vb_flat,vb_off,vb_modes,start,stop,sun,shade,solid,sine,cosine,directions,gate,solar_gate,sky_down,sky_side,surface_sun,surface_sh,lup,reflection_factor):
    rows=stop-start
    patches=sh_modes.shape[0]
    output=np.zeros((rows,7),dtype=np.float32)
    # The reflected sweep re-reads only the occlusion predicate, captured here
    # during the sky sweep so no payload is decoded twice.
    masks=np.empty((rows,patches),dtype=np.bool_)
    for row in prange(rows):
        pixel=start+row
        accum=np.zeros(10,dtype=np.float32)
        for patch in range(patches):
            sh_v=_decode_at(sh_flat,sh_off,sh_modes,patch,pixel)
            vs_v=_decode_at(vs_flat,vs_off,vs_modes,patch,pixel)
            vb_v=_decode_at(vb_flat,vb_off,vb_modes,patch,pixel)
            masks[row,patch]= sh_v==0 or vs_v==0 or vb_v==0
            sky=sh_v==1 and vs_v==1
            veg=vs_v==0 or vb_v==0
            building=np.float32(np.float32(1)-sh_v)*vb_v==1
            accum[0]=np.float32(accum[0]+np.float32(sky*sky_down[patch]))
            accum[5]=np.float32(accum[5]+np.float32(sky*sky_side[patch]))
            vegetation_side=((surface_sh*solid[patch])*cosine[patch])*veg
            vegetation_down=((surface_sh*solid[patch])*sine[patch])*veg
            accum[6]=np.float32(accum[6]+vegetation_side)
            accum[1]=np.float32(accum[1]+vegetation_down)
            if solar_gate[patch]:
                sun_side=((((surface_sun*sun[row,patch])*solid[patch])*cosine[patch])*building)
                shade_side=((((surface_sh*shade[row,patch])*solid[patch])*cosine[patch])*building)
                sun_down=((((surface_sun*sun[row,patch])*solid[patch])*sine[patch])*building)
                shade_down=((((surface_sh*shade[row,patch])*solid[patch])*sine[patch])*building)
                accum[8]=np.float32(accum[8]+sun_side)
                accum[7]=np.float32(accum[7]+shade_side)
                accum[3]=np.float32(accum[3]+sun_down)
                accum[2]=np.float32(accum[2]+shade_down)
            else:
                shade_side=(((surface_sh*solid[patch])*cosine[patch])*building)
                shade_down=(((surface_sh*solid[patch])*sine[patch])*building)
                accum[7]=np.float32(accum[7]+shade_side)
                accum[2]=np.float32(accum[2]+shade_down)
        # The reflection field depends on the completed ordered sky sweep.
        reflected=np.float32(np.float32(np.float32(np.float32(accum[0]+lup[row])*reflection_factor)*np.float32(.5))/np.float32(np.pi))
        for patch in range(patches):
            mask=masks[row,patch]
            side=np.float32(np.float32(np.float32(reflected*solid[patch])*cosine[patch])*mask)
            down=np.float32(np.float32(np.float32(reflected*solid[patch])*sine[patch])*mask)
            accum[9]=np.float32(accum[9]+side)
            accum[4]=np.float32(accum[4]+down)
        output[row,0]=np.float32(np.float32(np.float32(np.float32(accum[0]+accum[1])+accum[2])+accum[3])+accum[4])
        output[row,1]=np.float32(np.float32(np.float32(np.float32(accum[5]+accum[6])+accum[7])+accum[8])+accum[9])
        output[row,2:7]=accum[5:10]
    return output


@njit(cache=True, fastmath=False)
def _longwave_fused_primary_serial(sh_flat,sh_off,sh_modes,vs_flat,vs_off,vs_modes,vb_flat,vb_off,vb_modes,start,stop,sun,shade,solid,sine,cosine,directions,gate,solar_gate,sky_down,sky_side,surface_sun,surface_sh,lup,reflection_factor):
    rows=stop-start
    patches=sh_modes.shape[0]
    output=np.zeros((rows,7),dtype=np.float32)
    # The reflected sweep re-reads only the occlusion predicate, captured here
    # during the sky sweep so no payload is decoded twice.
    masks=np.empty((rows,patches),dtype=np.bool_)
    for row in range(rows):
        pixel=start+row
        accum=np.zeros(10,dtype=np.float32)
        for patch in range(patches):
            sh_v=_decode_at(sh_flat,sh_off,sh_modes,patch,pixel)
            vs_v=_decode_at(vs_flat,vs_off,vs_modes,patch,pixel)
            vb_v=_decode_at(vb_flat,vb_off,vb_modes,patch,pixel)
            masks[row,patch]= sh_v==0 or vs_v==0 or vb_v==0
            sky=sh_v==1 and vs_v==1
            veg=vs_v==0 or vb_v==0
            building=np.float32(np.float32(1)-sh_v)*vb_v==1
            accum[0]=np.float32(accum[0]+np.float32(sky*sky_down[patch]))
            accum[5]=np.float32(accum[5]+np.float32(sky*sky_side[patch]))
            vegetation_side=((surface_sh*solid[patch])*cosine[patch])*veg
            vegetation_down=((surface_sh*solid[patch])*sine[patch])*veg
            accum[6]=np.float32(accum[6]+vegetation_side)
            accum[1]=np.float32(accum[1]+vegetation_down)
            if solar_gate[patch]:
                sun_side=((((surface_sun*sun[row,patch])*solid[patch])*cosine[patch])*building)
                shade_side=((((surface_sh*shade[row,patch])*solid[patch])*cosine[patch])*building)
                sun_down=((((surface_sun*sun[row,patch])*solid[patch])*sine[patch])*building)
                shade_down=((((surface_sh*shade[row,patch])*solid[patch])*sine[patch])*building)
                accum[8]=np.float32(accum[8]+sun_side)
                accum[7]=np.float32(accum[7]+shade_side)
                accum[3]=np.float32(accum[3]+sun_down)
                accum[2]=np.float32(accum[2]+shade_down)
            else:
                shade_side=(((surface_sh*solid[patch])*cosine[patch])*building)
                shade_down=(((surface_sh*solid[patch])*sine[patch])*building)
                accum[7]=np.float32(accum[7]+shade_side)
                accum[2]=np.float32(accum[2]+shade_down)
        # The reflection field depends on the completed ordered sky sweep.
        reflected=np.float32(np.float32(np.float32(np.float32(accum[0]+lup[row])*reflection_factor)*np.float32(.5))/np.float32(np.pi))
        for patch in range(patches):
            mask=masks[row,patch]
            side=np.float32(np.float32(np.float32(reflected*solid[patch])*cosine[patch])*mask)
            down=np.float32(np.float32(np.float32(reflected*solid[patch])*sine[patch])*mask)
            accum[9]=np.float32(accum[9]+side)
            accum[4]=np.float32(accum[4]+down)
        output[row,0]=np.float32(np.float32(np.float32(np.float32(accum[0]+accum[1])+accum[2])+accum[3])+accum[4])
        output[row,1]=np.float32(np.float32(np.float32(np.float32(accum[5]+accum[6])+accum[7])+accum[8])+accum[9])
        output[row,2:7]=accum[5:10]
    return output


def _longwave_fused_primary_block(shadow,vegetation,vegetation_building,start,stop,patches,sun,shade,solid,sine,cosine,directions,gate,solar_gate,sky_down,sky_side,surface_sun,surface_sh,lup,reflection_factor,parallel):
    """Fused ordered decode+reduce for one block; None selects the retained route."""
    from .patch_radiation import _fused_enabled, _packed_leaves, _fused_guard
    if not _fused_enabled():
        return None
    from ..geometry.visibility_compiled import _fused_descriptor, _preflight_flat
    leaves=[_packed_leaves(channel) for channel in (shadow,vegetation,vegetation_building)]
    if any(pair is None or len(pair)!=1 for pair in leaves):
        return None
    flat=[pair[0] for pair in leaves]
    with _fused_guard(flat,start,stop,patches):
        sh_desc=_fused_descriptor(leaves[0][0])
        vs_desc=_fused_descriptor(leaves[1][0])
        vb_desc=_fused_descriptor(leaves[2][0])
        # Preflight every decode stream in the original observable read order.
        _preflight_flat(*sh_desc,start,stop,patches)
        _preflight_flat(*vs_desc,start,stop,patches)
        _preflight_flat(*vb_desc,start,stop,patches)
        kernel=_longwave_fused_primary if parallel else _longwave_fused_primary_serial
        return kernel(*sh_desc,*vs_desc,*vb_desc,start,stop,sun,shade,solid,sine,cosine,
                      directions,gate,solar_gate,sky_down,sky_side,surface_sun,surface_sh,
                      lup,reflection_factor)


def define_patch_characteristics_primary(*args,block_pixels=128,parallel=True,**kwargs):
    """Primary-output driver: identical to the public driver minus the four
    cardinal projection loops and the four cardinal output columns.

    ``parallel`` tri-state (N8-40b public seam): ``True`` is an explicit
    parallel demand; ``False`` is an explicit serial demand -- the stream
    route gate stays shut and the serial kernels run, never dispatching;
    ``None`` is no explicit demand -- serial kernels exactly as ``False``
    when the stream declines, but the single ``_lw_region_route`` consult
    fires so an admitted invocation can take the bounded Numba stream
    (the leaf's own prange carries the granted threading).
    """
    from . import engine as e
    from .patch_radiation import _reference, _block, _class_coefficients, _classes, _supported, patch_geometry
    reference=_reference('define_patch_characteristics')
    values=inspect.signature(reference).bind(*args,**kwargs).arguments
    if not isinstance(values['solar_altitude'],np.ndarray):
        return reference(*args,**kwargs)
    if any(np.asarray(values[key]).ndim != 0 for key in ('Ta','Tgwall','ewall','solar_altitude','solar_azimuth')) or not _supported([values[k] for k in ('patch_altitude','patch_azimuth','steradian','asvf','Lup','Lsky_down','Lsky_side')],[values[k] for k in ('shmat','vegshmat','vbshvegshmat')]):
        return reference(*args,**kwargs)
    rows,cols=values['rows'],values['cols']
    if not 0 < values['patch_altitude'].size <= 609:
        return reference(*args,**kwargs)
    geometry=patch_geometry(np.column_stack((values['patch_altitude'],values['patch_azimuth'])))
    count=geometry.altitude.size
    radians=e._array(np.pi/180.)
    ewall=e._array(values['ewall'])
    sbc=e._array(5.67051e-8)
    sun_surface=e._divide(e._operate(np.multiply,e._operate(np.multiply,ewall,sbc),e._operate(np.power,e._operate(np.add,e._operate(np.add,values['Ta'],values['Tgwall']),273.15),4)),e._array(np.pi))[()]
    shade_surface=e._divide(e._operate(np.multiply,e._operate(np.multiply,ewall,sbc),e._operate(np.power,e._operate(np.add,values['Ta'],273.15),4)),e._array(np.pi))[()]
    directions=geometry.longwave_cardinal_cosine
    azi=geometry.azimuth
    gate=geometry.reflection_cardinal
    difference=np.asarray([np.abs(e._operate(np.subtract,values['solar_azimuth'],value)) for value in azi])
    solar_gate=(difference>90)&(difference<270)&(values['solar_altitude']>0)
    output=np.empty((7,rows*cols),dtype=np.float32)
    kernel=_lw_kernel(parallel)
    factor=e._operate(np.subtract,1,ewall)[()]
    if block_pixels<1:raise ValueError('block_pixels must be positive')
    prepared=_class_coefficients(values['solar_altitude'],values['solar_azimuth'],geometry,values['asvf'],solar_gate) if rows*cols else None
    if parallel is not False and rows*cols:
        routed=_lw_region_route(values,geometry,solar_gate,prepared,rows*cols,block_pixels,factor,sun_surface,shade_surface)
        if routed is not None:
            return tuple(field.reshape(rows,cols) for field in routed)
    for start in range(0,rows*cols,block_pixels):
        stop=min(start+block_pixels,rows*cols)
        sun,shade=_classes(values['solar_altitude'],values['solar_azimuth'],geometry,values['asvf'],start,stop,active=solar_gate,prepared=prepared)
        reduced=_longwave_fused_primary_block(values['shmat'],values['vegshmat'],values['vbshvegshmat'],
                                              start,stop,count,sun,shade,values['steradian'],geometry.sine,
                                              geometry.cosine,directions,gate,solar_gate,
                                              values['Lsky_down'][:,2],values['Lsky_side'][:,2],
                                              sun_surface,shade_surface,values['Lup'].reshape(-1)[start:stop],factor,parallel)
        if reduced is None:
            sh,vs,vb=(_block(values[name],start,stop,count) for name in ('shmat','vegshmat','vbshvegshmat'))
            reduced=kernel(sh,vs,vb,sun,shade,values['steradian'],geometry.sine,geometry.cosine,directions,gate,solar_gate,values['Lsky_down'][:,2],values['Lsky_side'][:,2],sun_surface,shade_surface,values['Lup'].reshape(-1)[start:stop],factor)
        output[:,start:stop]=reduced.T
    return tuple(field.reshape(rows,cols) for field in output)


def Lcyl_v2022a_primary(*args,block_pixels=128,parallel=True,**kwargs):
    """Primary-output twin of the compiled ``Lcyl_v2022a`` fast path.

    Returns ``(Ldown, Lside, NOT_REQUESTED, NOT_REQUESTED, NOT_REQUESTED,
    NOT_REQUESTED)``; the four cardinal diagnostic fields are omitted by
    contract. Guard failures fall back to the untouched serial reference in
    the original order. ``parallel`` follows the N8-40b tri-state contract
    documented on ``define_patch_characteristics_primary``.
    """
    from . import engine as e
    from .patch_radiation import _reference, _model2, _supported, patch_geometry
    reference=_reference('Lcyl_v2022a')
    v=inspect.signature(reference).bind(*args,**kwargs).arguments
    if not isinstance(v['solar_altitude'],np.ndarray):
        return reference(*args,**kwargs)
    if any(np.asarray(v[key]).ndim != 0 for key in ('esky','Ta','Tgwall','ewall','solar_altitude','solar_azimuth')) or not _supported([v[k] for k in ('sky_patches','Lup','asvf')],[v[k] for k in ('shmat','vegshmat','vbshvegshmat')]):
        return reference(*args,**kwargs)
    table=v['sky_patches']
    if not 0 < table.shape[0] <= 609:
        return reference(*args,**kwargs)
    geometry=patch_geometry(table)
    sbc=e._array(5.67051e-8)
    radians=e._array(np.pi/180)
    bands=geometry.bands
    _,emissivity=_model2(geometry,v['esky'])
    down=np.zeros(table.shape[0],dtype=np.float32)
    side=np.zeros_like(down)
    normal=np.zeros_like(down)
    # Preserve the per-step model2 coefficient generation and band operations.
    for band_index,altitude in enumerate(bands):
        mask=geometry.band_membership[band_index]
        emission=emissivity[band_index:band_index+1]
        flux=e._divide(e._operate(np.multiply,e._operate(np.multiply,emission,sbc),e._operate(np.power,e._operate(np.add,v['Ta'],273.15),4)),e._array(np.pi))
        down[mask]=e._operate(np.multiply,e._operate(np.multiply,flux,geometry.solid_angle[mask]),np.sin(e._operate(np.multiply,altitude,radians)))
        side[mask]=e._operate(np.multiply,e._operate(np.multiply,flux,geometry.solid_angle[mask]),np.cos(e._operate(np.multiply,altitude,radians)))
        normal[mask]=e._operate(np.multiply,flux,geometry.solid_angle[mask])
    ldown,lside,lnormal=(table.copy() for _ in range(3))
    ldown[:,2],lside[:,2],lnormal[:,2]=down,side,normal
    result=define_patch_characteristics_primary(v['solar_altitude'],v['solar_azimuth'],geometry.altitude,geometry.azimuth,geometry.solid_angle,v['asvf'],v['shmat'],v['vegshmat'],v['vbshvegshmat'],ldown,lside,lnormal,v['Lup'],v['Ta'],v['Tgwall'],v['ewall'],v['rows'],v['cols'],block_pixels=block_pixels,parallel=parallel)
    return (result[0],result[1],NOT_REQUESTED,NOT_REQUESTED,NOT_REQUESTED,NOT_REQUESTED)


def Lcyl_v2022a_by_demand(*args,block_pixels=128,parallel=True,**kwargs):
    """Dispatch the cylinder longwave call on the private demand profile.

    ``FULL_DIAGNOSTICS`` (default) runs the untouched public full path.
    ``PIPELINE_CYLINDERS_ANISOTROPIC`` runs the primary-output reduction and
    reports the four cardinal fields as ``NOT_REQUESTED``.

    ``parallel`` follows the N8-40b tri-state contract documented on
    ``define_patch_characteristics_primary`` (``True``/``False`` explicit
    demands, ``None`` the route-eligible no-demand); the full path is
    entered with ``bool(parallel)`` so patch_radiation keeps its boolean
    domain (``None`` and ``False`` both select its serial kernels).
    """
    from .patch_radiation import Lcyl_v2022a as full
    demand=current_demand()
    if demand is CylinderLongwaveDemand.FULL_DIAGNOSTICS:
        return full(*args,block_pixels=block_pixels,parallel=bool(parallel),**kwargs)
    if demand is CylinderLongwaveDemand.PIPELINE_CYLINDERS_ANISOTROPIC:
        return Lcyl_v2022a_primary(*args,block_pixels=block_pixels,parallel=parallel,**kwargs)
    raise ValueError(f'unsupported cylinder longwave demand: {demand!r}')
