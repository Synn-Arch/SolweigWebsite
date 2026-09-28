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
"""Bounded patch geometry and ordered, pixel-owned anisotropic reductions.

Only immutable patch geometry is cached. Forcing coefficients and solar
classification retain the serial engine's NumPy operations on every call.
The compiled profile uses float32 raster/table/visibility fields, scalar
forcing, scalar albedo/wall temperature, and at most 609 patches. Other
profiles use the retained serial engine. Block size only bounds decode work.
"""
from dataclasses import dataclass
from functools import lru_cache
import inspect
import os

import numpy as np
from numba import njit, prange

# Private compiled decode helpers; importing this module is cycle-free
# (geometry/__init__ is docstring-only) and the kernels need them as globals.
from .._native_dispatch.aplus_decode import _decode_at_plus as _decode_at
# The exact-table classification kernel evaluates the same SLEEF scalar core
# that atan_array loops over; numba needs it as a module-level global.
from ._sleef_classifier import atan_fma


@dataclass(frozen=True)
class PatchGeometry:
    altitude: np.ndarray
    azimuth: np.ndarray
    solid_angle: np.ndarray
    sine: np.ndarray
    cosine: np.ndarray
    bands: np.ndarray
    band_counts: np.ndarray
    band_membership: np.ndarray
    diffuse_cardinal: np.ndarray
    reflection_cardinal: np.ndarray
    longwave_cardinal_cosine: np.ndarray


def _immutable(array):
    array=np.asarray(array)
    return np.frombuffer(array.tobytes(),dtype=array.dtype).reshape(array.shape)


@lru_cache(maxsize=8)
def _cached_geometry(shape,dtype,payload):
    from . import engine as e
    table=np.frombuffer(payload,dtype=dtype).reshape(shape)
    altitude,azimuth=table[:,0],table[:,1]
    bands,counts=np.unique(altitude,return_counts=True)
    radians=e._array(np.pi/180)
    solid=np.zeros(altitude.size,dtype=np.float32)
    for i in range(altitude.size):
        count=counts[bands==altitude[i]].item()
        if count>1:
            difference=e._operate(np.subtract,np.sin(e._operate(np.multiply,e._operate(np.add,altitude[i],altitude[0]),radians)),np.sin(e._operate(np.multiply,e._operate(np.subtract,altitude[i],altitude[0]),radians)))
        else:
            difference=e._operate(np.subtract,np.sin(e._operate(np.multiply,altitude[i],radians)),np.sin(e._operate(np.multiply,e._operate(np.add,altitude[i-1],altitude[0]),radians)))
        solid[i]=e._operate(np.multiply,e._operate(np.multiply,e._divide(360,count),radians),difference)
    angles=e._operate(np.multiply,altitude,radians)
    membership=bands[:,None]==altitude[None,:]
    diffuse=np.column_stack(((azimuth>360)|(azimuth<=180),(azimuth>90)&(azimuth<=270),(azimuth>180)&(azimuth<=360),(azimuth>270)|(azimuth<=90)))
    reflection=np.column_stack(((azimuth>360)|(azimuth<180),(azimuth>90)&(azimuth<270),(azimuth>180)&(azimuth<360),(azimuth>270)|(azimuth<90)))
    cardinal=np.empty((altitude.size,4),dtype=np.float32)
    for patch in range(altitude.size):
        for direction,origin in enumerate((90,180,270,0)):
            cardinal[patch,direction]=np.cos(e._operate(np.multiply,e._operate(np.subtract,origin,azimuth[patch]),radians))
    return PatchGeometry(*map(_immutable,(altitude,azimuth,solid,np.sin(angles),np.cos(angles),bands,counts,membership,diffuse,reflection,cardinal)))


def patch_geometry(table):
    table=np.asarray(table)
    if table.ndim!=2 or table.shape[1]<2 or not 0<table.shape[0]<=609:
        raise ValueError('cached patch geometry requires 1–609 ordered patches')
    positions=np.ascontiguousarray(table[:,:2])
    return _cached_geometry(positions.shape,positions.dtype.str,positions.tobytes())


def clear_geometry_cache():
    _cached_geometry.cache_clear()


def _reference(name):
    from . import engine as e
    for candidate in ('_serial_'+name,name+'_reference','reference_'+name,name+'_numpy',name):
        function=getattr(e,candidate,None)
        if function is not None and function.__module__!=__name__:
            return function
    raise RuntimeError('serial radiation reference unavailable for unsupported profile')


def _block(channel,start,stop,patches):
    from ..geometry.visibility import PackedVisibility, LazyDiffVisibility
    leaves=(channel.shadow,channel.vegetation) if isinstance(channel,LazyDiffVisibility) else (channel,)
    if all(isinstance(leaf,PackedVisibility) for leaf in leaves):
        from ..geometry.visibility_compiled import decode_block
        decoded=decode_block(channel,start,stop,patches)
        if decoded is not None:
            return decoded
    result=np.empty((stop-start,patches),dtype=np.float32)
    for patch in range(patches):
        if hasattr(channel,'decode_pixels'):
            result[:,patch]=channel.decode_pixels(patch,start,stop)
        else:
            indices=np.arange(start,stop,dtype=np.int64)
            result[:,patch]=channel[indices//channel.shape[1],indices%channel.shape[1],patch]
    return result


def _shortwave_visibility_blocks(shadow,vegetation,vegetation_building,diffuse,start,stop,patches):
    """Read shortwave visibility in the original observable order."""
    from ..geometry.visibility_prepared import decode_shortwave_block
    prepared=decode_shortwave_block(shadow,vegetation,vegetation_building,diffuse,start,stop,patches)
    if prepared is not None:
        return prepared
    sh=_block(shadow,start,stop,patches)
    vs=_block(vegetation,start,stop,patches)
    vb=_block(vegetation_building,start,stop,patches)
    from ..geometry.visibility_compiled import diff_from_shared_decoded
    diff=diff_from_shared_decoded(shadow,vegetation,diffuse,sh,vs,start,stop,patches)
    if diff is None:
        diff=_block(diffuse,start,stop,patches)
    return sh,vs,vb,diff


def _fused_enabled():
    """Fused radiation route is opt-in: SOLWEIG_LIGHT_FUSED_RAD=1, default OFF.

    Pipeline measurement showed the fused kernels regressing the radiation
    stage, so the dispatch hooks keep the retained route armed unless the
    environment explicitly requests the experimental path.
    """
    return os.environ.get('SOLWEIG_LIGHT_FUSED_RAD') == '1'


def _classes_exact_enabled():
    """Exact finite-state classification tables are opt-in:
    SOLWEIG_LIGHT_PATCH_CLASS_TABLES=1, default OFF.

    The retained per-block NumPy classification stays the dispatch default.
    The gate arms the exact-table route: deduplicated per-state coefficient
    construction (equal full source-state tuples share one verbatim scalar
    evaluation) plus a single compiled block pass that reproduces the
    retained route's elementwise arithmetic bitwise on admitted inputs.
    """
    return os.environ.get('SOLWEIG_LIGHT_PATCH_CLASS_TABLES') == '1'


def _packed_leaves(channel,allow_lazy=False):
    """Admitted packed leaves of a channel, or None when fusion is unsupported.

    Admission is deliberately restricted to the two known immutable packed
    implementations, never arbitrary PackedVisibility subclasses. Only the
    diffuse stream may be a LazyDiffVisibility pair: the fused kernels apply
    the leaf subtraction on that stream alone, so a lazy direct channel must
    fall back to the retained route.
    """
    from ..geometry.visibility import PackedVisibility, LazyDiffVisibility
    from ..geometry.visibility_native import MappedVisibility
    def admitted(leaf):
        return type(leaf) is PackedVisibility or isinstance(leaf,MappedVisibility)
    if isinstance(channel,LazyDiffVisibility):
        leaves=(channel.shadow,channel.vegetation)
        return leaves if allow_lazy and all(admitted(leaf) for leaf in leaves) else None
    return (channel,) if admitted(channel) else None


def _fused_guard(leaves,start,stop,patches):
    """Hold mapped-owner locks and restate decode_block's range contract."""
    from contextlib import ExitStack
    stack=ExitStack()
    try:
        owners=sorted({id(leaf):leaf for leaf in leaves if hasattr(leaf,'_lock')}.values(),key=id)
        for owner in owners:
            stack.enter_context(owner._lock)
            owner._check_open()
        for leaf in leaves:
            if not 0 <= start <= stop <= leaf.shape[0]*leaf.shape[1] or not 0 <= patches <= leaf.shape[2]:
                raise IndexError('Visibility block interval out of range')
    except BaseException:
        stack.close()
        raise
    return stack


def _shortwave_fused_block(shadow,vegetation,vegetation_building,diffuse,start,stop,patches,sun,shade,lum,solid,cosine,directions,diff_gate,ref_gate,box_gate,surface_sun,surface_sh,box,parallel):
    """Fused ordered decode+reduce for one block; None selects the retained route."""
    if not _fused_enabled():
        return None
    from ..geometry.visibility_compiled import _fused_descriptor, _preflight_flat
    leaves=[_packed_leaves(channel,allow_lazy=index==3)
            for index,channel in enumerate((shadow,vegetation,vegetation_building,diffuse))]
    if any(pair is None for pair in leaves):
        return None
    flat=[leaf for pair in leaves for leaf in pair]
    with _fused_guard(flat,start,stop,patches):
        sh_desc=_fused_descriptor(leaves[0][0])
        vs_desc=_fused_descriptor(leaves[1][0])
        vb_desc=_fused_descriptor(leaves[2][0])
        dsh_desc=_fused_descriptor(leaves[3][0])
        lazy=len(leaves[3])==2
        dveg_desc=_fused_descriptor(leaves[3][1]) if lazy else dsh_desc
        # A shared cached descriptor is the same payload stream: its decode
        # equals the base channel's, and it has already been preflighted.
        dsh_shared=dsh_desc is sh_desc
        dveg_shared=lazy and dveg_desc is vs_desc
        # Preflight every distinct decode stream in the original read order.
        _preflight_flat(*sh_desc,start,stop,patches)
        _preflight_flat(*vs_desc,start,stop,patches)
        _preflight_flat(*vb_desc,start,stop,patches)
        if not dsh_shared:
            _preflight_flat(*dsh_desc,start,stop,patches)
        if lazy and not dveg_shared:
            _preflight_flat(*dveg_desc,start,stop,patches)
        kernel=_shortwave_fused if parallel else _shortwave_fused_serial
        return kernel(*sh_desc,*vs_desc,*vb_desc,*dsh_desc,*dveg_desc,
                      lazy,dsh_shared,dveg_shared,start,stop,sun,shade,
                      lum,solid,cosine,directions,diff_gate,ref_gate,box_gate,
                      surface_sun,surface_sh,box)


def _longwave_fused_block(shadow,vegetation,vegetation_building,start,stop,patches,sun,shade,solid,sine,cosine,directions,gate,solar_gate,sky_down,sky_side,surface_sun,surface_sh,lup,reflection_factor,parallel):
    """Fused ordered decode+reduce for one block; None selects the retained route."""
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
        kernel=_longwave_fused if parallel else _longwave_fused_serial
        return kernel(*sh_desc,*vs_desc,*vb_desc,start,stop,sun,shade,solid,sine,cosine,
                      directions,gate,solar_gate,sky_down,sky_side,surface_sun,surface_sh,
                      lup,reflection_factor)


def _supported(values,cubes):
    return all(np.asarray(value).dtype==np.float32 for value in values) and all(getattr(value,'dtype',None)==np.dtype(np.float32) for value in cubes)


def _class_coefficient(azimuth,altitude,patch_azimuth):
    """One patch's scalar coefficient chain, verbatim from the retained loop.

    Kept as its own function so the exact-table route can evaluate it once
    per equal source state without changing any operation, operand order,
    intermediate dtype, or the separate coefficient cast.
    """
    from . import engine as e
    difference=np.abs(e._operate(np.subtract,azimuth,patch_azimuth))
    deg2rad=e._divide(np.pi,180.0)
    xi=np.cos(e._operate(np.multiply,difference,deg2rad))
    if not isinstance(altitude,np.ndarray):
        raise TypeError('tan(): solar_altitude must be a tensor-origin array')
    yi=e._operate(np.multiply,e._operate(np.multiply,2,xi),np.tan(e._operate(np.multiply,altitude,deg2rad)))
    return np.where(yi>0,0.0,yi)


def _class_coefficients(altitude,azimuth,geometry,asvf,active=None):
    """Evaluate scalar operations in patch order before broadcasting their results.

    A patch-axis array would change the wrapped-scalar promotion rules. Keep
    each scalar expression identical to the engine, then cast its result as
    the engine does when adding it to the raster. This is per-call state.
    """
    from . import engine as e
    field=np.asarray(asvf)
    if (field.dtype != np.float32 or geometry.altitude.dtype != np.float32
            or geometry.azimuth.dtype != np.float32 or np.ndim(altitude) != 0
            or np.ndim(azimuth) != 0):
        return None
    indices=np.flatnonzero(np.ones(geometry.altitude.size,dtype=bool) if active is None else active)
    coefficients=np.empty(indices.size,dtype=field.dtype)
    if _classes_exact_enabled():
        # Exact finite-state table (R04+G06): within one call the solar
        # scalars are fixed, so the coefficient bits are a pure function of
        # the patch azimuth source state. Patches with an equal full
        # source-state tuple (exact float32 bit pattern: signed zeros and NaN
        # payloads stay distinct) share one verbatim evaluation of the chain
        # and the stored float32 coefficient is scattered to the class
        # members. No reassociation: the original intermediate dtypes and the
        # separate coefficient cast run exactly as the retained loop does.
        states=np.ascontiguousarray(geometry.azimuth).view(np.uint32)
        evaluated={}
        for column,patch in enumerate(indices):
            key=int(states[patch])
            if key in evaluated:
                coefficients[column]=evaluated[key]
            else:
                coefficients[column]=_class_coefficient(azimuth,altitude,geometry.azimuth[patch])
                evaluated[key]=coefficients[column]
        return indices,coefficients,e._divide(180.0,np.pi)
    for column,patch in enumerate(indices):
        coefficients[column]=_class_coefficient(azimuth,altitude,geometry.azimuth[patch])
    return indices,coefficients,e._divide(180.0,np.pi)


def _classes(altitude,azimuth,geometry,asvf,start,stop,active=None,prepared=None):
    from . import engine as e
    field=np.asarray(asvf).reshape(-1)[start:stop].reshape(-1,1)
    sun=np.zeros((stop-start,geometry.altitude.size),dtype=np.bool_)
    shade=np.zeros_like(sun)
    if prepared is None:
        prepared=_class_coefficients(altitude,azimuth,geometry,asvf,active)
    if prepared is not None:
        indices,coefficients,rad2deg=prepared
        if indices.size:
            if _classes_exact_enabled():
                # Exact finite-state tables (R04): the retained tan32 field
                # pass feeds one compiled pass whose per-element sequence —
                # float32 add, the SLEEF scalar arctangent atan_array loops
                # over, float32 multiply, the two strict comparisons — is the
                # retained route's elementwise arithmetic without the array
                # temporaries and dispatch overhead. Typed arithmetic only.
                from ._math_profile import tan32
                heights=np.ascontiguousarray(tan32(field)).reshape(-1)
                if indices[0]==0 and indices.size==indices[-1]+1:
                    _classes_table(heights,coefficients,
                                   np.asarray(rad2deg,dtype=field.dtype)[()],
                                   geometry.altitude[indices],sun,shade)
                else:
                    _classes_table_masked(heights,coefficients,
                                          np.asarray(rad2deg,dtype=field.dtype)[()],
                                          geometry.altitude[indices],indices,sun,shade)
            else:
                # Both raster operations remain float32 and retain their grouping.
                from ._math_profile import tan32, atan32
                delta=np.add(tan32(field),coefficients[None,:])
                degrees=np.multiply(atan32(delta),np.asarray(rad2deg,dtype=field.dtype))
                sun[:,indices]=degrees<geometry.altitude[indices]
                shade[:,indices]=degrees>geometry.altitude[indices]
        return sun,shade
    for patch in range(geometry.altitude.size):
        if active is not None and not active[patch]:
            continue
        a,b=e.shaded_or_sunlit(altitude,azimuth,geometry.altitude[patch],geometry.azimuth[patch],field)
        sun[:,patch],shade[:,patch]=a[:,0],b[:,0]
    return sun,shade


@njit(cache=True, fastmath=False, error_model='numpy')
def _classes_table(heights,coefficients,radians_to_degrees,altitudes,sun,shade):
    """Exact finite-state classification of one block in a single compiled pass.

    Contiguous-column form (the whole-vault state set): per element this
    reproduces the retained route's elementwise NumPy operations exactly —
    the float32 add of the ``tan32`` field value and the per-state
    coefficient, the source-derived SLEEF scalar arctangent (the same
    ``atan_fma`` core ``atan_array`` loops over), the float32 multiply by
    the degree factor, and the two strict comparisons against the patch
    altitude. No reassociation, no fused multiply-add, serial over rows so
    the route adds no thread interactions; only the float32/bool temporaries
    and per-op dispatch of the retained route are removed.
    """
    for row in range(heights.shape[0]):
        value=heights[row]
        for column in range(coefficients.shape[0]):
            delta=np.float32(value+coefficients[column])
            degrees=np.float32(atan_fma(delta)*radians_to_degrees)
            sun[row,column]=degrees<altitudes[column]
            shade[row,column]=degrees>altitudes[column]


@njit(cache=True, fastmath=False, error_model='numpy')
def _classes_table_masked(heights,coefficients,radians_to_degrees,altitudes,indices,sun,shade):
    """Masked-column form of :func:`_classes_table` for solar-gate subsets.

    Identical per-element arithmetic; only the destination columns are
    indirections through the active-patch indices, as the retained route's
    ``sun[:,indices]`` assignment writes them.
    """
    for row in range(heights.shape[0]):
        value=heights[row]
        for column in range(coefficients.shape[0]):
            delta=np.float32(value+coefficients[column])
            degrees=np.float32(atan_fma(delta)*radians_to_degrees)
            sun[row,indices[column]]=degrees<altitudes[column]
            shade[row,indices[column]]=degrees>altitudes[column]


@njit(cache=True, fastmath=False, parallel=True)
def _shortwave(sh,vs,vb,diff,sun,shade,lum,solid,cosine,directions,diff_gate,ref_gate,box_gate,surface_sun,surface_sh,box):
    pixels,patches=sh.shape
    # diffuse and sunlit/shaded/vegetation reflections each have own accumulators.
    out=np.zeros((pixels,20),dtype=np.float32)
    for pixel in prange(pixels):
        for patch in range(patches):
            veg=vs[pixel,patch]==0 or vb[pixel,patch]==0
            building=np.float32(np.float32(1)-sh[pixel,patch])*vb[pixel,patch]==1
            contribution=np.float32(np.float32(np.float32(diff[pixel,patch]*lum[patch])*cosine[patch])*solid[patch])
            out[pixel,0]=np.float32(out[pixel,0]+contribution)
            v=((surface_sh*veg)*solid[patch])*cosine[patch]
            out[pixel,3]=np.float32(out[pixel,3]+v)
            if not box:
                a=((((surface_sun*sun[pixel,patch])*building)*solid[patch])*cosine[patch])
                b=((((surface_sh*shade[pixel,patch])*building)*solid[patch])*cosine[patch])
                out[pixel,1]=np.float32(out[pixel,1]+a)
                out[pixel,2]=np.float32(out[pixel,2]+b)
            else:
                if box_gate[patch]:
                    a=((((surface_sun*sun[pixel,patch])*building)*solid[patch])*cosine[patch])
                    b=((((surface_sh*shade[pixel,patch])*building)*solid[patch])*cosine[patch])
                else:
                    a=np.float32(0)
                    b=(((surface_sh*building)*solid[patch])*cosine[patch])
                out[pixel,1]=np.float32(out[pixel,1]+a)
                out[pixel,2]=np.float32(out[pixel,2]+b)
                for direction in range(4):
                    if diff_gate[patch,direction]:
                        inc=np.float32(cosine[patch]*directions[patch,direction])
                        d=np.float32(np.float32(np.float32(diff[pixel,patch]*lum[patch])*inc)*solid[patch])
                        out[pixel,4+direction]=np.float32(out[pixel,4+direction]+d)
                    if ref_gate[patch,direction]:
                        v=((((surface_sh*solid[patch])*cosine[patch])*veg)*directions[patch,direction])
                        out[pixel,12+direction]=np.float32(out[pixel,12+direction]+v)
                        if box_gate[patch]:
                            a=(((((surface_sun*sun[pixel,patch])*solid[patch])*cosine[patch])*building)*directions[patch,direction])
                            b=(((((surface_sh*shade[pixel,patch])*solid[patch])*cosine[patch])*building)*directions[patch,direction])
                        else:
                            a=np.float32(0)
                            b=((((surface_sh*solid[patch])*cosine[patch])*building)*directions[patch,direction])
                        out[pixel,8+direction]=np.float32(out[pixel,8+direction]+a)
                        # Shaded directional accumulation lives in separate final columns.
                        out[pixel,16+direction]=np.float32(out[pixel,16+direction]+b)
    return out


@njit(cache=True, fastmath=False)
def _shortwave_serial(sh,vs,vb,diff,sun,shade,lum,solid,cosine,directions,diff_gate,ref_gate,box_gate,surface_sun,surface_sh,box):
    pixels,patches=sh.shape
    # diffuse and sunlit/shaded/vegetation reflections each have own accumulators.
    out=np.zeros((pixels,20),dtype=np.float32)
    for pixel in range(pixels):
        for patch in range(patches):
            veg=vs[pixel,patch]==0 or vb[pixel,patch]==0
            building=np.float32(np.float32(1)-sh[pixel,patch])*vb[pixel,patch]==1
            contribution=np.float32(np.float32(np.float32(diff[pixel,patch]*lum[patch])*cosine[patch])*solid[patch])
            out[pixel,0]=np.float32(out[pixel,0]+contribution)
            v=((surface_sh*veg)*solid[patch])*cosine[patch]
            out[pixel,3]=np.float32(out[pixel,3]+v)
            if not box:
                a=((((surface_sun*sun[pixel,patch])*building)*solid[patch])*cosine[patch])
                b=((((surface_sh*shade[pixel,patch])*building)*solid[patch])*cosine[patch])
                out[pixel,1]=np.float32(out[pixel,1]+a)
                out[pixel,2]=np.float32(out[pixel,2]+b)
            else:
                if box_gate[patch]:
                    a=((((surface_sun*sun[pixel,patch])*building)*solid[patch])*cosine[patch])
                    b=((((surface_sh*shade[pixel,patch])*building)*solid[patch])*cosine[patch])
                else:
                    a=np.float32(0)
                    b=(((surface_sh*building)*solid[patch])*cosine[patch])
                out[pixel,1]=np.float32(out[pixel,1]+a)
                out[pixel,2]=np.float32(out[pixel,2]+b)
                for direction in range(4):
                    if diff_gate[patch,direction]:
                        inc=np.float32(cosine[patch]*directions[patch,direction])
                        d=np.float32(np.float32(np.float32(diff[pixel,patch]*lum[patch])*inc)*solid[patch])
                        out[pixel,4+direction]=np.float32(out[pixel,4+direction]+d)
                    if ref_gate[patch,direction]:
                        v=((((surface_sh*solid[patch])*cosine[patch])*veg)*directions[patch,direction])
                        out[pixel,12+direction]=np.float32(out[pixel,12+direction]+v)
                        if box_gate[patch]:
                            a=(((((surface_sun*sun[pixel,patch])*solid[patch])*cosine[patch])*building)*directions[patch,direction])
                            b=(((((surface_sh*shade[pixel,patch])*solid[patch])*cosine[patch])*building)*directions[patch,direction])
                        else:
                            a=np.float32(0)
                            b=((((surface_sh*solid[patch])*cosine[patch])*building)*directions[patch,direction])
                        out[pixel,8+direction]=np.float32(out[pixel,8+direction]+a)
                        # Shaded directional accumulation lives in separate final columns.
                        out[pixel,16+direction]=np.float32(out[pixel,16+direction]+b)
    return out



@njit(cache=True, fastmath=False, parallel=True)
def _shortwave_fused(sh_flat,sh_off,sh_modes,vs_flat,vs_off,vs_modes,vb_flat,vb_off,vb_modes,dsh_flat,dsh_off,dsh_modes,dveg_flat,dveg_off,dveg_modes,diff_is_lazy,dsh_shared,dveg_shared,start,stop,sun,shade,lum,solid,cosine,directions,diff_gate,ref_gate,box_gate,surface_sun,surface_sh,box):
    rows=stop-start
    patches=sh_modes.shape[0]
    out=np.zeros((rows,20),dtype=np.float32)
    for row in prange(rows):
        pixel=start+row
        for patch in range(patches):
            sh_v=_decode_at(sh_flat,sh_off,sh_modes,patch,pixel)
            vs_v=_decode_at(vs_flat,vs_off,vs_modes,patch,pixel)
            vb_v=_decode_at(vb_flat,vb_off,vb_modes,patch,pixel)
            if diff_is_lazy:
                # The lazy diffuse read decodes both leaves, then applies _diff;
                # leaves shared with the base channels reuse their decode.
                d_v=sh_v if dsh_shared else _decode_at(dsh_flat,dsh_off,dsh_modes,patch,pixel)
                t_v=vs_v if dveg_shared else _decode_at(dveg_flat,dveg_off,dveg_modes,patch,pixel)
                difference=np.float32(np.float32(1)-t_v)
                product=np.float32(difference*np.float32(1-.03))
                d_v=np.float32(d_v-product)
            else:
                d_v=sh_v if dsh_shared else _decode_at(dsh_flat,dsh_off,dsh_modes,patch,pixel)
            veg=vs_v==0 or vb_v==0
            building=np.float32(np.float32(1)-sh_v)*vb_v==1
            contribution=np.float32(np.float32(np.float32(d_v*lum[patch])*cosine[patch])*solid[patch])
            out[row,0]=np.float32(out[row,0]+contribution)
            v=((surface_sh*veg)*solid[patch])*cosine[patch]
            out[row,3]=np.float32(out[row,3]+v)
            if not box:
                a=((((surface_sun*sun[row,patch])*building)*solid[patch])*cosine[patch])
                b=((((surface_sh*shade[row,patch])*building)*solid[patch])*cosine[patch])
                out[row,1]=np.float32(out[row,1]+a)
                out[row,2]=np.float32(out[row,2]+b)
            else:
                if box_gate[patch]:
                    a=((((surface_sun*sun[row,patch])*building)*solid[patch])*cosine[patch])
                    b=((((surface_sh*shade[row,patch])*building)*solid[patch])*cosine[patch])
                else:
                    a=np.float32(0)
                    b=(((surface_sh*building)*solid[patch])*cosine[patch])
                out[row,1]=np.float32(out[row,1]+a)
                out[row,2]=np.float32(out[row,2]+b)
                for direction in range(4):
                    if diff_gate[patch,direction]:
                        inc=np.float32(cosine[patch]*directions[patch,direction])
                        d=np.float32(np.float32(np.float32(d_v*lum[patch])*inc)*solid[patch])
                        out[row,4+direction]=np.float32(out[row,4+direction]+d)
                    if ref_gate[patch,direction]:
                        v=((((surface_sh*solid[patch])*cosine[patch])*veg)*directions[patch,direction])
                        out[row,12+direction]=np.float32(out[row,12+direction]+v)
                        if box_gate[patch]:
                            a=(((((surface_sun*sun[row,patch])*solid[patch])*cosine[patch])*building)*directions[patch,direction])
                            b=(((((surface_sh*shade[row,patch])*solid[patch])*cosine[patch])*building)*directions[patch,direction])
                        else:
                            a=np.float32(0)
                            b=((((surface_sh*solid[patch])*cosine[patch])*building)*directions[patch,direction])
                        out[row,8+direction]=np.float32(out[row,8+direction]+a)
                        # Shaded directional accumulation lives in separate final columns.
                        out[row,16+direction]=np.float32(out[row,16+direction]+b)
    return out


@njit(cache=True, fastmath=False)
def _shortwave_fused_serial(sh_flat,sh_off,sh_modes,vs_flat,vs_off,vs_modes,vb_flat,vb_off,vb_modes,dsh_flat,dsh_off,dsh_modes,dveg_flat,dveg_off,dveg_modes,diff_is_lazy,dsh_shared,dveg_shared,start,stop,sun,shade,lum,solid,cosine,directions,diff_gate,ref_gate,box_gate,surface_sun,surface_sh,box):
    rows=stop-start
    patches=sh_modes.shape[0]
    out=np.zeros((rows,20),dtype=np.float32)
    for row in range(rows):
        pixel=start+row
        for patch in range(patches):
            sh_v=_decode_at(sh_flat,sh_off,sh_modes,patch,pixel)
            vs_v=_decode_at(vs_flat,vs_off,vs_modes,patch,pixel)
            vb_v=_decode_at(vb_flat,vb_off,vb_modes,patch,pixel)
            if diff_is_lazy:
                # The lazy diffuse read decodes both leaves, then applies _diff;
                # leaves shared with the base channels reuse their decode.
                d_v=sh_v if dsh_shared else _decode_at(dsh_flat,dsh_off,dsh_modes,patch,pixel)
                t_v=vs_v if dveg_shared else _decode_at(dveg_flat,dveg_off,dveg_modes,patch,pixel)
                difference=np.float32(np.float32(1)-t_v)
                product=np.float32(difference*np.float32(1-.03))
                d_v=np.float32(d_v-product)
            else:
                d_v=sh_v if dsh_shared else _decode_at(dsh_flat,dsh_off,dsh_modes,patch,pixel)
            veg=vs_v==0 or vb_v==0
            building=np.float32(np.float32(1)-sh_v)*vb_v==1
            contribution=np.float32(np.float32(np.float32(d_v*lum[patch])*cosine[patch])*solid[patch])
            out[row,0]=np.float32(out[row,0]+contribution)
            v=((surface_sh*veg)*solid[patch])*cosine[patch]
            out[row,3]=np.float32(out[row,3]+v)
            if not box:
                a=((((surface_sun*sun[row,patch])*building)*solid[patch])*cosine[patch])
                b=((((surface_sh*shade[row,patch])*building)*solid[patch])*cosine[patch])
                out[row,1]=np.float32(out[row,1]+a)
                out[row,2]=np.float32(out[row,2]+b)
            else:
                if box_gate[patch]:
                    a=((((surface_sun*sun[row,patch])*building)*solid[patch])*cosine[patch])
                    b=((((surface_sh*shade[row,patch])*building)*solid[patch])*cosine[patch])
                else:
                    a=np.float32(0)
                    b=(((surface_sh*building)*solid[patch])*cosine[patch])
                out[row,1]=np.float32(out[row,1]+a)
                out[row,2]=np.float32(out[row,2]+b)
                for direction in range(4):
                    if diff_gate[patch,direction]:
                        inc=np.float32(cosine[patch]*directions[patch,direction])
                        d=np.float32(np.float32(np.float32(d_v*lum[patch])*inc)*solid[patch])
                        out[row,4+direction]=np.float32(out[row,4+direction]+d)
                    if ref_gate[patch,direction]:
                        v=((((surface_sh*solid[patch])*cosine[patch])*veg)*directions[patch,direction])
                        out[row,12+direction]=np.float32(out[row,12+direction]+v)
                        if box_gate[patch]:
                            a=(((((surface_sun*sun[row,patch])*solid[patch])*cosine[patch])*building)*directions[patch,direction])
                            b=(((((surface_sh*shade[row,patch])*solid[patch])*cosine[patch])*building)*directions[patch,direction])
                        else:
                            a=np.float32(0)
                            b=((((surface_sh*solid[patch])*cosine[patch])*building)*directions[patch,direction])
                        out[row,8+direction]=np.float32(out[row,8+direction]+a)
                        # Shaded directional accumulation lives in separate final columns.
                        out[row,16+direction]=np.float32(out[row,16+direction]+b)
    return out

def Kside_veg_v2022a(*args,block_pixels=128,parallel=True,**kwargs):
    from . import engine as e
    reference=_reference('Kside_veg_v2022a')
    values=inspect.signature(reference).bind(*args,**kwargs).arguments
    if np.ndim(values['cyl'])==0 and values['cyl']!=1 and not isinstance(values['azimuth'],np.ndarray) and not isinstance(values['t'],np.ndarray):
        return reference(*args,**kwargs)
    if any(np.asarray(values[key]).ndim != 0 for key in ('radI','radD','radG','altitude','azimuth','psi','t','albedo','cyl','anisotropic_diffuse')) or values['anisotropic_diffuse']!=1 or not _supported([values[k] for k in ('shadow','asvf','KupE','KupS','KupW','KupN','lv')],[values[k] for k in ('diffsh','shmat','vegshmat','vbshvegshmat')]):
        return reference(*args,**kwargs)
    rows,cols=values['rows'],values['cols']
    pixels=rows*cols
    if not 0 < values['lv'].shape[0] <= 609:
        return reference(*args,**kwargs)
    geometry=patch_geometry(values['lv'])
    count=geometry.altitude.size
    radians=e._array(np.pi/180.)
    altitude=e._array(values['altitude'])
    azimuth=values['azimuth']
    t=values['t']
    box=values['cyl']!=1
    if not box and not _fused_enabled():
        # C6-22 narrow scratch specialization, admitted-profile only. The
        # entry re-verifies admission and returns None => fall through to the
        # untouched generic path below. Declined while the experimental fused
        # route is armed so SOLWEIG_LIGHT_FUSED_RAD=1 keeps base behavior.
        from .cylinder_shortwave import kside_cylinder_anisotropic as _narrow
        narrowed=_narrow(values,block_pixels=block_pixels,parallel=parallel)
        if narrowed is not None:
            return narrowed
    total=np.zeros(1,dtype=np.float32)
    luminance=values['lv'][:,2]
    for patch in range(count):
        total+=e._operate(np.multiply,e._operate(np.multiply,luminance[patch],geometry.solid_angle[patch]),geometry.sine[patch])
    lum=e._divide(e._operate(np.multiply,luminance,values['radD']),total)
    incident=np.cos(e._operate(np.multiply,altitude,radians))
    surface_sun=e._divide(e._operate(np.add,e._operate(np.multiply,values['albedo'],e._operate(np.multiply,values['radI'],incident)),e._operate(np.multiply,values['radD'],.5)),np.pi)[()]
    surface_sh=e._divide(e._operate(np.multiply,e._operate(np.multiply,values['albedo'],values['radD']),.5),np.pi)[()]
    directions=np.empty((count,4),dtype=np.float32)
    for patch in range(count):
        for direction,origin in enumerate((90,180,270,0)):
            directions[patch,direction]=np.cos(e._operate(np.multiply,e._operate(np.add,e._operate(np.subtract,origin,geometry.azimuth[patch]),t),radians))
    azi=geometry.azimuth
    diff_gate=geometry.diffuse_cardinal
    ref_gate=geometry.reflection_cardinal
    difference=np.asarray([np.abs(e._operate(np.subtract,azimuth,value)) for value in azi])
    box_gate=(difference>90)&(difference<270)
    output=np.zeros((7,pixels),dtype=np.float32)
    shadow=values['shadow']
    if not box:
        direct=e._operate(np.multiply,e._operate(np.multiply,shadow,values['radI']),incident)
        output[4]=direct.reshape(-1)
        for index,name in enumerate(('KupE','KupS','KupW','KupN')):
            output[index]=e._operate(np.multiply,values[name],.5).reshape(-1)
    else:
        angles=(azimuth+t,azimuth-90+t,azimuth-180+t,azimuth-270+t)
        gates=((azimuth>360-t)|(azimuth<=180-t),(azimuth>90-t)&(azimuth<=270-t),(azimuth>180-t)&(azimuth<=360-t),(azimuth<=90-t)|(azimuth>270-t))
        for index,angle in enumerate(angles):
            direct=e._operate(np.multiply,e._operate(np.multiply,e._operate(np.multiply,values['radI'],shadow),incident),np.sin(e._operate(np.multiply,angle,radians)))
            output[index]=np.where(gates[index],direct,0).reshape(-1)
    kernel=_shortwave if parallel else _shortwave_serial
    if block_pixels<1:raise ValueError('block_pixels must be positive')
    prepared=_class_coefficients(altitude,azimuth,geometry,values['asvf']) if pixels else None
    for start in range(0,pixels,block_pixels):
        stop=min(start+block_pixels,pixels)
        sun,shade=_classes(altitude,azimuth,geometry,values['asvf'],start,stop,prepared=prepared)
        reduced=_shortwave_fused_block(values['shmat'],values['vegshmat'],
                                       values['vbshvegshmat'],values['diffsh'],
                                       start,stop,count,sun,shade,lum,geometry.solid_angle,
                                       geometry.cosine,directions,diff_gate,ref_gate,box_gate,
                                       surface_sun,surface_sh,box,parallel)
        if reduced is None:
            sh,vs,vb,diff=_shortwave_visibility_blocks(values['shmat'],values['vegshmat'],
                                                       values['vbshvegshmat'],values['diffsh'],
                                                       start,stop,count)
            reduced=kernel(sh,vs,vb,diff,sun,shade,lum,geometry.solid_angle,geometry.cosine,directions,diff_gate,ref_gate,box_gate,surface_sun,surface_sh,box)
        if not box:
            output[5,start:stop]=reduced[:,0]
            combined=e._operate(np.add,e._operate(np.add,e._operate(np.add,e._operate(np.add,output[4,start:stop],reduced[:,0]),reduced[:,1]),reduced[:,2]),reduced[:,3])
            output[6,start:stop]=combined
        else:
            for direction,name in enumerate(('KupE','KupS','KupW','KupN')):
                result=output[direction,start:stop]
                for contribution in (reduced[:,4+direction],reduced[:,8+direction],reduced[:,16+direction],reduced[:,12+direction]):
                    result=e._operate(np.add,result,contribution)
                output[direction,start:stop]=e._operate(np.add,result,e._operate(np.multiply,values[name].reshape(-1)[start:stop],.5))
    return tuple(field.reshape(rows,cols) for field in output)


@njit(cache=True, fastmath=False, parallel=True)
def _longwave(sh,vs,vb,sun,shade,solid,sine,cosine,directions,gate,solar_gate,sky_down,sky_side,surface_sun,surface_sh,lup,reflection_factor):
    pixels,patches=sh.shape
    output=np.zeros((pixels,11),dtype=np.float32)
    for pixel in prange(pixels):
        accum=np.zeros(14,dtype=np.float32)
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
            for direction in range(4):
                if gate[patch,direction]:
                    sky_term=np.float32(np.float32(sky*sky_side[patch])*directions[patch,direction])
                    accum[10+direction]=np.float32(accum[10+direction]+sky_term)
                    vegetation_term=vegetation_side*directions[patch,direction]
                    accum[10+direction]=np.float32(accum[10+direction]+vegetation_term)
            if solar_gate[patch]:
                sun_side=((((surface_sun*sun[pixel,patch])*solid[patch])*cosine[patch])*building)
                shade_side=((((surface_sh*shade[pixel,patch])*solid[patch])*cosine[patch])*building)
                sun_down=((((surface_sun*sun[pixel,patch])*solid[patch])*sine[patch])*building)
                shade_down=((((surface_sh*shade[pixel,patch])*solid[patch])*sine[patch])*building)
                accum[8]=np.float32(accum[8]+sun_side)
                accum[7]=np.float32(accum[7]+shade_side)
                accum[3]=np.float32(accum[3]+sun_down)
                accum[2]=np.float32(accum[2]+shade_down)
                for direction in range(4):
                    if gate[patch,direction]:
                        accum[10+direction]=np.float32(accum[10+direction]+sun_side*directions[patch,direction])
                        accum[10+direction]=np.float32(accum[10+direction]+shade_side*directions[patch,direction])
            else:
                shade_side=(((surface_sh*solid[patch])*cosine[patch])*building)
                shade_down=(((surface_sh*solid[patch])*sine[patch])*building)
                accum[7]=np.float32(accum[7]+shade_side)
                accum[2]=np.float32(accum[2]+shade_down)
                for direction in range(4):
                    if gate[patch,direction]:
                        accum[10+direction]=np.float32(accum[10+direction]+shade_side*directions[patch,direction])
        # The reflection field depends on the completed ordered sky sweep.
        reflected=np.float32(np.float32(np.float32(np.float32(accum[0]+lup[pixel])*reflection_factor)*np.float32(.5))/np.float32(np.pi))
        for patch in range(patches):
            mask=sh[pixel,patch]==0 or vs[pixel,patch]==0 or vb[pixel,patch]==0
            side=np.float32(np.float32(np.float32(reflected*solid[patch])*cosine[patch])*mask)
            down=np.float32(np.float32(np.float32(reflected*solid[patch])*sine[patch])*mask)
            accum[9]=np.float32(accum[9]+side)
            accum[4]=np.float32(accum[4]+down)
            for direction in range(4):
                if gate[patch,direction]:
                    accum[10+direction]=np.float32(accum[10+direction]+np.float32(side*directions[patch,direction]))
        output[pixel,0]=np.float32(np.float32(np.float32(np.float32(accum[0]+accum[1])+accum[2])+accum[3])+accum[4])
        output[pixel,1]=np.float32(np.float32(np.float32(np.float32(accum[5]+accum[6])+accum[7])+accum[8])+accum[9])
        output[pixel,2:7]=accum[5:10]
        output[pixel,7]=accum[10]
        output[pixel,8]=accum[12]
        output[pixel,9]=accum[13]
        output[pixel,10]=accum[11]
    return output


@njit(cache=True, fastmath=False)
def _longwave_serial(sh,vs,vb,sun,shade,solid,sine,cosine,directions,gate,solar_gate,sky_down,sky_side,surface_sun,surface_sh,lup,reflection_factor):
    pixels,patches=sh.shape
    output=np.zeros((pixels,11),dtype=np.float32)
    for pixel in range(pixels):
        accum=np.zeros(14,dtype=np.float32)
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
            for direction in range(4):
                if gate[patch,direction]:
                    sky_term=np.float32(np.float32(sky*sky_side[patch])*directions[patch,direction])
                    accum[10+direction]=np.float32(accum[10+direction]+sky_term)
                    vegetation_term=vegetation_side*directions[patch,direction]
                    accum[10+direction]=np.float32(accum[10+direction]+vegetation_term)
            if solar_gate[patch]:
                sun_side=((((surface_sun*sun[pixel,patch])*solid[patch])*cosine[patch])*building)
                shade_side=((((surface_sh*shade[pixel,patch])*solid[patch])*cosine[patch])*building)
                sun_down=((((surface_sun*sun[pixel,patch])*solid[patch])*sine[patch])*building)
                shade_down=((((surface_sh*shade[pixel,patch])*solid[patch])*sine[patch])*building)
                accum[8]=np.float32(accum[8]+sun_side)
                accum[7]=np.float32(accum[7]+shade_side)
                accum[3]=np.float32(accum[3]+sun_down)
                accum[2]=np.float32(accum[2]+shade_down)
                for direction in range(4):
                    if gate[patch,direction]:
                        accum[10+direction]=np.float32(accum[10+direction]+sun_side*directions[patch,direction])
                        accum[10+direction]=np.float32(accum[10+direction]+shade_side*directions[patch,direction])
            else:
                shade_side=(((surface_sh*solid[patch])*cosine[patch])*building)
                shade_down=(((surface_sh*solid[patch])*sine[patch])*building)
                accum[7]=np.float32(accum[7]+shade_side)
                accum[2]=np.float32(accum[2]+shade_down)
                for direction in range(4):
                    if gate[patch,direction]:
                        accum[10+direction]=np.float32(accum[10+direction]+shade_side*directions[patch,direction])
        # The reflection field depends on the completed ordered sky sweep.
        reflected=np.float32(np.float32(np.float32(np.float32(accum[0]+lup[pixel])*reflection_factor)*np.float32(.5))/np.float32(np.pi))
        for patch in range(patches):
            mask=sh[pixel,patch]==0 or vs[pixel,patch]==0 or vb[pixel,patch]==0
            side=np.float32(np.float32(np.float32(reflected*solid[patch])*cosine[patch])*mask)
            down=np.float32(np.float32(np.float32(reflected*solid[patch])*sine[patch])*mask)
            accum[9]=np.float32(accum[9]+side)
            accum[4]=np.float32(accum[4]+down)
            for direction in range(4):
                if gate[patch,direction]:
                    accum[10+direction]=np.float32(accum[10+direction]+np.float32(side*directions[patch,direction]))
        output[pixel,0]=np.float32(np.float32(np.float32(np.float32(accum[0]+accum[1])+accum[2])+accum[3])+accum[4])
        output[pixel,1]=np.float32(np.float32(np.float32(np.float32(accum[5]+accum[6])+accum[7])+accum[8])+accum[9])
        output[pixel,2:7]=accum[5:10]
        output[pixel,7]=accum[10]
        output[pixel,8]=accum[12]
        output[pixel,9]=accum[13]
        output[pixel,10]=accum[11]
    return output



@njit(cache=True, fastmath=False, parallel=True)
def _longwave_fused(sh_flat,sh_off,sh_modes,vs_flat,vs_off,vs_modes,vb_flat,vb_off,vb_modes,start,stop,sun,shade,solid,sine,cosine,directions,gate,solar_gate,sky_down,sky_side,surface_sun,surface_sh,lup,reflection_factor):
    rows=stop-start
    patches=sh_modes.shape[0]
    output=np.zeros((rows,11),dtype=np.float32)
    # The reflected sweep re-reads only the occlusion predicate, captured here
    # during the sky sweep so no payload is decoded twice.
    masks=np.empty((rows,patches),dtype=np.bool_)
    for row in prange(rows):
        pixel=start+row
        accum=np.zeros(14,dtype=np.float32)
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
            for direction in range(4):
                if gate[patch,direction]:
                    sky_term=np.float32(np.float32(sky*sky_side[patch])*directions[patch,direction])
                    accum[10+direction]=np.float32(accum[10+direction]+sky_term)
                    vegetation_term=vegetation_side*directions[patch,direction]
                    accum[10+direction]=np.float32(accum[10+direction]+vegetation_term)
            if solar_gate[patch]:
                sun_side=((((surface_sun*sun[row,patch])*solid[patch])*cosine[patch])*building)
                shade_side=((((surface_sh*shade[row,patch])*solid[patch])*cosine[patch])*building)
                sun_down=((((surface_sun*sun[row,patch])*solid[patch])*sine[patch])*building)
                shade_down=((((surface_sh*shade[row,patch])*solid[patch])*sine[patch])*building)
                accum[8]=np.float32(accum[8]+sun_side)
                accum[7]=np.float32(accum[7]+shade_side)
                accum[3]=np.float32(accum[3]+sun_down)
                accum[2]=np.float32(accum[2]+shade_down)
                for direction in range(4):
                    if gate[patch,direction]:
                        accum[10+direction]=np.float32(accum[10+direction]+sun_side*directions[patch,direction])
                        accum[10+direction]=np.float32(accum[10+direction]+shade_side*directions[patch,direction])
            else:
                shade_side=(((surface_sh*solid[patch])*cosine[patch])*building)
                shade_down=(((surface_sh*solid[patch])*sine[patch])*building)
                accum[7]=np.float32(accum[7]+shade_side)
                accum[2]=np.float32(accum[2]+shade_down)
                for direction in range(4):
                    if gate[patch,direction]:
                        accum[10+direction]=np.float32(accum[10+direction]+shade_side*directions[patch,direction])
        # The reflection field depends on the completed ordered sky sweep.
        reflected=np.float32(np.float32(np.float32(np.float32(accum[0]+lup[row])*reflection_factor)*np.float32(.5))/np.float32(np.pi))
        for patch in range(patches):
            mask=masks[row,patch]
            side=np.float32(np.float32(np.float32(reflected*solid[patch])*cosine[patch])*mask)
            down=np.float32(np.float32(np.float32(reflected*solid[patch])*sine[patch])*mask)
            accum[9]=np.float32(accum[9]+side)
            accum[4]=np.float32(accum[4]+down)
            for direction in range(4):
                if gate[patch,direction]:
                    accum[10+direction]=np.float32(accum[10+direction]+np.float32(side*directions[patch,direction]))
        output[row,0]=np.float32(np.float32(np.float32(np.float32(accum[0]+accum[1])+accum[2])+accum[3])+accum[4])
        output[row,1]=np.float32(np.float32(np.float32(np.float32(accum[5]+accum[6])+accum[7])+accum[8])+accum[9])
        output[row,2:7]=accum[5:10]
        output[row,7]=accum[10]
        output[row,8]=accum[12]
        output[row,9]=accum[13]
        output[row,10]=accum[11]
    return output


@njit(cache=True, fastmath=False)
def _longwave_fused_serial(sh_flat,sh_off,sh_modes,vs_flat,vs_off,vs_modes,vb_flat,vb_off,vb_modes,start,stop,sun,shade,solid,sine,cosine,directions,gate,solar_gate,sky_down,sky_side,surface_sun,surface_sh,lup,reflection_factor):
    rows=stop-start
    patches=sh_modes.shape[0]
    output=np.zeros((rows,11),dtype=np.float32)
    # The reflected sweep re-reads only the occlusion predicate, captured here
    # during the sky sweep so no payload is decoded twice.
    masks=np.empty((rows,patches),dtype=np.bool_)
    for row in range(rows):
        pixel=start+row
        accum=np.zeros(14,dtype=np.float32)
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
            for direction in range(4):
                if gate[patch,direction]:
                    sky_term=np.float32(np.float32(sky*sky_side[patch])*directions[patch,direction])
                    accum[10+direction]=np.float32(accum[10+direction]+sky_term)
                    vegetation_term=vegetation_side*directions[patch,direction]
                    accum[10+direction]=np.float32(accum[10+direction]+vegetation_term)
            if solar_gate[patch]:
                sun_side=((((surface_sun*sun[row,patch])*solid[patch])*cosine[patch])*building)
                shade_side=((((surface_sh*shade[row,patch])*solid[patch])*cosine[patch])*building)
                sun_down=((((surface_sun*sun[row,patch])*solid[patch])*sine[patch])*building)
                shade_down=((((surface_sh*shade[row,patch])*solid[patch])*sine[patch])*building)
                accum[8]=np.float32(accum[8]+sun_side)
                accum[7]=np.float32(accum[7]+shade_side)
                accum[3]=np.float32(accum[3]+sun_down)
                accum[2]=np.float32(accum[2]+shade_down)
                for direction in range(4):
                    if gate[patch,direction]:
                        accum[10+direction]=np.float32(accum[10+direction]+sun_side*directions[patch,direction])
                        accum[10+direction]=np.float32(accum[10+direction]+shade_side*directions[patch,direction])
            else:
                shade_side=(((surface_sh*solid[patch])*cosine[patch])*building)
                shade_down=(((surface_sh*solid[patch])*sine[patch])*building)
                accum[7]=np.float32(accum[7]+shade_side)
                accum[2]=np.float32(accum[2]+shade_down)
                for direction in range(4):
                    if gate[patch,direction]:
                        accum[10+direction]=np.float32(accum[10+direction]+shade_side*directions[patch,direction])
        # The reflection field depends on the completed ordered sky sweep.
        reflected=np.float32(np.float32(np.float32(np.float32(accum[0]+lup[row])*reflection_factor)*np.float32(.5))/np.float32(np.pi))
        for patch in range(patches):
            mask=masks[row,patch]
            side=np.float32(np.float32(np.float32(reflected*solid[patch])*cosine[patch])*mask)
            down=np.float32(np.float32(np.float32(reflected*solid[patch])*sine[patch])*mask)
            accum[9]=np.float32(accum[9]+side)
            accum[4]=np.float32(accum[4]+down)
            for direction in range(4):
                if gate[patch,direction]:
                    accum[10+direction]=np.float32(accum[10+direction]+np.float32(side*directions[patch,direction]))
        output[row,0]=np.float32(np.float32(np.float32(np.float32(accum[0]+accum[1])+accum[2])+accum[3])+accum[4])
        output[row,1]=np.float32(np.float32(np.float32(np.float32(accum[5]+accum[6])+accum[7])+accum[8])+accum[9])
        output[row,2:7]=accum[5:10]
        output[row,7]=accum[10]
        output[row,8]=accum[12]
        output[row,9]=accum[13]
        output[row,10]=accum[11]
    return output

def define_patch_characteristics(*args,block_pixels=128,parallel=True,**kwargs):
    from . import engine as e
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
    output=np.empty((11,rows*cols),dtype=np.float32)
    kernel=_longwave if parallel else _longwave_serial
    factor=e._operate(np.subtract,1,ewall)[()]
    if block_pixels<1:raise ValueError('block_pixels must be positive')
    prepared=_class_coefficients(values['solar_altitude'],values['solar_azimuth'],geometry,values['asvf'],solar_gate) if rows*cols else None
    for start in range(0,rows*cols,block_pixels):
        stop=min(start+block_pixels,rows*cols)
        sun,shade=_classes(values['solar_altitude'],values['solar_azimuth'],geometry,values['asvf'],start,stop,active=solar_gate,prepared=prepared)
        reduced=_longwave_fused_block(values['shmat'],values['vegshmat'],values['vbshvegshmat'],
                                      start,stop,count,sun,shade,values['steradian'],geometry.sine,
                                      geometry.cosine,directions,gate,solar_gate,
                                      values['Lsky_down'][:,2],values['Lsky_side'][:,2],
                                      sun_surface,shade_surface,values['Lup'].reshape(-1)[start:stop],factor,parallel)
        if reduced is None:
            from ..geometry.visibility_prepared import decode_longwave_block
            decoded=decode_longwave_block(values['shmat'],values['vegshmat'],
                                          values['vbshvegshmat'],start,stop,count)
            if decoded is None:
                sh,vs,vb=(_block(values[name],start,stop,count)
                          for name in ('shmat','vegshmat','vbshvegshmat'))
            else:
                sh,vs,vb=decoded
            reduced=kernel(sh,vs,vb,sun,shade,values['steradian'],geometry.sine,geometry.cosine,directions,gate,solar_gate,values['Lsky_down'][:,2],values['Lsky_side'][:,2],sun_surface,shade_surface,values['Lup'].reshape(-1)[start:stop],factor)
        output[:,start:stop]=reduced.T
    return tuple(field.reshape(rows,cols) for field in output)


def Lcyl_v2022a(*args,block_pixels=128,parallel=True,**kwargs):
    from . import engine as e
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
    result=define_patch_characteristics(v['solar_altitude'],v['solar_azimuth'],geometry.altitude,geometry.azimuth,geometry.solid_angle,v['asvf'],v['shmat'],v['vegshmat'],v['vbshvegshmat'],ldown,lside,lnormal,v['Lup'],v['Ta'],v['Tgwall'],v['ewall'],v['rows'],v['cols'],block_pixels=block_pixels,parallel=parallel)
    return tuple(result[index] for index in (0,1,7,8,9,10))


def _model2(geometry,esky):
    """Original model2 arithmetic with immutable band lookup supplied by cache."""
    from . import engine as e
    deg2rad=e._array(e._divide(np.pi,180))
    skyzen=e._operate(np.subtract,90,geometry.bands)
    b_c=.308
    esky_band=e._operate(np.subtract,1,e._operate(np.multiply,e._operate(np.subtract,1,esky),np.exp(e._operate(np.multiply,b_c,e._operate(np.subtract,1.7,e._divide(1,np.cos(e._operate(np.multiply,skyzen,deg2rad))))))))
    patch_emissivity=e._zeros(geometry.altitude.size)
    for index in range(geometry.bands.size):
        patch_emissivity[geometry.band_membership[index]]=esky_band[index:index+1]
    normalized=e._divide(patch_emissivity,np.sum(patch_emissivity))
    return normalized,esky_band
