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

import numpy as np
from numba import njit, prange


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


def _supported(values,cubes):
    return all(np.asarray(value).dtype==np.float32 for value in values) and all(getattr(value,'dtype',None)==np.dtype(np.float32) for value in cubes)


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
    for column,patch in enumerate(indices):
        difference=np.abs(e._operate(np.subtract,azimuth,geometry.azimuth[patch]))
        deg2rad=e._divide(np.pi,180.0)
        xi=np.cos(e._operate(np.multiply,difference,deg2rad))
        if not isinstance(altitude,np.ndarray):
            raise TypeError('tan(): solar_altitude must be a tensor-origin array')
        yi=e._operate(np.multiply,e._operate(np.multiply,2,xi),np.tan(e._operate(np.multiply,altitude,deg2rad)))
        coefficients[column]=np.where(yi>0,0.0,yi)
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
            # Both raster operations remain float32 and retain their grouping.
            delta=np.add(np.tan(field),coefficients[None,:])
            degrees=np.multiply(np.arctan(delta),np.asarray(rad2deg,dtype=field.dtype))
            sun[:,indices]=degrees<geometry.altitude[indices]
            shade[:,indices]=degrees>geometry.altitude[indices]
        return sun,shade
    for patch in range(geometry.altitude.size):
        if active is not None and not active[patch]:
            continue
        a,b=e.shaded_or_sunlit(altitude,azimuth,geometry.altitude[patch],geometry.azimuth[patch],field)
        sun[:,patch],shade[:,patch]=a[:,0],b[:,0]
    return sun,shade


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
        sh,vs,vb,diff=(_block(values[name],start,stop,count) for name in ('shmat','vegshmat','vbshvegshmat','diffsh'))
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
        sh,vs,vb=(_block(values[name],start,stop,count) for name in ('shmat','vegshmat','vbshvegshmat'))
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
