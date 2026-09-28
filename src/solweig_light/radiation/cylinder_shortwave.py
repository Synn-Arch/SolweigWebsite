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
"""Narrow cylinder shortwave scratch for the admitted pipeline demand profile.

R-C (dossiers/03): in cylinder mode (cyl==1) the seven demanded public results
of Kside_veg_v2022a consume only reduction columns 0..3 of the generic kernel
scratch; the generic kernel allocates 20 columns and its wrapper builds
box-only direction cosines and gates that no cylinder route reads.

Backward-slice verification at base 5e1fab46:
  - patch_radiation._shortwave writes columns 4..19 only in its ``box``
    branch; with box=False they stay at the np.zeros initialiser.
  - the wrapper's cylinder assembly reads reduced[:,0..3] only
    (KsideD, Kref_sun, Kref_sh, Kref_veg), and builds KsideI plus the four
    Kup/2 fields outside the kernel.
  - the serial engine reference returns the same seven fields from the same
    cylinder branch (engine.Kside_veg_v2022a, cyl==1 and anisotropic_diffuse==1).

Removed box-only work (directions cosine table, box_gate difference list,
gate pass-through, 20-column scratch) can raise or warn for exotic inputs,
so the private entry below admits only the verified finite-scalar cylinder
profile and returns None otherwise. None selects the untouched generic
wrapper path, preserving original failure order and warning behaviour.
Omitted diagnostics are not requested under this private profile; the public
full path is never altered here.
"""
import numpy as np
from numba import njit, prange

PIPELINE_CYLINDER_ANISOTROPIC='PIPELINE_CYLINDER_ANISOTROPIC'
FULL_DIAGNOSTICS='FULL_DIAGNOSTICS'

_demand_profile=FULL_DIAGNOSTICS


def set_demand_profile(profile):
    """Select the private radiation demand profile for pipeline runs."""
    global _demand_profile
    if profile not in (PIPELINE_CYLINDER_ANISOTROPIC,FULL_DIAGNOSTICS):
        raise ValueError('unknown radiation demand profile: %r' % (profile,))
    _demand_profile=profile


def demand_profile():
    """Return the currently selected private demand profile."""
    return _demand_profile


@njit(cache=True, fastmath=False, parallel=True)
def _shortwave_cylinder(sh,vs,vb,diff,sun,shade,lum,solid,cosine,surface_sun,surface_sh):
    pixels,patches=sh.shape
    # Cylinder scratch: exactly the four demanded reduction columns.
    out=np.zeros((pixels,4),dtype=np.float32)
    for pixel in prange(pixels):
        for patch in range(patches):
            veg=vs[pixel,patch]==0 or vb[pixel,patch]==0
            building=np.float32(np.float32(1)-sh[pixel,patch])*vb[pixel,patch]==1
            contribution=np.float32(np.float32(np.float32(diff[pixel,patch]*lum[patch])*cosine[patch])*solid[patch])
            out[pixel,0]=np.float32(out[pixel,0]+contribution)
            v=((surface_sh*veg)*solid[patch])*cosine[patch]
            out[pixel,3]=np.float32(out[pixel,3]+v)
            a=((((surface_sun*sun[pixel,patch])*building)*solid[patch])*cosine[patch])
            b=((((surface_sh*shade[pixel,patch])*building)*solid[patch])*cosine[patch])
            out[pixel,1]=np.float32(out[pixel,1]+a)
            out[pixel,2]=np.float32(out[pixel,2]+b)
    return out


@njit(cache=True, fastmath=False)
def _shortwave_cylinder_serial(sh,vs,vb,diff,sun,shade,lum,solid,cosine,surface_sun,surface_sh):
    pixels,patches=sh.shape
    # Cylinder scratch: exactly the four demanded reduction columns.
    out=np.zeros((pixels,4),dtype=np.float32)
    for pixel in range(pixels):
        for patch in range(patches):
            veg=vs[pixel,patch]==0 or vb[pixel,patch]==0
            building=np.float32(np.float32(1)-sh[pixel,patch])*vb[pixel,patch]==1
            contribution=np.float32(np.float32(np.float32(diff[pixel,patch]*lum[patch])*cosine[patch])*solid[patch])
            out[pixel,0]=np.float32(out[pixel,0]+contribution)
            v=((surface_sh*veg)*solid[patch])*cosine[patch]
            out[pixel,3]=np.float32(out[pixel,3]+v)
            a=((((surface_sun*sun[pixel,patch])*building)*solid[patch])*cosine[patch])
            b=((((surface_sh*shade[pixel,patch])*building)*solid[patch])*cosine[patch])
            out[pixel,1]=np.float32(out[pixel,1]+a)
            out[pixel,2]=np.float32(out[pixel,2]+b)
    return out


def _admitted(values,geometry):
    """Restate the wrapper guards and audit the removed box-only work.

    The removed cylinder-route evaluations (direction cosines, box gate) read
    only the solar azimuth, rotation t and patch azimuths. Nonfinite values
    there can raise or emit invalid-value warnings in the original generic
    path, so anything nonfinite falls back instead of being skipped silently.
    """
    if np.ndim(values['cyl'])!=0 or values['cyl']!=1 or values['anisotropic_diffuse']!=1:
        return False
    if not 0 < values['lv'].shape[0] <= 609:
        return False
    from .patch_radiation import _supported
    if any(np.asarray(values[key]).ndim != 0 for key in ('radI','radD','radG','altitude','azimuth','psi','t','albedo','cyl','anisotropic_diffuse')) or not _supported([values[k] for k in ('shadow','asvf','KupE','KupS','KupW','KupN','lv')],[values[k] for k in ('diffsh','shmat','vegshmat','vbshvegshmat')]):
        return False
    try:
        if not (np.isfinite(values['t']) and np.isfinite(values['azimuth'])):
            return False
        if not bool(np.isfinite(geometry.azimuth).all()):
            return False
    except Exception:
        # Unauditable exotic input: the generic wrapper keeps its own
        # original failure order on the full path.
        return False
    return True


def kside_cylinder_anisotropic(values,block_pixels=128,parallel=True):
    """Narrow cylinder branch of Kside_veg_v2022a, or None when not admitted.

    ``values`` is the wrapper's bound-argument mapping. Returns the same
    seven public fields (KupE/2..KupN/2, KsideI, KsideD, Kside) with
    bitwise-identical bits, or None to select the untouched generic path.
    """
    if _demand_profile!=PIPELINE_CYLINDER_ANISOTROPIC:
        return None
    from .patch_radiation import _class_coefficients,_classes,_shortwave_visibility_blocks,patch_geometry
    from . import engine as e
    try:
        geometry=patch_geometry(values['lv'])
    except Exception:
        return None
    if not _admitted(values,geometry):
        return None
    count=geometry.altitude.size
    radians=e._array(np.pi/180.)
    altitude=e._array(values['altitude'])
    azimuth=values['azimuth']
    t=values['t']
    rows,cols=values['rows'],values['cols']
    pixels=rows*cols
    total=np.zeros(1,dtype=np.float32)
    luminance=values['lv'][:,2]
    for patch in range(count):
        total+=e._operate(np.multiply,e._operate(np.multiply,luminance[patch],geometry.solid_angle[patch]),geometry.sine[patch])
    lum=e._divide(e._operate(np.multiply,luminance,values['radD']),total)
    incident=np.cos(e._operate(np.multiply,altitude,radians))
    surface_sun=e._divide(e._operate(np.add,e._operate(np.multiply,values['albedo'],e._operate(np.multiply,values['radI'],incident)),e._operate(np.multiply,values['radD'],.5)),np.pi)[()]
    surface_sh=e._divide(e._operate(np.multiply,e._operate(np.multiply,values['albedo'],values['radD']),.5),np.pi)[()]
    output=np.zeros((7,pixels),dtype=np.float32)
    shadow=values['shadow']
    # Cylinder public fields: direct term and the four Kup/2 fields.
    direct=e._operate(np.multiply,e._operate(np.multiply,shadow,values['radI']),incident)
    output[4]=direct.reshape(-1)
    for index,name in enumerate(('KupE','KupS','KupW','KupN')):
        output[index]=e._operate(np.multiply,values[name],.5).reshape(-1)
    if block_pixels<1:raise ValueError('block_pixels must be positive')
    prepared=_class_coefficients(altitude,azimuth,geometry,values['asvf']) if pixels else None
    kernel=_shortwave_cylinder if parallel else _shortwave_cylinder_serial
    for start in range(0,pixels,block_pixels):
        stop=min(start+block_pixels,pixels)
        sun,shade=_classes(altitude,azimuth,geometry,values['asvf'],start,stop,prepared=prepared)
        sh,vs,vb,diff=_shortwave_visibility_blocks(values['shmat'],values['vegshmat'],
                                                   values['vbshvegshmat'],values['diffsh'],
                                                   start,stop,count)
        reduced=kernel(sh,vs,vb,diff,sun,shade,lum,geometry.solid_angle,geometry.cosine,surface_sun,surface_sh)
        output[5,start:stop]=reduced[:,0]
        combined=e._operate(np.add,e._operate(np.add,e._operate(np.add,e._operate(np.add,output[4,start:stop],reduced[:,0]),reduced[:,1]),reduced[:,2]),reduced[:,3])
        output[6,start:stop]=combined
    return tuple(field.reshape(rows,cols) for field in output)
