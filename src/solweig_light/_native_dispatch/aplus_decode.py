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
"""N9 F1D A-plus comparator: mode-specialized ``_decode_at`` for the row-A
fused decode path. CLEARLY SEPARATED from the A8 control: nothing dispatches
here; the shipped default path (``cylinder_longwave._longwave_fused_primary*``
via ``visibility_compiled._decode_at``) is byte-for-byte unchanged. This
module exists so the same constant-mode transformation the producer kernels
received (F1D) is applied to the strongest shipped comparator too -- a native
gain must not be credited to withholding shared improvements from Numba.

``_decode_at_plus`` preserves ``_decode_at``'s exact bits and typing: raw
mode keeps the original little-endian byte assembly, the binary/ternary arms
use literal shifts/masks that are exactly ``pixel//8, pixel%8`` and
``pixel//4, pixel%4`` for the non-negative pixels the range contract
guarantees, and the original generic expression is retained verbatim as the
fallback arm for any other mode value. Like ``_decode_at``, reserved codes
are NOT checked here (``_preflight_flat`` owns that contract and its
patch-major first-error order, unchanged). The fused kernels are the frozen
A8 graphs transcribed with ``_decode_at`` -> ``_decode_at_plus`` and NO other
change (char-identical arithmetic; ``inline='always'`` kept so the LLVM
inlining position matches).
"""
import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=False, inline='always')
def _decode_at_plus(flat, offsets, modes, patch, pixel):
    """``_decode_at`` with the mode branch specialized to literal ops."""
    mode = modes[patch]
    base = offsets[patch]
    if mode == 4:
        offset = base + pixel*4
        bits = (np.uint32(flat[offset]) | (np.uint32(flat[offset+1]) << 8)
                | (np.uint32(flat[offset+2]) << 16) | (np.uint32(flat[offset+3]) << 24))
        return np.uint32(bits).view(np.float32)
    if mode == 1:
        return np.float32((flat[base + (pixel >> 3)] >> (pixel & 7)) & 1)
    if mode == 2:
        return np.float32((flat[base + (pixel >> 2)] >> ((pixel & 3) << 1)) & 3)
    code = (flat[base + pixel // (8 // mode)]
            >> ((pixel % (8 // mode))*mode)) & ((1 << mode)-1)
    return np.float32(code)


@njit(cache=True, fastmath=False, parallel=True)
def _longwave_fused_primary_aplus(sh_flat,sh_off,sh_modes,vs_flat,vs_off,vs_modes,vb_flat,vb_off,vb_modes,start,stop,sun,shade,solid,sine,cosine,directions,gate,solar_gate,sky_down,sky_side,surface_sun,surface_sh,lup,reflection_factor):
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
            sh_v=_decode_at_plus(sh_flat,sh_off,sh_modes,patch,pixel)
            vs_v=_decode_at_plus(vs_flat,vs_off,vs_modes,patch,pixel)
            vb_v=_decode_at_plus(vb_flat,vb_off,vb_modes,patch,pixel)
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
def _longwave_fused_primary_aplus_serial(sh_flat,sh_off,sh_modes,vs_flat,vs_off,vs_modes,vb_flat,vb_off,vb_modes,start,stop,sun,shade,solid,sine,cosine,directions,gate,solar_gate,sky_down,sky_side,surface_sun,surface_sh,lup,reflection_factor):
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
            sh_v=_decode_at_plus(sh_flat,sh_off,sh_modes,patch,pixel)
            vs_v=_decode_at_plus(vs_flat,vs_off,vs_modes,patch,pixel)
            vb_v=_decode_at_plus(vb_flat,vb_off,vb_modes,patch,pixel)
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
