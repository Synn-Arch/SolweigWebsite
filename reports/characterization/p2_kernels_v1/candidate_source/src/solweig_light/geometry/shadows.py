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
# CPU adaptation preserves the pinned upstream compatibility behavior.
"""Serial, step-major sky visibility recurrence; no shortened ray support."""
import numpy as np


def create_patches(patch_option):
    if patch_option in (1, 2, 3):
        ann = np.array([0,12,24,36,48,60,72,84,90], dtype=np.int64)
        alts = np.array([6,18,30,42,54,66,78,90], dtype=np.int64)
        starts = np.array([0,4,2,5,8,0,10,0], dtype=np.int64)
        counts = {1:[30,30,24,24,18,12,6,1],2:[31,30,28,24,19,13,7,1],3:[62,60,56,48,38,26,14,1]}[patch_option]
    elif patch_option == 4:
        ann = np.array([0,4.5,9,15,21,27,33,39,45,51,57,63,69,75,81,90],dtype=np.float32)
        alts = np.array([3,9,15,21,27,33,39,45,51,57,63,69,75,81,90],dtype=np.int64)
        starts = np.array([0,0,4,4,2,2,5,5,8,8,0,0,10,10,0],dtype=np.int64)
        counts = [62,62,60,60,56,56,48,48,38,38,26,26,14,14,1]
    else:
        raise UnboundLocalError("unsupported patch option: upstream has no patch table")
    counts = np.array(counts,dtype=np.int64)
    # Torch integer true division uses its default float32 dtype.
    steps = np.reciprocal(counts.astype(np.float32)) * np.float32(360)
    altitude = np.repeat(alts, counts).astype(np.float32)
    azi = np.concatenate([np.array([np.float32(k)*steps[j]+np.float32(starts[j]) for k in range(counts[j])],dtype=np.float32) for j in range(len(counts))])
    return altitude, azi, ann, alts, counts, steps, starts


def shadow(amaxvalue, a, vegdem, vegdem2, bush, azimuth, altitude, scale):
    a, vegdem, vegdem2, bush = map(np.asarray,(a,vegdem,vegdem2,bush))
    # Python scalar inputs become default float32 tensors upstream, while
    # supplied NumPy scalars retain dtype. Integer-angle arithmetic promotes
    # to default float32. Fixture callers restore the original argument types.
    def angle(x):
        if isinstance(x,(float,int)) or np.asarray(x).dtype.kind in 'iu':
            return np.float32(x)
        return np.asarray(x)[()]
    if azimuth == 0:
        azimuth = 1e-12
    azimuth = angle(azimuth) * (np.pi/180)
    altitude = angle(altitude) * (np.pi/180)
    sx, sy = a.shape
    zeros = lambda: np.zeros((sx,sy),dtype=np.float32)
    temp, tv, tv2, sh, vb, tb, g = [zeros() for _ in range(7)]
    f = a.copy()
    bp = (bush > 1).astype(np.float32)
    vs = bp.copy()
    sin,cos,tan = np.sin(azimuth),np.cos(azimuth),np.tan(azimuth)
    ss,sc = np.sign(sin),np.sign(cos)
    ds_s,ds_c = abs(1/sin),abs(1/cos)
    tas = np.tan(altitude)/scale
    dx = dy = dz = np.float32(0)
    index = 1
    while amaxvalue >= dz and abs(dx)<sx and abs(dy)<sy:
        if np.pi/4 <= azimuth < 3*np.pi/4 or 5*np.pi/4 <= azimuth < 7*np.pi/4:
            dy = ss*index
            dx = -sc*abs(np.round(index/tan))
            ds = ds_s
        else:
            dy = ss*abs(np.round(index*tan))
            dx = -sc*index
            ds = ds_c
        dz = ds*index*tas
        tv.fill(0);tv2.fill(0);temp.fill(0)
        ax,ay=abs(dx),abs(dy)
        xc1,xc2=int((dx+ax)/2),int(sx+(dx-ax)/2)
        yc1,yc2=int((dy+ay)/2),int(sy+(dy-ay)/2)
        xp1,xp2=int(-(dx-ax)/2),int(sx-(dx+ax)/2)
        yp1,yp2=int(-(dy-ay)/2),int(sy-(dy+ay)/2)
        src=np.s_[xc1:xc2,yc1:yc2];dst=np.s_[xp1:xp2,yp1:yp2]
        # A scalar tensor is weak against a nonzero-dimensional source array.
        tv[dst]=vegdem[src]-np.asarray(dz,dtype=vegdem.dtype)
        tv2[dst]=vegdem2[src]-np.asarray(dz,dtype=vegdem2.dtype)
        temp[dst]=a[src]-np.asarray(dz,dtype=a.dtype)
        f=np.maximum(f,temp)
        sh[f>a]=1;sh[f<=a]=0
        fa=tv>a;ga=tv2>a
        vs=np.maximum(vs,fa.astype(np.float32)-ga.astype(np.float32))
        vs[vs*sh>0]=0
        vb=vs+vb
        if index==1:
            first=tv-temp;first[first<=0]=1000
            vs[first<np.asarray(dz,dtype=first.dtype)]=1
            vs=vs*(vegdem2>a).astype(np.float32)
            vb.fill(0)
        if bush.max()>0 and np.max(fa*bush)>0:
            tb.fill(0);tb[dst]=bush[src]-np.asarray(dz,dtype=bush.dtype)
            g=np.maximum(g,tb);g*=bp
        index += 1
    sh=1-sh
    vb[vb>0]=1;vb=vb-vs
    if bush.max()>0:
        g=g-bush;g[g>0]=1;g[g<0]=0
        vs=vs-bp+g;vs[vs<0]=0
    vs[vs>0]=1
    return sh,1-vs,1-vb
