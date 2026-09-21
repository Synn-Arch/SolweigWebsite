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
"""All fifteen SVF fields and three lossless visibility channels."""
import os
import tempfile
import zipfile
import numpy as np
from numba import njit
from .shadows import shadow, create_patches


def _annulus_weight_numpy(altitude, aziinterval):
    n=np.float32(90)
    altitude=np.asarray(altitude)
    # Integer inputs become float32 when mixed with the default floating scalar.
    annulus=np.float32(91)-np.float32(altitude)
    interval=np.float32(aziinterval)
    steprad=(np.reciprocal(interval)*np.float32(360))*np.float32(np.pi/180)
    w=np.float32(1/(2*np.pi))*np.sin(np.float32(np.pi)/(np.float32(2)*n))*np.sin((np.float32(np.pi)*(np.float32(2)*annulus-np.float32(1)))/(np.float32(2)*n))
    return steprad*w


@njit(cache=True, fastmath=False, error_model="numpy")
def _annulus_weight_scalar(altitude, interval):
    """Preserve reflected scalar division and native float32 sine calls."""
    n = np.float32(90)
    annulus = np.float32(91) - altitude
    steprad = ((np.float32(1) / interval) * np.float32(360)) * np.float32(np.pi / 180)
    # Upstream Python scalar / tensor uses reciprocal then multiply.
    angle = (np.float32(1) / (np.float32(2) * n)) * np.float32(np.pi)
    other = (np.float32(np.pi) * (np.float32(2) * annulus - np.float32(1))) / (np.float32(2) * n)
    weight = np.float32(1 / (2 * np.pi)) * np.sin(angle) * np.sin(other)
    return np.float32(steprad * weight)


def annulus_weight(altitude, aziinterval):
    if np.ndim(altitude) or np.ndim(aziinterval):
        return _annulus_weight_numpy(altitude, aziinterval)
    return np.float32(_annulus_weight_scalar(np.float32(altitude), np.float32(aziinterval)))


def save_raster_like_gdal(template,path,array):
    from osgeo import gdal
    gdal.UseExceptions()
    rows,cols=array.shape
    ds=gdal.GetDriverByName('GTiff').Create(str(path),cols,rows,1,gdal.GDT_Float32)
    ds.SetGeoTransform(template.GetGeoTransform());ds.SetProjection(template.GetProjection())
    ds.GetRasterBand(1).WriteArray(np.asarray(array,dtype=np.float32));ds.FlushCache()
    ds=None


def save_svf_zip_npz_outputs(output_dir,gdal_dsm,svf,svfE,svfS,svfW,svfN,svfveg,svfEveg,svfSveg,svfWveg,svfNveg,svfaveg,svfEaveg,svfSaveg,svfWaveg,svfNaveg,shmat,vegshmat,vbshvegshmat,svftotal,number=None):
    if output_dir is None or gdal_dsm is None:
        raise ValueError('output_dir and gdal_dsm are required to save SVF rasters')
    os.makedirs(output_dir,exist_ok=True)
    suffix=f'_{number}' if number is not None else ''
    names=['svf','svfE','svfS','svfW','svfN','svfveg','svfEveg','svfSveg','svfWveg','svfNveg','svfaveg','svfEaveg','svfSaveg','svfWaveg','svfNaveg']
    arrays=[svf,svfE,svfS,svfW,svfN,svfveg,svfEveg,svfSveg,svfWveg,svfNveg,svfaveg,svfEaveg,svfSaveg,svfWaveg,svfNaveg]
    # Isolate export scratch for different logical tiles/calls.
    with tempfile.TemporaryDirectory(prefix='.svf-',dir=output_dir) as tmp:
        zpath=os.path.join(tmp,'svfs.zip')
        with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name,array in zip(names,arrays):
                p=os.path.join(tmp,name+'.tif');save_raster_like_gdal(gdal_dsm,p,array);z.write(p,name+'.tif')
        total=os.path.join(tmp,'total.tif');save_raster_like_gdal(gdal_dsm,total,svftotal)
        mats=os.path.join(tmp,'mats.npz')
        from .visibility import PackedVisibility, export_visibility_npz
        channels = [value if isinstance(value, PackedVisibility) else
                    PackedVisibility.from_dense(np.asarray(value, dtype=np.float32))
                    for value in (shmat, vegshmat, vbshvegshmat)]
        export_visibility_npz(mats, *channels)
        for src,dst in [(zpath,f'svfs{suffix}.zip'),(total,f'SkyViewFactor{suffix}.tif'),(mats,f'shadowmats{suffix}.npz')]:
            os.replace(src,os.path.join(output_dir,dst))


def _calculate_svf(patch_option,amaxvalue=None,a=None,vegdem=None,vegdem2=None,bush=None,scale=None,save_rasters=False,building_dsm_path=None,tree_path=None,dem_path=None,output_dir=None,number=None,gdal_dsm=None,*,compact=False):
    template=gdal_dsm
    required=[amaxvalue,a,vegdem,vegdem2,bush,scale]
    if save_rasters:
        from osgeo import gdal
        gdal.UseExceptions()
        if output_dir is None:raise ValueError('When save_rasters=True, output_dir must be provided.')
        if any(x is None for x in required):
            if any(x is None for x in [building_dsm_path,tree_path,dem_path]):raise ValueError('building_dsm_path, tree_path, and dem_path must be provided')
            template=gdal.Open(str(building_dsm_path));a=template.GetRasterBand(1).ReadAsArray().astype(np.float32)
            tree=gdal.Open(str(tree_path)).ReadAsArray().astype(np.float32)
            dem=gdal.Open(str(dem_path)).ReadAsArray().astype(np.float32)
            scale=1/template.GetGeoTransform()[1];tree[tree<0]=0
            height=tree+dem;trunkheight=tree*np.float32(.25)+dem
            bush=np.logical_not(trunkheight*height)*height
            vegdem=tree+a;vegdem[vegdem==a]=0
            vegdem2=tree*np.float32(.25)+a;vegdem2[vegdem2==a]=0
            amaxvalue=np.maximum(a.max(),height.max())
        elif template is None:
            if building_dsm_path is None:raise ValueError('provide either gdal_dsm or building_dsm_path')
            template=gdal.Open(str(building_dsm_path))
    elif any(x is None for x in required):
        raise ValueError('amaxvalue, a, vegdem, vegdem2, bush and scale must be provided')
    a=np.asarray(a);shape=a.shape
    fields=[np.zeros(shape,dtype=np.float32) for _ in range(15)]
    _,_,ann,alts,counts,steps,starts=create_patches(patch_option)
    # Option 4 reaches a floating range bound upstream and raises TypeError.
    if ann.dtype.kind=='f':raise TypeError('only integer tensors of a single element can be converted to an index')
    azis=np.zeros(sum(counts),dtype=np.float32)
    index=0
    for band in range(len(alts)):
        for k in range(int(np.reciprocal(steps[band])*np.float32(360))):
            azis[index]=np.float32(k)*steps[band]+np.float32(starts[band])
            if azis[index]>360:azis[index]-=np.float32(360)
            index+=1
    if compact:
        from .visibility import VisibilityBuilder
        mats = [VisibilityBuilder((*shape, int(sum(counts)))) for _ in range(3)]
    else:
        mats=[np.zeros((*shape,sum(counts)),dtype=np.float32) for _ in range(3)]
    anis=np.ceil(counts.astype(np.float32)/np.float32(2))
    index=0
    for band in range(len(alts)):
        for j in range(counts[band]):
            azi=azis[index]
            masks=shadow(amaxvalue,a,vegdem,vegdem2,bush,azi,alts[band],scale)
            for mat,mask in zip(mats,masks):
                if compact: mat.append(mask)
                else: mat[:,:,index]=mask
            directions=[0<=azi<180,90<=azi<270,180<=azi<360,azi>=270 or azi<90]
            for k in range(ann[band]+1,ann[band+1]+1):
                wt=annulus_weight(k,counts[band]);wd=annulus_weight(k,anis[band])
                for channel,mask in enumerate(masks):
                    offset=channel*5
                    fields[offset]+=wt*mask
                    for d,active in enumerate(directions):
                        if active:fields[offset+d+1]+=wd*mask
            index+=1
    fields[2]+=np.float32(3.0459e-4);fields[3]+=np.float32(3.0459e-4)
    last=np.zeros(shape,dtype=np.float32);last[np.asarray(vegdem2)==0]=np.float32(3.0459e-4)
    for i in (7,8,12,13):fields[i]+=last
    for field in fields:field[field>1]=1
    total=fields[0]-(np.float32(1)-fields[5])*(np.float32(1)-np.float32(.03))
    shmat,vegmat,vbmat = [mat.finish() for mat in mats] if compact else mats
    if save_rasters:save_svf_zip_npz_outputs(output_dir,template,*fields,shmat,vegmat,vbmat,total,number)
    # Preserve the actual executed upstream tuple order (not its stale docstring).
    order=[0,10,1,11,6,4,14,9,2,12,7,5,3,13,8]
    return tuple(fields[i] for i in order)+(vegmat,vbmat,shmat,total)


def svf_calculator(patch_option,amaxvalue=None,a=None,vegdem=None,vegdem2=None,bush=None,scale=None,save_rasters=False,building_dsm_path=None,tree_path=None,dem_path=None,output_dir=None,number=None,gdal_dsm=None):
    """Legacy low-level ndarray return contract, with bounded export scratch."""
    return _calculate_svf(patch_option,amaxvalue,a,vegdem,vegdem2,bush,scale,save_rasters,
                         building_dsm_path,tree_path,dem_path,output_dir,number,gdal_dsm)


def svf_calculator_compact(*args, **kwargs):
    """Native geometry path: encode each patch immediately without float cubes."""
    return _calculate_svf(*args, **kwargs, compact=True)
