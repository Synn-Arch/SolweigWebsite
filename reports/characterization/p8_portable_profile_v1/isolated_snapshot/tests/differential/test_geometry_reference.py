"""Candidate comparisons against source-provenanced original CPU artifacts."""
import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
from osgeo import gdal
from solweig_light.geometry.shadows import shadow,create_patches
from solweig_light.geometry.walls import findwalls,filter1Goodwin_as_aspect_v3
from solweig_light.geometry.svf import svf_calculator

gdal.UseExceptions()

REF=Path(__file__).parents[1]/'reference'
GEOM=REF/'geometry_original_cpu'
CASES=json.loads((GEOM/'reference_manifest.json').read_text())['cases']

@pytest.mark.parametrize('case',CASES,ids=[c['name'] for c in CASES])
def test_original_geometry(case):
    inputs=dict(np.load(GEOM/case['input_artifact']['path']))
    gold=np.load(GEOM/case['output_artifact']['path'])
    if case['function']=='create_patches':actual=create_patches(int(inputs['patch_option']))
    else:
        # Oracle characterization called these as Python float arguments.
        for key in ('azimuth','altitude','scale'):inputs[key]=float(inputs[key])
        actual=shadow(**inputs)
    for key,value in zip(gold.files,actual,strict=True):
        assert value.dtype==gold[key].dtype
        np.testing.assert_array_equal(value,gold[key],err_msg=f'{case["name"]}/{key}')


def assert_svf_tiff_equal(new,old,name):
    assert (new.RasterXSize,new.RasterYSize,new.RasterCount)==(old.RasterXSize,old.RasterYSize,old.RasterCount)
    assert new.RasterCount==1
    assert new.GetGeoTransform()==old.GetGeoTransform()
    assert new.GetProjection()==old.GetProjection()
    assert new.GetMetadata()==old.GetMetadata()
    for index in range(1,new.RasterCount+1):
        candidate=new.GetRasterBand(index);reference=old.GetRasterBand(index)
        assert candidate.DataType==reference.DataType==gdal.GDT_Float32
        assert candidate.GetNoDataValue()==reference.GetNoDataValue()
        assert candidate.GetMetadata()==reference.GetMetadata()
        assert candidate.GetDescription()==reference.GetDescription()
        np.testing.assert_allclose(candidate.ReadAsArray(),reference.ReadAsArray(),rtol=0,atol=1e-6,err_msg=name)


def test_small_original_walls_and_svf(tmp_path):
    scene=REF/'small_original_cpu'/'scene'
    def read(path):return gdal.Open(str(path)).ReadAsArray().astype(np.float32)
    a=read(scene/'Building_DSM.tif')
    walls=findwalls(a,3.)
    np.testing.assert_array_equal(walls.astype(np.float32),read(scene/'processed_inputs/walls/walls_0_0.tif'))
    aspect=filter1Goodwin_as_aspect_v3(walls,1.,a)
    np.testing.assert_array_equal(aspect.astype(np.float32),read(scene/'processed_inputs/aspect/aspect_0_0.tif'))
    actual=svf_calculator(2,save_rasters=True,building_dsm_path=scene/'Building_DSM.tif',tree_path=scene/'Trees.tif',dem_path=scene/'DEM.tif',output_dir=tmp_path,number='0_0')
    svfdir=scene/'processed_inputs/SVF'
    with np.load(svfdir/'shadowmats_0_0.npz',allow_pickle=False) as gold, np.load(tmp_path/'shadowmats_0_0.npz',allow_pickle=False) as saved:
        assert set(saved.files)=={'shadowmat','vegshadowmat','vbshmat'}
        assert saved.files==gold.files
        for key in gold.files:
            assert saved[key].dtype==gold[key].dtype==np.float32
            assert saved[key].shape==gold[key].shape
            np.testing.assert_array_equal(saved[key],gold[key],err_msg=f'written NPZ/{key}')
        for key,value in zip(['vegshadowmat','vbshmat','shadowmat'],actual[15:18]):
            np.testing.assert_array_equal(value,gold[key],err_msg=key)
    np.testing.assert_allclose(actual[18],read(svfdir/'SkyViewFactor_0_0.tif'),rtol=0,atol=1e-6)
    standalone=gdal.Open(str(tmp_path/'SkyViewFactor_0_0.tif'))
    original_standalone=gdal.Open(str(svfdir/'SkyViewFactor_0_0.tif'))
    assert_svf_tiff_equal(standalone,original_standalone,'SkyViewFactor_0_0.tif')
    standalone=original_standalone=None
    with zipfile.ZipFile(svfdir/'svfs_0_0.zip') as orig,zipfile.ZipFile(tmp_path/'svfs_0_0.zip') as candidate:
        assert orig.namelist()==candidate.namelist()
        for name in orig.namelist():
            gdal.FileFromMemBuffer('/vsimem/reference-'+name,orig.read(name));gdal.FileFromMemBuffer('/vsimem/candidate-'+name,candidate.read(name))
            old=gdal.Open('/vsimem/reference-'+name);new=gdal.Open('/vsimem/candidate-'+name)
            assert_svf_tiff_equal(new,old,name)
            old=new=None;gdal.Unlink('/vsimem/reference-'+name);gdal.Unlink('/vsimem/candidate-'+name)
