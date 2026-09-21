"""Own-met preprocessing artifacts and exact source-extracted tiling oracle."""
import ast
import inspect
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
from osgeo import gdal, osr
from solweig_light.preprocessor import preprocess, create_tiles_to_folder, find_windcoeff_files

gdal.UseExceptions()
ROOT=Path(__file__).resolve().parents[2]
SCENE=ROOT/'tests/reference/small_original_cpu/scene'


def assert_raster_equal(candidate,reference):
    new=gdal.Open(str(candidate));old=gdal.Open(str(reference))
    assert (new.RasterXSize,new.RasterYSize,new.RasterCount)==(old.RasterXSize,old.RasterYSize,old.RasterCount)
    assert new.GetGeoTransform()==old.GetGeoTransform()
    assert new.GetProjection()==old.GetProjection()
    assert new.GetMetadata()==old.GetMetadata()
    np.testing.assert_array_equal(new.ReadAsArray(),old.ReadAsArray())
    for i in range(1,new.RasterCount+1):
        nb,ob=new.GetRasterBand(i),old.GetRasterBand(i)
        assert nb.DataType==ob.DataType
        assert nb.GetNoDataValue()==ob.GetNoDataValue()
        assert nb.GetDescription()==ob.GetDescription()
        assert nb.GetMetadata()==ob.GetMetadata()


def test_packaged_original_own_met_preprocessing(tmp_path):
    result=preprocess(str(SCENE),'2025-06-21',own_met_file=str(SCENE/'met.txt'),preprocess_dir=str(tmp_path))
    assert result==str(tmp_path)
    for kind in ['Building_DSM','DEM','Trees']:
        assert_raster_equal(tmp_path/kind/f'{kind}_0_0.tif',SCENE/'processed_inputs'/kind/f'{kind}_0_0.tif')
    assert (tmp_path/'metfiles/metfile_0_0.txt').read_bytes()==(SCENE/'processed_inputs/metfiles/metfile_0_0.txt').read_bytes()
    # Own-met mode copies the complete forcing file; selected_date_str does not
    # filter timestamps or normalize meteorology upstream.
    assert (tmp_path/'metfiles/metfile_0_0.txt').read_bytes()==(SCENE/'met.txt').read_bytes()


@pytest.mark.parametrize('tile_size,overlap',[(6,0),(6,2),(50,3)])
def test_exact_upstream_source_tiling(tmp_path,tile_size,overlap):
    path=tmp_path/'source.tif'
    ds=gdal.GetDriverByName('GTiff').Create(str(path),17,13,2,gdal.GDT_Float32)
    ds.SetGeoTransform((400000,2,.1,4500000,.2,-2))
    crs=osr.SpatialReference();crs.ImportFromEPSG(32618);ds.SetProjection(crs.ExportToWkt())
    ds.SetMetadata({'fixture':'non-square rotated multiband'})
    for i in (1,2):
        band=ds.GetRasterBand(i);band.WriteArray(np.arange(221,dtype=np.float32).reshape(13,17)*i)
        band.SetNoDataValue(-9999);band.SetDescription(f'band-{i}')
    ds=None
    # Execute the exact pinned function body, with real GDAL and filesystem
    # dependencies. No module-import repairs or candidate-generated goldens.
    source=(ROOT/'.upstream/SOLWEIG-GPU/solweig_gpu/preprocessor.py').read_text()
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='create_tiles_to_folder')
    namespace={'gdal':gdal,'os':os,'shutil':shutil}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(ROOT/'.upstream/SOLWEIG-GPU/solweig_gpu/preprocessor.py'),'exec'),namespace)
    oracle=tmp_path/'oracle';candidate=tmp_path/'candidate'
    namespace['create_tiles_to_folder'](str(path),tile_size,overlap,str(oracle),'DEM',True)
    create_tiles_to_folder(str(path),tile_size,overlap,str(candidate),'DEM',True)
    assert sorted(p.name for p in candidate.iterdir())==sorted(p.name for p in oracle.iterdir())
    for p in oracle.iterdir():assert_raster_equal(candidate/p.name,p)


def test_optional_forcing_requires_original_parameters(tmp_path, capsys):
    with pytest.raises(SystemExit) as failure:
        preprocess(str(SCENE), '2025-06-21', use_own_met=False, preprocess_dir=str(tmp_path))
    assert failure.value.code == 1
    assert 'provide data_folder, data_source_type, start_time, and end_time' in capsys.readouterr().out


def test_incomplete_directional_wind_rejected(tmp_path):
    (tmp_path/'WindCoeff_dir030.tif').touch()
    with pytest.raises(FileNotFoundError,match='Missing directional'):
        find_windcoeff_files(str(tmp_path),'WindCoeff_dir*.tif')


def test_public_signature_matches_pinned_upstream():
    from typing import Optional
    source=(ROOT/'.upstream/SOLWEIG-GPU/solweig_gpu/solweig_gpu.py').read_text()
    function=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='preprocess')
    namespace={'Optional':Optional}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'upstream-public-preprocess','exec'),namespace)
    assert inspect.signature(preprocess)==inspect.signature(namespace['preprocess'])
