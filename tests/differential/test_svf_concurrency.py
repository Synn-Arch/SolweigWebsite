"""Concurrent logical tiles own complete, separate legacy export paths."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import zipfile
import numpy as np
from osgeo import gdal
from solweig_light.geometry.svf import svf_calculator_compact
from solweig_light.geometry.visibility import import_visibility_npz

REFERENCE=Path(__file__).resolve().parents[1]/'reference/small_original_cpu/scene'


def test_concurrent_distinct_tile_svf_exports(tmp_path):
    def compute(tile):
        return svf_calculator_compact(2,save_rasters=True,
            building_dsm_path=REFERENCE/'Building_DSM.tif',tree_path=REFERENCE/'Trees.tif',
            dem_path=REFERENCE/'DEM.tif',output_dir=tmp_path,number=tile)
    tiles=['3_7','9_2']
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(compute,tiles))
    assert {p.name for p in tmp_path.iterdir()}=={
        f'{prefix}_{tile}.{suffix}' for tile in tiles for prefix,suffix in
        [('svfs','zip'),('SkyViewFactor','tif'),('shadowmats','npz')]}
    for tile,outputs in zip(tiles,results):
        channels=import_visibility_npz(tmp_path/f'shadowmats_{tile}.npz',max_workspace_bytes=4096)
        for key,index in [('shadowmat',17),('vegshadowmat',15),('vbshmat',16)]:
            for patch in range(outputs[index].shape[2]):
                np.testing.assert_array_equal(channels[key][:,:,patch],outputs[index][:,:,patch])
        with zipfile.ZipFile(tmp_path/f'svfs_{tile}.zip') as archive:
            with zipfile.ZipFile(REFERENCE/'processed_inputs/SVF/svfs_0_0.zip') as original:
                assert archive.namelist()==original.namelist()
            for member in archive.namelist():
                actual=gdal.Open(f'/vsizip/{(tmp_path/f"svfs_{tile}.zip").resolve()}/{member}')
                expected=gdal.Open(f'/vsizip/{(REFERENCE/"processed_inputs/SVF/svfs_0_0.zip").resolve()}/{member}')
                np.testing.assert_allclose(actual.ReadAsArray(),expected.ReadAsArray(),rtol=0,atol=1e-6)
                assert actual.GetGeoTransform()==expected.GetGeoTransform()
                assert actual.GetProjection()==expected.GetProjection()
    for first,second in zip(results[0][:15],results[1][:15]):np.testing.assert_array_equal(first,second)
