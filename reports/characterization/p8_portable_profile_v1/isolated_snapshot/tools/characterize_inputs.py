"""Collect original pinned input helpers, without executing auto-install bootstrap.

Functions are AST-extracted unchanged; no candidate imports. This is an original
helper CPU reference, not evidence for authenticated acquisition.
"""
import ast
import math
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import box
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from dataclasses import dataclass
from typing import Any, List

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.upstream/SOLWEIG-GPU/solweig_gpu/create_inputs.py'
EXTRACTION_POLICY = 'unchanged AST-selected definitions; upstream import/bootstrap statements excluded; original helper CPU reference, not full-module execution'
NAMES = {'GridSpec', 'BBoxes', 'tlog', 'timed', 'utm_epsg_from_latlon', 'compute_bounding_boxes', 'compute_reference_grid', '_resample_to_grid', 'reclassify_esa_worldcover_inplace', 'create_building_dsm_and_clean_trees', 'rasterize_polygons_to_array', '_clip_polygons'}

def original():
    tree = ast.parse(SOURCE.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in NAMES]
    namespace = dict(globals(), time=__import__('time'))
    exec(compile(tree, str(SOURCE), 'exec'), namespace)
    return namespace

def write(path, array):
    with rasterio.open(path, 'w', driver='GTiff', height=array.shape[0], width=array.shape[1], count=1, dtype=array.dtype, crs='EPSG:32614', transform=from_origin(0, 4, 1, 1)) as dst:
        dst.write(array, 1)

def exercise(ns, work):
    worldcover=np.array([[0,10,20,30],[40,50,60,70],[80,90,95,100],[255,10,50,80]], dtype='uint8')
    building=np.array([[0,1,1.01,10],[0,0,0,0],[3,0,0,0],[0,0,0,0]],dtype='float32')
    trees=np.full((4,4),7,dtype='float32'); dem=np.arange(16,dtype='float32').reshape(4,4)
    for name, arr in [('wc',worldcover),('building',building),('tree',trees),('dem',dem)]: write(work/f'{name}.tif',arr)
    ns['reclassify_esa_worldcover_inplace'](work/'wc.tif',work/'building.tif')
    ns['create_building_dsm_and_clean_trees'](work/'building.tif',work/'tree.tif',work/'dem.tif',work/'dsm.tif')
    grid=ns['compute_reference_grid'](box(0,0,4,4),1.5)
    ns['_resample_to_grid'](work/'dem.tif',work/'aligned.tif','EPSG:32614',grid.transform,grid.width,grid.height,Resampling.bilinear)
    polygons=gpd.GeoDataFrame({'value':[1]},geometry=[box(0,0,2,4)],crs='EPSG:32614')
    raster=ns['rasterize_polygons_to_array'](polygons,from_origin(0,4,1,1),4,4,'EPSG:32614')
    result={'polygon':raster}
    for name in ('wc','tree','dsm','aligned'):
        with rasterio.open(work/f'{name}.tif') as src: result[name]=src.read(1)
    bounds=ns['compute_bounding_boxes'](30.2857,-97.7396,3,1,1)
    result['bounds']=np.array(bounds.bbox4326+list(bounds.bbox_utm_solweig.bounds))
    return result

if __name__=='__main__':
    out=ROOT/'tests/reference/inputs_original_cpu'; out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp: np.savez(out/'local.npz',**exercise(original(),Path(tmp)))
    import subprocess,sys
    script = Path(__file__).read_bytes()
    fixture_nodes = [node for node in ast.parse(script).body
                     if isinstance(node, ast.FunctionDef) and node.name in {'write', 'exercise'}]
    fixture_ast = ast.dump(ast.Module(body=fixture_nodes, type_ignores=[]), include_attributes=False)
    manifest = {
        'upstream_commit': '0d7fe742abeeddd890dd58fc76ed7f78bd47faec',
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'collector_sha256': hashlib.sha256(script).hexdigest(),
        'fixture_ast_sha256': hashlib.sha256(fixture_ast.encode()).hexdigest(),
        'artifacts': {'local.npz': hashlib.sha256((out/'local.npz').read_bytes()).hexdigest()},
        'extraction_policy': EXTRACTION_POLICY,
        'extracted_definitions': sorted(NAMES),
        'evidence_class': 'original upstream CPU helpers; unchanged AST-extracted definitions, import bootstrap excluded',
        'invocation': '.venv-oracle/bin/python tools/characterize_inputs.py',
        'python': sys.version,
        'dependencies': subprocess.check_output(['uv','pip','freeze','--python',sys.executable],text=True),
        'live_services_verified': False,
    }
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
