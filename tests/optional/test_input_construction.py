"""Real local input transforms; acquisition tests are explicitly offline mocks."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest
import rasterio
from solweig_light.inputs import construction as inputs

ROOT=Path(__file__).resolve().parents[2]

def harness():
    spec=importlib.util.spec_from_file_location('input_fixture',ROOT/'tools/characterize_inputs.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def test_original_local_transform_reference(tmp_path):
    actual=harness().exercise(vars(inputs),tmp_path)
    with np.load(ROOT/'tests/reference/inputs_original_cpu/local.npz') as expected:
        for name in expected.files:
            np.testing.assert_array_equal(actual[name],expected[name],err_msg=name)

def test_clip_reproject_and_missing_vector(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box
    polygons=gpd.GeoDataFrame({'HEIGHT_ROOF':[7.]},geometry=[box(0,0,4,4)],crs='EPSG:32614')
    clip=gpd.GeoDataFrame(geometry=[box(0,0,2,2)],crs='EPSG:32614')
    clipped=inputs._clip_polygons(polygons,clip)
    assert clipped.geometry.area.sum()==4
    grid=inputs.compute_reference_grid(box(0,0,4,4),1)
    out=tmp_path/'empty.tif'
    inputs.rasterize_vector_checked(tmp_path/'absent.geojson',out,'HEIGHT_ROOF',grid.transform,4,4,'EPSG:32614','buildings',dtype='float32')
    with rasterio.open(out) as src:
        assert not src.read(1).any()
        assert src.nodata==0

def test_builder_offline_mock_acquisition_real_processing(tmp_path,monkeypatch):
    monkeypatch.setattr(inputs,'safe_initialize_ee',lambda:None)
    def vector(paths,bbox):
        import geopandas as gpd
        b=inputs.compute_bounding_boxes(30.2857,-97.7396,.003,0,0)
        gpd.GeoDataFrame({'HEIGHT_ROOF':[8.]},geometry=[b.bbox_utm_solweig],crs=b.crs_utm).to_file(paths.bld_fp,driver='GeoJSON')
    monkeypatch.setattr(inputs,'build_vectors_from_osm_gba',vector)
    def download(path,*args):
        b=inputs.compute_bounding_boxes(30.2857,-97.7396,.003,0,0)
        grid=inputs.compute_reference_grid(b.bbox_utm_solweig,2)
        value=10 if 'WorldCover' in path.name else 3
        with rasterio.open(path,'w',driver='GTiff',height=grid.height,width=grid.width,count=1,dtype='float32',crs=b.crs_utm,transform=grid.transform) as dst: dst.write(np.full((grid.height,grid.width),value,dtype='float32'),1)
    for name in ('download_worldcover','download_tree_dsm','download_dem'): monkeypatch.setattr(inputs,name,download)
    from solweig_light import build_inputs
    result=build_inputs(30.2857,-97.7396,'offline',.003,0,0,str(tmp_path),2)
    output=Path(result)
    assert {p.name for p in output.glob('*.tif')}=={'Landuse.tif','Trees.tif','DEM.tif','Building_DSM.tif','Buildings.tif'}
    inputs.check_raster_alignment(list(output.glob('*.tif')))
    with rasterio.open(output/'Buildings.tif') as src: buildings=src.read(1)
    mask=buildings>1
    assert mask.any() and (~mask).any()  # ceil grid extends beyond polygon
    with rasterio.open(output/'Landuse.tif') as src: np.testing.assert_array_equal(src.read(1),np.where(mask,2,4))
    with rasterio.open(output/'Trees.tif') as src: np.testing.assert_array_equal(src.read(1),np.where(mask,0,3))
    with rasterio.open(output/'Building_DSM.tif') as src: np.testing.assert_array_equal(src.read(1),buildings+3)

def test_reverse_geocode_offline_mock(monkeypatch):
    class Client:
        def reverse(self,*a,**k): return type('Location',(),{'raw':{'address':{'town':'Town','state':'State','country_code':'US'}}})()
    monkeypatch.setattr(inputs,'Nominatim',lambda **k:Client())
    assert inputs.reverse_geocode(0,0)==('Town','State','us')

def test_era5_netcdf_offline_mock_ee_real_normalization(tmp_path,monkeypatch):
    import xarray as xr
    from types import SimpleNamespace
    bands=['temperature_2m','dewpoint_temperature_2m','surface_pressure','u_component_of_wind_10m','v_component_of_wind_10m','surface_solar_radiation_downwards','surface_thermal_radiation_downwards','geopotential']
    class Response:
        def __init__(self,value): self.value=value
        def getInfo(self): return self.value
    class Collection:
        def first(self): return self
        def bandNames(self): return Response(bands)
        def filterDate(self,*a): return self
        def filterBounds(self,*a): return self
        def select(self,*a): return self
        def getRegion(self,*a): return Response([['id','longitude','latitude','time']+bands,['a',-97,30,1704067200000,300,290,100000,2,3,360000,720000,98.0665],['b',-97,30,1704070800000,301,291,100010,3,4,720000,1080000,196.133]])
    class Date:
        def advance(self,*a): return self
    ee=SimpleNamespace(Geometry=SimpleNamespace(Point=lambda p:p),ImageCollection=lambda n:Collection(),Date=SimpleNamespace(fromYMD=lambda *a:Date()))
    monkeypatch.setattr(inputs,'ee',ee)
    out=tmp_path/'met.nc'
    inputs.download_and_embed_era5(out,[29,-98,31,-96],2024,2024,tmp_path)
    with xr.open_dataset(out) as ds:
        assert set(ds.data_vars)=={'T2M','RH2M','SWDOWN','HGT','LWDOWN','PSFC','U10M','V10M'}
        assert ds.sizes=={'time':2,'lat':1,'lon':1}
        np.testing.assert_array_equal(ds.SWDOWN.values[:,0,0],[100,200])
        np.testing.assert_array_equal(ds.LWDOWN.values[:,0,0],[200,300])
        np.testing.assert_allclose(ds.HGT.values,[[10]])
        assert np.all((ds.RH2M.values>0)&(ds.RH2M.values<100))

def test_osmnx_offline_mock_filters_polygon_features(monkeypatch):
    import geopandas as gpd
    from shapely.geometry import box, Point
    from types import SimpleNamespace
    response=gpd.GeoDataFrame({'kind':['area','point']},geometry=[box(0,0,1,1),Point(.5,.5)],crs='EPSG:4326')
    calls=[]
    def fetch(poly,tags):
        calls.append((poly.bounds,tags)); return response
    monkeypatch.setattr(inputs,'ox',SimpleNamespace(features_from_polygon=fetch))
    actual=inputs._get_osm_polygons([0,0,1,1],{'building':True})
    assert len(actual)==1 and actual.iloc[0]['kind']=='area'
    assert calls==[((0.,0.,1.,1.),{'building':True})]

def test_original_reference_provenance():
    import ast
    import hashlib
    import json
    reference=ROOT/'tests/reference/inputs_original_cpu'
    manifest=json.loads((reference/'manifest.json').read_text())
    source=ROOT/'.upstream/SOLWEIG-GPU/solweig_gpu/create_inputs.py'
    collector=ROOT/'tools/characterize_inputs.py'
    sha=lambda data:hashlib.sha256(data).hexdigest()
    assert manifest['upstream_commit']=='0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
    assert manifest['source_sha256']=='38b570dac030463b4801d5c10ff52dd9d2ea661820c9b1f34b13a1fb33206149'
    if source.exists():
        assert sha(source.read_bytes())==manifest['source_sha256']
    assert sha(collector.read_bytes())==manifest['collector_sha256']
    fixture_nodes=[node for node in ast.parse(collector.read_bytes()).body
                   if isinstance(node,ast.FunctionDef) and node.name in {'write','exercise'}]
    fixture_ast=ast.dump(ast.Module(body=fixture_nodes,type_ignores=[]),include_attributes=False)
    assert sha(fixture_ast.encode())==manifest['fixture_ast_sha256']
    assert set(manifest['artifacts'])=={'local.npz'}
    for name,digest in manifest['artifacts'].items():
        assert sha((reference/name).read_bytes())==digest
    assert manifest['extraction_policy']=='unchanged AST-selected definitions; upstream import/bootstrap statements excluded; original helper CPU reference, not full-module execution'
    assert manifest['extracted_definitions']==sorted(harness().NAMES)
    assert manifest['evidence_class']=='original upstream CPU helpers; unchanged AST-extracted definitions, import bootstrap excluded'
    assert manifest['live_services_verified'] is False
