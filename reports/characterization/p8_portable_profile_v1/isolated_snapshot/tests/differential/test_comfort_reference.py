"""Compare CPU comfort and solar ports with packaged original upstream CPU outputs."""
from pathlib import Path
import ast
import hashlib
import inspect
import json
import numpy as np
import pytest
from osgeo import gdal
gdal.UseExceptions()
from solweig_light.comfort import utci_calculator, black_globe_temperature, isobaric_wet_bulb_temperature_from_rh
from solweig_light.radiation.solar import Solweig_2015a_metdata_noload
from solweig_light.radiation.materials import Tgmaps_v1

ROOT=Path(__file__).resolve().parents[2]
REF=ROOT/'tests/reference'
CASES=['base','uhi','landcover','landcover_normalized','directional','legacy_direct','legacy_wrapper']

def raster(path):
    ds=gdal.Open(str(path)); a=ds.ReadAsArray(); ds=None; return a

def compare(actual,expected,budget):
    assert actual.shape==expected.shape
    for f in (np.isnan,np.isposinf,np.isneginf): np.testing.assert_array_equal(f(actual),f(expected))
    np.testing.assert_array_equal(actual==-999,expected==-999)
    finite=np.isfinite(expected)
    errors=np.abs(actual[finite]-expected[finite])
    worst=np.unravel_index(np.nanargmax(np.where(finite,np.abs(actual-expected),np.nan)),actual.shape)
    assert errors.max(initial=0)<=budget, f'max_abs={errors.max()} worst={worst} actual={actual[worst]} reference={expected[worst]}'

@pytest.mark.parametrize('case',CASES)
def test_utci_original_rasters(case):
    base=REF/'optional_original_cpu'/case
    met=np.loadtxt(base/'processed_inputs/metfiles/metfile_0_0_2020-07-18.txt',skiprows=1)
    output=base/'output_folder/0_0'
    ta=raster(output/'Ta_0_0.tif'); wind=raster(output/'Wind_0_0.tif'); tmrt=raster(output/'TMRT_0_0.tif')
    rh=np.broadcast_to(met[:,10].astype(np.float32)[:,None,None],ta.shape)
    result=utci_calculator(ta,rh,tmrt,wind)
    dsm=raster(base/'processed_inputs/Building_DSM/Building_DSM_0_0.tif')
    dem=raster(base/'processed_inputs/DEM/DEM_0_0.tif')
    buildings=dsm-dem; buildings[buildings<2]=1; buildings[buildings>=2]=0
    result[:,buildings!=1]=np.nan
    compare(result,raster(output/'UTCI_0_0.tif'),.02)

@pytest.mark.parametrize('case',CASES)
def test_wbgt_original_rasters(case):
    base=REF/'optional_original_cpu'/case
    met=np.loadtxt(base/'processed_inputs/metfiles/metfile_0_0_2020-07-18.txt',skiprows=1)
    output=base/'output_folder/0_0'
    ta=raster(output/'Ta_0_0.tif'); wind=raster(output/'Wind_0_0.tif'); tmrt=raster(output/'TMRT_0_0.tif')
    wbt=isobaric_wet_bulb_temperature_from_rh(p=met[:,12]*1000.,T=met[:,11]+met[:,24]+273.15,rh=met[:,10],phase='liquid',method='Romps',limit=True)
    hcg=(6.3/.46821)*wind**.6
    globe=black_globe_temperature(hcg,tmrt,ta,emissivity=.95)
    # Wet bulb is a float64 scalar in upstream: raster arithmetic retains float32.
    wbt=wbt.astype(np.float32)[:,None,None]
    sun=.7*wbt+.3*globe
    shade=.7*wbt+.2*globe+.1*ta
    result=np.where(raster(output/'Shadow_0_0.tif')<.1,sun,shade)
    compare(result,raster(output/'WBGT_0_0.tif'),.02)

def test_solar_original_captured_forcing():
    base=REF/'small_original_cpu'
    met=np.loadtxt(base/'scene/met.txt',skiprows=1)
    boundaries=base/'boundaries'
    events=json.loads((boundaries/'manifest.json').read_text())['events']
    inputs=[e for e in events if e['function']=='Solweig_2022a_calc' and e['boundary']=='input']
    with np.load(boundaries/inputs[0]['path']) as first:
        location={k:float(first['location/'+k]) for k in ['latitude','longitude','altitude']}
    values=Solweig_2015a_metdata_noload(met,location,-4)
    for event in inputs:
        with np.load(boundaries/event['path']) as ref:
            i=event['timestep']
            for output,index in [('altitude',1),('azimuth',2),('zen',3),('jday',4),('dectime',6),('altmax',7)]:
                actual=values[index][i] if index==6 else values[index][0,i]
                np.testing.assert_allclose(actual,ref[output],rtol=0,atol=1e-10,err_msg=f'{output} timestep={i}')
    np.testing.assert_array_equal(values[5],np.ones((1,len(met))))

def test_utci_sentinel_and_nan_policy():
    ta=np.array([25,-999,-1000,25,25,25,np.nan],np.float32)
    rh=np.array([50,50,50,-999,50,50,50],np.float32)
    tmrt=np.array([30,30,30,30,-999,30,30],np.float32)
    wind=np.array([1,1,1,1,1,-999,1],np.float32)
    result=utci_calculator(ta,rh,tmrt,wind)
    assert result.dtype==np.float32
    np.testing.assert_array_equal(result[1:6],np.full(5,-999,np.float32))
    assert np.isnan(result[6]) and np.isfinite(result[0])

def test_material_current_value_lookup_collision_is_preserved():
    grid=np.array([[1,2]],np.float32)
    classes=np.array([[1,2,2,2,2,2],[2,.2,.95,.37,-3.41,15],[99,.2,.9,.4,-3,16]])
    outputs=Tgmaps_v1(grid,classes)
    # Class 1 properties initially equal class 2 and are replaced in its later pass.
    for result,expected in zip(outputs[:4],[.37,-3.41,.2,.95]):
        np.testing.assert_array_equal(result,np.full(grid.shape,expected,np.float32))
    np.testing.assert_array_equal(outputs[6],np.full(grid.shape,15,np.float32))
    np.testing.assert_array_equal(grid,np.array([[1,2]],np.float32))


def test_explicit_utci_polynomial_matches_original_expression_checksum():
    # Frozen from the original pinned source AST, preserving coefficients and order.
    from solweig_light.comfort.utci import utci_polynomial
    function=ast.parse(inspect.getsource(utci_polynomial)).body[0]
    expression=next(n.value for n in function.body if isinstance(n,ast.Assign))
    checksum=hashlib.sha256(ast.dump(expression).encode()).hexdigest()
    assert checksum=='33cec1d5f69ef606d602e0e25d45c462701cc049e90a91d8f704f65b2dfd4907'
