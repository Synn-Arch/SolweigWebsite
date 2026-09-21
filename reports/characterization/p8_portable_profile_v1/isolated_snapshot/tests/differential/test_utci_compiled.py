"""Frozen original-CPU UTCI fixtures; serial compiled parity precedes parallel use."""
from pathlib import Path
import ast
import hashlib
import io
import json
import inspect
import numpy as np
import pytest
from numba import get_num_threads, set_num_threads
from osgeo import gdal
from solweig_light.comfort.utci import (utci_calculator,utci_polynomial,utci_calculator_compiled,
    _utci_points_serial,_utci_points_parallel)
from solweig_light.comfort._utci_scalar import polynomial_scalar,saturation_scalar

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'tests/reference/utci_original_cpu'
CASES=['base','uhi','landcover','landcover_normalized','directional','legacy_direct','legacy_wrapper']
gdal.UseExceptions()

def values():
    with np.load(REFERENCE/'cases.npz',allow_pickle=False) as f:
        return {key:f[key] for key in f.files}

def arrays(f):return tuple(f[key] for key in ['Ta','RH','Tmrt','wind'])
def domain(f):
    ta,rh,tmrt,wind=arrays(f)
    return (ta>=-50)&(ta<=50)&(rh>=0)&(rh<=100)&(tmrt>=-50)&(tmrt<=100)&(wind>=0)&(wind<=17)

def compare(actual,expected,mask=None,budget=.02):
    assert actual.shape==expected.shape
    for function in (np.isnan,np.isposinf,np.isneginf):np.testing.assert_array_equal(function(actual),function(expected))
    np.testing.assert_array_equal(actual==-999,expected==-999)
    finite=np.isfinite(expected)
    if mask is not None:finite&=mask
    errors=np.where(finite,np.abs(actual-expected),np.nan)
    maximum=float(np.nanmax(errors))
    worst=np.unravel_index(np.nanargmax(errors),errors.shape)
    assert maximum<=budget,f'max_abs={maximum} worst={worst} candidate={actual[worst]} original={expected[worst]}'


def test_original_reference_provenance_and_coefficient_reconstruction():
    manifest=json.loads((REFERENCE/'manifest.json').read_text())
    assert manifest['evidence_class']=='original_upstream_cpu' and manifest['oracle_patch_hash'] is None
    assert manifest['source_commit']=='0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
    assert hashlib.sha256((REFERENCE/'cases.npz').read_bytes()).hexdigest()==manifest['fixture_sha256']
    for name,checksum in manifest['typed_fixture_sha256'].items():assert hashlib.sha256((REFERENCE/name).read_bytes()).hexdigest()==checksum
    coefficients=json.loads((ROOT/'src/solweig_light/comfort/utci_coefficients.json').read_text())
    assert coefficients['source_sha256']==manifest['source_package_sha256']['calculate_utci.py']
    assert coefficients['term_count']==211 and coefficients['reconstruction_ast_equal']
    terms=[ast.parse(term['expression'],mode='eval').body for term in coefficients['ordered_terms']]
    reconstructed=terms[0]
    for term in terms[1:]:reconstructed=ast.BinOp(reconstructed,ast.Add(),term)
    digest=hashlib.sha256(ast.dump(reconstructed).encode()).hexdigest()
    assert digest==coefficients['original_expression_ast_sha256']=='33cec1d5f69ef606d602e0e25d45c462701cc049e90a91d8f704f65b2dfd4907'
    assert hashlib.sha256((ROOT/'src/solweig_light/comfort/_utci_scalar.py').read_bytes()).hexdigest()==coefficients['scalar_sha256']


def test_serial_scalar_matches_original_function_and_masks():
    f=values()
    with np.errstate(all='ignore'):result=utci_calculator_compiled(*arrays(f))
    compare(result,f['UTCI'],domain(f))
    # Unsupported finite/nonfinite positions use the unchanged float32 evaluator.
    outside=~domain(f)&~f['invalid']
    with np.errstate(all='ignore'):fallback=utci_calculator(*(x[outside] for x in arrays(f)))
    np.testing.assert_array_equal(result[outside],fallback)
    assert _utci_points_serial.nopython_signatures and _utci_points_serial.targetoptions['fastmath'] is False


def test_scalar_polynomial_separately_with_original_vapour_pressure():
    f=values();valid=domain(f)
    result=np.array([polynomial_scalar(np.float32(t-a),a,w,p) for a,t,w,p in zip(f['Ta'][valid],f['Tmrt'][valid],f['wind'][valid],f['Pa'][valid])],np.float32)
    compare(result,f['polynomial'][valid])


def test_scalar_saturation_separately_with_original_humidity_intermediate():
    f=values();valid=domain(f)
    result=np.array([saturation_scalar(t) for t in f['Ta'][valid]],np.float32)
    # This is an intermediate diagnostic, not a replacement for the full 0.02C gate.
    assert np.isfinite(result).all()
    relative=np.abs(result-f['saturation'][valid])/f['saturation'][valid]
    assert relative.max()<2e-5


@pytest.mark.parametrize('threads',[1,4,10])
def test_parallel_pixel_ownership_determinism_and_original_parity(threads):
    f=values();previous=get_num_threads()
    try:
        set_num_threads(threads)
        with np.errstate(all='ignore'):
            serial=utci_calculator_compiled(*arrays(f))
            parallel=utci_calculator_compiled(*arrays(f),parallel=True)
        np.testing.assert_array_equal(parallel,serial)
        compare(parallel,f['UTCI'],domain(f))
    finally:set_num_threads(previous)
    assert _utci_points_parallel.nopython_signatures
    assert _utci_points_parallel.targetoptions['fastmath'] is False


def test_numba_types_and_parallel_diagnostics():
    f=values()
    with np.errstate(all='ignore'):utci_calculator_compiled(*(x[:2] for x in arrays(f)),parallel=True)
    text=io.StringIO();_utci_points_parallel.inspect_types(file=text)
    assert 'pyobject' not in text.getvalue()
    # Recompile to retain diagnostic metadata when a dispatcher was loaded from cache.
    _utci_points_parallel.recompile()
    _utci_points_parallel.parallel_diagnostics(level=2)


@pytest.mark.parametrize('profile',[f'mixed_{mask:04b}' if mask!=9 else 'mixed' for mask in range(1,15)]+['float64'])
def test_characterized_double_assignment_failure_is_preserved(profile):
    manifest=json.loads((REFERENCE/'manifest.json').read_text())
    outcome=manifest['typed_outcomes'][profile]
    assert outcome['status']=='failed_original_upstream'
    assert 'Index put requires the source and destination dtypes match' in outcome['traceback']
    with np.load(REFERENCE/(profile+'.npz'),allow_pickle=False) as f:
        inputs=arrays(f)
        with pytest.raises(RuntimeError,match='Float for the destination and Double for the source'):utci_calculator(*inputs)
        with pytest.raises(RuntimeError,match='Float for the destination and Double for the source'):utci_calculator_compiled(*inputs)
        # The original direct polynomial remains supported and separately captured.
        valid=domain(f)
        result=utci_polynomial(inputs[2][valid]-inputs[0][valid],inputs[0][valid],inputs[3][valid],f['Pa'][valid])
        compare(result,f['polynomial'][valid],budget=.02)


def raster(path):
    ds=gdal.Open(str(path));a=ds.ReadAsArray();ds=None;return a


@pytest.mark.parametrize('case',CASES)
def test_serial_and_parallel_original_packaged_rasters(case):
    base=ROOT/'tests/reference/optional_original_cpu'/case;out=base/'output_folder/0_0'
    forcing=np.loadtxt(base/'processed_inputs/metfiles/metfile_0_0_2020-07-18.txt',skiprows=1)
    ta=raster(out/'Ta_0_0.tif');tmrt=raster(out/'TMRT_0_0.tif');wind=raster(out/'Wind_0_0.tif')
    rh=np.broadcast_to(forcing[:,10].astype(np.float32)[:,None,None],ta.shape)
    original=raster(out/'UTCI_0_0.tif')
    dsm=raster(base/'processed_inputs/Building_DSM/Building_DSM_0_0.tif');dem=raster(base/'processed_inputs/DEM/DEM_0_0.tif')
    buildings=dsm-dem;buildings[buildings<2]=1;buildings[buildings>=2]=0
    with np.errstate(all='ignore'):
        result=utci_calculator_compiled(ta,rh,tmrt,wind)
        parallel=utci_calculator_compiled(ta,rh,tmrt,wind,parallel=True)
    np.testing.assert_array_equal(result,parallel)
    result[:,buildings!=1]=np.nan
    compare(result,original)


@pytest.mark.parametrize('case',CASES)
def test_uniform_forcing_original_fields_and_variable_loop(case):
    from solweig_light.comfort.utci import utci_calculator_uniform
    base=ROOT/'tests/reference/optional_original_cpu'/case;out=base/'output_folder/0_0'
    forcing=np.loadtxt(base/'processed_inputs/metfiles/metfile_0_0_2020-07-18.txt',skiprows=1)
    ta=raster(out/'Ta_0_0.tif');tmrt=raster(out/'TMRT_0_0.tif');wind=raster(out/'Wind_0_0.tif')
    result=np.empty_like(tmrt)
    for i in range(len(tmrt)):
        humidity=np.float32(forcing[i,10])
        result[i]=utci_calculator_uniform(np.float32(ta[i,0,0]),humidity,tmrt[i],wind[i])
        variable=utci_calculator_compiled(ta[i],np.full_like(ta[i],humidity),tmrt[i],wind[i])
        parallel=utci_calculator_uniform(np.float32(ta[i,0,0]),humidity,tmrt[i],wind[i],parallel=True)
        np.testing.assert_array_equal(result[i],variable)
        np.testing.assert_array_equal(result[i],parallel)
    buildings=raster(base/'processed_inputs/Building_DSM/Building_DSM_0_0.tif')-raster(base/'processed_inputs/DEM/DEM_0_0.tif')
    buildings[buildings<2]=1;buildings[buildings>=2]=0;result[:,buildings!=1]=np.nan
    compare(result,raster(out/'UTCI_0_0.tif'))


@pytest.mark.parametrize('ta,rh',[(np.nan,50),(25,np.nan),(-999,50),(25,-999),(-np.inf,50),(np.inf,50),(25,np.inf)])
def test_uniform_scalar_nonfinite_and_sentinel_policy(ta,rh):
    from solweig_light.comfort.utci import utci_calculator_uniform
    ta,rh=np.float32(ta),np.float32(rh)
    tmrt=np.array([30,-999,np.nan,30],np.float32);wind=np.array([1,1,1,-999],np.float32)
    with np.errstate(all='ignore'):
        actual=utci_calculator_uniform(ta,rh,tmrt,wind)
        reference=utci_calculator(np.full_like(tmrt,ta),np.full_like(tmrt,rh),tmrt,wind)
    np.testing.assert_array_equal(actual,reference)


@pytest.mark.parametrize('threads',[1,4,10])
def test_uniform_parallel_thread_determinism(threads):
    from solweig_light.comfort.utci import utci_calculator_uniform
    f=values();tmrt=f['Tmrt'];wind=f['wind'];previous=get_num_threads()
    try:
        set_num_threads(threads)
        with np.errstate(all='ignore'):
            serial=utci_calculator_uniform(np.float32(25),np.float32(50),tmrt,wind)
            parallel=utci_calculator_uniform(np.float32(25),np.float32(50),tmrt,wind,parallel=True)
        np.testing.assert_array_equal(serial,parallel)
    finally:set_num_threads(previous)


def test_generated_scalar_reconstructs_original_written_expression():
    class UnwrapFloat32(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='np' and node.func.attr=='float32':
                assert len(node.args)==1
                return self.visit(node.args[0])
            raise AssertionError('Unexpected generated scalar function call')
    tree=ast.parse((ROOT/'src/solweig_light/comfort/_utci_scalar.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='polynomial_scalar')
    assignments=[n for n in function.body if isinstance(n,ast.Assign)]
    expression=UnwrapFloat32().visit(assignments[0].value)
    for assignment in assignments[1:]:
        operation=UnwrapFloat32().visit(assignment.value)
        assert isinstance(operation,ast.BinOp) and isinstance(operation.op,ast.Add)
        assert isinstance(operation.left,ast.Name) and operation.left.id=='result'
        expression=ast.BinOp(expression,ast.Add(),operation.right)
    assert hashlib.sha256(ast.dump(expression).encode()).hexdigest()=='33cec1d5f69ef606d602e0e25d45c462701cc049e90a91d8f704f65b2dfd4907'
