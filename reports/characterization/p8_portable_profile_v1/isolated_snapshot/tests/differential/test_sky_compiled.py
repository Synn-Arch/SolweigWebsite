"""Compiled sky recurrence: original oracle plus labeled P1 diagnostics."""
import hashlib
import json
from pathlib import Path

import numba
import numpy as np
import pytest
from solweig_light.geometry.sky_compiled import (shadow_serial, shadow_parallel,
                                                shadow_pixel_serial, shadow_pixel_parallel)
from solweig_light.geometry import shadows as numpy_module

REFERENCE=Path(__file__).parents[1]/'reference/geometry_original_cpu'
MANIFEST=json.loads((REFERENCE/'reference_manifest.json').read_text())
CASES=[case for case in MANIFEST['cases'] if case['function']=='shadow']
assert MANIFEST['evidence_class']=='original_upstream_cpu'
assert MANIFEST['source_commit']=='0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
assert MANIFEST['oracle_patch_hash'] is None


def assert_exact(actual,expected):
    assert actual.dtype==expected.dtype
    assert actual.shape==expected.shape
    np.testing.assert_array_equal(actual,expected)
    for mask in (np.isnan,np.isposinf,np.isneginf):
        np.testing.assert_array_equal(mask(actual),mask(expected))
    finite=np.isfinite(expected)
    np.testing.assert_array_equal(np.signbit(actual[finite]),np.signbit(expected[finite]))


@pytest.mark.parametrize('variant',['step','pixel'])
@pytest.mark.parametrize('case',CASES,ids=[case['name'] for case in CASES])
@pytest.mark.parametrize('threads',[None,1,4,10],ids=['serial','parallel1','parallel4','parallel10'])
def test_original_sky_reference(case,threads,variant):
    input_path=REFERENCE/case['input_artifact']['path']
    output_path=REFERENCE/case['output_artifact']['path']
    for path,key in [(input_path,'input_artifact'),(output_path,'output_artifact')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==case[key]['sha256']
    with np.load(input_path,allow_pickle=False) as saved, np.load(output_path,allow_pickle=False) as golden:
        inputs=dict(saved)
        for key in ('azimuth','altitude','scale'):inputs[key]=float(inputs[key])
        snapshots={key:np.asarray(value).copy() for key,value in inputs.items()}
        previous=numba.get_num_threads()
        try:
            if threads is not None:numba.set_num_threads(threads)
            kernel=({'step':shadow_serial,'pixel':shadow_pixel_serial} if threads is None else
                    {'step':shadow_parallel,'pixel':shadow_pixel_parallel})[variant]
            values=kernel(**inputs)
            for name,value in zip(['sh','vegsh','vbshvegsh'],values,strict=True):assert_exact(value,golden[name])
            for key,before in snapshots.items():assert_exact(np.asarray(inputs[key]),before)
        finally:
            numba.set_num_threads(previous)


@pytest.mark.parametrize('variant',['step','pixel'])
@pytest.mark.parametrize('kind',['fractional_bush','nan_dsm','nan_canopy','nan_bush','infinite_bush','signed_zero','negative_altitude','zero_altitude','zero_scale','nan_bound'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('threads',[None,1,4,10])
def test_additional_edges_against_p1_numpy_diagnostic(kind,dtype,threads,variant):
    # Secondary evidence only: this compares the established P1 NumPy port,
    # not an original upstream oracle and not a scientific expectation.
    reference=getattr(numpy_module,'shadow_numpy',numpy_module.shadow)
    assert reference.__module__!='solweig_light.geometry.sky_compiled'
    a=np.full((9,11),-2,dtype=dtype);a[3:6,4:7]=4
    canopy=np.zeros_like(a);canopy[2:7,3:8]=9
    trunk=np.zeros_like(a);trunk[2:7,3:8]=2
    bush=np.zeros_like(a)
    if kind=='fractional_bush':bush[2:7,3:8]=1.00001
    if kind=='nan_dsm':a[4,5]=np.nan
    if kind=='nan_canopy':canopy[4,5]=np.nan
    if kind=='nan_bush':bush[4,5]=np.nan;bush[3,4]=3
    if kind=='infinite_bush':bush[4,5]=np.inf
    if kind=='signed_zero':a.fill(0);a[::2]=-0.;canopy.fill(-0.);trunk.fill(0.)
    inputs=dict(amaxvalue=np.asarray(np.nan if kind=='nan_bound' else 9,dtype=dtype),
                a=a,vegdem=canopy,vegdem2=trunk,bush=bush,
                azimuth=45.,altitude=-.5 if kind=='negative_altitude' else 0. if kind=='zero_altitude' else 35.,
                scale=0. if kind=='zero_scale' else 1.)
    previous=numba.get_num_threads()
    try:
        if threads is not None:numba.set_num_threads(threads)
        kernel=({'step':shadow_serial,'pixel':shadow_pixel_serial} if threads is None else
                    {'step':shadow_parallel,'pixel':shadow_pixel_parallel})[variant]
        with np.errstate(invalid='ignore',divide='ignore'):
            expected=reference(**inputs)
            actual=kernel(**inputs)
        for value,gold in zip(actual,expected,strict=True):assert_exact(value,gold)
    finally:
        numba.set_num_threads(previous)
