"""Original ground-view oracle, including observable mutation and failures."""
import hashlib
import json
from pathlib import Path

import numba
import numpy as np
import pytest
from solweig_light.radiation import ground_view

REFERENCE=Path(__file__).parents[1]/'reference/ground_view_original_cpu'
MANIFEST=json.loads((REFERENCE/'manifest.json').read_text())
CASES=MANIFEST['cases']
assert MANIFEST['evidence_class']=='original_upstream_cpu'
assert MANIFEST['source_commit']=='0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
assert MANIFEST['patch'] is None and not MANIFEST['candidate_import']


def load_artifact(case,key):
    artifact=case[key];path=REFERENCE/artifact['path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==artifact['sha256']
    with np.load(path,allow_pickle=False) as arrays:return dict(arrays)


def assert_exact(value,expected):
    assert value.shape==expected.shape
    assert value.dtype==expected.dtype
    np.testing.assert_array_equal(value,expected)
    finite=np.isfinite(expected)
    np.testing.assert_array_equal(np.signbit(value[finite]),np.signbit(expected[finite]))


@pytest.mark.parametrize('case',CASES,ids=[case['name'] for case in CASES])
@pytest.mark.parametrize('threads',[None,1,4,10],ids=['serial','parallel1','parallel4','parallel10'])
def test_original_ground_view(case,threads):
    before=load_artifact(case,'input_artifact')
    inputs={key:value[()] if value.ndim==0 else value for key,value in before.items()}
    after=load_artifact(case,'after_artifact')
    previous=numba.get_num_threads()
    try:
        if threads is not None:numba.set_num_threads(threads)
        function=getattr(ground_view,case['function']+('_parallel' if threads is not None else ''))
        with np.errstate(invalid='ignore',divide='ignore'):
            if case['status']=='failed':
                exception={'RuntimeError':RuntimeError,'UnboundLocalError':UnboundLocalError}[case['exception']]
                with pytest.raises(exception):function(**inputs)
            else:
                actual=function(**inputs)
                golden=load_artifact(case,'output_artifact')
                assert len(actual)==len(golden)
                fluxes=[1] if case['function']=='sunonsurface_2018a' else [0,3,6,9,12]
                for index,value in enumerate(actual):
                    expected=golden[f'output_{index}']
                    assert value.dtype==expected.dtype
                    assert value.shape==expected.shape
                    for mask in (np.isnan,np.isposinf,np.isneginf):
                        np.testing.assert_array_equal(mask(value),mask(expected))
                    if index in fluxes:
                        np.testing.assert_allclose(value,expected,atol=.05,rtol=1e-5,equal_nan=True)
                    else:
                        # Stronger than the frozen 1e-6 gate on these fixtures.
                        assert_exact(value,expected)
        # Verify all original before/after inputs, including failed invocations.
        # Only Tg/class3 and normalized sunwall may change, exactly as upstream.
        for key,expected in after.items():assert_exact(np.asarray(inputs[key]),expected)
    finally:
        numba.set_num_threads(previous)


def test_cached_descriptors_are_bounded_immutable_and_field_free():
    ground_view.clear_schedule_cache()
    args=((17,23),np.float64(np.pi/4),np.float32(1),np.float32(1),np.float32(14))
    first=ground_view.ray_schedule(*args)
    assert first is ground_view.ray_schedule(*args)
    assert ground_view._schedule_cached.cache_info().maxsize==36
    assert first.shape==(14,8)
    assert first.nbytes<=ground_view.MAX_CACHED_SCHEDULE_BYTES
    assert not first.flags.writeable
    with pytest.raises(ValueError):first.setflags(write=True)
    with pytest.raises(ValueError):first[0,0]=9
    changed=(args[0],args[1],args[2],args[3],np.float32(13))
    assert ground_view.ray_schedule(*changed) is not first


def test_changed_dynamic_fields_reuse_addresses_without_reusing_results():
    case=next(case for case in CASES if case['name']=='sun_direction_45')
    original=load_artifact(case,'input_artifact')
    def fresh():return {key:value[()] if value.ndim==0 else value.copy() for key,value in original.items()}
    first=fresh();second=fresh()
    second['shadow'].fill(0);second['Tg'].fill(2);second['sunwall'].fill(0)
    ground_view.clear_schedule_cache()
    a=ground_view.sunonsurface_2018a(**first)
    initial=ground_view._schedule_cached.cache_info()
    b=ground_view.sunonsurface_2018a(**second)
    updated=ground_view._schedule_cached.cache_info()
    assert updated.hits==initial.hits+1
    assert any(not np.array_equal(x,y,equal_nan=True) for x,y in zip(a,b))
