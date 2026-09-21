"""Original-only wall-shadow packet and frozen chronological boundary checks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import numba
import pytest
from solweig_light.radiation import wall_shadows as compiled

ROOT=Path(__file__).resolve().parents[1]/'reference/wall_shadows_original_cpu'
MANIFEST=json.loads((ROOT/'manifest.json').read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(case):
    path=ROOT/case['input']
    assert digest(path)==case['input_sha256']
    with np.load(path) as data:
        inputs={k:data[k].copy() for k in data.files}
    for key,kind in case['scalar_types'].items():
        inputs[key]=inputs[key][()] if kind=='float64' else inputs[key].item()
    return inputs


def check(actual,expected,family):
    categories={0,3,4} if family==13 else {0,1,2,6,7}
    for index,(a,b) in enumerate(zip(actual,expected,strict=True)):
        assert a.shape==b.shape
        assert a.dtype==b.dtype
        for mask in (np.isnan,np.isposinf,np.isneginf):
            np.testing.assert_array_equal(mask(a),mask(b))
        if index in categories:
            np.testing.assert_array_equal(a,b,err_msg=f'family{family} output{index}')
        else:
            np.testing.assert_allclose(a,b,atol=1e-6,rtol=0,equal_nan=True,err_msg=f'family{family} output{index}')


@pytest.mark.parametrize('case',MANIFEST['cases'],ids=lambda case:case['name'])
@pytest.mark.parametrize('threads',[0,1,4,10],ids=['serial','one-thread','four-threads','ten-threads'])
def test_original_wall_shadow_packet(case,threads):
    assert MANIFEST['evidence_class']=='original_upstream_cpu'
    assert MANIFEST['upstream_patch_hash'] is None
    assert MANIFEST['upstream_commit']=='0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
    # Frozen original source identity; candidate tests need no upstream checkout.
    assert MANIFEST['source_sha256']=='1ac27bcd56edcd690d11fce756bedc17cc53d1259e0fe60313e868921a28095c'
    inputs=load(case)
    function=getattr(compiled,f'exact_{case["family"]}')
    previous=numba.get_num_threads()
    try:
        if threads:numba.set_num_threads(threads)
        if case['status']=='original_failure':
            with pytest.raises(getattr(__import__('builtins'),case['exception'])):
                function(**inputs,parallel=bool(threads))
            return
        actual=function(**inputs,parallel=bool(threads))
    finally:
        numba.set_num_threads(previous)
    output=ROOT/case['output']
    assert digest(output)==case['output_sha256']
    with np.load(output) as expected:
        check(actual,[expected[k] for k in expected.files],case['family'])


def test_parallel_diagnostics(capsys):
    for family in (13,23):
        case=next(c for c in MANIFEST['cases'] if c['family']==family and c['status']=='captured')
        getattr(compiled,f'exact_{family}')(**load(case))
        kernel=getattr(compiled,f'_wall{family}')
        kernel.parallel_diagnostics(level=4)
        output=capsys.readouterr().out
        assert 'Parallel' in output
        assert kernel.nopython_signatures

BOUNDARIES=ROOT.parent/'small_original_cpu/boundaries'
EVENTS=json.loads((BOUNDARIES/'manifest.json').read_text())['events']
WALL_EVENTS=[event for event in EVENTS if event['function']=='shadowingfunction_wallheight_23' and event['boundary']=='output']

@pytest.mark.parametrize('event',WALL_EVENTS,ids=lambda event:f'step-{event["timestep"]}')
@pytest.mark.parametrize('threads',[0,1,4,10])
def test_original_chronological_wall_boundaries(event,threads):
    from solweig_light.radiation import engine
    input_event=next(item for item in EVENTS if item['function']=='Solweig_2022a_calc' and item['boundary']=='input' and item['timestep']==event['timestep'])
    assert digest(BOUNDARIES/input_event['path'])==input_event['sha256']
    with np.load(BOUNDARIES/input_event['path']) as data:
        inputs={name:data[name].copy() for name in ('vegdem','vegdem2','bush','walls')}
        inputs['a']=data['dsm'].copy()
        inputs.update(azimuth=data['azimuth'].item(),altitude=data['altitude'].item(),scale=data['scale'].item(),amaxvalue=data['amaxvalue'].item())
        inputs['aspect']=engine._divide(engine._operate(np.multiply,data['dirwalls'],np.pi),180.)
    previous=numba.get_num_threads()
    try:
        if threads:numba.set_num_threads(threads)
        actual=compiled.exact_23(**inputs,parallel=bool(threads))
    finally:
        numba.set_num_threads(previous)
    assert digest(BOUNDARIES/event['path'])==event['sha256']
    with np.load(BOUNDARIES/event['path']) as expected:
        check(actual,[expected[name] for name in event['fields']],23)
