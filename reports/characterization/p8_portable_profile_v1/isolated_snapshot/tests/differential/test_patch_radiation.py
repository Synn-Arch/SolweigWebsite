"""Compiled patch reductions compared against original CPU boundary artifacts."""
import importlib.util
from pathlib import Path

import numba
import numpy as np
import pytest
from solweig_light.radiation import engine,patch_radiation as compiled
from solweig_light.geometry.visibility import PackedVisibility

_spec=importlib.util.spec_from_file_location('radiation_reference_helpers',Path(__file__).with_name('test_radiation_reference.py'))
_helpers=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
EVENTS=_helpers.MANIFEST['events']
DAYTIME=[event for event in EVENTS if event['boundary']=='input' and any(item['function']=='Kside_veg_v2022a' and item['timestep']==event['timestep'] for item in EVENTS)]


@pytest.mark.parametrize('event',DAYTIME,ids=lambda event:f'step-{event["timestep"]}')
@pytest.mark.parametrize('threads',[0,1,4,10])
@pytest.mark.parametrize('packed',[False,True])
def test_original_shortwave_boundaries(event,threads,packed,monkeypatch):
    reference=engine.Kside_veg_v2022a
    monkeypatch.setattr(engine,'Kside_veg_v2022a_reference',reference,raising=False)
    expected=next(item for item in EVENTS if item['function']=='Kside_veg_v2022a' and item['timestep']==event['timestep'])
    def call(*args,**kwargs):
        values=__import__('inspect').signature(reference).bind(*args,**kwargs).arguments
        if packed:
            for key in ('shmat','vegshmat','vbshvegshmat','diffsh'):
                values[key]=PackedVisibility.from_dense(values[key])
        previous=numba.get_num_threads()
        try:
            if threads:numba.set_num_threads(threads)
            output=compiled.Kside_veg_v2022a(**values,parallel=bool(threads),block_pixels=17)
        finally:numba.set_num_threads(previous)
        _helpers.compare(expected,output)
        return output
    monkeypatch.setattr(engine,'Kside_veg_v2022a',call)
    with np.errstate(all='ignore'):
        engine.Solweig_2022a_calc(**_helpers.load_input(event))

ALL_STEPS=[event for event in EVENTS if event['boundary']=='input']

@pytest.mark.parametrize('event',ALL_STEPS,ids=lambda event:f'step-{event["timestep"]}')
@pytest.mark.parametrize('threads',[0,1,4,10])
@pytest.mark.parametrize('packed',[False,True])
def test_original_longwave_boundaries(event,threads,packed,monkeypatch):
    reference=engine.Lcyl_v2022a
    patch_reference=engine.define_patch_characteristics
    monkeypatch.setattr(engine,'Lcyl_v2022a_reference',reference,raising=False)
    monkeypatch.setattr(engine,'define_patch_characteristics_reference',patch_reference,raising=False)
    expected=next(item for item in EVENTS if item['function']=='Lcyl_v2022a' and item['timestep']==event['timestep'])
    def call(*args,**kwargs):
        values=__import__('inspect').signature(reference).bind(*args,**kwargs).arguments
        if packed:
            for key in ('shmat','vegshmat','vbshvegshmat'):
                values[key]=PackedVisibility.from_dense(values[key])
        previous=numba.get_num_threads()
        try:
            if threads:numba.set_num_threads(threads)
            output=compiled.Lcyl_v2022a(**values,parallel=bool(threads),block_pixels=17)
        finally:numba.set_num_threads(previous)
        _helpers.compare(expected,output)
        return output
    monkeypatch.setattr(engine,'Lcyl_v2022a',call)
    with np.errstate(all='ignore'):
        engine.Solweig_2022a_calc(**_helpers.load_input(event))

PACKET_ROOT=_helpers.ROOT.parent.parent/'patch_radiation_original_cpu'
# Existing helper ROOT is small_original_cpu/boundaries; its parent.parent is reference.
PACKET=__import__('json').loads((PACKET_ROOT/'manifest.json').read_text())
CAPTURED=[case for case in PACKET['cases'] if case['status']=='captured']


def load_packet(case):
    import hashlib
    path=PACKET_ROOT/case['input']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==case['input_sha256']
    with np.load(path) as data:
        values={}
        for key,spec in case['fields'].items():
            if spec['kind']=='none':values[key]=None
            elif spec['kind'] in ('tensor','ndarray'):values[key]=data[key].copy()
            elif spec['kind']=='numpy_scalar':values[key]=data[key][()]
            else:values[key]=data[key].item()
    return values


@pytest.mark.parametrize('case',CAPTURED,ids=lambda case:case['label']+'-'+case['function'])
@pytest.mark.parametrize('threads',[0,1,4,10])
@pytest.mark.parametrize('packed',[False,True])
def test_original_typed_patch_packet(case,threads,packed):
    import hashlib
    values=load_packet(case)
    if packed:
        for key in ('shmat','vegshmat','vbshvegshmat','diffsh'):
            if key in values:values[key]=PackedVisibility.from_dense(values[key])
    previous=numba.get_num_threads()
    try:
        if threads:numba.set_num_threads(threads)
        with np.errstate(all='ignore'):
            actual=getattr(compiled,case['function'])(**values,parallel=bool(threads),block_pixels=7)
    finally:numba.set_num_threads(previous)
    path=PACKET_ROOT/case['output']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==case['output_sha256']
    with np.load(path) as data:
        for index,value in enumerate(actual):
            expected=data[f'output_{index}']
            assert value.shape==expected.shape
            for mask in (np.isnan,np.isposinf,np.isneginf):
                np.testing.assert_array_equal(mask(value),mask(expected))
            np.testing.assert_allclose(value,expected,atol=.05,rtol=1e-5,equal_nan=True)


def test_original_packet_provenance_and_failures():
    assert PACKET['evidence_class']=='original_upstream_cpu_instrumented'
    assert PACKET['upstream_commit']=='0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
    assert PACKET['upstream_patch_hash'] is None
    assert PACKET['source_sha256']=='1ac27bcd56edcd690d11fce756bedc17cc53d1259e0fe60313e868921a28095c'
    assert len(PACKET['failures'])==12
    assert all(failure['exception']=='TypeError' for failure in PACKET['failures'])
    assert sum(case['status']=='original_failure' for case in PACKET['cases'])==20


def test_geometry_cache_is_immutable_bounded_and_has_no_forcing():
    from dataclasses import FrozenInstanceError
    case=next(c for c in CAPTURED if c['function']=='Kside_veg_v2022a')
    table=load_packet(case)['lv']
    compiled.clear_geometry_cache()
    geometry=compiled.patch_geometry(table)
    changed=table.copy()
    changed[:,2]*=2
    assert compiled.patch_geometry(changed) is geometry
    for field in ('altitude','azimuth','solid_angle','sine','cosine','bands','band_counts','band_membership','diffuse_cardinal','reflection_cardinal','longwave_cardinal_cosine'):
        array=getattr(geometry,field)
        assert not array.flags.writeable
        with pytest.raises(ValueError):array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):geometry.altitude=np.zeros(1)
    for index in range(9):
        changed=table.copy();changed[:,1]+=np.float32(index*.01)
        compiled.patch_geometry(changed)
    assert compiled._cached_geometry.cache_info().currsize==8


@pytest.mark.parametrize('block_pixels',[1,7,128])
@pytest.mark.parametrize('function',['Kside_veg_v2022a','Lcyl_v2022a'])
def test_packed_visibility_never_decodes_a_full_plane_or_cube(block_pixels,function):
    case=next(c for c in CAPTURED if c['function']==function and c['label'].startswith('option2-binary'))
    values=load_packet(case)
    decoded=[]
    class Tracked:
        dtype=np.dtype(np.float32)
        def __init__(self,value):self.value=PackedVisibility.from_dense(value);self.shape=self.value.shape
        def decode_pixels(self,patch,start,stop):
            decoded.append(stop-start)
            assert stop-start<=block_pixels
            return self.value.decode_pixels(patch,start,stop)
        def __array__(self,*args,**kwargs):raise AssertionError('dense conversion')
        def __getitem__(self,index):raise AssertionError('full patch slice')
        def decode_patch(self,patch):raise AssertionError('full plane decode')
    for key in ('shmat','vegshmat','vbshvegshmat','diffsh'):
        if key in values:values[key]=Tracked(values[key])
    with np.errstate(all='ignore'):
        output=getattr(compiled,function)(**values,block_pixels=block_pixels,parallel=False)
    assert decoded
    with np.load(PACKET_ROOT/case['output']) as expected:
        for index,value in enumerate(output):np.testing.assert_allclose(value,expected[f'output_{index}'],atol=.05,rtol=1e-5,equal_nan=True)


def test_unsupported_float64_profile_retains_serial_reference():
    case=next(c for c in CAPTURED if c['function']=='Kside_veg_v2022a' and c['label'].startswith('option2-binary'))
    values=load_packet(case)
    values['shadow']=values['shadow'].astype(np.float64)
    with np.errstate(all='ignore'):
        expected=engine.Kside_veg_v2022a(**values)
        actual=compiled.Kside_veg_v2022a(**values)
    for a,b in zip(actual,expected,strict=True):
        assert a.dtype==b.dtype
        np.testing.assert_array_equal(a,b)


def test_compiled_patch_parallel_diagnostics(capsys):
    for function,kernel in [('Kside_veg_v2022a',compiled._shortwave),('Lcyl_v2022a',compiled._longwave)]:
        case=next(c for c in CAPTURED if c['function']==function)
        with np.errstate(all='ignore'):getattr(compiled,function)(**load_packet(case))
        kernel.parallel_diagnostics(level=4)
        output=capsys.readouterr().out
        assert 'Parallel' in output
        assert kernel.nopython_signatures
        assert kernel.targetoptions['fastmath'] is False

@pytest.mark.parametrize('option',[1,2,3,4])
def test_cached_model2_preserves_per_step_coefficient_arithmetic(option):
    case=next(c for c in CAPTURED if c['function']=='Lcyl_v2022a' and c['label']==f'option{option}-binary-longwave-tensor-solar')
    table=load_packet(case)['sky_patches']
    geometry=compiled.patch_geometry(table)
    for emissivity in (np.float32(.2),np.float32(.8),np.float32(1),np.float64(.8),np.float32(np.nan)):
        with np.errstate(all='ignore'):
            expected=engine.model2(table,emissivity,np.float32(24))
            actual=compiled._model2(geometry,emissivity)
        for a,b in zip(actual,expected,strict=True):
            assert a.dtype==b.dtype
            np.testing.assert_array_equal(a,b)


def test_warm_longwave_uses_cached_bands_and_fixed_cardinal_geometry(monkeypatch):
    case=next(c for c in CAPTURED if c['function']=='Lcyl_v2022a' and c['label']=='option2-binary-longwave-tensor-solar')
    values=load_packet(case)
    compiled.clear_geometry_cache()
    compiled.patch_geometry(values['sky_patches'])
    def unexpected(*args,**kwargs):raise AssertionError('per-step unique recomputation')
    monkeypatch.setattr(np,'unique',unexpected)
    with np.errstate(all='ignore'):
        actual=compiled.Lcyl_v2022a(**values,parallel=False)
    with np.load(PACKET_ROOT/case['output']) as data:
        for index,value in enumerate(actual):np.testing.assert_allclose(value,data[f'output_{index}'],atol=.05,rtol=1e-5,equal_nan=True)

FAILED=[case for case in PACKET['cases'] if case['status']=='original_failure']

@pytest.mark.parametrize('case',FAILED,ids=lambda case:case['label']+'-'+case['function'])
@pytest.mark.parametrize('parallel',[False,True])
def test_every_original_failed_frame_preserves_exception(case,parallel):
    values=load_packet(case)
    with pytest.raises(TypeError) as error:
        getattr(compiled,case['function'])(**values,parallel=parallel)
    assert ('where' if case['function']=='Kside_veg_v2022a' else 'tan') in str(error.value)


@pytest.mark.parametrize('altitude',[0.,-1.])
@pytest.mark.parametrize('function',['Lcyl_v2022a','define_patch_characteristics'])
def test_python_longwave_solar_is_not_rejected_when_night_branch_avoids_tan(altitude,function):
    case=next(c for c in CAPTURED if c['function']==function and c['label']=='option2-binary-longwave-tensor-solar')
    values=load_packet(case)
    values['solar_altitude']=altitude
    values['solar_azimuth']=180.
    with np.errstate(all='ignore'):
        expected=compiled._reference(function)(**values)
        actual=getattr(compiled,function)(**values,parallel=False)
    for a,b in zip(actual,expected,strict=True):
        np.testing.assert_allclose(a,b,atol=.05,rtol=1e-5,equal_nan=True)

@pytest.mark.parametrize('case',FAILED,ids=lambda case:case['label']+'-'+case['function'])
def test_engine_dispatch_preserves_every_original_failed_frame(case):
    with pytest.raises(TypeError) as error:
        getattr(engine,case['function'])(**load_packet(case))
    assert ('where' if case['function']=='Kside_veg_v2022a' else 'tan') in str(error.value)


@pytest.mark.parametrize('function',['Lcyl_v2022a','define_patch_characteristics'])
def test_native_positive_longwave_solar_with_no_active_patch_uses_reference_order(function):
    case=next(c for c in CAPTURED if c['function']==function and c['label']=='option2-binary-longwave-tensor-solar')
    values=load_packet(case)
    values['solar_altitude']=35.
    values['solar_azimuth']=0.
    for key in ('shmat','vegshmat','vbshvegshmat'):values[key]=values[key][:,:,-1:].copy()
    if function=='Lcyl_v2022a':values['sky_patches']=values['sky_patches'][-1:].copy()
    else:
        for key in ('patch_altitude','patch_azimuth','steradian'):values[key]=values[key][-1:].copy()
        for key in ('Lsky_down','Lsky_side','Lsky'):values[key]=values[key][-1:].copy()
    with np.errstate(all='ignore'):
        expected=compiled._reference(function)(**values)
        actual=getattr(compiled,function)(**values,parallel=False)
    for a,b in zip(actual,expected,strict=True):np.testing.assert_array_equal(a,b)


def test_python_box_azimuth_with_tensorlike_t_remains_valid():
    case=next(c for c in CAPTURED if c['function']=='Kside_veg_v2022a' and c['label']=='option2-binary-box-tensor-azimuth')
    values=load_packet(case)
    values['azimuth']=180.
    values['t']=np.asarray(0.,dtype=np.float32)
    with np.errstate(all='ignore'):actual=compiled.Kside_veg_v2022a(**values,parallel=False)
    with np.load(PACKET_ROOT/case['output']) as expected:
        for index,value in enumerate(actual):np.testing.assert_allclose(value,expected[f'output_{index}'],atol=.05,rtol=1e-5,equal_nan=True)


def test_cylinder_python_azimuth_remains_valid():
    case=next(c for c in CAPTURED if c['function']=='Kside_veg_v2022a' and c['label']=='option2-binary-cylinder')
    values=load_packet(case)
    assert isinstance(values['azimuth'],float)
    with np.errstate(all='ignore'):actual=compiled.Kside_veg_v2022a(**values,parallel=False)
    with np.load(PACKET_ROOT/case['output']) as expected:
        for index,value in enumerate(actual):np.testing.assert_allclose(value,expected[f'output_{index}'],atol=.05,rtol=1e-5,equal_nan=True)
