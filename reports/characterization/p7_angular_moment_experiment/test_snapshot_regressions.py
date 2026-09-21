import json
from pathlib import Path

import numpy as np
import pytest

from solweig_light.radiation import patch_radiation as candidate
from solweig_light.radiation.angular_moments import build_bundle, validate_bundle, TileMomentOwner
from tools.experiments.p7_angular_moment import load_packet

ROOT=Path(__file__).resolve().parents[3]
PACKET_ROOT=ROOT/'tests/reference/patch_radiation_original_cpu'
MANIFEST=json.loads((PACKET_ROOT/'manifest.json').read_text())
CASE=next(c for c in MANIFEST['cases'] if c['status']=='captured' and c['function']=='Lcyl_v2022a')


def fixture():
    values=load_packet(CASE,MANIFEST)
    geometry=candidate.patch_geometry(values['sky_patches'])
    sources=[]
    for name in ('shmat','vegshmat','vbshvegshmat'):
        original=values[name]
        value=np.frombuffer(original.tobytes(order='C'),dtype=np.float32).reshape(original.shape)
        values[name]=value
        sources.append(value)
    bundle=build_bundle(*sources,geometry.solid_angle,geometry.sine,geometry.cosine,
        geometry.longwave_cardinal_cosine,geometry.reflection_cardinal,
        logical_shape=(values['rows'],values['cols']),profile='strict-f32-v1',
        source_fingerprint='p7-angular-snapshot-v2')
    return values,geometry,bundle


@pytest.mark.parametrize('parallel',[False,True])
@pytest.mark.parametrize('mode',['nan','inf','high','signed_zero'])
def test_unsafe_per_pixel_state_uses_exact_original_sweep(parallel,mode):
    values,_,bundle=fixture()
    if mode=='nan': values['Lup']=values['Lup'].copy();values['Lup'].flat[0]=np.nan
    elif mode=='inf': values['Lup']=values['Lup'].copy();values['Lup'].flat[0]=np.inf
    elif mode=='high': values['Lup']=values['Lup'].copy();values['Lup'].flat[0]=np.finfo(np.float32).max
    else: values['ewall']=np.float32(1)  # reflection factor +0, including signed-zero-sensitive accumulation
    with np.errstate(all='ignore'):
        expected=candidate.Lcyl_v2022a(**values,parallel=parallel,block_pixels=7)
        actual=candidate.Lcyl_v2022a(**values,parallel=parallel,block_pixels=7,_longwave_moments=bundle)
    for a,b in zip(actual,expected,strict=True):
        if mode=='signed_zero':
            np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
        else:
            np.testing.assert_array_equal(a.reshape(-1)[:1].view(np.uint32),b.reshape(-1)[:1].view(np.uint32))
            np.testing.assert_allclose(a,b,atol=.05,rtol=1e-5,equal_nan=True)


@pytest.mark.parametrize('parallel',[False,True])
def test_wrong_tile_same_shape_and_angles_is_rejected_and_falls_back(parallel):
    values,geometry,bundle=fixture()
    changed=values['shmat'].copy();changed.flat[0]=np.float32(1-changed.flat[0]);changed.setflags(write=False)
    values['shmat']=changed
    assert not validate_bundle(bundle,(values['rows'],values['cols']),changed,values['vegshmat'],
        values['vbshvegshmat'],geometry.solid_angle,geometry.sine,geometry.cosine,
        geometry.longwave_cardinal_cosine,geometry.reflection_cardinal,
        profile='strict-f32-v1',source_fingerprint='p7-angular-snapshot-v2')
    with np.errstate(all='ignore'):
        expected=candidate.Lcyl_v2022a(**values,parallel=parallel,block_pixels=7)
        actual=candidate.Lcyl_v2022a(**values,parallel=parallel,block_pixels=7,_longwave_moments=bundle)
    for a,b in zip(actual,expected,strict=True):np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))


def test_identity_shape_profile_immutability_and_dense_policy():
    values,geometry,bundle=fixture()
    with pytest.raises(ValueError):bundle.values.setflags(write=True)
    assert not validate_bundle(bundle,(1,values['rows']*values['cols']),values['shmat'],values['vegshmat'],
        values['vbshvegshmat'],geometry.solid_angle,geometry.sine,geometry.cosine,
        geometry.longwave_cardinal_cosine,geometry.reflection_cardinal,
        profile='strict-f32-v1',source_fingerprint='p7-angular-snapshot-v2')
    mutable=values['shmat'].copy()
    with pytest.raises(ValueError):build_bundle(mutable,values['vegshmat'],values['vbshvegshmat'],
        geometry.solid_angle,geometry.sine,geometry.cosine,geometry.longwave_cardinal_cosine,
        geometry.reflection_cardinal,logical_shape=(values['rows'],values['cols']))
    owner=np.array(values['shmat'],copy=True);alias=owner.view();alias.setflags(write=False)
    with pytest.raises(ValueError):build_bundle(alias,values['vegshmat'],values['vbshvegshmat'],
        geometry.solid_angle,geometry.sine,geometry.cosine,geometry.longwave_cardinal_cosine,
        geometry.reflection_cardinal,logical_shape=(values['rows'],values['cols']))
    owner.flat[0]=np.float32(1-owner.flat[0])


def test_integrated_owner_failure_cleanup_and_close():
    values,geometry,_=fixture();owner=TileMomentOwner()
    with pytest.raises(RuntimeError,match='injected'):
        owner.build(values['shmat'],values['vegshmat'],values['vbshvegshmat'],geometry.solid_angle,
            geometry.sine,geometry.cosine,geometry.longwave_cardinal_cosine,geometry.reflection_cardinal,
            block_pixels=1,fail_after_blocks=1,logical_shape=(values['rows'],values['cols']))
    assert owner.count==0 and owner.allocation_bytes==0
    owner.close();assert owner.closed and owner.count==0
