import json
import platform

import numpy as np
import pytest

from solweig_light.cache import ARRAY_FIELDS, VISIBILITY_FIELDS, GeometryStore
from solweig_light.geometry.visibility import PackedVisibility
from solweig_light.io.rasters import RasterMetadata
from solweig_light.persistence import PersistenceError, TransactionalOutputs
from solweig_light.radiation import _math_profile as profile


def geometry_fixture():
    plane = np.zeros((2, 3), np.float32)
    cube = np.zeros((2, 3, 2), np.float32)
    return {**{name: plane.copy() for name in ARRAY_FIELDS},
            **{name: PackedVisibility.from_dense(cube) for name in VISIBILITY_FIELDS}}


def test_profile_identity_is_explicit_complete_and_deterministic():
    first = profile.profile_identity()
    second = profile.profile_identity()
    assert first == second
    assert first['id'] == 'solweig-portable-sleef-5a1d179d-v1'
    assert first['implementation']['sleef_commit'] == '5a1d179df9cf652951b59010a2d2075372d67f68'
    assert first['implementation']['coefficient_policy'] == 'numpy-float64-cos-tan-scalar-v1'
    assert set(first['implementation']['source_sha256']) == set(profile.SOURCE_FILES)
    assert first['runtime']['fastmath'] is False
    assert first['runtime']['fma'] == 'explicit-llvm.fma.f32'


def test_runtime_target_changes_fingerprint(monkeypatch):
    before = profile.profile_identity()
    monkeypatch.setattr(platform, 'machine', lambda: 'different-isa')
    after = profile.profile_identity()
    assert before['id'] == after['id']
    assert before['fingerprint'] != after['fingerprint']


def test_float32_tan_complete_accepted_domain_and_explicit_fallback():
    values = np.array([-np.inf, -10000, -125, -124.999, -0.0, 0.0, 124.999, 125, 10000, np.inf, np.nan], np.float32)
    with np.errstate(all='ignore'):
        actual = profile.tan32(values)
        legacy = np.tan(values)
    fallback = ~np.isfinite(values) | (np.abs(values) >= np.float32(125))
    np.testing.assert_array_equal(np.isnan(actual[fallback]), np.isnan(legacy[fallback]))
    finite = fallback & np.isfinite(actual) & np.isfinite(legacy)
    np.testing.assert_array_equal(actual[finite].view(np.uint32), legacy[finite].view(np.uint32))
    assert np.signbit(actual[4]) and not np.signbit(actual[5])


@pytest.mark.parametrize('sign', [0x00000000, 0x80000000])
def test_float32_tan_wide_fallback_matches_numpy_vector_bits(sign):
    rng = np.random.default_rng(20260920)
    bits = rng.integers(0x42FA0000, 0x7F800000, 100_000, dtype=np.uint32)
    values = (bits | np.uint32(sign)).view(np.float32)
    expected = np.tan(values)
    actual = profile.tan32(values)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_float32_tan_mixed_strided_scalar_and_nonfinite_dispatch():
    from solweig_light.radiation._sleef_classifier import tan_array

    rng = np.random.default_rng(20260920)
    values = rng.integers(0, 0x100000000, 200_000, dtype=np.uint32).view(np.float32)
    values = values.reshape(400, 500)[::3, 1::4]
    assert not values.flags.c_contiguous
    fast = np.isfinite(values) & (np.abs(values) < np.float32(125.0))
    with np.errstate(all='ignore'):
        expected_fallback = np.tan(values)
        actual = profile.tan32(values)
    np.testing.assert_array_equal(actual[~fast].view(np.uint32), expected_fallback[~fast].view(np.uint32))
    np.testing.assert_array_equal(actual[fast].view(np.uint32), tan_array(values[fast].ravel()).view(np.uint32))
    for scalar in (np.float32(0.5), np.float32(10_000), np.float32(np.inf), np.float32(np.nan)):
        with np.errstate(all='ignore'):
            got = profile.tan32(scalar)
            legacy = np.tan(np.asarray(scalar))
        assert np.asarray(got).shape == ()
        if np.isnan(legacy):
            assert np.isnan(got)
        elif abs(scalar) >= np.float32(125.0):
            assert np.asarray(got).view(np.uint32) == np.asarray(legacy).view(np.uint32)


def test_promoted_inputs_preserve_numpy_path_and_layout():
    base = np.linspace(-2, 2, 18, dtype=np.float64).reshape(3, 6)[:, ::2]
    assert not base.flags.c_contiguous
    np.testing.assert_array_equal(profile.tan32(base), np.tan(base))
    np.testing.assert_array_equal(profile.atan32(base), np.arctan(base))
    svf = np.array([0.0, .25, 1.0], np.float64)
    np.testing.assert_array_equal(profile.asvf(svf), np.arccos(np.sqrt(svf)))


def test_asvf_invalid_and_nonfinite_masks_match_legacy():
    values = np.array([-np.inf, -1, -0.0, 0.0, .25, 1.0, np.inf, np.nan], np.float32)
    with np.errstate(all='ignore'):
        actual = profile.asvf(values)
        legacy = np.arccos(np.sqrt(values))
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(legacy))
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(legacy))
    assert actual.dtype == np.float32 and actual.shape == values.shape


def test_profile_change_creates_fresh_geometry_generation(tmp_path):
    store = GeometryStore(tmp_path)
    identity = {'math_profile': profile.profile_identity(), 'scene': 'same'}
    first = store.get_or_create(identity, geometry_fixture)
    first_key = first.key
    first.close()
    changed = json.loads(json.dumps(identity))
    changed['math_profile']['fingerprint'] = '0' * 64
    called = []
    with store.get_or_create(changed, lambda: (called.append(True) or geometry_fixture())) as second:
        assert not second.hit and called == [True]
        assert second.key != first_key
    assert (tmp_path / first_key / 'manifest.json').exists()


def test_geometry_manifest_without_profile_is_stale(tmp_path):
    store = GeometryStore(tmp_path)
    legacy = {'scene': 'same'}
    with store.get_or_create(legacy, geometry_fixture) as old:
        old_key = old.key
    current = {'scene': 'same', 'math_profile': profile.profile_identity()}
    produced = []
    with store.get_or_create(current, lambda: (produced.append(True) or geometry_fixture())) as fresh:
        assert produced == [True]
        assert not fresh.hit and fresh.key != old_key


def test_checkpoint_resume_rejects_profile_mismatch(tmp_path):
    metadata = RasterMetadata(1, 1, (0., 1., 0., 0., 0., -1.), '')
    met = np.zeros((1, 4))
    common = dict(directory=tmp_path/'out', tile='0_0', metadata=metadata,
                  met=met, selected_date='2020-01-01', fields=('UTCI',),
                  transaction_dir=tmp_path/'transactions')
    first = {'scene': 'same', 'math_profile': profile.profile_identity()}
    with TransactionalOutputs(identity=first, resume=False, **common):
        pass
    changed = json.loads(json.dumps(first)); changed['math_profile']['fingerprint'] = 'f' * 64
    with pytest.raises(PersistenceError, match='identity'):
        TransactionalOutputs(identity=changed, resume=True, **common)


def test_checkpoint_without_profile_cannot_resume_profiled_run(tmp_path):
    metadata = RasterMetadata(1, 1, (0., 1., 0., 0., 0., -1.), '')
    common = dict(directory=tmp_path/'out', tile='0_0', metadata=metadata,
                  met=np.zeros((1, 4)), selected_date='2020-01-01', fields=('UTCI',),
                  transaction_dir=tmp_path/'transactions')
    with TransactionalOutputs(identity={'scene': 'same'}, resume=False, **common):
        pass
    current = {'scene': 'same', 'math_profile': profile.profile_identity()}
    with pytest.raises(PersistenceError, match='identity'):
        TransactionalOutputs(identity=current, resume=True, **common)

def test_lowered_helpers_use_explicit_fma_without_fast_flags():
    from solweig_light.radiation._sleef_acos import asvf_fma
    from solweig_light.radiation._sleef_classifier import tan_array, atan_array
    asvf_fma(np.array([.25], np.float32)); tan_array(np.array([.25], np.float32)); atan_array(np.array([.25], np.float32))
    for dispatcher in (asvf_fma, tan_array, atan_array):
        llvm = dispatcher.inspect_llvm(dispatcher.signatures[0])
        assert 'llvm.fma.f32' in llvm
        assert ' fast ' not in llvm

def test_simulation_identity_embeds_full_profile(tmp_path):
    from solweig_light.identities import simulation_identity
    met = tmp_path / 'met.txt'; met.write_text('fixture')
    identity = simulation_identity({'metfiles': met}, {}, '2020-01-01', '0_0',
                                   {'utci': True}, {'lat': 1.0, 'lon': 2.0}, 0.0)
    assert identity['math_profile'] == profile.profile_identity()
