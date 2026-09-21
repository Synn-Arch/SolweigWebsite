import json
import multiprocessing
import os
from pathlib import Path
import time
import zipfile

import numpy as np
import pytest

from solweig_light.cache import (
    GeometryStore, content_fingerprint, array_fingerprint, raster_fingerprint,
    validate_legacy_geometry, load_legacy_geometry, ARRAY_FIELDS, VISIBILITY_FIELDS, SVF_FIELDS,
)
from solweig_light.cache import geometry as storage
from solweig_light.geometry.visibility import PackedVisibility, export_visibility_npz


def fixture():
    bits = np.array([0, 0x80000000, 0x7fc00023, 0x3eaaaaab, 0x3f800000, 0x40000000], dtype=np.uint32)
    dense = np.empty((2, 3, 3), dtype=np.float32)
    dense[:, :, 0] = np.arange(6).reshape(2, 3) % 2
    dense[:, :, 1] = np.arange(6).reshape(2, 3) % 3
    dense[:, :, 2] = bits.view(np.float32).reshape(2, 3)
    return {**{name: (bits.view(np.float32).reshape(2, 3).copy() if name == 'svf' else np.full((2, 3), i, np.float32))
               for i, name in enumerate(ARRAY_FIELDS)},
            **{name: PackedVisibility.from_dense(dense) for name in VISIBILITY_FIELDS}}


def assert_fields(actual, expected):
    assert set(actual) == set(expected)
    for name in ARRAY_FIELDS:
        np.testing.assert_array_equal(actual[name].view(np.uint32), expected[name].view(np.uint32))
    for name in VISIBILITY_FIELDS:
        np.testing.assert_array_equal(actual[name].to_dense().view(np.uint32), expected[name].to_dense().view(np.uint32))


def test_fingerprints_content_not_stat(tmp_path):
    path = tmp_path / 'input.bin'
    path.write_bytes(b'abc')
    stamp = path.stat()
    before = content_fingerprint(path, max_workspace_bytes=128)
    path.write_bytes(b'xyz')
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert path.stat().st_size == stamp.st_size
    assert path.stat().st_mtime_ns == stamp.st_mtime_ns
    assert content_fingerprint(path, max_workspace_bytes=128) != before
    a = np.arange(60, dtype=np.float32).reshape(6, 10)
    assert array_fingerprint(a.T, max_workspace_bytes=128) == array_fingerprint(a.T.copy(), max_workspace_bytes=128)
    assert array_fingerprint(a) != array_fingerprint(a.reshape(3, 20))
    assert array_fingerprint(np.zeros(2, np.float32)) != array_fingerprint(np.zeros(2, np.float64))
    assert array_fingerprint(np.array([0], np.uint32).view(np.float32)) != array_fingerprint(np.array([0x80000000], np.uint32).view(np.float32))


def test_roundtrip_owned_mappings_close(tmp_path):
    store = GeometryStore(tmp_path, max_workspace_bytes=128)
    source = fixture()
    with store.get_or_create({'source': 'a', 'policy': {'one': 1}}, lambda: source) as handle:
        assert not handle.hit
        assert_fields(handle.fields, source)
        mapping = handle.fields['svf']
        visibility = handle.fields['shmat']
        plane = visibility.decode_patch(2)
        assert not mapping.flags.writeable
        assert isinstance(mapping, np.memmap)
        assert mapping._mmap.closed is False
    assert handle.closed and handle.fields == {}
    assert mapping._mmap.closed and visibility.closed
    handle.close()
    with pytest.raises(RuntimeError):
        visibility.decode_patch(0)
    np.testing.assert_array_equal(plane.view(np.uint32), source['shmat'].decode_patch(2).view(np.uint32))
    with store.get_or_create({'policy': {'one': 1}, 'source': 'a'}, lambda: pytest.fail('producer on hit')) as hit:
        assert hit.hit
        assert_fields(hit.fields, source)
        assert hit.key == handle.key


@pytest.mark.parametrize('field', ['input', 'metadata', 'nodata', 'domain', 'normalization', 'patches', 'canopy', 'trunk', 'transmission', 'dtype', 'implementation', 'construction'])
def test_every_declared_dependency_affects_key(tmp_path, field):
    identity = {name: 'old' for name in ('input', 'metadata', 'nodata', 'domain', 'normalization', 'patches', 'canopy', 'trunk', 'transmission', 'dtype', 'implementation', 'construction')}
    store = GeometryStore(tmp_path)
    key = store.key_for(identity)
    identity[field] = 'new'
    assert store.key_for(identity) != key
    assert GeometryStore(tmp_path, model_version='different').key_for(identity) != store.key_for(identity)


@pytest.mark.parametrize('identity', [{1: 'value'}, {'a': np.float32(1)}, {'a': float('nan')}, {'a': object()}])
def test_non_json_identity_rejected(tmp_path, identity):
    with pytest.raises((TypeError, ValueError)):
        GeometryStore(tmp_path).get_or_create(identity, lambda: pytest.fail('invalid identity producer'))


@pytest.mark.parametrize('kind', ['missing_manifest', 'bad_json', 'stale_identity', 'bad_schema', 'missing_array', 'corrupt_array', 'missing_visibility', 'corrupt_visibility', 'missing_native_payload'])
def test_invalid_entries_recomputed_and_old_payloads_retained(tmp_path, kind):
    store = GeometryStore(tmp_path, max_workspace_bytes=128)
    identity = {'scene': 1}
    old = store.get_or_create(identity, fixture)
    directory = tmp_path / old.key
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    old_array = directory / manifest['arrays']['svf']['name']
    path = old_array
    if kind == 'missing_manifest':
        manifest_path.unlink()
    elif kind == 'bad_json':
        manifest_path.write_text('{')
    elif kind in ('stale_identity', 'bad_schema'):
        if kind == 'stale_identity':
            manifest['identity'] = {'scene': 2}
        else:
            manifest['arrays'].pop('svfE')
        manifest['manifest_sha256'] = storage._manifest_digest(manifest)
        manifest_path.write_text(json.dumps(manifest))
    elif kind == 'missing_array':
        path.unlink()
    elif kind == 'corrupt_array':
        path.chmod(0o644)
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    else:
        path = directory / manifest['visibility']['shmat']['name']
        if kind == 'missing_visibility':
            path.unlink()
        elif kind == 'corrupt_visibility':
            path.write_text('{}')
        else:
            native = json.loads(path.read_text())
            (path.parent / native['payload']['name']).unlink()
    called = []
    def producer():
        called.append(1)
        return fixture()
    with store.get_or_create(identity, producer) as fresh:
        assert not fresh.hit and called == [1]
        assert_fields(fresh.fields, fixture())
        assert json.loads(manifest_path.read_text())['arrays']['svf']['name'] != manifest['arrays']['svf']['name']
    # Corruption is externally imposed; cache recomputation must leave the
    # original mapped generation alone (even when its manifest is missing).
    if kind not in ('missing_array', 'corrupt_array'):
        np.testing.assert_array_equal(old.fields['svf'].view(np.uint32), fixture()['svf'].view(np.uint32))
        assert old_array.exists()
    old.close()


def test_interrupted_producer_and_publication(tmp_path, monkeypatch):
    store = GeometryStore(tmp_path)
    def fail():
        raise RuntimeError('producer interrupted')
    with pytest.raises(RuntimeError, match='interrupted'):
        store.get_or_create({'a': 1}, fail)
    assert not (tmp_path / store.key_for({'a': 1}) / 'manifest.json').exists()
    original = storage.os.replace
    def interrupt(source, target):
        if Path(target).name == 'manifest.json':
            raise OSError('publication interrupted')
        return original(source, target)
    monkeypatch.setattr(storage.os, 'replace', interrupt)
    with pytest.raises(OSError, match='publication interrupted'):
        store.get_or_create({'a': 1}, fixture)
    assert not (tmp_path / store.key_for({'a': 1}) / 'manifest.json').exists()
    monkeypatch.setattr(storage.os, 'replace', original)
    with store.get_or_create({'a': 1}, fixture) as result:
        assert not result.hit
        assert_fields(result.fields, fixture())


def _process_request(root, calls, queue):
    def producer():
        with open(calls, 'a') as stream:
            stream.write('produced\n')
        time.sleep(.1)
        return fixture()
    with GeometryStore(root).get_or_create({'same': 'identity'}, producer) as handle:
        queue.put(handle.hit)


def _crash_request(root):
    GeometryStore(root).get_or_create({'same': 'identity'}, lambda: os._exit(9))


def test_process_lock_single_producer_and_crash_release(tmp_path):
    context = multiprocessing.get_context('spawn')
    crash = context.Process(target=_crash_request, args=(tmp_path,))
    crash.start()
    crash.join(10)
    assert crash.exitcode == 9
    queue = context.Queue()
    calls = tmp_path / 'calls.txt'
    processes = [context.Process(target=_process_request, args=(tmp_path, calls, queue)) for _ in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert not process.is_alive()
        assert process.exitcode == 0
    assert sorted(queue.get(timeout=2) for _ in processes) == [False, True, True, True]
    assert calls.read_text() == 'produced\n'


def test_original_reference_fields_cache_roundtrip(tmp_path):
    from solweig_light.geometry.svf import svf_calculator_compact
    root = Path(__file__).parents[1] / 'reference' / 'svf_original_cpu'
    with np.load(root / 'building_vegetation_option1_inputs.npz') as inputs:
        values = dict(inputs)
    values['patch_option'] = int(values['patch_option'])
    values['scale'] = float(values['scale'])
    result = svf_calculator_compact(**values)
    names = 'svf svfaveg svfE svfEaveg svfEveg svfN svfNaveg svfNveg svfS svfSaveg svfSveg svfveg svfW svfWaveg svfWveg vegshmat vbshvegshmat shmat svftotal'.split()
    fields = dict(zip(names, result))
    with np.load(root / 'building_vegetation_option1_outputs.npz') as reference:
        # P2 frozen tolerances remain untouched; persistence itself is bit exact.
        for i, value in enumerate(result):
            actual = value.to_dense() if isinstance(value, PackedVisibility) else value
            if isinstance(value, PackedVisibility):
                np.testing.assert_array_equal(actual, reference[names[i]])
            else:
                np.testing.assert_allclose(actual, reference[names[i]], rtol=0, atol=1e-6)
    with GeometryStore(tmp_path, max_workspace_bytes=128).get_or_create({'original_fixture': content_fingerprint(root / 'building_vegetation_option1_inputs.npz')}, lambda: fields) as cached:
        assert_fields(cached.fields, fields)


def legacy_fixture(tmp_path):
    from osgeo import gdal, osr
    geotransform = (10, 2, 0, 40, 0, -2)
    crs = osr.SpatialReference()
    crs.ImportFromEPSG(32618)
    projection = crs.ExportToWkt()
    zip_path = tmp_path / 'svfs.zip'
    fields = fixture()
    def tiff(path, value, dtype=gdal.GDT_Float32, nodata=None):
        ds = gdal.GetDriverByName('GTiff').Create(str(path), 3, 2, 1, dtype)
        ds.SetGeoTransform(geotransform)
        ds.SetProjection(projection)
        if nodata is not None:
            ds.GetRasterBand(1).SetNoDataValue(nodata)
        ds.GetRasterBand(1).WriteArray(value)
        ds = None
    with zipfile.ZipFile(zip_path, 'w') as archive:
        for name in SVF_FIELDS:
            path = tmp_path / (name + '.tif')
            tiff(path, fields[name])
            archive.write(path, path.name)
    total_path = tmp_path / 'SkyViewFactor.tif'
    tiff(total_path, fields['svftotal'])
    npz_path = tmp_path / 'shadowmats.npz'
    export_visibility_npz(npz_path, *(fields[name] for name in VISIBILITY_FIELDS), max_workspace_bytes=128)
    validation = dict(shape=(2, 3), patch_count=3, geotransform=geotransform, projection=projection, nodata=None, max_workspace_bytes=128)
    return zip_path, npz_path, total_path, validation, fields


def test_legacy_explicit_trust_metadata_and_no_modification(tmp_path, caplog):
    zip_path, npz_path, total_path, validation, fields = legacy_fixture(tmp_path)
    before = [content_fingerprint(p) for p in (zip_path, npz_path, total_path)]
    report = validate_legacy_geometry(zip_path, npz_path, total_path, **validation)
    assert not report['identity_present']
    assert load_legacy_geometry(zip_path, npz_path, total_path, **validation) is None
    loaded = load_legacy_geometry(zip_path, npz_path, total_path, trust=True, **validation)
    assert 'without dependency identity' in caplog.text
    assert_fields(loaded, fields)
    assert [content_fingerprint(p) for p in (zip_path, npz_path, total_path)] == before
    fingerprint = raster_fingerprint(total_path, max_workspace_bytes=128)
    assert fingerprint['geotransform'] == list(validation['geotransform'])
    assert fingerprint['bands'][0]['nodata'] is None


@pytest.mark.parametrize('kind', ['zip_extra', 'zip_duplicate', 'npz_extra', 'patch_count', 'shape', 'geotransform', 'projection', 'nodata', 'total_missing', 'total_dtype', 'npz_dtype'])
def test_invalid_legacy_rejected(tmp_path, kind):
    zip_path, npz_path, total_path, validation, fields = legacy_fixture(tmp_path)
    if kind.startswith('zip_'):
        with zipfile.ZipFile(zip_path, 'a') as archive:
            archive.writestr('other.tif' if kind == 'zip_extra' else 'svf.tif', b'invalid')
    elif kind == 'npz_extra':
        with zipfile.ZipFile(npz_path, 'a') as archive:
            archive.writestr('other.npy', b'invalid')
    elif kind == 'npz_dtype':
        np.savez(npz_path, shadowmat=np.ones((2, 3, 3), dtype=np.float64), vegshadowmat=np.ones((2, 3, 3), np.float32), vbshmat=np.ones((2, 3, 3), np.float32))
    elif kind == 'total_missing':
        total_path.unlink()
    elif kind == 'total_dtype':
        from osgeo import gdal
        ds = gdal.GetDriverByName('GTiff').Create(str(total_path), 3, 2, 1, gdal.GDT_Float64)
        ds.SetGeoTransform(validation['geotransform'])
        ds.SetProjection(validation['projection'])
        ds = None
    else:
        validation[kind] = {'patch_count': 4, 'shape': (3, 2), 'geotransform': (0, 1, 0, 0, 0, -1), 'projection': '', 'nodata': -9999}[kind]
    with pytest.raises((ValueError, OSError, RuntimeError, zipfile.BadZipFile)):
        validate_legacy_geometry(zip_path, npz_path, total_path, **validation)


def test_threads_single_producer_and_bounded_io(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    original_hash = storage._hash_file
    original_iter = storage.np.nditer
    hash_chunks, array_chunks = [], []
    def check_hash(path, budget):
        assert budget == 128
        hash_chunks.append(budget)
        return original_hash(path, budget)
    def check_iter(*args, **kwargs):
        for chunk in original_iter(*args, **kwargs):
            assert chunk.nbytes <= 128
            array_chunks.append(chunk.nbytes)
            yield chunk
    monkeypatch.setattr(storage, '_hash_file', check_hash)
    monkeypatch.setattr(storage.np, 'nditer', check_iter)
    source = fixture()
    source = {name: np.tile(source[name], (12, 14)) if name in ARRAY_FIELDS else
              PackedVisibility.from_dense(np.tile(source[name].to_dense(), (12, 14, 1)))
              for name in source}
    array_fingerprint(source['svf'], max_workspace_bytes=128)
    calls = []
    def producer():
        calls.append(1)
        time.sleep(.05)
        return source
    def request(_):
        with GeometryStore(tmp_path, max_workspace_bytes=128).get_or_create({'threads': 1}, producer) as result:
            assert_fields(result.fields, source)
            return result.hit
    with ThreadPoolExecutor(4) as pool:
        hits = list(pool.map(request, range(4)))
    assert sorted(hits) == [False, True, True, True]
    assert calls == [1]
    assert hash_chunks and array_chunks


def test_final_publication_failure_keeps_previous_generation(tmp_path, monkeypatch):
    store = GeometryStore(tmp_path)
    identity = {'previous': 'valid'}
    with store.get_or_create(identity, fixture) as old:
        directory = tmp_path / old.key
        before = (directory / 'manifest.json').read_bytes()
        original = storage.os.replace
        def interrupt(source, target):
            if Path(target).name == 'manifest.json':
                raise OSError('interrupted publication')
            return original(source, target)
        monkeypatch.setattr(storage.os, 'replace', interrupt)
        with pytest.raises(OSError):
            store._publish(directory, identity, old.key, fixture())
        assert (directory / 'manifest.json').read_bytes() == before
        assert_fields(old.fields, fixture())
        with store.get_or_create(identity, lambda: pytest.fail('previous manifest lost')) as hit:
            assert hit.hit
            assert_fields(hit.fields, fixture())


@pytest.mark.parametrize('kind', ['dtype', 'nodata', 'transform'])
def test_each_zip_tiff_metadata_is_validated(tmp_path, kind):
    from osgeo import gdal
    zip_path, npz_path, total_path, validation, _ = legacy_fixture(tmp_path)
    path = tmp_path / 'svfE.tif'
    if kind == 'dtype':
        ds = gdal.GetDriverByName('GTiff').Create(str(path), 3, 2, 1, gdal.GDT_Int16)
        ds.SetGeoTransform(validation['geotransform'])
        ds.SetProjection(validation['projection'])
    else:
        ds = gdal.Open(str(path), gdal.GA_Update)
        if kind == 'nodata':
            ds.GetRasterBand(1).SetNoDataValue(-9999)
        else:
            ds.SetGeoTransform((0, 1, 0, 0, 0, -1))
    ds = None
    with zipfile.ZipFile(zip_path, 'w') as archive:
        for name in SVF_FIELDS:
            archive.write(tmp_path / (name + '.tif'), name + '.tif')
    with pytest.raises(ValueError):
        validate_legacy_geometry(zip_path, npz_path, total_path, **validation)
