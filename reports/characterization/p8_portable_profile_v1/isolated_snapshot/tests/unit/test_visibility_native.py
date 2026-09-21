import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from solweig_light.geometry.visibility import PackedVisibility, LazyDiffVisibility, export_visibility_npz, import_visibility_npz
from solweig_light.geometry import visibility_native as native


def fixture():
    bits = np.array([0, 0x80000000, 0x7fc00023, 0x7fa00045, 0x7f800000, 0xff800000, 0x3eaaaaab, 0x3f800000, 0x40000000], dtype=np.uint32)
    dense = np.empty((3, 3, 3), dtype=np.float32)
    dense[:, :, 0] = np.arange(9).reshape(3, 3) % 2
    dense[:, :, 1] = np.arange(9).reshape(3, 3) % 3
    dense[:, :, 2] = bits.view(np.float32).reshape(3, 3)
    return dense


def rewrite(path, change):
    value = json.loads(path.read_text())
    change(value)
    value['manifest_sha256'] = native._digest_manifest(value)
    path.write_text(json.dumps(value))


def test_roundtrip_lifetime_and_legacy(tmp_path):
    dense = fixture()
    packed = PackedVisibility.from_dense(dense)
    path = tmp_path / 'visibility.json'
    manifest = native.save_native_visibility(path, packed, max_workspace_bytes=128)
    with native.open_native_visibility(path, max_workspace_bytes=128) as mapped:
        assert isinstance(mapped, PackedVisibility)
        assert isinstance(mapped._mapping, np.memmap)
        assert not mapped._mapping.flags.writeable
        assert mapped.modes == ('binary', 'ternary', 'raw')
        assert mapped.nbytes == packed.nbytes
        plane = mapped.decode_patch(2)
        np.testing.assert_array_equal(mapped.to_dense().view(np.uint32), dense.view(np.uint32))
        for start in range(9):
            np.testing.assert_array_equal(mapped.decode_pixels(2, start, 9).view(np.uint32), dense[:, :, 2].ravel()[start:].view(np.uint32))
        with pytest.raises(TypeError):
            np.asarray(mapped)
        diff = LazyDiffVisibility(mapped, mapped)
        np.testing.assert_array_equal(diff[:, :, 0], dense[:, :, 0] - (np.float32(1) - dense[:, :, 0]) * np.float32(.97))
        legacy = tmp_path / 'legacy.npz'
        export_visibility_npz(legacy, mapped, mapped, mapped, max_workspace_bytes=128)
        imported = import_visibility_npz(legacy, max_workspace_bytes=128)
        np.testing.assert_array_equal(imported['vbshmat'].to_dense().view(np.uint32), dense.view(np.uint32))
        native.save_native_visibility(tmp_path / 'copy.json', mapped, max_workspace_bytes=128)
    assert mapped.closed
    mapped.close()
    np.testing.assert_array_equal(plane.view(np.uint32), dense[:, :, 2].view(np.uint32))
    for operation in (lambda: mapped.decode_patch(0), lambda: mapped.decode_pixels(0, 0, 1), mapped.to_dense, lambda: mapped[:, :, 0], mapped.__enter__):
        with pytest.raises(RuntimeError, match='closed'):
            operation()
    payload = tmp_path / manifest['payload']['name']
    assert np.load(payload, mmap_mode='r').dtype == np.uint8


@pytest.mark.parametrize('shape', [(0, 3, 2), (2, 0, 2), (2, 2, 0), (1, 1, 1), (1, 7, 1), (1, 8, 1), (1, 9, 1)])
def test_boundaries(tmp_path, shape):
    dense = np.zeros(shape, dtype=np.float32)
    path = tmp_path / 'v.json'
    native.save_native_visibility(path, PackedVisibility.from_dense(dense))
    with native.open_native_visibility(path) as mapped:
        np.testing.assert_array_equal(mapped.to_dense(), dense)


@pytest.mark.parametrize('change', [
    lambda m: m.update(native_version=2),
    lambda m: m.update(dtype='<f8'),
    lambda m: m.update(codebook_bits=[0, 1, 2]),
    lambda m: m['patches'][0].update(offset=1),
    lambda m: m['patches'][0].update(length=20),
    lambda m: m['patches'][0].update(mode='boolean'),
    lambda m: m['payload'].update(name='../bad.npy'),
    lambda m: m.update(shape=[True, 3, 3]),
])
def test_bad_metadata(tmp_path, change):
    path = tmp_path / 'v.json'
    native.save_native_visibility(path, PackedVisibility.from_dense(fixture()))
    rewrite(path, change)
    with pytest.raises(ValueError):
        native.open_native_visibility(path)


@pytest.mark.parametrize('kind', ['digest', 'truncate', 'missing', 'reserved', 'padding'])
def test_corruption(tmp_path, kind):
    path = tmp_path / 'v.json'
    m = native.save_native_visibility(path, PackedVisibility.from_dense(fixture()))
    payload = tmp_path / m['payload']['name']
    if kind == 'missing':
        payload.unlink()
    elif kind == 'digest':
        path.write_text(path.read_text().replace('"native_version":1', '"native_version":2'))
    else:
        payload.chmod(0o644)
        if kind == 'truncate':
            payload.write_bytes(payload.read_bytes()[:-1])
        else:
            mapping = np.load(payload, mmap_mode='r+')
            if kind == 'reserved':
                mapping[m['patches'][1]['offset']] = 3
            else:
                mapping[m['patches'][0]['length'] - 1] = 128
            mapping.flush()
            mapping._mmap.close()
            rewrite(path, lambda value: value['payload'].update(sha256=native._hash_file(payload, 128)))
    with pytest.raises((ValueError, FileNotFoundError)):
        native.open_native_visibility(path, max_workspace_bytes=128)


def test_atomic_failure_and_concurrent_tiles(tmp_path, monkeypatch):
    packed = PackedVisibility.from_dense(fixture())
    paths = [tmp_path / f'{i}.json' for i in range(4)]
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda p: native.save_native_visibility(p, packed, max_workspace_bytes=128), paths))
    names = [json.loads(p.read_text())['payload']['name'] for p in paths]
    assert len(set(names)) == 4
    before = paths[0].read_bytes()
    def fail(*args):
        raise OSError('interrupted')
    monkeypatch.setattr(native.os, 'replace', fail)
    with pytest.raises(OSError):
        native.save_native_visibility(paths[0], packed)
    assert paths[0].read_bytes() == before
    with native.open_native_visibility(paths[0]) as mapped:
        np.testing.assert_array_equal(mapped.to_dense().view(np.uint32), fixture().view(np.uint32))


def test_bounded_hash_reads(tmp_path, monkeypatch):
    path = tmp_path / 'v.json'
    dense = np.full((80, 80, 1), np.float32(.37))
    native.save_native_visibility(path, PackedVisibility.from_dense(dense), max_workspace_bytes=128)
    original = native._hash_file
    chunks = []
    def check(path, chunk):
        chunks.append(chunk)
        assert chunk <= 128
        return original(path, chunk)
    monkeypatch.setattr(native, '_hash_file', check)
    with native.open_native_visibility(path, max_workspace_bytes=128) as mapped:
        assert mapped.nbytes == 80 * 80 * 4
    assert chunks == [128]


def test_random_bits_and_concurrent_close(tmp_path):
    rng = np.random.default_rng(703)
    bits = rng.integers(0, 2**32, size=(17, 19, 3), dtype=np.uint32)
    dense = bits.view(np.float32)
    path = tmp_path / 'v.json'
    native.save_native_visibility(path, PackedVisibility.from_dense(dense), max_workspace_bytes=128)
    mapped = native.open_native_visibility(path, max_workspace_bytes=128)
    def read(_):
        try:
            plane = mapped.decode_patch(2)
        except RuntimeError:
            return
        np.testing.assert_array_equal(plane.view(np.uint32), bits[:, :, 2])
    with ThreadPoolExecutor(4) as pool:
        futures = [pool.submit(read, i) for i in range(30)]
        mapped.close()
        for future in futures:
            future.result()
    assert mapped.closed
