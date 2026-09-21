"""Bounded native decoding parity, lifetime and duck-type compatibility."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from solweig_light.geometry.visibility import PackedVisibility, LazyDiffVisibility
from solweig_light.geometry.visibility_native import save_native_visibility, open_native_visibility
from solweig_light.radiation.patch_radiation import _block


def scene():
    rng = np.random.default_rng(57)
    array = rng.integers(0, 2, (9, 13, 5)).astype(np.float32)
    array[:, :, 1] = rng.integers(0, 3, (9, 13))
    array[:, :, 2] = rng.normal(size=(9, 13)).astype(np.float32)
    array.reshape(-1, 5)[:5, 3] = np.array([0x80000000, 0x7fc01234, 0xffc05678, 0x7f800000, 1], np.uint32).view(np.float32)
    return array


@pytest.mark.parametrize('mapped', [False, True])
@pytest.mark.parametrize('size', [1, 7, 128])
def test_blocks_bits_and_diff(tmp_path, mapped, size):
    source = scene()
    packed = PackedVisibility.from_dense(source)
    if mapped:
        path = tmp_path/'visibility.json'
        save_native_visibility(path, packed)
        packed = open_native_visibility(path)
    vegetation = PackedVisibility.from_dense(source[:, :, ::-1].copy())
    diffuse = LazyDiffVisibility(packed, vegetation)
    with np.errstate(invalid='ignore'):
        expected = source - (np.float32(1)-source[:, :, ::-1])*np.float32(1-.03)
        for start in range(0, 117, size):
            stop = min(117, start+size)
            for channel, target in [(packed, source), (diffuse, expected)]:
                result = _block(channel, start, stop, 5)
                np.testing.assert_array_equal(result.view('u4'), target.reshape(-1, 5)[start:stop].view('u4'))
    descriptor = packed._block_descriptor
    assert sum(v.nbytes for v in descriptor[0]) == packed.nbytes
    _block(packed, 0, 1, 5)
    assert packed._block_descriptor is descriptor
    if mapped:
        result = _block(packed, 0, 3, 5)
        packed.close()
        np.testing.assert_array_equal(result.view('u4'), source.reshape(-1, 5)[:3].view('u4'))
        with pytest.raises(RuntimeError, match='closed'):
            _block(packed, 0, 3, 5)
        with pytest.raises(RuntimeError, match='closed'):
            _block(diffuse, 0, 3, 5)


def test_concurrent_mapped_reads_and_close(tmp_path):
    source = scene()
    path = tmp_path/'visibility.json'
    save_native_visibility(path, PackedVisibility.from_dense(source))
    channel = open_native_visibility(path)
    _block(channel, 0, 1, 5)
    def read(_):
        try:
            result = _block(channel, 3, 83, 5)
            np.testing.assert_array_equal(result.view('u4'), source.reshape(-1, 5)[3:83].view('u4'))
            return True
        except RuntimeError:
            assert channel.closed
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(read, index) for index in range(20)]
        channel.close()
        [future.result() for future in futures]


def test_dense_views_and_duck_types():
    source = scene()[:, ::-1, :]
    class Duck:
        shape = source.shape
        def decode_pixels(self, patch, start, stop):
            return source[:, :, patch].reshape(-1)[start:stop]
    for channel in (source, Duck()):
        np.testing.assert_array_equal(_block(channel, 5, 17, 5).view('u4'), source.reshape(-1, 5)[5:17].view('u4'))


def test_empty_intervals_and_invalid_ranges():
    channel = PackedVisibility.from_dense(scene())
    assert _block(channel, 4, 4, 5).shape == (0, 5)
    assert _block(channel, 0, 5, 0).shape == (5, 0)
    for start, stop, patches in [(-1, 1, 5), (0, 118, 5), (7, 5, 5), (0, 1, 6)]:
        with pytest.raises(IndexError):
            _block(channel, start, stop, patches)


def test_reserved_ternary_code_preserves_exception():
    from solweig_light.geometry.visibility import _EncodedPatch
    channel = PackedVisibility((1, 1, 1), (_EncodedPatch('ternary', b'\x03'),))
    with pytest.raises(IndexError):
        _block(channel, 0, 1, 1)
