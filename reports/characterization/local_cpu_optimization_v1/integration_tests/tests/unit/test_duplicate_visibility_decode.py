import numpy as np
import pytest
from pathlib import Path
import solweig_light
import solweig_light.geometry.visibility_compiled as visibility_compiled
import solweig_light.radiation.patch_radiation as patch_radiation

from solweig_light.geometry.visibility import (
    LazyDiffVisibility, PackedVisibility, _EncodedPatch,
)
from solweig_light.geometry.visibility_native import (
    open_native_visibility, save_native_visibility,
)
from solweig_light.geometry.visibility_compiled import (
    decode_block, diff_from_shared_decoded,
)

PACKAGE = Path(solweig_light.__file__).resolve().parent


def test_active_visibility_modules_share_installed_package_origin():
    for module in (visibility_compiled, patch_radiation):
        assert Path(module.__file__).resolve().is_relative_to(PACKAGE)


def scene():
    bits = np.array([
        0x00000000, 0x80000000, 0x3f800000,
        0x40000000, 0x7fc01234, 0x7fa05678,
        0x7f800000, 0xff800000, 0x3eaaaaab,
    ], dtype=np.uint32)
    raw = bits.view(np.float32).reshape(3, 3)
    shadow = np.empty((3, 3, 3), dtype=np.float32)
    vegetation = np.empty_like(shadow)
    shadow[:, :, 0] = np.arange(9).reshape(3, 3) % 2
    shadow[:, :, 1] = np.arange(9).reshape(3, 3) % 3
    shadow[:, :, 2] = raw
    vegetation[:, :, 0] = shadow[::-1, :, 0]
    vegetation[:, :, 1] = shadow[:, ::-1, 1]
    vegetation[:, :, 2] = raw[::-1, ::-1]
    return shadow, vegetation


@pytest.mark.parametrize('mapped', [False, True])
def test_shared_lazy_leaves_preserve_bits_and_shadow(tmp_path, mapped):
    shadow_dense, vegetation_dense = scene()
    shadow = PackedVisibility.from_dense(shadow_dense)
    vegetation = PackedVisibility.from_dense(vegetation_dense)
    owners = []
    if mapped:
        for name, channel in (('shadow', shadow), ('vegetation', vegetation)):
            path = tmp_path / f'{name}.json'
            save_native_visibility(path, channel, max_workspace_bytes=128)
            owners.append(open_native_visibility(path, max_workspace_bytes=128))
        shadow, vegetation = owners
    diffuse = LazyDiffVisibility(shadow, vegetation)
    vegetation_building = PackedVisibility.from_dense(np.zeros_like(shadow_dense))
    try:
        with np.errstate(all='ignore'):
            expected = decode_block(diffuse, 1, 9, 3)
            actual_shadow, actual_vegetation, _, actual = patch_radiation._shortwave_visibility_blocks(
                shadow, vegetation, vegetation_building, diffuse, 1, 9, 3,
            )
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(
            actual_shadow.view(np.uint32),
            shadow_dense.reshape(-1, 3)[1:9].view(np.uint32),
        )
    finally:
        for owner in owners:
            owner.close()


def test_independent_and_different_diffuse_inputs_fall_back():
    shadow_dense, vegetation_dense = scene()
    shadow = PackedVisibility.from_dense(shadow_dense)
    vegetation = PackedVisibility.from_dense(vegetation_dense)
    independent = _independent_diff(shadow_dense, vegetation_dense)
    other_shadow = PackedVisibility.from_dense(vegetation_dense)
    other_vegetation = PackedVisibility.from_dense(shadow_dense)
    different = LazyDiffVisibility(other_shadow, other_vegetation)

    class Subclass(LazyDiffVisibility):
        pass

    decoded_shadow = decode_block(shadow, 0, 9, 3)
    decoded_vegetation = decode_block(vegetation, 0, 9, 3)
    assert diff_from_shared_decoded(
        shadow, vegetation, independent, decoded_shadow, decoded_vegetation, 0, 9, 3,
    ) is None
    assert diff_from_shared_decoded(
        shadow, vegetation, different, decoded_shadow, decoded_vegetation, 0, 9, 3,
    ) is None
    assert diff_from_shared_decoded(
        shadow, vegetation, Subclass(shadow, vegetation),
        decoded_shadow, decoded_vegetation, 0, 9, 3,
    ) is None
    np.testing.assert_array_equal(
        decode_block(different, 0, 9, 3).view(np.uint32),
        _independent_diff(vegetation_dense, shadow_dense).reshape(-1, 3).view(np.uint32),
    )


def _independent_diff(shadow, vegetation):
    with np.errstate(all='ignore'):
        return shadow - (np.float32(1) - vegetation) * np.float32(.97)


def test_reserved_shadow_precedes_closed_vegetation_owner(tmp_path):
    bad = PackedVisibility((1, 1, 1), (_EncodedPatch('ternary', b'\x03'),))
    dense = np.zeros((1, 1, 1), dtype=np.float32)
    path = tmp_path / 'mapped.json'
    save_native_visibility(path, PackedVisibility.from_dense(dense))
    mapped = open_native_visibility(path)
    mapped.close()
    diffuse = LazyDiffVisibility(bad, mapped)
    with pytest.raises(IndexError, match='Reserved visibility code'):
        patch_radiation._shortwave_visibility_blocks(
            bad, mapped, PackedVisibility.from_dense(dense), diffuse, 0, 1, 1,
        )


def test_intervening_duck_close_is_observed_at_lazy_read(tmp_path):
    dense = np.zeros((1, 1, 1), dtype=np.float32)
    owners = []
    for name in ('shadow', 'vegetation'):
        path = tmp_path / f'{name}.json'
        save_native_visibility(path, PackedVisibility.from_dense(dense))
        owners.append(open_native_visibility(path))
    shadow, vegetation = owners

    class ClosingDuck:
        dtype = np.dtype(np.float32)
        shape = dense.shape
        def decode_pixels(self, patch, start, stop):
            shadow.close()
            return np.zeros(stop-start, dtype=np.float32)

    try:
        with pytest.raises(RuntimeError, match='closed'):
            patch_radiation._shortwave_visibility_blocks(
                shadow, vegetation, ClosingDuck(),
                LazyDiffVisibility(shadow, vegetation), 0, 1, 1,
            )
    finally:
        shadow.close()
        vegetation.close()


@pytest.mark.parametrize('subclass_target', ['leaf', 'lazy'])
def test_subclasses_force_original_fourth_read(monkeypatch, subclass_target):
    dense = np.zeros((1, 1, 1), dtype=np.float32)
    base_shadow = PackedVisibility.from_dense(dense)
    vegetation = PackedVisibility.from_dense(dense)
    vegetation_building = PackedVisibility.from_dense(dense)

    class PackedSubclass(PackedVisibility):
        pass
    class LazySubclass(LazyDiffVisibility):
        pass

    shadow = (PackedSubclass(base_shadow.shape, base_shadow._patches)
              if subclass_target == 'leaf' else base_shadow)
    lazy_type = LazySubclass if subclass_target == 'lazy' else LazyDiffVisibility
    diffuse = lazy_type(shadow, vegetation)
    original = patch_radiation._block
    reads = []
    def tracked(channel, start, stop, patches):
        reads.append(channel)
        return original(channel, start, stop, patches)
    monkeypatch.setattr(patch_radiation, '_block', tracked)
    patch_radiation._shortwave_visibility_blocks(
        shadow, vegetation, vegetation_building, diffuse, 0, 1, 1,
    )
    assert all(actual is expected for actual, expected in zip(
        reads, [shadow, vegetation, vegetation_building, diffuse], strict=True,
    ))


def test_shared_path_retains_range_validation():
    dense = np.zeros((1, 2, 1), dtype=np.float32)
    packed = PackedVisibility.from_dense(dense)
    diffuse = LazyDiffVisibility(packed, packed)
    decoded = np.zeros((1, 1), dtype=np.float32)
    for start, stop, patches in ((-1, 1, 1), (0, 3, 1), (2, 1, 1), (0, 1, 2)):
        with pytest.raises(IndexError, match='Visibility block interval out of range'):
            diff_from_shared_decoded(
                packed, packed, diffuse, decoded, decoded, start, stop, patches,
            )
