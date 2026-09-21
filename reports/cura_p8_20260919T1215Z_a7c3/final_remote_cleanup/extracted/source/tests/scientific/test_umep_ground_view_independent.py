"""Independent, unchanged-UMEP ground-view comparison.

Authored during the P7 benchmark quiet window; execution is intentionally deferred.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from solweig_light.radiation import ground_view


ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "reports" / "characterization" / "p8_umep_source"
MANIFEST_PATH = SOURCE / "manifest.json"
GVF_PATH = SOURCE / "gvf_2018a.py"
SUN_PATH = SOURCE / "sunonsurface_2018a.py"
GVF_SHA256 = "9bfb49dd281924bff5352d57d9c28a8274d1a2c4660c4fbca532afa6de167c94"
SUN_SHA256 = "b5abe6568ffad18781b1011e13681238c63a2ebfba21c11c61e40d986089e089"
UMEP_COMMIT = "3fcc0c3dca67d1d5644a6d34b9148d7a365743ba"
LONGWAVE_FIELDS = frozenset((0, 3, 6, 9, 12))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_unchanged_umep():
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["repository"] == "https://github.com/UMEP-dev/UMEP"
    assert manifest["commit"] == UMEP_COMMIT
    entries = {entry["local_path"]: entry for entry in manifest["files"]}
    expected = {
        "reports/characterization/p8_umep_source/gvf_2018a.py": (GVF_PATH, GVF_SHA256, 4160),
        "reports/characterization/p8_umep_source/sunonsurface_2018a.py": (SUN_PATH, SUN_SHA256, 9396),
    }
    for local_path, (path, digest, size) in expected.items():
        entry = entries[local_path]
        assert entry["sha256"] == digest
        assert entry["bytes"] == size
        assert _sha256(path) == digest
        assert path.stat().st_size == size

    package_name = "_solweig_test_pinned_umep_ground_view"
    for name in (f"{package_name}.gvf_2018a", f"{package_name}.sunonsurface_2018a", package_name):
        sys.modules.pop(name, None)
    package = types.ModuleType(package_name)
    package.__path__ = [str(SOURCE)]
    sys.modules[package_name] = package
    try:
        sun_name = f"{package_name}.sunonsurface_2018a"
        sun_spec = importlib.util.spec_from_file_location(sun_name, SUN_PATH)
        assert sun_spec is not None and sun_spec.loader is not None
        sun_module = importlib.util.module_from_spec(sun_spec)
        sys.modules[sun_name] = sun_module
        sun_spec.loader.exec_module(sun_module)

        gvf_name = f"{package_name}.gvf_2018a"
        gvf_spec = importlib.util.spec_from_file_location(gvf_name, GVF_PATH)
        assert gvf_spec is not None and gvf_spec.loader is not None
        gvf_module = importlib.util.module_from_spec(gvf_spec)
        sys.modules[gvf_name] = gvf_module
        gvf_spec.loader.exec_module(gvf_module)
        return sun_module, gvf_module
    finally:
        for name in (f"{package_name}.gvf_2018a", f"{package_name}.sunonsurface_2018a", package_name):
            sys.modules.pop(name, None)


def _fixture(landcover):
    rows, cols = 9, 11
    row, col = np.indices((rows, cols))
    buildings = np.ones((rows, cols), dtype=np.float32)
    buildings[(row + 2 * col) % 7 == 0] = 0
    walls = (1.0 + ((3 * row + col) % 5) * 0.4).astype(np.float32)
    wallsun = np.where((2 * row + col) % 4 == 0, walls, 0).astype(np.float32)
    shadow_values = np.asarray((0.0, 0.35, 1.0), dtype=np.float32)
    shadow = shadow_values[(row + col) % len(shadow_values)]
    dirwall_values = np.asarray((2, 44, 89, 91, 179, 181, 269, 271, 358), dtype=np.float32)
    dirwalls = dirwall_values[(2 * row + 3 * col) % len(dirwall_values)]
    tg = (1.5 + row * 0.23 - col * 0.07).astype(np.float32)
    emis_grid = (0.91 + ((row + col) % 5) * 0.012).astype(np.float32)
    alb_grid = (0.08 + ((2 * row + col) % 7) * 0.045).astype(np.float32)
    lc_grid = ((row + 2 * col) % 6 + 1).astype(np.int16)
    return {
        "wallsun": wallsun,
        "walls": walls,
        "buildings": buildings,
        "scale": np.float32(1),
        "shadow": shadow,
        "first": np.float32(2),
        "second": np.float32(4),
        "dirwalls": dirwalls,
        "Tg": tg,
        "Tgwall": np.float32(4.25),
        "Ta": np.float32(21.5),
        "emis_grid": emis_grid,
        "ewall": np.float32(0.90),
        "alb_grid": alb_grid,
        "SBC": np.float32(5.67051e-8),
        "albedo_b": np.float32(0.20),
        "rows": rows,
        "cols": cols,
        "Twater": np.float32(18.0),
        "lc_grid": lc_grid,
        "landcover": np.float32(landcover),
    }


def _copy(arguments):
    return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in arguments.items()}


def _direct_arguments(arguments, azimuth):
    result = _copy(arguments)
    result["azimuthA"] = np.float32(azimuth)
    result["sunwall"] = ((result["wallsun"] / result["walls"]) * result["buildings"] == 1).astype(np.float32)
    result["aspect"] = (result.pop("dirwalls") * np.float32(np.pi / 180)).astype(np.float32)
    for key in ("wallsun", "rows", "cols"):
        result.pop(key)
    return result


def _assert_field(actual, expected, longwave):
    assert actual.shape == expected.shape
    for mask in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(mask(actual), mask(expected))
    if longwave:
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=0.05, equal_nan=True)
    else:
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6, equal_nan=True)


def _assert_mutations(before, reference, candidate, landcover, include_sunwall):
    np.testing.assert_array_equal(candidate["Tg"], reference["Tg"])
    if landcover:
        water = before["Twater"] - before["Ta"]
        water_cells = candidate["Tg"][before["lc_grid"] == 3]
        np.testing.assert_array_equal(water_cells, np.full_like(water_cells, water))
        np.testing.assert_array_equal(
            candidate["Tg"][before["lc_grid"] != 3], before["Tg"][before["lc_grid"] != 3]
        )
    else:
        np.testing.assert_array_equal(candidate["Tg"], before["Tg"])
    if include_sunwall:
        expected = before["sunwall"].copy()
        expected[expected > 0] = 1
        np.testing.assert_array_equal(reference["sunwall"], expected)
        np.testing.assert_array_equal(candidate["sunwall"], expected)
    else:
        np.testing.assert_array_equal(reference["wallsun"], before["wallsun"])
        np.testing.assert_array_equal(candidate["wallsun"], before["wallsun"])


@pytest.mark.parametrize("landcover", (0, 1))
@pytest.mark.parametrize("azimuth", (5, 105, 285))
def test_umep_sunonsurface_direct(landcover, azimuth):
    umep_sun, _ = _load_unchanged_umep()
    base = _direct_arguments(_fixture(landcover), azimuth)
    reference_args = _copy(base)
    serial_args = _copy(base)
    parallel_args = _copy(base)

    reference = umep_sun.sunonsurface_2018a(**reference_args)
    serial = ground_view.sunonsurface_2018a(**serial_args)
    parallel = ground_view.sunonsurface_2018a_parallel(**parallel_args)

    assert len(reference) == len(serial) == len(parallel) == 5
    for index, expected in enumerate(reference):
        _assert_field(serial[index], expected, longwave=index == 1)
        _assert_field(parallel[index], expected, longwave=index == 1)
        _assert_field(parallel[index], serial[index], longwave=index == 1)
    _assert_mutations(base, reference_args, serial_args, landcover, include_sunwall=True)
    _assert_mutations(base, reference_args, parallel_args, landcover, include_sunwall=True)


@pytest.mark.parametrize("landcover", (0, 1))
def test_umep_full_ground_view(landcover):
    _, umep_gvf = _load_unchanged_umep()
    base = _fixture(landcover)
    reference_args = _copy(base)
    serial_args = _copy(base)
    parallel_args = _copy(base)

    reference = umep_gvf.gvf_2018a(**reference_args)
    serial = ground_view.gvf_2018a(**serial_args)
    parallel = ground_view.gvf_2018a_parallel(**parallel_args)

    assert len(reference) == len(serial) == len(parallel) == 17
    for index, expected in enumerate(reference):
        longwave = index in LONGWAVE_FIELDS
        _assert_field(serial[index], expected, longwave=longwave)
        _assert_field(parallel[index], expected, longwave=longwave)
        _assert_field(parallel[index], serial[index], longwave=longwave)
    _assert_mutations(base, reference_args, serial_args, landcover, include_sunwall=False)
    _assert_mutations(base, reference_args, parallel_args, landcover, include_sunwall=False)
