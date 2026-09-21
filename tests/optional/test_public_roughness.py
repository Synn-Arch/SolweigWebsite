"""Real public local roughness generation and consumption; no numerical mocks."""
import hashlib
import json
from pathlib import Path
import sys

import pytest
import numpy as np
import rasterio

from solweig_light import thermal_comfort
from solweig_light.runtime import runtime_options

REFERENCE = Path(__file__).resolve().parents[1] / 'reference/public_roughness_original_cpu'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with rasterio.open(path) as ds:
        return ds.read()


@pytest.mark.parametrize('workers', [1, 2, 4])
def test_public_roughness_generates_and_consumes_all_directions(tmp_path, capsys, workers):
    manifest = json.loads((REFERENCE / 'manifest.json').read_text())
    assert manifest['evidence_class'] == 'original_upstream_cpu'
    scene = tmp_path / 'scene'
    scene.mkdir()
    for name, checksum in manifest['input_sha256'].items():
        source = REFERENCE / 'scene' / name
        assert sha(source) == checksum
        (scene / name).write_bytes(source.read_bytes())
    era5 = REFERENCE / 'era5'
    assert sha(era5 / 'data_stream-oper_stepType-instant.nc') == manifest['era5_sha256']
    with runtime_options(cpu_budget=workers, workers=workers, threads_per_worker=1, memory_budget_bytes=4 * 1024**3):
        result = thermal_comfort(str(scene), '2020-07-18', ERA_5_z0_find=True,
                                 tile_size=64, overlap=0, use_own_met=True,
                                 own_met_file=str(scene / 'met.txt'), use_uhi=False,
                                 data_folder=str(era5), save_wind=True, save_tmrt=False)
    assert result is None
    assert 'Could not find ERA-5 file with roughness length' not in capsys.readouterr().out
    names = [f'WindCoeff_dir{direction:03d}.tif' for direction in range(0, 360, 30)]
    assert sorted(p.name for p in scene.glob('WindCoeff_dir*.tif')) == names
    pre = scene / 'processed_inputs'
    assert sorted(p.name for p in (pre / 'WindCoeff').glob('*.tif')) == [f'WindCoeff_dir{d:03d}_0_0.tif' for d in range(0, 360, 30)]
    for name in names:
        source = REFERENCE / 'scene' / name
        assert sha(source) == manifest['artifact_sha256'][name]
        np.testing.assert_array_equal(read(scene / name), read(source))
        with rasterio.open(scene / name) as actual, rasterio.open(source) as expected:
            assert actual.crs == expected.crs and actual.transform == expected.transform
            assert actual.dtypes == expected.dtypes and actual.block_shapes == expected.block_shapes
            assert actual.compression == expected.compression and np.isnan(actual.nodata)
    met = np.loadtxt(pre / 'metfiles/metfile_0_0.txt', skiprows=1)
    assert set(met[:, 23]) == set(range(0, 360, 30))
    expected_wind = []
    for row in met:
        coefficient = read(pre / f'WindCoeff/WindCoeff_dir{int(row[23]):03d}_0_0.tif')[0]
        expected_wind.append(np.maximum(coefficient * np.float32(row[9]), np.float32(.15)))
    expected_wind = np.stack(expected_wind)
    actual_wind = read(scene / 'output_folder/0_0/Wind_0_0.tif')
    np.testing.assert_array_equal(actual_wind, expected_wind)
    original_wind = REFERENCE / 'scene/output_folder/0_0/Wind_0_0.tif'
    assert sha(original_wind) == manifest['artifact_sha256']['output_folder/0_0/Wind_0_0.tif']
    np.testing.assert_array_equal(actual_wind, read(original_wind))
    assert np.any(actual_wind[np.isfinite(actual_wind)] < 2)
    assert not np.array_equal(actual_wind[0], actual_wind[1], equal_nan=True)
    original_utci = REFERENCE / 'scene/output_folder/0_0/UTCI_0_0.tif'
    assert sha(original_utci) == manifest['artifact_sha256']['output_folder/0_0/UTCI_0_0.tif']
    actual, expected = read(scene / 'output_folder/0_0/UTCI_0_0.tif'), read(original_utci)
    for mask in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(mask(actual), mask(expected))
    finite = np.isfinite(expected)
    assert np.max(np.abs(actual[finite] - expected[finite])) <= .02
    assert 'torch' not in sys.modules
