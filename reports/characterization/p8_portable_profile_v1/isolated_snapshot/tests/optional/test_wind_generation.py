"""Real local generation versus unmodified original TIFF/NetCDF oracles."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import rasterio
from solweig_light import wind, build_wind_ext_coeff
from solweig_light.runtime import runtime_options

ROOT = Path(__file__).resolve().parents[1] / 'reference/wind_original_cpu'
MANIFEST = json.loads((ROOT / 'manifest.json').read_text())
CASES = [case for case in MANIFEST['cases'] if case['status'] == 'executed']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(path):
    with rasterio.open(path) as ds:
        return {'shape': list(ds.shape), 'count': ds.count, 'dtype': ds.dtypes[0], 'crs': ds.crs.to_wkt(),
                'transform': list(ds.transform), 'nodata': 'NaN' if np.isnan(ds.nodata) else ds.nodata,
                'compression': ds.compression.value, 'tiled': ds.profile['tiled'],
                'block_shapes': [list(s) for s in ds.block_shapes], 'tags': ds.tags(), 'band_tags': ds.tags(1)}


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['name'])
@pytest.mark.parametrize('workers', [1, 2, 4])
def test_original_twelve_direction_outputs(case, workers, tmp_path):
    original = ROOT / case['name']
    inputs = tmp_path / 'inputs'
    shutil.copytree(original / 'inputs', inputs, ignore=shutil.ignore_patterns('WindCoeff_*'))
    for name, checksum in case['input_sha256'].items():
        assert sha(inputs / name) == checksum
    assert sha(original / 'era5/data_stream-oper_stepType-instant.nc') == case['met_sha256']
    with runtime_options(cpu_budget=workers, workers=workers, threads_per_worker=1, memory_budget_bytes=4 * 1024**3):
        returned = build_wind_ext_coeff(inputs, original / 'era5', max_workers=workers)
    assert returned == str(inputs)
    actual = sorted(inputs.glob('WindCoeff_*.tif'))
    assert [p.name for p in actual] == case['returned_names'] == [f'WindCoeff_dir{d:03d}.tif' for d in range(0, 360, 30)]
    with np.load(original / 'fields.npz') as expected:
        assert sha(original / 'fields.npz') == case['fields_sha256']
        for path in actual:
            with rasterio.open(path) as ds:
                field = ds.read(1)
            reference = expected[path.stem]
            for mask in (np.isnan, np.isposinf, np.isneginf):
                np.testing.assert_array_equal(mask(field), mask(reference))
            np.testing.assert_array_equal(field, reference)
            assert metadata(path) == case['metadata'][path.name]
            finite = field[np.isfinite(field)]
            assert finite.min() >= np.float32(.1) and finite.max() <= 1


def test_original_block_failure_preserved(tmp_path):
    original = ROOT / 'invalid_blocks'
    shutil.copytree(original / 'inputs', tmp_path / 'inputs', ignore=shutil.ignore_patterns('WindCoeff_*'))
    assert json.loads((original / 'outcome.json').read_text())['status'] == 'original_failure'
    with pytest.raises(RuntimeError, match='Wind coefficient direction .* failed'):
        wind.build_wind_ext_coeff(tmp_path / 'inputs', original / 'era5', max_workers=1)


def test_building_precedence_fallback_and_roughness():
    preferred = ROOT / 'precedence/inputs'
    fallback = ROOT / 'dsm_fallback/inputs'
    assert wind._find_building_raster(preferred).name == 'Buildings.tif'
    assert wind._find_building_raster(fallback).name == 'Building_DSM.tif'
    raw, _, _ = wind._read_building_height(fallback / 'Building_DSM.tif', hmin_b=1)
    with rasterio.open(fallback / 'Building_DSM.tif') as ds:
        original = ds.read(1)
    np.testing.assert_array_equal(raw, np.where(np.isfinite(original) & (original >= 1), original, 0))
    for case in CASES:
        path = ROOT / case['name']
        building = wind._find_building_raster(path / 'inputs')
        assert wind._read_z0_from_fsr_at_raster_midpoint(path / 'era5/data_stream-oper_stepType-instant.nc', building, .03) == case['selected_z0']


def test_import_keeps_optional_dependencies_lazy():
    subprocess.run([sys.executable, '-c', "import sys; import solweig_light.wind; assert 'rasterio' not in sys.modules; assert 'xarray' not in sys.modules; assert 'torch' not in sys.modules"], check=True)


def test_top_level_discovery_and_directory_errors(tmp_path):
    with pytest.raises(NotADirectoryError, match='input_dir'):
        wind.build_wind_ext_coeff(tmp_path / 'absent', tmp_path)
    (tmp_path / 'nested').mkdir()
    (tmp_path / 'nested/Buildings.tif').touch()
    with pytest.raises(FileNotFoundError, match='building raster'):
        wind.build_wind_ext_coeff(tmp_path, tmp_path)
