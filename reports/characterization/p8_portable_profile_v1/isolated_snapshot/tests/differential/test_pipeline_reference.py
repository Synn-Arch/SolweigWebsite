"""Real chronological TIFF-to-TIFF execution against original CPU artifacts."""
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
from osgeo import gdal
import pytest

from solweig_light.pipeline import files_by_key, run_tile

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'tests/reference/small_original_cpu/scene'
PROTOCOL = json.loads((ROOT / 'benchmarks/protocols/comparison_v1.json').read_text())


@pytest.mark.parametrize('case,cold', [('small', False), ('small', True), ('raw', True)] +
                         [(name, False) for name in ('base', 'uhi', 'landcover',
                          'landcover_normalized', 'directional', 'legacy_wrapper')])
def test_chronological_tiff_pipeline(tmp_path, case, cold):
    assert importlib.util.find_spec('torch') is None, 'Run candidate checks in the isolated CPU environment'
    reference = REFERENCE if case in ('small', 'raw') else ROOT / 'tests/reference/optional_original_cpu' / case
    prepared = tmp_path / 'processed_inputs'
    shutil.copytree(reference / 'processed_inputs', prepared)
    if cold:
        shutil.rmtree(prepared / 'SVF')
    paths = {name: files_by_key(prepared / name)['0_0'] for name in
             ('Building_DSM', 'Trees', 'DEM', 'walls', 'aspect', 'metfiles')}
    if (prepared / 'Landcover').is_dir():
        paths['Landcover'] = files_by_key(prepared / 'Landcover')['0_0']
    flags = {f'save_{name}': True for name in
             ('tmrt', 'svf', 'kup', 'kdown', 'lup', 'ldown', 'shadow', 'wbgt', 'ta', 'wind')}
    if case == 'raw':
        from solweig_light import thermal_comfort
        shutil.rmtree(prepared)
        for filename in ('Building_DSM.tif', 'Trees.tif', 'DEM.tif', 'met.txt'):
            shutil.copy2(reference / filename, tmp_path / filename)
        assert thermal_comfort(str(tmp_path), '2020-07-18', own_met_file=str(tmp_path / 'met.txt'),
                               tile_size=64, overlap=0, ERA_5_z0_find=False, **flags) is None
    else:
        run_tile(tmp_path, prepared, '2020-07-18', '0_0', paths, flags)
    output = tmp_path / 'output_folder/0_0'
    expected_files = {p.name for p in (reference / 'output_folder/0_0').glob('*.tif')}
    assert {p.name for p in output.glob('*.tif')} == expected_files | ({'SVF_0_0.tif'} if cold and case != 'raw' else set())
    errors = {}
    for filename in sorted(expected_files):
        actual = gdal.Open(str(output / filename))
        expected = gdal.Open(str(reference / 'output_folder/0_0' / filename))
        assert actual.GetGeoTransform() == expected.GetGeoTransform()
        assert actual.GetProjection() == expected.GetProjection()
        assert actual.GetMetadata() == expected.GetMetadata()
        assert actual.RasterCount == expected.RasterCount == 24
        name = filename.split('_')[0]
        field = {'TMRT': 'Tmrt', 'Shadow': 'shadow'}.get(name, name)
        rule = PROTOCOL['other_outputs'].get(name, PROTOCOL['field_rules'].get('Solweig_2022a_calc/' + field))
        for i in range(1, 25):
            a, b = actual.GetRasterBand(i), expected.GetRasterBand(i)
            assert a.DataType == b.DataType
            assert a.GetMetadata() == b.GetMetadata()
            assert a.GetNoDataValue() == b.GetNoDataValue()
            x, y = a.ReadAsArray(), b.ReadAsArray()
            for mask in (np.isnan, np.isposinf, np.isneginf):
                np.testing.assert_array_equal(mask(x), mask(y))
            np.testing.assert_allclose(x, y, atol=rule.get('max_abs', rule.get('atol', 0)),
                                       rtol=rule.get('rtol', 0), equal_nan=True,
                                       err_msg=f'{filename} band {i}')
            delta = np.where(np.isfinite(y), abs(x-y), 0)
            index = np.unravel_index(np.argmax(delta), delta.shape)
            if float(delta[index]) >= errors.get(name, {}).get('max_abs', -1):
                errors[name] = {'max_abs': float(delta[index]), 'band': i,
                                'row': int(index[0]), 'col': int(index[1])}
        actual = expected = None
    (tmp_path / 'comparison.json').write_text(json.dumps(errors, indent=2) + '\n')
