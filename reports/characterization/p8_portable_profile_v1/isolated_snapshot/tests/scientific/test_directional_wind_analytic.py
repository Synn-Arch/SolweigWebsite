"""Explicit direction/sector expectations through a genuine chronological run.

The lookup table below is independently enumerated, not calculated with the
implementation's rounding expression. Expected speed is forcing times the
named sector coefficient, subject to the documented compatibility floor.
"""
from pathlib import Path
import shutil

import numpy as np
from osgeo import gdal

from solweig_light import RuntimeOptions, run_utci_tiles, runtime_options


def test_directional_sector_centres_edges_wrap_and_missing_values(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = root / 'tests/reference/small_original_cpu/scene/processed_inputs'
    prepared = tmp_path / 'processed_inputs'
    shutil.copytree(source, prepared, ignore=shutil.ignore_patterns('SVF', 'WindCoeff'))
    met_path = next((prepared / 'metfiles').glob('*.txt'))
    header = met_path.read_text().splitlines()[0]
    met = np.loadtxt(met_path, skiprows=1)
    assert met.shape[0] == 24 and met.shape[1] >= 24
    met[:, 9] = 2.0
    met[:, 23] = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330,
                  14.999, 15, 44.999, 45, 344.999, 345, 359.999, 360, 720,
                  -999, -1, np.nan]
    np.savetxt(met_path, met, header=header, comments='', fmt='%.8f')
    expected_speeds = [.15, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5,
                       .15, .5, .5, 1, 5.5, .15, .15, .15, .15, 2, 2, 2]

    template = gdal.Open(str(next((prepared / 'Building_DSM').glob('*.tif'))))
    wind = prepared / 'WindCoeff'
    wind.mkdir()
    for index, direction in enumerate(range(0, 360, 30)):
        dataset = gdal.GetDriverByName('GTiff').Create(
            str(wind / f'WindCoeff_dir{direction:03d}_0_0.tif'),
            template.RasterXSize, template.RasterYSize, 1, gdal.GDT_Float32)
        dataset.SetGeoTransform(template.GetGeoTransform())
        dataset.SetProjection(template.GetProjection())
        dataset.GetRasterBand(1).Fill(index / 4)
        dataset.FlushCache()
        dataset = None
    template = None

    with runtime_options(RuntimeOptions(cpu_budget=1, workers=1, threads_per_worker=1,
                                        cache_enabled=False, legacy_cache_policy='recompute')):
        run_utci_tiles(str(tmp_path), str(prepared), '2020-07-18',
                       tile_keys=['0_0'], save_tmrt=False, save_wind=True)
    result = gdal.Open(str(tmp_path / 'output_folder/0_0/Wind_0_0.tif'))
    assert result is not None and result.RasterCount == 24
    for band, expected in enumerate(expected_speeds, 1):
        actual = result.GetRasterBand(band).ReadAsArray()
        # Coefficients and forcing are binary-exact; the only nonbinary value
        # is the explicitly float32 compatibility floor, so exact equality is
        # appropriate rather than a fitted numerical tolerance.
        np.testing.assert_array_equal(actual, np.full_like(actual, np.float32(expected)))
    result = None
