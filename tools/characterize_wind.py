"""Real original-only wind generation fixtures; no performance evidence."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import numpy as np
import rasterio
from rasterio.transform import from_origin
import xarray as xr
import solweig_gpu
from solweig_gpu.wind_ext_coeff import calculate_wind_ext_coeff, _read_z0_from_fsr_at_raster_midpoint

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '0d7fe742abeeddd890dd58fc76ed7f78bd47faec'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def metadata(path):
    with rasterio.open(path) as ds:
        return {'shape': list(ds.shape), 'count': ds.count, 'dtype': ds.dtypes[0], 'crs': ds.crs.to_wkt(),
                'transform': list(ds.transform), 'nodata': 'NaN' if np.isnan(ds.nodata) else ds.nodata,
                'compression': ds.compression.value, 'tiled': ds.profile['tiled'],
                'block_shapes': [list(s) for s in ds.block_shapes], 'tags': ds.tags(), 'band_tags': ds.tags(1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT / 'tests/reference/wind_original_cpu')
    args = parser.parse_args()
    source = ROOT / '.upstream/SOLWEIG-GPU'
    package = Path(solweig_gpu.__file__).parent
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', '-C', str(source), 'diff', '--name-only'], text=True).strip()
    hashes = {}
    for path in sorted((source / 'solweig_gpu').iterdir()):
        if path.suffix in ('.py', '.txt'):
            assert sha(path) == sha(package / path.name), path.name
            hashes[path.name] = sha(path)
    reference = args.reference.resolve()
    reference.mkdir(parents=True, exist_ok=False)
    cases = []
    for name in ('precedence', 'dsm_fallback', 'missing_fsr', 'invalid_fsr', 'invalid_blocks'):
        directory = reference / name
        inputs, era5 = directory / 'inputs', directory / 'era5'
        inputs.mkdir(parents=True)
        era5.mkdir()
        shape = (32, 35) if name == 'invalid_blocks' else (32, 48)
        rows, cols = np.indices(shape)
        building = np.zeros(shape, np.float32)
        building[5:10, 8:14] = 12
        building[15:22, 24:28] = 6
        building[10, 15] = 3  # Diagonal disconnected component.
        building[11, 16] = 7
        building[0, 0] = np.nan
        trees = np.zeros(shape, np.float32)
        trees[4:11, 20:25] = 16
        trees[21:28, 9:15] = 7
        trees[18:22, 25:28] = 11  # Overlapping obstacle masks.
        trees[-1, -1] = np.nan
        profile = {'driver': 'GTiff', 'height': shape[0], 'width': shape[1], 'count': 1,
                   'dtype': 'float32', 'crs': 'EPSG:32631', 'transform': from_origin(500000, 5700000, 2, 2), 'nodata': np.nan}
        values = {'Buildings.tif': building, 'Building_DSM.tif': np.where(building > 0, building + 9, 0).astype(np.float32),
                  'DEM.tif': np.full(shape, 9, np.float32), 'Trees.tif': trees}
        if name == 'dsm_fallback':
            del values['Buildings.tif']
        for filename, value in values.items():
            with rasterio.open(inputs / filename, 'w', **profile) as ds:
                ds.write(value, 1)
        times = np.array(['2020-07-18T00:00', '2020-07-18T01:00'], dtype='datetime64[m]')
        fsr = np.full((2, 2, 2), .27, np.float64)
        fsr[1] = .91
        if name == 'invalid_fsr':
            fsr[0] = np.nan
        ds = xr.Dataset({'other' if name == 'missing_fsr' else 'fsr': (('valid_time', 'latitude', 'longitude'), fsr)},
                        coords={'valid_time': times, 'latitude': [51.4, 51.5], 'longitude': [2.9, 3.1]})
        met = era5 / 'data_stream-oper_stepType-instant.nc'
        ds.to_netcdf(met)
        building_path = inputs / ('Building_DSM.tif' if name == 'dsm_fallback' else 'Buildings.tif')
        selected_z0 = _read_z0_from_fsr_at_raster_midpoint(met, building_path, .03)
        assert selected_z0 == (.03 if name in ('missing_fsr', 'invalid_fsr') else .27)
        invocation = {'input_dir': str(inputs), 'era5_dir': str(era5), 'max_workers': 1}
        entry = {'name': name, 'invocation': invocation, 'selected_z0': selected_z0,
                 'input_sha256': {p.name: sha(p) for p in inputs.glob('*.tif')}, 'met_sha256': sha(met)}
        with (directory / 'stdout.log').open('w') as out, (directory / 'stderr.log').open('w') as err, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                outputs = calculate_wind_ext_coeff(**invocation)
                entry.update(status='executed', returned_names=[p.name for p in outputs],
                             output_sha256={p.name: sha(p) for p in outputs},
                             metadata={p.name: metadata(p) for p in outputs})
                np.savez_compressed(directory / 'fields.npz', **{p.stem: rasterio.open(p).read(1) for p in outputs})
                entry['fields_sha256'] = sha(directory / 'fields.npz')
            except BaseException:
                entry.update(status='original_failure', traceback=traceback.format_exc())
        assert entry['status'] == ('original_failure' if name == 'invalid_blocks' else 'executed')
        dump(directory / 'outcome.json', entry)
        cases.append(entry)
    dump(reference / 'manifest.json', {'evidence_class': 'original_upstream_cpu', 'source_commit': COMMIT,
        'oracle_patch_hash': None, 'source_sha256': hashes, 'harness_sha256': sha(__file__), 'invocation': sys.argv,
        'scope': 'Real tiny TIFF and roughness NetCDF twelve-direction generation; no timing claim.', 'cases': cases,
        'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                        'package_path': str(package), 'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}})
    print(json.dumps({c['name']: c['status'] for c in cases}))


if __name__ == '__main__':
    main()
