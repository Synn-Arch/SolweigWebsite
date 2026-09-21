"""Original public roughness generation/consumption oracle, no patched code."""
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import rasterio
import solweig_gpu

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'tests/reference/public_roughness_original_cpu'
COMMIT = '0d7fe742abeeddd890dd58fc76ed7f78bd47faec'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = ROOT / '.upstream/SOLWEIG-GPU'
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', '-C', str(source), 'diff', '--name-only'], text=True).strip()
    package = Path(solweig_gpu.__file__).parent
    hashes = {}
    for path in (source / 'solweig_gpu').iterdir():
        if path.suffix in ('.py', '.txt'):
            assert sha(path) == sha(package / path.name)
            hashes[path.name] = sha(path)
    REF.mkdir(parents=True, exist_ok=False)
    scene = REF / 'scene'
    scene.mkdir()
    original = ROOT / 'tests/reference/small_original_cpu/scene'
    for name in ('Building_DSM.tif', 'DEM.tif', 'Trees.tif'):
        with rasterio.open(original / name) as ds:
            data = np.pad(ds.read(1), ((0, 0), (0, 13)), mode='edge')
            profile = ds.profile.copy()
        profile.update(width=48, height=32)
        with rasterio.open(scene / name, 'w', **profile) as ds:
            ds.write(data, 1)
    with rasterio.open(scene / 'Building_DSM.tif') as ds:
        dsm, profile = ds.read(1), ds.profile.copy()
    with rasterio.open(scene / 'DEM.tif') as ds:
        dem = ds.read(1)
    with rasterio.open(scene / 'Buildings.tif', 'w', **profile) as ds:
        ds.write(np.maximum(dsm - dem, 0).astype(np.float32), 1)
    met = np.loadtxt(original / 'met.txt', skiprows=1)
    met[:, 23] = np.arange(24) % 12 * 30
    np.savetxt(scene / 'met.txt', met, header=(original / 'met.txt').read_text().splitlines()[0], comments='', fmt='%.8f')
    era = REF / 'era5'
    shutil.copytree(ROOT / 'tests/reference/wind_original_cpu/precedence/era5', era)
    kwargs = dict(base_path=str(scene), selected_date_str='2020-07-18', ERA_5_z0_find=True,
                  tile_size=64, overlap=0, use_own_met=True, own_met_file=str(scene / 'met.txt'),
                  use_uhi=False, data_folder=str(era), save_wind=True, save_tmrt=False)
    input_hashes = {p.name: sha(p) for p in scene.iterdir()}
    with (REF / 'stdout.log').open('w') as out, (REF / 'stderr.log').open('w') as err, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        returned = solweig_gpu.thermal_comfort(**kwargs)
    assert sorted(p.name for p in scene.glob('WindCoeff_dir*.tif')) == [f'WindCoeff_dir{d:03d}.tif' for d in range(0, 360, 30)]
    assert 'Could not find ERA-5' not in (REF / 'stdout.log').read_text()
    with rasterio.open(scene / 'output_folder/0_0/Wind_0_0.tif') as ds:
        wind = ds.read()
    met_tile = np.loadtxt(scene / 'processed_inputs/metfiles/metfile_0_0.txt', skiprows=1)
    expected = []
    for row in met_tile:
        with rasterio.open(scene / f'processed_inputs/WindCoeff/WindCoeff_dir{int(row[23]):03d}_0_0.tif') as ds:
            coefficient = ds.read(1)
        expected.append(np.maximum(coefficient * np.float32(row[9]), np.float32(.15)))
    np.testing.assert_array_equal(wind, np.stack(expected))
    report = {'evidence_class': 'original_upstream_cpu', 'source_commit': COMMIT, 'oracle_patch_hash': None,
              'source_sha256': hashes, 'harness_sha256': sha(__file__), 'invocation': kwargs, 'return_repr': repr(returned),
              'input_sha256': input_hashes, 'era5_sha256': sha(era / 'data_stream-oper_stepType-instant.nc'),
              'artifact_sha256': {str(p.relative_to(scene)): sha(p) for p in scene.rglob('*') if p.is_file()},
              'checks': {'twelve_directions_generated': True, 'broad_catch_not_triggered': True,
                         'Wind_all_24_bands_matches_directional_coefficient_times_met_wind_with_float32_floor': True},
              'environment': {'python': sys.version, 'executable': sys.executable,
                              'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}},
              'scope': 'Real public thermal_comfort own-met local roughness generation and directional wind consumption; no remote ERA5 acquisition or performance claim.'}
    (REF / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Original public roughness generation and 24-band consumption verified')


if __name__ == '__main__':
    main()
