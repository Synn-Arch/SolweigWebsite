"""Capture 48 original-driver hours; read-only instrumentation, never a benchmark."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

import numpy as np
import torch
from osgeo import gdal
import solweig_gpu
from solweig_gpu import solweig
from solweig_gpu.utci_process import compute_utci
from capture_reference import BoundaryCapture

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
STATE = ('firstdaytime', 'timeadd', 'timestepdec', 'Tgmap1', 'Tgmap1E', 'Tgmap1S', 'Tgmap1W', 'Tgmap1N', 'TgOut1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class StateCapture(BoundaryCapture):
    def __init__(self, destination):
        super().__init__(solweig, destination)
        self.codes = {solweig.Solweig_2022a_calc.__code__: 'Solweig_2022a_calc'}
        self.driver_context = []

    def save(self, function, boundary, values):
        super().save(function, boundary, values)
        event = self.events[-1]
        for name, value in values.items():
            event['fields'][name]['original_python_type'] = type(value).__module__ + '.' + type(value).__qualname__
            if isinstance(value, torch.Tensor):
                event['fields'][name]['original_tensor_shape'] = list(value.shape)

    def profile(self, frame, event, arg):
        if frame.f_code in self.codes and event == 'call':
            driver = frame.f_back.f_locals
            i = int(driver['i'])
            midnight = bool(driver['dectime'][i] % 1 == 0)
            self.driver_context.append({'step': i, 'dectime': float(driver['dectime'][i]),
                'midnight': midnight, 'Twater_recomputed': midnight or i == 0,
                'Twater': float(driver['Twater']), 'CI_entering_engine': float(driver['CI']),
                'daylines_tuple_length': len(np.where(np.floor(driver['dectime']) == driver['dectime'][i])) if midnight else None,
                'CI_reset_branch': 'else: CI = 1.' if midnight else 'no midnight reset'})
        super().profile(frame, event, arg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT / 'tests/reference/state_sequence_original_cpu')
    args = parser.parse_args()
    reference = args.reference.resolve()
    source = ROOT / '.upstream/SOLWEIG-GPU'
    package = Path(solweig_gpu.__file__).parent
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', '-C', str(source), 'diff', '--name-only'], text=True).strip()
    hashes = {}
    for path in sorted((source / 'solweig_gpu').iterdir()):
        if path.suffix in ('.py', '.txt'):
            assert sha(path) == sha(package / path.name), path.name
            hashes[path.name] = sha(path)
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    reference.mkdir(parents=True, exist_ok=False)
    original = ROOT / 'tests/reference/small_original_cpu/scene'
    scene = reference / 'scene'
    shutil.copytree(original, scene, ignore=shutil.ignore_patterns('output_folder'))
    met = np.loadtxt(original / 'met.txt', skiprows=1)
    assert met.shape[0] == 24
    second = met.copy()
    second[:, 1] += 1
    second[:, 11] += 4
    forcing = np.concatenate((met, second))
    header = (original / 'met.txt').read_text().splitlines()[0]
    np.savetxt(scene / 'met.txt', forcing, header=header, comments='', fmt='%.8f')
    pre = scene / 'processed_inputs'
    shutil.rmtree(pre / 'metfiles')
    (pre / 'metfiles').mkdir()
    shutil.copyfile(scene / 'met.txt', pre / 'metfiles/metfile_0_0_2020-07-18.txt')
    template = pre / 'Building_DSM/Building_DSM_0_0.tif'
    lc = pre / 'Landcover/Landcover_0_0.tif'
    lc.parent.mkdir(exist_ok=True)
    ds = gdal.GetDriverByName('GTiff').CreateCopy(str(lc), gdal.Open(str(template)))
    ds.GetRasterBand(1).WriteArray(np.full((ds.RasterYSize, ds.RasterXSize), 7, dtype=np.float32))
    ds = None
    output = scene / 'output_folder/0_0'
    output.mkdir(parents=True)
    flags = {f'save_{name}': True for name in ('tmrt', 'kup', 'kdown', 'lup', 'ldown', 'shadow', 'wbgt', 'ta', 'wind')}
    positional = [str(pre / p) for p in ('Building_DSM/Building_DSM_0_0.tif', 'Trees/Trees_0_0.tif', 'DEM/DEM_0_0.tif', 'walls/walls_0_0.tif', 'aspect/aspect_0_0.tif')]
    dump(reference / 'kwargs.json', {'entrypoint': 'solweig_gpu.utci_process.compute_utci', 'geometry_paths': positional,
        'landcover_path': str(lc), 'windcoeff_path': None, 'met_file': 'scene/met.txt', 'output_path': str(output),
        'number': '0_0', 'selected_date_str': '2020-07-18', **flags})
    dump(reference / 'fixture_hashes.json', {str(p.relative_to(scene)): sha(p) for p in sorted(scene.rglob('*')) if p.is_file()})
    dump(reference / 'environment.json', {'evidence_class': 'original_upstream_cpu', 'source_commit': COMMIT,
        'oracle_patch_hash': None, 'source_sha256': hashes, 'python': sys.version, 'executable': sys.executable,
        'package_path': str(package), 'platform': platform.platform(), 'torch_threads': 1, 'cuda': 'not available',
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'harness_sha256': sha(__file__), 'collector_sha256': sha(ROOT / 'tools/capture_reference.py'),
        'invocation': sys.argv})
    capture = StateCapture(reference / 'boundaries')
    try:
        with (reference / 'stdout.log').open('w') as stdout, (reference / 'stderr.log').open('w') as stderr, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), capture:
            returned = compute_utci(*positional, str(lc), None, forcing, str(output), '0_0', '2020-07-18', **flags)
    except BaseException:
        dump(reference / 'outcome.json', {'status': 'original_failure', 'traceback': traceback.format_exc()})
        raise
    inputs, outputs = {}, {}
    for event in capture.events:
        with np.load(reference / 'boundaries' / event['path']) as archive:
            fields = {key: archive[key].copy() for key in archive.files}
        (inputs if event['boundary'] == 'input' else outputs)[event['timestep']] = fields
    assert set(inputs) == set(outputs) == set(range(48))
    links = 0
    for i in range(1, 48):
        for name in STATE:
            np.testing.assert_array_equal(inputs[i][name], outputs[i - 1][name])
            links += 1
        if i != 24:
            np.testing.assert_array_equal(inputs[i]['CI'], outputs[i - 1]['CI'])
    expected_water = [float(np.mean(forcing[:24, 11])), float(np.mean(forcing[24:, 11]))]
    for i in range(48):
        assert float(inputs[i]['Twater']) == expected_water[i // 24]
        if i in (0, 24):
            assert float(inputs[i]['CI']) == 1
    altitude = [float(inputs[i]['altitude']) for i in range(48)]
    for start in (0, 24):
        assert altitude[start] < 0 and altitude[start + 23] < 0 and any(a > 0 for a in altitude[start:start + 24])
    output_meta = {}
    for path in sorted(output.glob('*.tif')):
        ds = gdal.Open(str(path))
        assert ds.RasterCount == 48
        output_meta[path.name] = {'sha256': sha(path), 'bands': ds.RasterCount,
            'band_metadata': [ds.GetRasterBand(i).GetMetadata() for i in range(1, 49)]}
        ds = None
    dump(reference / 'outcome.json', {'status': 'verified_original_capture', 'return_repr': repr(returned),
        'timesteps': 48, 'engine_boundary_events': len(capture.events), 'state_links_checked': links,
        'daily_Twater_means': expected_water, 'driver_context': capture.driver_context,
        'engine_altitudes': altitude, 'outputs': output_meta,
        'scope': 'Unmodified original compute_utci tile driver executes 48 met rows continuously; public run_utci_tiles selected-day discovery is not exercised.',
        'quirks': ['Midnight daylines is a one-element np.where tuple, so daylines.__len__()>1 is false and original CI resets to Python float 1.',
                   'selected_date_str is a base date for raster labels; original solar forcing and daily water mean use met day-of-year.'],
        'limitations': ['Read-only capture changes runtime; no benchmark or scientific validation claim.',
                        'No persistent restart API is exercised; full original inputs, outputs and carried state are recorded for candidate restart comparison.']})
    print(f'Verified original 48-hour driver: {len(capture.events)} boundaries, {links} carried-state links, Twater {expected_water}')


if __name__ == '__main__':
    main()
