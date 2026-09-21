"""Original-only synthetic component packets; no timing or simulation claim."""
import argparse
import ast
import contextlib
from copy import deepcopy
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
from solweig_gpu import solweig
import solweig_gpu
from solweig_gpu.utci_process import compute_utci

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
FUNCTIONS = ('gvf_2018a', 'Kside_veg_v2022a', 'Lcyl_v2022a')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clone(value):
    return value.clone() if isinstance(value, torch.Tensor) else deepcopy(value)


def array(value):
    return value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else np.asarray(value).copy()


def schema(value):
    result = {'kind': 'array' if isinstance(value, (torch.Tensor, np.ndarray)) else 'scalar',
              'original_python_type': type(value).__module__ + '.' + type(value).__qualname__,
              'dtype': str(array(value).dtype), 'shape': list(array(value).shape)}
    if isinstance(value, torch.Tensor):
        result['torch_dtype'] = str(value.dtype)
        result['torch_device'] = str(value.device)
    return result


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT / 'tests/reference/p4_benchmark_inputs')
    args = parser.parse_args()
    reference = args.reference.resolve()
    protocol_path = ROOT / 'benchmarks/protocols/p4_radiation_v1.json'
    protocol = json.loads(protocol_path.read_text())
    assert protocol['spatial_repeat'] == [4, 4] and protocol['patch_count'] == 153
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
    old = ROOT / 'tests/reference/small_original_cpu'
    boundaries = json.loads((old / 'boundaries/manifest.json').read_text())
    selected = next(e for e in boundaries['events'] if e['boundary'] == 'input' and float(np.load(old / 'boundaries' / e['path'])['altitude']) > 30)
    step = selected['timestep']
    captures = {}
    selected_main = {}
    codes = {getattr(solweig, name).__code__: name for name in FUNCTIONS}
    def profile(frame, event, arg):
        if event != 'call':
            return
        if frame.f_code == solweig.Solweig_2022a_calc.__code__ and int(frame.f_locals['i']) == step:
            selected_main.update({name: clone(value) for name, value in frame.f_locals.items()})
        name = codes.get(frame.f_code)
        if name and int(frame.f_back.f_locals.get('i', -1)) == step and name not in captures:
            captures[name] = {key: clone(frame.f_locals[key]) for key in inspect.signature(getattr(solweig, name)).parameters}
    with tempfile.TemporaryDirectory(prefix='.capture-', dir=reference) as working:
        scene = Path(working) / 'scene'
        shutil.copytree(old / 'scene', scene, ignore=shutil.ignore_patterns('output_folder'))
        pre = scene / 'processed_inputs'
        output = scene / 'output_folder/0_0'
        output.mkdir(parents=True)
        positional = [str(pre / p) for p in ('Building_DSM/Building_DSM_0_0.tif', 'Trees/Trees_0_0.tif', 'DEM/DEM_0_0.tif', 'walls/walls_0_0.tif', 'aspect/aspect_0_0.tif')]
        previous = sys.getprofile()
        assert previous is None
        try:
            with (reference / 'capture_stdout.log').open('w') as out, (reference / 'capture_stderr.log').open('w') as err, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                sys.setprofile(profile)
                compute_utci(*positional, None, None, np.loadtxt(scene / 'met.txt', skiprows=1), str(output), '0_0', '2020-07-18')
        finally:
            sys.setprofile(previous)
    assert set(captures) == set(FUNCTIONS)
    # Confirm this fresh original-driver call matches the archived original main
    # forcing/geometry/state bit-for-bit, while retaining actual Python types.
    with np.load(old / 'boundaries' / selected['path']) as archive:
        for name, value in selected_main.items():
            if isinstance(value, dict):
                for key, child in value.items():
                    np.testing.assert_array_equal(array(child), archive[name + '/' + key])
            elif isinstance(value, list):
                assert value == []
            elif value is not None:
                np.testing.assert_array_equal(array(value), archive[name], err_msg=name)
    rows, cols = int(selected_main['rows']), int(selected_main['cols'])
    assert (rows, cols) == (32, 35)
    tree = ast.parse(Path(solweig.__file__).read_text())
    returns = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            value = max((n for n in ast.walk(node) if isinstance(n, ast.Return)), key=lambda n: n.lineno).value
            returns[node.name] = [ast.unparse(n) for n in value.elts] if isinstance(value, ast.Tuple) else [ast.unparse(value)]
    cases = []
    for name in FUNCTIONS:
        kwargs = captures[name]
        for key, value in tuple(kwargs.items()):
            shape = tuple(value.shape) if isinstance(value, (torch.Tensor, np.ndarray)) else ()
            if len(shape) >= 2 and shape[:2] == (rows, cols):
                repeats = (4, 4) + (1,) * (len(shape) - 2)
                kwargs[key] = value.repeat(*repeats) if isinstance(value, torch.Tensor) else np.tile(value, repeats)
            elif key in ('rows', 'cols'):
                kwargs[key] = value * 4
        fields = {key: schema(value) for key, value in kwargs.items()}
        before = {key: array(value) for key, value in kwargs.items()}
        input_path = reference / (name + '_inputs.npz')
        np.savez_compressed(input_path, **before)
        with (reference / (name + '_stdout.log')).open('w') as out, contextlib.redirect_stdout(out):
            result = getattr(solweig, name)(**kwargs)
        result = result if isinstance(result, tuple) else (result,)
        assert len(result) == len(returns[name])
        output_path = reference / (name + '_outputs.npz')
        np.savez_compressed(output_path, **{key: array(value) for key, value in zip(returns[name], result, strict=True)})
        mutated = {key: array(value) for key, value in kwargs.items() if before[key].tobytes() != array(value).tobytes()}
        mutation_path = reference / (name + '_mutations.npz')
        np.savez_compressed(mutation_path, **mutated)
        cases.append({'function': name, 'input': input_path.name, 'input_sha256': sha(input_path),
            'fields': fields, 'output': output_path.name, 'output_sha256': sha(output_path),
            'output_names': returns[name], 'output_fields': {key: schema(value) for key, value in zip(returns[name], result, strict=True)},
            'mutations': {key: {'snapshot': mutation_path.name, 'key': key, **fields[key]} for key in mutated},
            'mutation_snapshot': mutation_path.name, 'mutation_snapshot_sha256': sha(mutation_path)})
    dump(reference / 'manifest.json', {'evidence_class': 'original_upstream_cpu_component_synthetic_repeated_spatial_fixture',
        'scope': 'Original component calls on repeated entry arguments; not full simulation goldens or benchmark timings.',
        'source_commit': COMMIT, 'oracle_patch_hash': None, 'source_sha256': hashes,
        'protocol': str(protocol_path.relative_to(ROOT)), 'protocol_sha256': sha(protocol_path),
        'harness_sha256': sha(__file__), 'invocation': sys.argv,
        'original_main_input': selected, 'original_main_input_sha256': sha(old / 'boundaries' / selected['path']),
        'original_main_matches_archive': True, 'original_shape': [rows, cols], 'repeated_shape': [rows * 4, cols * 4],
        'patch_count': 153, 'cases': cases,
        'environment': {'python': sys.version, 'executable': sys.executable, 'package_path': str(package),
                        'platform': platform.platform(), 'torch_threads': 1, 'cuda': 'not available',
                        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}})
    print(json.dumps({'step': step, 'cases': [{'function': c['function'], 'mutations': list(c['mutations'])} for c in cases]}))


if __name__ == '__main__':
    main()
