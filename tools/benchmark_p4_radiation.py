"""One isolated frozen P4 component measurement; outer runner guards sources."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
from check_p4_benchmarks import compare_fields, compare_mutations, sha

ROOT = Path(__file__).resolve().parents[1]


def array(value):
    return value.detach().cpu().numpy().copy() if hasattr(value, 'detach') else np.asarray(value).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['upstream', 'candidate'], required=True)
    parser.add_argument('--function', choices=['gvf_2018a', 'Kside_veg_v2022a', 'Lcyl_v2022a'], required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--visibility', choices=['dense', 'compact'], default='dense')
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--guard', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    guard = json.loads(args.guard.read_text())
    def check_guard():
        assert all(sha(ROOT / name) == digest for name, digest in guard['source_sha256'].items()), 'Sources changed during frozen run'
    check_guard()
    protocol_path = ROOT / 'benchmarks/protocols/p4_radiation_v1.json'
    comparison_path = ROOT / 'benchmarks/protocols/comparison_v1.json'
    protocol = json.loads(protocol_path.read_text())
    rules = json.loads(comparison_path.read_text())
    packets = json.loads((args.fixture / 'manifest.json').read_text())
    case = next(c for c in packets['cases'] if c['function'] == args.function)
    for filename, checksum in [(case['input'], case['input_sha256']), (case['output'], case['output_sha256']), (case['mutation_snapshot'], case['mutation_snapshot_sha256'])]:
        assert sha(args.fixture / filename) == checksum
    with np.load(args.fixture / case['input']) as archive:
        base = {key: archive[key].copy() for key in archive.files}
    if args.backend == 'upstream':
        assert args.visibility == 'dense' and not args.parallel
        import torch
        torch.set_num_threads(args.threads)
        import solweig_gpu.solweig as module
        assert sha(module.__file__) == sha(ROOT / '.upstream/SOLWEIG-GPU/solweig_gpu/solweig.py')
        function = getattr(module, args.function)
    else:
        from numba import set_num_threads
        set_num_threads(args.threads)
        if args.function == 'gvf_2018a':
            assert args.visibility == 'dense'
            from solweig_light.radiation import ground_view as module
            function = getattr(module, args.function + ('_parallel' if args.parallel else ''))
        else:
            from solweig_light.radiation import patch_radiation as module
            original_function = getattr(module, args.function)
            function = lambda **kwargs: original_function(**kwargs, parallel=args.parallel, block_pixels=protocol['candidate_block_pixels'])
        if args.visibility == 'compact':
            from solweig_light.geometry.visibility import PackedVisibility, LazyDiffVisibility
            for key in ('shmat', 'vegshmat', 'vbshvegshmat'):
                dense = base[key]
                packed = PackedVisibility.from_dense(dense)
                for patch in range(dense.shape[2]):
                    assert packed.decode_patch(patch).tobytes() == dense[:, :, patch].tobytes()
                base[key] = packed
            if 'diffsh' in base:
                dense = base['diffsh']
                lazy = LazyDiffVisibility(base['shmat'], base['vegshmat'])
                for patch in range(dense.shape[2]):
                    assert lazy[:, :, patch].tobytes() == dense[:, :, patch].tobytes()
                base['diffsh'] = lazy
            del dense, packed
    def prepare():
        result = {}
        for key, value in base.items():
            spec = case['fields'][key]
            typ = spec['original_python_type']
            if not isinstance(value, np.ndarray):
                result[key] = value  # Immutable encoded channels / lazy expression.
            elif typ.startswith('builtins.'):
                result[key] = value.item()
            elif typ.startswith('numpy.') and spec['kind'] == 'scalar':
                result[key] = value[()]
            elif args.backend == 'upstream' and typ == 'torch.Tensor':
                result[key] = torch.from_numpy(value.copy())
            else:
                result[key] = value.copy()
        return result
    with np.load(args.fixture / case['output']) as archive:
        expected = dict(archive)
    with np.load(args.fixture / case['mutation_snapshot']) as archive:
        expected_mutations = dict(archive)
    times = []
    mutation_fields = {}
    for iteration in range(1 + protocol['warm_repetitions']):
        kwargs = prepare()
        before = {key: array(value) for key, value in kwargs.items() if isinstance(value, (np.ndarray, int, float, bool, np.number)) or hasattr(value, 'detach')}
        gc.collect()
        start = time.perf_counter()
        result = function(**kwargs)
        times.append(time.perf_counter() - start)
        check_guard()
        result = result if isinstance(result, tuple) else (result,)
        output_fields = {key: array(value) for key, value in zip(case['output_names'], result, strict=True)}
        compare_fields(args.function, output_fields, expected, rules)
        mutation_fields = {key: array(kwargs[key]) for key in before if array(kwargs[key]).tobytes() != before[key].tobytes()}
        compare_mutations(mutation_fields, expected_mutations)
        del kwargs, before, result
        if iteration != protocol['warm_repetitions']:
            del output_fields
    np.savez_compressed(args.output / 'outputs.npz', **output_fields)
    np.savez_compressed(args.output / 'mutations.npz', **mutation_fields)
    check_guard()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {'backend': args.backend, 'function': args.function, 'parallel': args.parallel, 'visibility': args.visibility,
        'threads': args.threads, 'candidate_block_pixels': protocol['candidate_block_pixels'] if args.backend == 'candidate' else None,
        'input_sha256': case['input_sha256'], 'protocol_sha256': sha(protocol_path),
        'captured_fixture_protocol_sha256': packets['protocol_sha256'],
        'captured_protocol_note': 'Fixture captured before candidate_block_pixels=128 was added; no spatial/forcing/patch policy changed.',
        'comparison_protocol_sha256': sha(comparison_path), 'source_sha256': guard['source_sha256'],
        'first_call_seconds': times[0], 'warm_seconds': times[1:],
        'process_lifetime_peak_rss_bytes': int(peak if sys.platform == 'darwin' else peak * 1024),
        'memory_scope': protocol['memory'], 'python': sys.version, 'platform': platform.platform(), 'invocation': sys.argv,
        'dependencies': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        'thread_environment': {key: os.environ.get(key) for key in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
        'numba_cache_dir': os.environ.get('NUMBA_CACHE_DIR'), 'all_four_calls_outputs_and_mutations_passed': True,
        'artifacts': {p.name: sha(p) for p in args.output.glob('*.npz')}}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
