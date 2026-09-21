"""Run one frozen P2 kernel case; results are not end-to-end benchmarks."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['upstream', 'candidate'], required=True)
    parser.add_argument('--kernel', choices=['walls', 'aspect', 'utci', 'utci_uniform'], required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--callable', dest='function', help='Candidate module:function')
    parser.add_argument('--kwargs', default='{}')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    import numpy as np
    protocol_path = ROOT / 'benchmarks/protocols/p2_kernels_v1.json'
    protocol = json.loads(protocol_path.read_text())
    assert args.threads in protocol['thread_counts']
    case = protocol['cases'][args.kernel]
    shape = tuple(case['shape'])
    rng = np.random.default_rng(protocol['seed'])
    if args.kernel == 'walls':
        inputs = (rng.integers(0, 40, shape).astype(np.float32), 3.)
    elif args.kernel == 'aspect':
        r, c = np.indices(shape)
        dsm = (((r % 16 >= 4) & (r % 16 < 12)) & ((c % 16 >= 4) & (c % 16 < 12))).astype(np.float32) * 12
        # Fixture wall construction independent of the timed candidate kernel.
        walls = np.zeros(shape, np.float64)
        for row in range(1, shape[0]-1):
            for col in range(1, shape[1]-1):
                walls[row,col] = float(np.max(dsm[[row-1,row,row,row+1],[col,col-1,col+1,col]])) - float(dsm[row,col])
        walls[walls < 3] = 0
        inputs = (walls, 1., dsm)
    elif args.kernel == 'utci_uniform':
        inputs = (np.full(shape, case['Ta'], np.float32), np.full(shape, case['RH'], np.float32),
                  rng.uniform(*case['Tmrt_range'], size=shape).astype(np.float32),
                  rng.uniform(*case['wind_range'], size=shape).astype(np.float32))
    else:
        inputs = tuple(rng.uniform(*case[key], size=shape).astype(np.float32)
                       for key in ['Ta_range', 'RH_range', 'Tmrt_range', 'wind_range'])
    input_hash = hashlib.sha256()
    for value in inputs:
        array = np.asarray(value)
        input_hash.update(str(array.dtype).encode())
        input_hash.update(str(array.shape).encode())
        input_hash.update(array.tobytes())
    if args.backend == 'upstream':
        import torch
        torch.set_num_threads(args.threads)
        module_name, name = ('solweig_gpu.calculate_utci', 'utci_calculator') if args.kernel.startswith('utci') else (
            'solweig_gpu.walls_aspect', 'findwalls' if args.kernel == 'walls' else 'filter1Goodwin_as_aspect_v3')
        module = importlib.import_module(module_name)
        source = Path(module.__file__)
        original = ROOT / '.upstream/SOLWEIG-GPU' / 'solweig_gpu' / source.name
        assert source.read_bytes() == original.read_bytes()
        assert subprocess.check_output(['git', '-C', str(original.parent.parent), 'rev-parse', 'HEAD'], text=True).strip() == '0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
        assert not subprocess.check_output(['git', '-C', str(original.parent.parent), 'diff', '--name-only'], text=True).strip()
        function = getattr(module, name)
        if args.kernel.startswith('utci'):
            inputs = tuple(torch.from_numpy(x) for x in inputs)
        invocation = module_name + ':' + name
    else:
        import numba
        numba.set_num_threads(args.threads)
        assert args.function
        module_name, name = args.function.split(':')
        module = importlib.import_module(module_name)
        source = Path(module.__file__)
        function = getattr(module, name)
        invocation = args.function
        if args.kernel == 'utci_uniform':
            inputs = (inputs[0][0, 0], inputs[1][0, 0], inputs[2], inputs[3])
    kwargs = json.loads(args.kwargs)
    start = time.perf_counter()
    result = function(*inputs, **kwargs)
    first = time.perf_counter() - start
    samples = []
    for _ in range(protocol['repetitions']):
        start = time.perf_counter()
        result = function(*inputs, **kwargs)
        samples.append(time.perf_counter() - start)
    if hasattr(result, 'detach'):
        result = result.detach().cpu().numpy()
    result = np.asarray(result)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    output_path = args.report.with_suffix('.npz')
    np.savez_compressed(output_path, output=result)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = dict(backend=args.backend, kernel=args.kernel, threads=args.threads,
        upstream_commit='0d7fe742abeeddd890dd58fc76ed7f78bd47faec', candidate_commit=None,
        input_sha256=input_hash.hexdigest(),
        callable=invocation, kwargs=kwargs, protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), first_call_seconds=first,
        warm_seconds=samples, process_lifetime_peak_rss_bytes=int(peak if sys.platform == 'darwin' else peak*1024),
        output_file=output_path.name, output_sha256=hashlib.sha256(output_path.read_bytes()).hexdigest(),
        shape=list(result.shape), dtype=str(result.dtype), python=sys.version,
        platform=platform.platform(), invocation=sys.argv, numba_cache_dir=os.environ.get('NUMBA_CACHE_DIR'),
        thread_environment={name:os.environ.get(name) for name in
                            ('NUMBA_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
        affinity='not explicitly set; process scheduled by host OS',
        dependencies={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        candidate_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted((ROOT/'src/solweig_light').rglob('*.py'))} if args.backend == 'candidate' else None,
        scope='kernel only; first call excludes imports/fixture creation; lifetime RSS includes them')
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(args.report)


if __name__ == '__main__':
    main()
