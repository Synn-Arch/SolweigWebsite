"""Run 29 frozen P4 configurations sequentially with isolated fresh JIT caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configurations():
    jobs = []
    for function in ('gvf_2018a', 'Kside_veg_v2022a', 'Lcyl_v2022a'):
        jobs.extend((function, 'upstream', False, threads, 'dense') for threads in (1, 4, 10))
        for visibility in (('dense',) if function == 'gvf_2018a' else ('dense', 'compact')):
            jobs.extend((function, 'candidate', parallel, threads, visibility) for parallel, threads in ((False, 1), (True, 1), (True, 4), (True, 10)))
    assert len(jobs) == 29
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, default=ROOT / 'tests/reference/p4_benchmark_inputs')
    args = parser.parse_args()
    target = args.run.resolve()
    target.mkdir(parents=True, exist_ok=False)
    protocol = ROOT / 'benchmarks/protocols/p4_radiation_v1.json'
    comparison = ROOT / 'benchmarks/protocols/comparison_v1.json'
    sources = list((ROOT / 'src/solweig_light').rglob('*.py'))
    sources.extend(p for p in (ROOT / '.upstream/SOLWEIG-GPU/solweig_gpu').iterdir() if p.suffix in ('.py', '.txt'))
    sources.extend(ROOT / name for name in ('tools/benchmark_p4_radiation.py', 'tools/run_p4_benchmarks.py', 'tools/check_p4_benchmarks.py'))
    sources.extend((protocol, comparison))
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in sorted(sources)}
    fixture_hashes = {path.name: sha(path) for path in sorted(args.fixture.iterdir()) if path.is_file()}
    shutil.copytree(args.fixture, target / 'fixtures')
    for relative in source_hashes:
        destination = target / 'snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    guard = {'source_sha256': source_hashes}
    guard_path = target / 'guard.json'
    guard_path.write_text(json.dumps(guard, indent=2) + '\n')
    failures = []
    completed = []
    jobs = configurations()
    try:
        for function, backend, parallel, threads, visibility in jobs:
            name = f'{function}_{backend}_{"parallel" if parallel else "serial"}_{threads}_{visibility}'
            assert all(sha(ROOT / path) == digest for path, digest in source_hashes.items()), 'Source change before configuration'
            assert all(sha(target / 'fixtures' / path) == digest for path, digest in fixture_hashes.items()), 'Fixture change'
            command = [str(ROOT / ('.venv-oracle' if backend == 'upstream' else '.venv-light') / 'bin/python'),
                       str(ROOT / 'tools/benchmark_p4_radiation.py'), '--backend', backend, '--function', function,
                       '--threads', str(threads), '--visibility', visibility, '--fixture', str(target / 'fixtures'),
                       '--guard', str(guard_path), '--output', str(target / name)]
            if parallel:
                command.append('--parallel')
            cache = target / (name + '_jit_cache')
            assert not cache.exists()
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(ROOT / 'src') if backend == 'candidate' else '', NUMBA_CACHE_DIR=str(cache))
            for key in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
                environment[key] = str(threads)
            with (target / (name + '.log')).open('w') as log:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            completed.append({'name': name, 'command': command, 'returncode': result.returncode})
            if result.returncode:
                failures.append(completed[-1])
            assert all(sha(ROOT / path) == digest for path, digest in source_hashes.items()), 'Source change during configuration'
            print(name, result.returncode, flush=True)
    except BaseException as error:
        failures.append({'guard_or_runner_failure': repr(error)})
        raise
    finally:
        (target / 'execution_manifest.json').write_text(json.dumps({'scope': 'Frozen synthetic component workload; no end-to-end claim',
            'jobs': len(jobs), 'completed': completed, 'failures': failures, 'source_sha256': source_hashes,
            'fixture_sha256': fixture_hashes, 'protocol_sha256': source_hashes[str(protocol.relative_to(ROOT))],
            'comparison_protocol_sha256': source_hashes[str(comparison.relative_to(ROOT))]}, indent=2, allow_nan=False) + '\n')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
