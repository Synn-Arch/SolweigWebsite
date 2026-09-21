"""Execute the frozen wind resource characterization with original verification."""
import argparse
import json
import os
import platform
from pathlib import Path
import runpy
import sys

from measure_p6_runtime import (ROOT, monitor, write_json, sha256,
                                candidate_source_hashes, assert_source_unchanged)

PROTOCOL = ROOT / 'benchmarks/protocols/p6_wind_resources_v1.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', type=int, choices=(1, 2, 4))
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    if args.child:
        from importlib import metadata
        import solweig_light
        assert 'site-packages' in Path(solweig_light.__file__).parts
        # This test executes the public workflow on fresh copied inputs and
        # checks every original field, mask, artifact name and metadata value.
        namespace = runpy.run_path(str(ROOT / 'tests/optional/test_wind_generation.py'))
        case = next(item for item in namespace['CASES'] if item['name'] == 'precedence')
        namespace['test_original_twelve_direction_outputs'](case, args.child, args.directory)
        write_json(args.directory / 'outcome.json', {
            'status': 'passed', 'workers': args.child,
            'installed_module': solweig_light.__file__,
            'dependencies': {d.metadata['Name']: d.version for d in metadata.distributions()},
            'reference_case': case,
        })
        return
    args.directory.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(PROTOCOL.read_text())
    sources = candidate_source_hashes()
    write_json(args.directory / 'provenance.json', {
        'protocol': protocol, 'protocol_sha256': sha256(PROTOCOL),
        'harness_sha256': sha256(Path(__file__)),
        'verification_sha256': sha256(ROOT / 'tests/optional/test_wind_generation.py'),
        'monitor_sha256': sha256(ROOT / 'tools/measure_p6_runtime.py'),
        'python': sys.version, 'platform': platform.platform(),
        'processor': platform.processor(), 'logical_cpus': os.cpu_count(),
        'source_sha256': sources,
        'reference_manifest_sha256': sha256(ROOT / 'tests/reference/wind_original_cpu/manifest.json'),
        'wheel_sha256': sha256(ROOT / 'dist/solweig_light-0.1.0.dev0-py3-none-any.whl'),
    })
    results = []
    for workers in protocol['workers']:
        directory = args.directory / f'workers{workers}'
        directory.mkdir()
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS',
                     'NUMBA_NUM_THREADS', 'GDAL_NUM_THREADS'):
            env[name] = '1'
        write_json(directory / 'environment.json', {k: v for k, v in env.items() if 'THREAD' in k})
        result = monitor([str(ROOT / '.venv-optional/bin/python'), str(Path(__file__).resolve()),
                          '--child', str(workers), '--directory', str(directory.resolve())], directory, env)
        write_json(directory / 'measurement.json', result)
        assert_source_unchanged(sources, directory)
        results.append(result)
        assert result['exit_code'] == 0 and not result['memory_budget_exceeded'], result
        assert json.loads((directory / 'outcome.json').read_text())['status'] == 'passed'
    write_json(args.directory / 'summary.json', results)


if __name__ == '__main__':
    main()
