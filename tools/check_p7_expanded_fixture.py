"""Run preserved P6 on one frozen fixture and compare original upstream outputs."""
import argparse
import json
from pathlib import Path
import shutil
import sys

from benchmark_p7_pipeline import hashes, digest, dump, expand, measure, compare, LIMIT

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--compare-only', action='store_true', help='Reuse an already executed preserved-P6 run after comparator repair')
    parser.add_argument('--source', type=Path, default=ROOT / 'reports/characterization/p7_p6_baseline/src')
    parser.add_argument('--label', help='Explicit candidate variant label; never labels a candidate as an upstream reference')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    fixture = Path(protocol['fixture'])
    assert hashes(fixture) == protocol['fixture_hashes']
    oracle_environment = json.loads((args.oracle / 'environment.json').read_text())
    assert oracle_environment['evidence_class'] == 'original_upstream_cpu'
    assert oracle_environment['source_commit'] == '0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
    assert json.loads((args.oracle / 'outcome.json').read_text())['status'] == 'executed_not_yet_verified'
    # Compare physical inputs, tolerating only non-model manifest bookkeeping.
    oracle_inputs = json.loads((args.oracle / 'fixture_hashes.json').read_text())
    assert all(oracle_inputs[key] == value for key, value in hashes(fixture).items()
               if key.endswith(('.tif', '.txt', 'kwargs.json')))
    run = args.run.resolve()
    if not args.compare_only:
        run.mkdir(parents=True, exist_ok=False)
    scene = run / 'scene'
    if not args.compare_only:
        shutil.copytree(fixture, scene)
    kwargs = expand(json.loads(Path(protocol['kwargs_manifest']).read_text()), scene)
    source_hashes = hashes(args.source)
    spec = dict(source=str(args.source.resolve()), budget=1, entrypoint='thermal_comfort',
                kwargs=kwargs, runtime=expand(protocol['runtime_options'], scene))
    if args.compare_only:
        # Reuse is limited to the hash-preserved P6 baseline, not mutable src.
        baseline = ROOT / 'reports/characterization/p7_p6_baseline'
        assert args.source.resolve() == (baseline / 'src').resolve()
        recorded = json.loads((baseline / 'manifest.json').read_text())['files']
        assert all(digest(baseline / key) == value for key, value in recorded.items())
        assert json.loads((run / 'execution/spec.json').read_text()) == spec
        result = json.loads((run / 'execution/measurement.json').read_text())
        assert result['exit_code'] == 0
    else:
        dump(run / 'source_before.json', source_hashes)
        result = measure(spec, run / 'execution', sys.executable)
    label = args.label or ('preserved P6' if args.source.resolve() == (ROOT / 'reports/characterization/p7_p6_baseline/src').resolve() else 'candidate source')
    report = {'evidence_class': 'original_upstream_cpu_vs_candidate',
              'candidate_label': label, 'candidate_source': str(args.source.resolve()),
              'protocol_sha256': digest(args.protocol), 'source_sha256': source_hashes,
              'oracle_environment_sha256': digest(args.oracle / 'environment.json'),
              'oracle': str(args.oracle.resolve()), 'measurement': result,
              'measurement_scope': 'Single correctness characterization, not paired performance evidence',
              'memory_limit_bytes': LIMIT, 'passed': False}
    if result['exit_code'] == 0:
        comparison = compare(args.oracle / 'scene', scene, protocol['tiff_field_rules'], protocol['artifact_globs'])
        comparison['evidence_class'] = report['evidence_class']
        report['comparison'] = comparison
        report['passed'] = comparison['passed']
    report['source_unchanged'] = hashes(args.source) == source_hashes
    report['oracle_artifacts'] = hashes(args.oracle / 'scene/output_folder')
    dump(run / 'verification.json', report)
    if not report['passed'] or not report['source_unchanged']:
        raise SystemExit(1)
    print(run / 'verification.json')


if __name__ == '__main__':
    main()
