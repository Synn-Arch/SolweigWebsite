"""Check frozen P4 fields, masks, mutations and equal-work provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_fields(function, actual, expected, rules):
    assert set(actual) == set(expected)
    results = {}
    for name in expected:
        x, y = np.asarray(actual[name]), np.asarray(expected[name])
        assert x.shape == y.shape and x.dtype == y.dtype, name
        for mask in (np.isnan, np.isposinf, np.isneginf):
            np.testing.assert_array_equal(mask(x), mask(y), err_msg=name)
        budget = rules['field_rules'][function + '/' + name]
        if 'columns' in budget:
            np.testing.assert_array_equal(x[:, :2], y[:, :2], err_msg=name)
            x, y = x[:, 2], y[:, 2]
            budget = budget['columns'][2]
        if budget.get('rule', '').startswith('exact'):
            np.testing.assert_array_equal(x, y, err_msg=name)
        else:
            finite = np.isfinite(y)
            error = np.abs(x[finite] - y[finite])
            limit = budget.get('max_abs', budget.get('atol', 0)) + budget.get('rtol', 0) * np.abs(y[finite])
            assert np.all(error <= limit), f'{function}/{name}: max error {error.max()}'
        delta = np.where(np.isfinite(y), np.abs(x - y), 0)
        index = np.unravel_index(np.argmax(delta), delta.shape) if delta.size else ()
        finite = np.isfinite(y)
        signed = x[finite].astype(np.float64) - y[finite].astype(np.float64)
        absolute = np.abs(signed)
        nonzero = np.abs(y[finite]) > 0
        relative = absolute[nonzero] / np.abs(y[finite][nonzero])
        results[name] = {
            'max_abs': float(delta[index]) if delta.size else 0.,
            'worst_coordinate': [int(i) for i in index],
            'finite_count': int(signed.size),
            'bias': float(np.mean(signed)) if signed.size else None,
            'mae': float(np.mean(absolute)) if signed.size else None,
            'rmse': float(np.sqrt(np.mean(signed ** 2))) if signed.size else None,
            'p95_abs': float(np.percentile(absolute, 95)) if signed.size else None,
            'p99_abs': float(np.percentile(absolute, 99)) if signed.size else None,
            'max_relative_nonzero_reference': float(relative.max()) if relative.size else None,
        }
    return results


def compare_mutations(actual, expected):
    assert set(actual) == set(expected), ('mutated argument set', set(actual), set(expected))
    for name in expected:
        x, y = actual[name], expected[name]
        assert x.dtype == y.dtype and x.shape == y.shape
        assert x.tobytes() == y.tobytes(), name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.directory / 'execution_manifest.json').read_text())
    assert manifest['jobs'] == 29 and not manifest['failures']
    reference = args.directory / 'fixtures'
    for name, digest in manifest['fixture_sha256'].items():
        assert sha(reference / name) == digest, ('fixture integrity', name)
    for name, digest in manifest['source_sha256'].items():
        assert sha(args.directory / 'snapshot' / name) == digest, ('snapshot integrity', name)
    packets = json.loads((reference / 'manifest.json').read_text())
    rules = json.loads((args.directory / 'snapshot/benchmarks/protocols/comparison_v1.json').read_text())
    entries = list(args.directory.glob('*/report.json'))
    assert len(entries) == 29
    summaries = []
    from run_p4_benchmarks import configurations
    expected_configurations = set(configurations())
    observed_configurations = set()
    for path in sorted(entries):
        report = json.loads(path.read_text())
        configuration = (report['function'], report['backend'], report['parallel'], report['threads'], report['visibility'])
        assert configuration not in observed_configurations
        observed_configurations.add(configuration)
        assert len(report['warm_seconds']) == 3 and report['all_four_calls_outputs_and_mutations_passed']
        case = next(c for c in packets['cases'] if c['function'] == report['function'])
        assert report['input_sha256'] == case['input_sha256']
        assert report['protocol_sha256'] == manifest['protocol_sha256']
        assert report['comparison_protocol_sha256'] == manifest['comparison_protocol_sha256']
        assert report['source_sha256'] == manifest['source_sha256']
        for name, checksum in report['artifacts'].items():
            assert sha(path.parent / name) == checksum
        with np.load(path.parent / 'outputs.npz') as a, np.load(reference / case['output']) as e:
            errors = compare_fields(case['function'], dict(a), dict(e), rules)
        with np.load(path.parent / 'mutations.npz') as a, np.load(reference / case['mutation_snapshot']) as e:
            compare_mutations(dict(a), dict(e))
        summaries.append({**{key: report[key] for key in ('function', 'backend', 'parallel', 'visibility', 'threads', 'first_call_seconds', 'warm_seconds', 'process_lifetime_peak_rss_bytes')},
                          'warm_median_seconds': statistics.median(report['warm_seconds']), 'errors': errors})
    assert observed_configurations == expected_configurations
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({'scope': 'Synthetic P4 component calls; not full simulation or end-to-end speedup',
                                      'comparisons_passed': len(summaries), 'results': summaries}, indent=2, allow_nan=False) + '\n')
    print(f'Passed {len(summaries)} frozen P4 configurations')


if __name__ == '__main__':
    main()
