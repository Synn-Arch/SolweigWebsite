"""Execute all seven frozen original comparisons for one candidate source.

This is a correctness gate, not a timing experiment. Failed cases remain on
disk and do not remove subsequent cases from the matrix.
"""
import argparse
from pathlib import Path
import subprocess
import sys

from benchmark_p7_pipeline import digest, dump, hashes

CASES = {
    'real_vegetation_rich': 'p7_real_vegetation_rich_original_cpu',
    'real_sparse': 'p7_real_sparse_original_cpu',
    'small': 'dependencies_cpu',
    'repeated_block_256': 'p7_repeated_original_cpu',
    'dense_urban_256': 'p7_dense_original_cpu',
    'vegetation_rich_256': 'p7_vegetation_original_cpu',
    'real_dense_urban': 'p7_real_dense_urban_original_cpu',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    before = hashes(args.source)
    records = []
    for case, oracle_name in CASES.items():
        run = args.output / case
        oracle = root / 'reports/runs' / oracle_name
        command = [sys.executable, str(root / 'tools/check_p7_expanded_fixture.py'),
                   '--protocol', str(root / 'benchmarks/protocols/p7_pairs_v2' / case / 'protocol.json'),
                   '--oracle', str(oracle), '--run', str(run),
                   '--source', str(args.source), '--label', args.label]
        result = subprocess.run(command, cwd=root)
        geometry_command = [sys.executable, str(root / 'tools/check_p7_geometry_exports.py'),
                            str(oracle / 'scene/processed_inputs/SVF'),
                            str(run / 'scene/processed_inputs/SVF'),
                            '--report', str(run / 'geometry_exports.json')]
        geometry = subprocess.run(geometry_command, cwd=root)
        record = dict(case_id=case, command=command, exit_code=result.returncode,
                      geometry_command=geometry_command, geometry_exit_code=geometry.returncode)
        for name in ('verification.json', 'geometry_exports.json'):
            path = run / name
            if path.exists():
                record[name] = {'path': str(path.resolve()), 'sha256': digest(path)}
        records.append(record)
        dump(args.output / 'matrix.json', dict(status='running', records=records))
        print(case, 'pipeline', result.returncode, 'geometry', geometry.returncode, flush=True)
    unchanged = before == hashes(args.source)
    passed = unchanged and all(r['exit_code'] == r['geometry_exit_code'] == 0 for r in records)
    dump(args.output / 'matrix.json', dict(status='passed' if passed else 'failed',
         source=str(args.source.resolve()), source_hashes=before, source_unchanged=unchanged,
         label=args.label, scope='original-reference correctness; no performance claim', records=records))
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
