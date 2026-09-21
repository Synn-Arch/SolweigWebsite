"""Bind unchanged P7 workloads to the independently checked repair-v2 baseline."""
import argparse
import json
from pathlib import Path

from benchmark_p7_pipeline import digest, dump, validate_protocol_baseline

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--harden-instrument',action='store_true')
    args=parser.parse_args()
    parent = ROOT / ('benchmarks/protocols/p7_pairs_v3' if args.harden_instrument else 'benchmarks/protocols/p7_pairs_v2')
    target = ROOT / ('benchmarks/protocols/p7_pairs_v4' if args.harden_instrument else 'benchmarks/protocols/p7_pairs_v3')
    baseline = ROOT / 'reports/characterization/p7_repaired_p6_baseline_v2'
    manifest_path = baseline / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    gate = ROOT / 'reports/characterization/p7_repair_v2_correctness/gate.json'
    assert manifest['repair_policy'] == 'angular_and_svf_weights_v1'
    assert json.loads(gate.read_text())['status'] == 'passed'
    target.mkdir(exist_ok=False)
    records = []
    for original_path in sorted(parent.glob('*/protocol.json')):
        protocol = json.loads(original_path.read_text())
        amended = dict(protocol)
        amended.update(
            schema_version=4 if args.harden_instrument else 3,
            status='workload_frozen_before_tuning; baseline_amended_after_correctness_repair_before_paired_timing',
            scope='Repaired P6 versus optimized candidate; not unchanged P6 or upstream speedup evidence',
            parent_protocol=str(original_path), parent_protocol_sha256=digest(original_path),
            baseline_manifest=str(manifest_path), baseline_manifest_sha256=digest(manifest_path),
            baseline_label=manifest['label'], repair_policy=manifest['repair_policy'],
            repair_patch_sha256=manifest['repair_patch_sha256'], correctness_report=str(gate),
            correctness_report_sha256=digest(gate),
            instrument_sha256=digest(ROOT / 'tools/benchmark_p7_pipeline.py'),
            amendment_script_sha256=digest(Path(__file__)),
            amendment_reason='Original-reference failures required angular and annulus arithmetic repairs in both variants; original failed reports retained',
        )
        # Only baseline/provenance framing is amended. Every workload and
        # numerical-policy field from v2 must remain byte-equivalent as JSON.
        changed = {'schema_version', 'status', 'scope'}
        if args.harden_instrument:
            changed.update({'parent_protocol','parent_protocol_sha256','instrument_sha256','amendment_script_sha256'})
        assert all(amended[k] == v for k, v in protocol.items() if k not in changed)
        assert amended['repetitions'] == 5 and amended['native_budgets'] == [1, 4]
        assert amended['regimes'] == ['cold', 'geometry_warm']
        validation = validate_protocol_baseline(baseline / 'src', amended, ROOT / 'src')
        directory = target / original_path.parent.name
        directory.mkdir()
        dump(directory / 'protocol.json', amended)
        dump(directory / 'baseline_validation.json', validation)
        records.append({'case_id': directory.name, 'protocol_sha256': digest(directory / 'protocol.json')})
    assert len(records) == 7
    dump(target / 'manifest.json', {'status': 'frozen_not_measured', 'protocols': records})
    print(target)


if __name__ == '__main__':
    main()
