"""Index measured evidence without promoting incomplete release/performance gates."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    return json.loads((ROOT / relative).read_text())


def reference(relative):
    path = ROOT / relative
    return {'path': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write(relative, data):
    (ROOT / relative).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def main():
    component = 'reports/characterization/p7_repaired_v2_component_matrix'
    checks = load(component + '/comparison.json')
    executions = load(component + '/execution_manifest.json')
    assert checks['comparisons_passed'] == 29
    assert executions['jobs'] == len(executions['completed']) == 29
    assert not executions['failures'] and all(z['returncode'] == 0 for z in executions['completed'])
    completed = []
    for path in sorted((ROOT / 'reports/characterization/p7_pairs_v4').glob('*/summary.json')):
        completed.append({'case_id': path.parent.name,
                          'evidence': reference(str(path.relative_to(ROOT))),
                          'global_evidence_eligible': json.loads(path.read_text())['global_evidence_eligible']})
    shared = {'status': 'incomplete_release_evidence_index', 'release_gate_passed': False,
              'upstream_commit': '0d7fe742abeeddd890dd58fc76ed7f78bd47faec',
              'candidate_commit': None, 'candidate_state': 'uncommitted worktree; individual records bind exact sources',
              'cuda': {'historical_m1': 'not_available',
                       'cura': 'original_warm_passed_original_cold_resource_failed_patched_cold_passed_performance_unmeasured',
                       'patched_cold_correctness': reference('reports/cura_p8_20260919T1215Z_a7c3/cuda_admission/patched_workerlimit/manifest.json'),
                       'warm_correctness': reference('reports/cura_p8_20260919T1215Z_a7c3/cuda_admission/cuda_geometry_warm_admission.json'),
                       'cold_failure': reference('reports/cura_p8_20260919T1215Z_a7c3/cuda_admission/cuda_feasibility_manifest.json'),
                       'memory_diagnosis': reference('reports/cura_p8_20260919T1215Z_a7c3/cuda_admission/cuda_startup_memory_diagnostic.json'),
                       'assessment': reference('reports/cura_p8_20260919T1215Z_a7c3/assessment.md')},
              'remote_ci': 'configured_not_executed',
              'recorder': reference('tools/record_release_measurement_indices.py')}
    performance = dict(shared, published_speedup_claims=[], evidence=[
        {'scope': 'historical original CPU thread tuning; see own protocol and limitations',
         'report': reference('reports/upstream_cpu_tuning.json')},
        {'scope': 'historical P2 kernel-only comparisons, not end-to-end',
         'report': reference('reports/p2_kernel_comparison.json')},
        {'scope': 'historical P3 geometry component comparisons, not end-to-end',
         'report': reference('reports/p3_geometry_comparison.json')},
        {'scope': 'historical P4 radiation component comparisons, including recorded regressions',
         'report': reference('reports/p4_radiation_comparison.json')},
        {'scope': 'current repaired-v2 candidate, frozen 29-configuration component matrix only',
         'report': reference(component + '/comparison.json'),
         'execution': reference(component + '/execution_manifest.json'),
         'comparisons_passed': 29}],
        p7_full_pipeline={'scope': 'repair-only P6 versus optimized candidate; not upstream speedup',
                          'protocol_manifest': reference('benchmarks/protocols/p7_pairs_v4/manifest.json'),
                          'completed_scene_summaries': completed,
                          'required_scenes': 7, 'repetitions_per_regime_budget': 5,
                          'partial_v3_not_claim_eligible': reference('reports/characterization/p7_pairs_v3/interruption.json')},
        p7_persistent_jit=reference('reports/characterization/p7_persistent_jit_promotion/report.json'),
        p7_rejected_angular_moments=reference('reports/characterization/p7_angular_moment_experiment/pairs_v2/summary.json'),
        p7_checkpoint_review=reference('reports/p7_checkpoint_review.json'),
        outstanding=['Current-main cross-platform numerical repair and final P7 gate',
                     'P8 equal-work original default/best CPU versus candidate release matrix',
                     'Release hardware/workload coverage and uncertainty assessment'])
    write('reports/performance_results.json', performance)
    runtime_root = 'reports/characterization/p6_runtime_v1_retry2_psutil'
    wind_root = 'reports/characterization/p6_wind_resources_v1'
    core = load(runtime_root + '/summary.json')['cases']
    wind = load(wind_root + '/summary.json')
    assert len(core) == 8 and len(wind) == 3
    rows = []
    for name, measurement in ([(case['name'], case['measurement']) for case in core]
                              + [(f'wind_workers{workers}', value) for workers, value in zip((1, 2, 4), wind)]):
        assert measurement['exit_code'] == 0 and not measurement['memory_budget_exceeded']
        peak = measurement['sampled_process_tree_peak_rss_bytes']
        assert 0 < peak <= measurement['memory_budget_bytes']
        rows.append({'case': name, 'sampled_peak_process_tree_rss_bytes': peak,
                     'memory_budget_bytes': measurement['memory_budget_bytes'],
                     'sample_count': measurement['sample_count'],
                     'sample_interval_seconds': measurement['sample_interval_seconds'],
                     'command': measurement['command']})
    memory = dict(shared, historical_sampled_cases=rows,
        evidence=[reference(runtime_root + '/summary.json'), reference(runtime_root + '/provenance.json'),
                  reference(wind_root + '/summary.json'), reference(wind_root + '/provenance.json'),
                  reference('reports/p3_visibility_memory.json')],
        limitations=['P6 core samples predate final wind/cgroup fixes and current P7 repairs; they are not measurements of the current wheel',
                     'Historical small-fixture single trials do not establish a universal scaling bound',
                     'Summed process RSS may double-count shared pages and miss between-sample peaks',
                     'Component lifetime RSS and process-tree sampled RSS are different measurements'],
        current_p7={'raw_directory': 'reports/characterization/p7_pairs_v4',
                    'completed_scene_summaries': completed, 'status': 'full_matrix_not_yet_assessed'},
        outstanding=['Current full-workload memory/regression assessment',
                     'P8 required large-workload memory matrix with failures and unavailable cases retained'])
    write('reports/peak_memory.json', memory)
    print('Recorded measurement indices; release gates remain incomplete')


if __name__ == '__main__':
    main()
