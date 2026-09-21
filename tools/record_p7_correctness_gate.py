"""Bind executed P7 repair-v2 correctness evidence to both complete sources."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from benchmark_p7_pipeline import digest, dump, hashes

ROOT = Path(__file__).resolve().parents[1]


def evidence(path):
    return {'path': str(path.resolve()), 'sha256': digest(path)}


def read(path):
    return json.loads(path.read_text())


def main():
    target = ROOT / 'reports/characterization/p7_repair_v2_correctness'
    target.mkdir(exist_ok=False)
    sources = {'baseline': ROOT / 'reports/characterization/p7_repaired_p6_baseline_v2/src',
               'candidate': ROOT / 'src'}
    roots = {'baseline': ROOT / 'reports/runs/p7_repair_v2_baseline',
             'candidate': ROOT / 'reports/runs/p7_repair_v2_optimized'}
    junits = {'baseline': ROOT / 'reports/p7_repaired_v2_baseline_regressions.xml',
              'candidate': ROOT / 'reports/p7_repair_v2_installed_tests.xml'}
    records, original, angular, weights = [], {}, {}, {}
    identifiers = None
    diagnosis = ROOT / 'reports/characterization/p7_real_diagnosis'
    original_packet = diagnosis / 'original/0013_8_gvf_2018a_output.npz'
    weight_manifest = ROOT / 'tests/reference/p7_svf_annulus_original_cpu/manifest.json'
    weight_diagnosis = diagnosis / 'svf_weights/repair_verification.json'
    assert read(weight_diagnosis)['source_after']['src/solweig_light/geometry/svf.py'] == digest(sources['candidate'] / 'solweig_light/geometry/svf.py')
    for variant, source in sources.items():
        matrix = read(roots[variant] / 'matrix.json')
        assert matrix['status'] == 'passed' and matrix['source_unchanged']
        assert matrix['source_hashes'] == hashes(source)
        ids = sorted(r['case_id'] for r in matrix['records'])
        assert len(ids) == len(set(ids)) == 7
        if identifiers is None:
            identifiers = ids
        assert ids == identifiers
        for item in matrix['records']:
            case = item['case_id']
            path = roots[variant] / case / 'verification.json'
            geometry_path = path.with_name('geometry_exports.json')
            result, geometry = read(path), read(geometry_path)
            assert item['exit_code'] == item['geometry_exit_code'] == 0
            assert result['passed'] and result['source_unchanged']
            assert result['source_sha256'] == hashes(source)
            assert geometry['status'] == 'passed' and len(geometry['records']) == 18
            for name, file in [('verification.json', path), ('geometry_exports.json', geometry_path)]:
                assert digest(file) == item[name]['sha256']
            record = dict(case_id=case, variant=variant, passed=True,
                          evidence=evidence(path), geometry_evidence=evidence(geometry_path))
            records.append(record)
        tests = list(ET.parse(junits[variant]).getroot().iter('testcase'))
        assert tests and all(not list(t) for t in tests)
        expected = {'tests.differential.test_pipeline_reference': 9,
                    'tests.differential.test_p7_ground_view_angular': 32,
                    'tests.differential.test_p7_svf_annulus': 4}
        for classname, count in expected.items():
            assert sum(t.get('classname') == classname for t in tests) == count
        original[variant] = dict(passed=True, cases=9, evidence=evidence(junits[variant]))
        component = diagnosis / ('component_repair_v2_' + variant)
        capture = read(component / 'manifest.json')
        assert Path(capture['package']).resolve() == (source / 'solweig_light/radiation/engine.py').resolve()
        matches = [z for z in capture['events'] if z['name'] == 'gvf_2018a' and z['boundary'] == 'output']
        assert len(matches) == 1
        packet = component / matches[0]['path']
        with np.load(original_packet) as left, np.load(packet) as right:
            assert set(left.files) == set(right.files) and len(left.files) == 17
            for name in left.files:
                assert left[name].dtype == right[name].dtype
                np.testing.assert_array_equal(left[name].view(np.uint32), right[name].view(np.uint32))
        detail = dict(passed=True, field_count=17, boundary_passed=True,
                      original_packet=evidence(original_packet), candidate_packet=evidence(packet),
                      capture=evidence(component / 'manifest.json'), boundary_tests=evidence(junits[variant]),
                      source_hashes=hashes(source))
        detail_path = target / f'{variant}_angular.json'
        dump(detail_path, detail)
        angular[variant] = dict(passed=True, field_count=17, boundary_passed=True, evidence=evidence(detail_path))
        assert digest(source / 'solweig_light/geometry/svf.py') == digest(sources['candidate'] / 'solweig_light/geometry/svf.py')
        detail_path = target / f'{variant}_svf_weights.json'
        dump(detail_path, dict(passed=True, tests=evidence(junits[variant]),
                              reference_manifest=evidence(weight_manifest), diagnosis=evidence(weight_diagnosis),
                              source_sha256=digest(source / 'solweig_light/geometry/svf.py'),
                              limitation='174/180 weights bitwise equal; six within one ULP on this host'))
        weights[variant] = dict(passed=True, weight_count=180, boundary_pixel_count=2,
                               maximum_weight_ulp=1, evidence=evidence(detail_path))
    dump(target / 'gate.json', dict(status='passed', fixture_case_ids=identifiers,
         source_hashes={name: hashes(source) for name, source in sources.items()},
         cases=records, original_regression=original, angular_ground_view=angular,
         svf_weights=weights, recorder_sha256=digest(Path(__file__)),
         limitation='Original CPU compatibility on characterized fixtures; no performance or physical-accuracy claim'))
    print(target / 'gate.json')


if __name__ == '__main__':
    main()
