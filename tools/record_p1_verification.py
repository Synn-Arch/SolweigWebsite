"""Record a completed installed-wheel pytest gate; never run or infer tests."""
import hashlib
import getpass
import argparse
from importlib import metadata, util
import json
from pathlib import Path
import platform
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
import solweig_light

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--milestone', choices=['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'], default='p1')
    parser.add_argument('--run-label', help='Distinct report prefix; preserves earlier milestone evidence')
    parser.add_argument('--include-scientific', action='store_true')
    args = parser.parse_args()
    label = args.run_label or args.milestone
    assert label.replace('_', '').replace('-', '').isalnum()
    installed = Path(solweig_light.__file__).parent
    assert 'site-packages' in installed.parts
    wheel = ROOT / 'dist/solweig_light-0.1.0.dev0-py3-none-any.whl'
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith('solweig_light/') and name.endswith(('.py', '.txt', '.json')):
                content = archive.read(name)
                assert (installed.parent / name).read_bytes() == content, name
                assert (ROOT / 'src' / name).read_bytes() == content, name
    absent = ['torch', 'xarray', 'cdsapi', 'wrf']
    assert all(util.find_spec(name) is None for name in absent)
    junit = ROOT / f'reports/{label}_installed_tests.xml'
    suites = ET.parse(junit).getroot()
    counts = {key: sum(int(s.attrib[key]) for s in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    assert counts['tests'] > 0 and not any(counts[k] for k in ('failures', 'errors', 'skipped'))
    latest = (Path(tempfile.gettempdir()) / f'pytest-of-{getpass.getuser()}/pytest-current').resolve()
    cases = {p.parent.name: json.loads(p.read_text()) for p in latest.glob('test_chronological*/comparison.json')
             if not p.parent.is_symlink()}
    assert len(cases) == 9
    scope = 'tests/differential tests/unit' + (' tests/integration' if args.milestone in ('p6', 'p7') else '')
    if args.include_scientific:
        scope += ' tests/scientific'
    report = dict(counts, evidence_class='candidate installed wheel compared with original upstream CPU',
        upstream_commit='0d7fe742abeeddd890dd58fc76ed7f78bd47faec', candidate_commit=None,
        candidate_state='uncommitted worktree; tested wheel matches installed and source files bytewise',
        command=f'.venv-wheel/bin/python -m pytest {scope} -q --junitxml=reports/{label}_installed_tests.xml',
        junit_sha256=sha(junit), python=sys.version, platform=platform.platform(), machine=platform.machine(),
        installed_module=solweig_light.__file__, absent_dependencies=absent,
        dependencies={d.metadata['Name']: d.version for d in metadata.distributions()},
        source_sha256={str(p.relative_to(ROOT)): sha(p) for folder in
                       (('src/solweig_light', 'compat/src', 'tests/unit', 'tests/differential', 'tests/integration')
                        + (('tests/scientific',) if args.include_scientific else ()))
                       for p in sorted((ROOT / folder).rglob('*.py'))},
        wheel_sha256={wheel.name: sha(wheel)},
        reference_manifests={str(p.relative_to(ROOT)): sha(p) for p in
            (ROOT / 'tests/reference/small_original_cpu/reference_manifest.json',
             ROOT / 'tests/reference/optional_original_cpu/manifest.json',
             ROOT / 'tests/reference/ground_view_original_cpu/manifest.json',
             ROOT / 'tests/reference/patch_radiation_original_cpu/manifest.json',
             ROOT / 'tests/reference/delay_original_cpu/manifest.json',
             ROOT / 'tests/reference/state_sequence_original_cpu/boundaries/manifest.json') if p.exists()},
        pipeline_cases=cases, remote_ci='configured_not_executed', cuda='not_available', performance_claim=None)
    (ROOT / f'reports/{label}_installed_verification.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f"Recorded {counts['tests']} installed-wheel tests and {len(cases)} pipeline cases")


if __name__ == '__main__':
    main()
