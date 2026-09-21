"""Record the executed optional installed-wheel gate without inferring live services."""
from importlib import metadata, util
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET
import zipfile

import solweig_light

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forcing-only', action='store_true')
    parser.add_argument('--milestone', choices=('p5', 'p6', 'p7'), default='p5')
    parser.add_argument('--run-label', help='Distinct report prefix; preserves earlier evidence')
    args = parser.parse_args()
    label = args.run_label or args.milestone
    assert label.replace('_', '').replace('-', '').isalnum()
    installed = Path(solweig_light.__file__).parent
    assert 'site-packages' in installed.parts
    absent = ['torch'] + (['rasterio', 'geopandas', 'ee', 'osmnx', 'geemap'] if args.forcing_only else [])
    assert all(util.find_spec(name) is None for name in absent)
    wheel = ROOT / 'dist/solweig_light-0.1.0.dev0-py3-none-any.whl'
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith('solweig_light/') and name.endswith(('.py', '.txt', '.json')):
                content = archive.read(name)
                assert (installed.parent / name).read_bytes() == content, name
                assert (ROOT / 'src' / name).read_bytes() == content, name
    prefix = f'{label}_forcing_extra' if args.forcing_only else f'{label}_optional_installed'
    junit = ROOT / f'reports/{prefix}_tests.xml'
    counts = {key: sum(int(s.attrib[key]) for s in ET.parse(junit).getroot())
              for key in ('tests', 'failures', 'errors', 'skipped')}
    assert counts['tests'] > 0
    assert not any(counts[key] for key in ('failures', 'errors', 'skipped'))
    packets = ('wind_original_cpu', 'forcing_original_cpu', 'wrf_patched_cpu',
               'inputs_original_cpu', 'public_forcing_original_cpu', 'public_roughness_original_cpu')
    references = {}
    for packet in packets:
        manifest = ROOT / 'tests/reference' / packet / 'manifest.json'
        data = json.loads(manifest.read_text())
        assert data.get('upstream_commit', data.get('source_commit')) == '0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
        references[str(manifest.relative_to(ROOT))] = sha(manifest)
    environment = 'forcing' if args.forcing_only else 'optional'
    scope = ('tests/optional/test_forcing_optional.py tests/optional/test_public_forcing.py'
             if args.forcing_only else 'tests/optional')
    command = (f'.venv-{environment}/bin/python -m pytest {scope} -q '
               f'--basetemp=reports/runs/{label}-{environment}-pytest '
               f'--junitxml=reports/{prefix}_tests.xml')
    report = dict(counts, command=command,
        candidate_commit=None,
        candidate_state='uncommitted; installed wheel and source match bytewise',
        evidence_classes=['original upstream CPU', 'patched upstream CPU: wrf_timestamp_v1',
                          'original AST-extracted input helpers', 'candidate offline service mocks'],
        live_service_verification='not executed', cuda='not available',
        remote_ci='configured, not executed', performance_claim=None,
        installed_module=solweig_light.__file__, absent_dependencies=absent,
        python=sys.version, platform=platform.platform(),
        dependencies={d.metadata['Name']: d.version for d in metadata.distributions()},
        junit_sha256=sha(junit), wheel_sha256=sha(wheel), reference_manifests=references,
        source_sha256={str(p.relative_to(ROOT)): sha(p)
                       for folder in ('src/solweig_light', 'compat/src', 'tests/optional')
                       for p in sorted((ROOT / folder).rglob('*.py'))},
        warning_policy='Known netCDF4 import and rasterio deprecation warnings remain visible; see docs/optional_environment.md')
    destination = ROOT / f'reports/{prefix}_verification.json'
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f"Recorded {counts['tests']} optional installed-wheel tests")


if __name__ == '__main__':
    main()
