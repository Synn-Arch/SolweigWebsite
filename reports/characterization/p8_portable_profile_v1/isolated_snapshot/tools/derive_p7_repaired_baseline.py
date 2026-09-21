"""Preserve P6 and derive a distinct baseline containing only the angular repair."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-svf-weight-repair', action='store_true')
    args = parser.parse_args()
    parent = ROOT / 'reports/characterization/p7_p6_baseline'
    target = ROOT / ('reports/characterization/p7_repaired_p6_baseline_v2'
                     if args.include_svf_weight_repair else
                     'reports/characterization/p7_repaired_p6_baseline')
    manifest = json.loads((parent / 'manifest.json').read_text())
    assert all(sha(parent / key) == value for key, value in manifest['files'].items())
    target.mkdir(exist_ok=False)
    shutil.copytree(parent / 'src', target / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.nbc', '*.nbi'))
    repair = ('radiation/ground_view.py', 'radiation/engine.py')
    if args.include_svf_weight_repair:
        repair += ('geometry/svf.py',)
    differences, changed = [], {}
    for name in repair:
        relative = 'src/solweig_light/' + name
        before, after = parent / relative, ROOT / relative
        assert before.read_bytes() != after.read_bytes(), name
        differences.extend(difflib.unified_diff(before.read_text().splitlines(True), after.read_text().splitlines(True),
                                                fromfile='a/' + relative, tofile='b/' + relative))
        shutil.copy2(after, target / relative)
        changed[relative] = {'before_sha256': sha(before), 'after_sha256': sha(after)}
    files = {str(p.relative_to(target)): sha(p) for p in sorted((target / 'src').rglob('*')) if p.is_file()}
    assert set(files) == {key for key in manifest['files'] if key.startswith('src/')}
    assert {key for key in files if files[key] != manifest['files'][key]} == set(changed)
    policy = 'angular_and_svf_weights_v1' if args.include_svf_weight_repair else 'angular_v1'
    patch = target / ('compatibility_repair.patch' if args.include_svf_weight_repair
                      else 'angular_compatibility_repair.patch')
    patch.write_text(''.join(differences))
    record = {
        'label': ('P6 plus angular and SVF weight compatibility repairs; not unchanged accepted P6'
                  if args.include_svf_weight_repair else
                  'P6 plus angular compatibility repair; not unchanged accepted P6'),
        'repair_policy': policy, 'repair_patch_file': patch.name,
        'parent_snapshot': str(parent), 'parent_manifest_sha256': sha(parent / 'manifest.json'),
        'repair_patch_sha256': sha(patch), 'changed_files': changed, 'files': files,
        'status': 'derived; expanded correctness gate pending before benchmark use',
        'excluded_optimizations': ['P7 visibility decoding', 'P7 classification'],
        'evidence_class': 'candidate comparison baseline, not upstream reference',
        'derivation_script_sha256': sha(Path(__file__)),
    }
    (target / 'manifest.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(target)


if __name__ == '__main__':
    main()
