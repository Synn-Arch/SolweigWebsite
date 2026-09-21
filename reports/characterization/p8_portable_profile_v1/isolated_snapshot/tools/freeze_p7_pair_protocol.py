"""Freeze explicit existing inputs for the P7 P6-versus-candidate instrument."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmark_p7_pipeline import hashes

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--kwargs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    kwargs = json.loads(args.kwargs.read_text())
    kwargs['base_path'] = '{scene}'
    kwargs['own_met_file'] = '{scene}/' + Path(kwargs['own_met_file']).name
    kwargs_path = args.output.resolve() / 'kwargs.json'
    kwargs_path.write_text(json.dumps(kwargs, indent=2, sort_keys=True) + '\n')
    comparison_path = ROOT / 'benchmarks/protocols/comparison_v1.json'
    comparison = json.loads(comparison_path.read_text())
    rules = {}
    for name in ('UTCI', 'TMRT', 'Kup', 'Kdown', 'Lup', 'Ldown', 'Shadow', 'WBGT', 'Ta', 'Wind'):
        field = {'TMRT': 'Tmrt', 'Shadow': 'shadow'}.get(name, name)
        rules[name] = comparison['other_outputs'].get(name, comparison['field_rules'].get('Solweig_2022a_calc/' + field))
        assert rules[name] is not None, name
    rules['SVF'] = rules['svf'] = rules['SkyViewFactor'] = comparison['other_outputs']['SVF_all_15_fields']
    protocol = {
        'schema_version': 1, 'status': 'frozen_before_optimization',
        'scope': 'P6 versus candidate, full chronological workload; not upstream speedup evidence',
        'fixture': str(args.fixture.resolve()), 'fixture_hashes': hashes(args.fixture),
        'kwargs_manifest': str(kwargs_path), 'repetitions': 5, 'seed': 20260918,
        'native_budgets': [1, 4], 'regimes': ['cold', 'geometry_warm'],
        'runtime_options': {'cache_dir': '{scene}/.runtime_cache', 'cache_enabled': True,
                            'legacy_cache_policy': 'recompute', 'block_pixels': 128,
                            'checkpoint_interval': 1},
        'preprocess_relative': 'processed_inputs',
        'cold_remove_relative': ['processed_inputs', 'output_folder', '.runtime_cache', '.solweig-light', '.solweig-light-locks'],
        'warm_remove_relative': ['output_folder', '.solweig-light', '.solweig-light-locks'],
        'artifact_globs': ['output_folder/**/*.tif', 'processed_inputs/SVF/*.tif',
                           'processed_inputs/SVF/*.zip', 'processed_inputs/SVF/*.npz'],
        'tiff_field_rules': rules,
        'comparison_protocol_sha256': hashlib.sha256(comparison_path.read_bytes()).hexdigest(),
        'freezer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'memory_budget_bytes': 12 * 1024**3,
        'limitations': ['Development host; OS page cache uncontrolled',
                       'Warm means validated geometry warm, not JIT warm',
                       'No reduced patches, resolution, timesteps or output computation'],
    }
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2, sort_keys=True) + '\n')
    print(args.output / 'protocol.json')


if __name__ == '__main__':
    main()
