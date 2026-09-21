"""Attribute warm patch-radiation costs on the unchanged P4 component fixture."""
import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import sys

import numpy as np

from solweig_light.geometry.visibility import PackedVisibility, LazyDiffVisibility
from solweig_light.radiation import patch_radiation

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = ROOT / 'tests/reference/p4_benchmark_inputs'
    manifest = json.loads((fixture / 'manifest.json').read_text())
    paths = list((ROOT / 'src/solweig_light').rglob('*.py')) + [
        Path(__file__), fixture / 'manifest.json',
        ROOT / 'benchmarks/protocols/p4_radiation_v1.json']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for name in ('Kside_veg_v2022a', 'Lcyl_v2022a'):
        case = next(c for c in manifest['cases'] if c['function'] == name)
        path = fixture / case['input']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case['input_sha256']
        with np.load(path) as archive:
            values = {key: archive[key].copy() for key in archive.files}
        for key, value in list(values.items()):
            spec = case['fields'][key]
            typ = spec['original_python_type']
            if typ.startswith('builtins.'):
                values[key] = value.item()
            elif typ.startswith('numpy.') and spec['kind'] == 'scalar':
                values[key] = value[()]
        for key in ('shmat', 'vegshmat', 'vbshvegshmat'):
            values[key] = PackedVisibility.from_dense(values[key])
        if 'diffsh' in values:
            values['diffsh'] = LazyDiffVisibility(values['shmat'], values['vegshmat'])
        function = getattr(patch_radiation, name)
        function(**values, parallel=False, block_pixels=128)
        profile = cProfile.Profile()
        profile.runcall(function, **values, parallel=False, block_pixels=128)
        profile.dump_stats(str(args.output / (name + '.prof')))
        with (args.output / (name + '.txt')).open('w') as stream:
            pstats.Stats(profile, stream=stream).sort_stats('cumulative').print_stats(25)
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
               for p, digest in hashes.items())
    (args.output / 'provenance.json').write_text(json.dumps({
        'source_sha256': hashes, 'command': sys.argv,
        'scope': 'one warmed profiled call; attribution only, not benchmark or numerical verification',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
