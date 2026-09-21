"""Generate original, unmodified wall/aspect fixtures in the isolated oracle."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import importlib.metadata as metadata

ROOT = Path(__file__).resolve().parents[1]
PIN = '0d7fe742abeeddd890dd58fc76ed7f78bd47faec'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    import solweig_gpu.walls_aspect as original
    source = ROOT / '.upstream/SOLWEIG-GPU'
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == PIN
    assert not subprocess.check_output(['git', '-C', str(source), 'diff', '--name-only'], text=True).strip()
    installed = Path(original.__file__)
    assert sha(installed) == sha(source / 'solweig_gpu/walls_aspect.py')
    target = ROOT / 'tests/reference/walls_original_cpu'
    target.mkdir(exist_ok=False)
    rng = np.random.default_rng(20260918)
    scenes = {}
    scenes['open_negative'] = np.full((23, 29), -4, np.float32)
    block = np.zeros((25, 31), np.float32)
    block[7:17, 9:19] = 12
    scenes['block'] = block
    scenes['threshold'] = np.zeros((13, 17), np.float32)
    scenes['threshold'][4, 4:7] = [np.nextafter(np.float32(3), np.float32(0)), 3, np.nextafter(np.float32(3), np.float32(4))]
    scenes['random'] = rng.integers(-10, 35, (27, 33)).astype(np.float32)
    scenes['near_ties'] = np.full((27, 33), np.float32(2**20))
    scenes['near_ties'] += rng.choice(np.array([0, .125, -.125, 8], np.float32), size=(27, 33))
    scenes['nan'] = block.copy(); scenes['nan'][8, 10] = np.nan
    scenes['infinity'] = block.copy(); scenes['infinity'][8, 10] = np.inf; scenes['infinity'][12, 12] = -np.inf
    scenes['signed_zero'] = np.zeros((13, 17), np.float32); scenes['signed_zero'][::2] = -0.
    scenes['tiny'] = np.array([[0, 3, 8], [0, 0, 0]], np.float32)
    records = []
    for name, dsm in scenes.items():
        scales = [.2, .5, 1., 1.5, 2.] if name in ('block', 'random', 'near_ties') else [1.]
        for scale in scales:
            walls = original.findwalls(dsm, 3.)
            aspect = original.filter1Goodwin_as_aspect_v3(walls, scale, dsm)
            filename = f'{name}_{scale:g}.npz'
            np.savez_compressed(target / filename, dsm=dsm, walls=walls, aspect=aspect,
                                scale=np.float64(scale), walllimit=np.float64(3))
            records.append(dict(name=name, scale=scale, file=filename, sha256=sha(target / filename),
                                walls_dtype=str(walls.dtype), aspect_dtype=str(aspect.dtype)))
    manifest = dict(evidence_class='original_upstream_cpu', source_commit=PIN,
                    source_sha256=sha(installed), generator_sha256=sha(Path(__file__)),
                    command='.venv-oracle/bin/python tools/characterize_walls.py',
                    environment={d.metadata['Name']:d.version for d in metadata.distributions()},
                    python=sys.version, platform=platform.platform(), fixtures=records,
                    repair=None, candidate_outputs=False)
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Wrote {len(records)} original wall/aspect fixtures to {target}')


if __name__ == '__main__':
    main()
