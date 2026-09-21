"""Generate original, unmodified wall/aspect fixtures in the isolated oracle."""
import hashlib
import ast
import inspect
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
    target = ROOT / 'tests/reference/walls_expanded_original_cpu'
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
    scenes['random64'] = scenes['random'].astype(np.float64) + rng.uniform(-1e-8, 1e-8, (27, 33))
    for dy, dx in [(0,0), (-1,0), (0,-1), (0,1), (1,0)]:
        scene = block.copy(); scene[12+dy, 15+dx] = np.nan
        scenes[f'nan_{dy}_{dx}'] = scene
    for name in ('cross', 'tee', 'corner', 'diagonal'):
        scene = np.zeros((27, 33), np.float32)
        if name == 'diagonal':
            np.fill_diagonal(scene, 8)
        else:
            scene[13, 7:26] = 8
            scene[5:22 if name == 'cross' else 14, 16] = 8
            if name == 'corner': scene[13, :16] = 0
        scenes[name] = scene
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
    # Read-only trace of the unmodified original captures tables after all
    # special-angle edits, before pixel processing. No candidate code is used.
    tree = ast.parse(installed.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'filter1Goodwin_as_aspect_v3')
    line = next(n.lineno for n in ast.walk(function) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'i')
    filters = []
    for scale in [.2, .5, 1., 1.5, 2.]:
        captured = {}
        def trace(frame, event, arg):
            if event == 'line' and frame.f_code is original.filter1Goodwin_as_aspect_v3.__code__ and frame.f_lineno == line:
                values = frame.f_locals
                captured[values['h']] = (values['filtmatrix1'].copy(), values['filtmatrixbuild'].copy())
            return trace
        sys.settrace(trace)
        try:
            original.filter1Goodwin_as_aspect_v3(np.zeros((2, 3)), scale, np.zeros((2, 3)))
        finally:
            sys.settrace(None)
        assert list(captured) == list(range(180))
        filename = f'filters_{scale:g}.npz'
        np.savez_compressed(target / filename, score=np.stack([captured[h][0] for h in range(180)]),
                            sides=np.stack([captured[h][1] for h in range(180)]), scale=np.float64(scale))
        filters.append(dict(file=filename, scale=scale, sha256=sha(target / filename)))
    (target / 'generator.py').write_bytes(Path(__file__).read_bytes())
    manifest = dict(evidence_class='original_upstream_cpu', source_commit=PIN,
                    source_sha256=sha(installed), generator_sha256=sha(Path(__file__)),
                    command='.venv-oracle/bin/python tools/characterize_walls.py',
                    environment={d.metadata['Name']:d.version for d in metadata.distributions()},
                    python=sys.version, platform=platform.platform(), fixtures=records, filters=filters,
                    repair=None, candidate_outputs=False)
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Wrote {len(records)} original wall/aspect fixtures to {target}')


if __name__ == '__main__':
    main()
