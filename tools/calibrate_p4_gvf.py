"""Original-engine GVF sensitivity experiment; perturbed runs are not goldens."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from solweig_gpu import solweig as original

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT/'tests/reference/small_original_cpu/boundaries'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(original.__file__) == sha(ROOT/'.upstream/SOLWEIG-GPU/solweig_gpu/solweig.py')
assert 'solweig_light' not in sys.modules
torch.set_num_threads(1)
manifest = json.loads((REF/'manifest.json').read_text())
inputs = [e for e in manifest['events'] if e['boundary'] == 'input']
outputs = {e['timestep']: e for e in manifest['events'] if e['boundary'] == 'output' and e['function'] == 'Solweig_2022a_calc'}
state_names = ('firstdaytime','timeadd','timestepdec','Tgmap1','Tgmap1E','Tgmap1S','Tgmap1W','Tgmap1N','CI','TgOut1')


def load(event):
    assert sha(REF/event['path']) == event['sha256']
    result = {}
    with np.load(REF/event['path']) as a:
        for name, spec in event['fields'].items():
            if '/' in name: continue
            kind = spec['kind']
            if kind == 'dict': result[name] = {k: a[name+'/'+k].item() for k in spec['keys']}
            elif kind == 'list': result[name] = []
            elif kind == 'none': result[name] = None
            elif kind == 'array' or name in ('jday','Ta','RH','radG','radD','radI','P','amaxvalue'): result[name] = torch.from_numpy(a[name].copy())
            elif name in ('altitude','azimuth','zen','dectime','altmax'): result[name] = a[name][()]
            else: result[name] = a[name].item()
    return result


base_gvf = original.gvf_2018a
runs = []
# Coherent signed and checkerboard perturbations test propagation, not a proof
# for every possible error pattern or scene. No candidate output selects budgets.
for mode in ('baseline','positive','negative','checkerboard'):
    def altered(*args, **kwargs):
        fields = list(base_gvf(*args, **kwargs))
        if mode != 'baseline':
            for index in (1,2,4,5,7,8,10,11,13,14,15,16):
                value = fields[index]
                sign = -1 if mode == 'negative' else 1
                if mode == 'checkerboard':
                    yy, xx = torch.meshgrid(torch.arange(value.shape[0]), torch.arange(value.shape[1]), indexing='ij')
                    sign = ((xx+yy)%2)*2-1
                fields[index] = value + sign * 1e-6
        return tuple(fields)
    original.gvf_2018a = altered
    carried = {}
    errors = []
    for event in inputs:
        arguments = load(event)
        arguments.update(carried)
        result = original.Solweig_2022a_calc(**arguments)
        names = list(outputs[event['timestep']]['fields'])
        carried = {name: result[names.index(name)] for name in state_names}
        with np.load(REF/outputs[event['timestep']]['path']) as expected:
            tmrt = result[names.index('Tmrt')].detach().numpy()
            assert np.array_equal(np.isnan(tmrt), np.isnan(expected['Tmrt']))
            errors.append(float(np.nanmax(abs(tmrt-expected['Tmrt']))))
    runs.append(dict(mode=mode, tmrt_max_abs_by_step=errors, tmrt_max_abs=max(errors)))
    print(mode, max(errors), flush=True)
original.gvf_2018a = base_gvf
report = dict(evidence_class='original-engine controlled sensitivity experiment; perturbed outputs are not reference goldens',upstream_commit='0d7fe742abeeddd890dd58fc76ed7f78bd47faec',source_sha256=sha(original.__file__),script_sha256=sha(__file__),reference_manifest_sha256=sha(REF/'manifest.json'),perturbation=1e-6,fields='all dimensionless GVF outputs together',runs=runs,scope='one original 24-step scene; does not establish arbitrary-scene or arbitrary-error bounds',python=sys.version,torch=torch.__version__,numpy=np.__version__)
(ROOT/'reports/p4_gvf_sensitivity.json').write_text(json.dumps(report,indent=2)+'\n')
