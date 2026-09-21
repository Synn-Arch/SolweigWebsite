"""Lossless, original-upstream CPU patch/shadow characterization; no repairs."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

COMMIT = '0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', type=Path, default=ROOT/'reports/geometry_characterization.json')
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    args.run.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    import solweig_gpu
    from solweig_gpu import shadow as module
    from osgeo import gdal
    if torch.cuda.is_available():
        raise RuntimeError('CPU reference requires unavailable CUDA')
    source = ROOT/'.upstream/SOLWEIG-GPU'
    revision = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(source),'diff','--name-only'],text=True).strip()
    if revision != COMMIT or dirty:
        raise RuntimeError('upstream pin/cleanliness check failed')
    package = Path(solweig_gpu.__file__).parent
    hashes = {}
    for p in sorted((source/'solweig_gpu').iterdir()):
        if p.is_file() and p.suffix in {'.py','.txt'}:
            if sha(p) != sha(package/p.name):
                raise RuntimeError(f'installed source differs: {p.name}')
            hashes[p.name] = sha(p)
    torch.set_num_threads(1)
    def array(value):
        return value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)
    def describe(value):
        a = array(value)
        finite = np.isfinite(a)
        unique = np.unique(a)
        return {'shape':list(a.shape),'dtype':str(a.dtype),
                'sha256_array_bytes':hashlib.sha256(a.tobytes()).hexdigest(),
                'nan_count':int(np.isnan(a).sum()),'positive_inf_count':int(np.isposinf(a).sum()),
                'negative_inf_count':int(np.isneginf(a).sum()),
                'finite_min':float(a[finite].min()) if finite.any() else None,
                'finite_max':float(a[finite].max()) if finite.any() else None,
                'exact_values':unique.tolist() if len(unique)<=32 else None,
                'exact_value_count':len(unique)}
    report = {'evidence_class':'original_upstream_cpu','source_commit':revision,
              'oracle_patch_hash':None,'source_package_sha256':hashes,
              'environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                  'package_path':str(package),'gdal_version':gdal.VersionInfo(),
                  'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
                  'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
                  'torch_default_dtype':str(torch.get_default_dtype()),'cuda':'not available',
                  'thread_environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES']},
                  'invocation':sys.argv,'script_sha256':sha(__file__),
                  'candidate_git_status':subprocess.check_output(['git','-C',str(ROOT),'status','--short'],text=True)},
              'signatures':{n:str(inspect.signature(getattr(module,n))) for n in ['shadow','create_patches']},
              'cases':[], 'scope':'targeted fixtures; observed binary values are not proof for every accepted input'}
    def execute(name, function, inputs, names, independent=False):
        entry={'name':name,'function':function.__name__,'inputs':{k:describe(v) for k,v in inputs.items()}}
        inp=args.run/(name+'_inputs.npz')
        np.savez_compressed(inp,**{k:array(v) for k,v in inputs.items()})
        entry['input_artifact']={'path':str(inp),'sha256':sha(inp)}
        try:
            values=function(**inputs)
            out=args.run/(name+'_outputs.npz')
            np.savez_compressed(out,**{k:array(v) for k,v in zip(names,values,strict=True)})
            entry.update(status='executed',outputs={k:describe(v) for k,v in zip(names,values,strict=True)},
                         output_artifact={'path':str(out),'sha256':sha(out)})
            if independent:
                checks={k:bool(np.all(array(v)==1)) for k,v in zip(names,values,strict=True)}
                entry['independent_no_obstacle_expectation']={'preconditions':'flat nonnegative DSM, zero canopy/trunk/bush, positive scale, altitude > 0',
                       'expected':'all three visibility channels exactly one','checks':checks,'passed':all(checks.values())}
        except Exception:
            entry.update(status='failed',traceback=traceback.format_exc())
        entry['input_mutations']={k:describe(v)['sha256_array_bytes']!=entry['inputs'][k]['sha256_array_bytes'] for k,v in inputs.items()}
        report['cases'].append(entry)
    for option in range(1,5):
        execute(f'patch_option_{option}',module.create_patches,{'patch_option':option},
                ['skyvaultalt','skyvaultazi','annulino','skyvaultaltint','patches_in_band','skyvaultaziint','azistart'])
    def scene(kind,dtype):
        a=torch.zeros((16,19),dtype=dtype)
        canopy=torch.zeros_like(a); trunk=torch.zeros_like(a); bush=torch.zeros_like(a)
        if kind=='block': a[5:10,7:12]=8
        if kind in {'tree','trunk','bush'}:
            canopy[5:10,7:12]=9
            if kind=='trunk': trunk[5:10,7:12]=4
            if kind=='bush': bush[5:10,7:12]=3
        if kind=='negative': a.fill_(-2); a[5:10,7:12]=-1
        return a,canopy,trunk,bush
    angles=[('normal',125.,35.),('north',0.,35.),('near_north',1e-7,35.),
            ('east',90.,35.),('near_east',89.99999,35.),('south',180.,35.),
            ('west',270.,35.),('zenith',125.,90.),('short_ray',125.,89.99),('low',45.,0.5)]
    for dtype_name,dtype in [('float32',torch.float32),('float64',torch.float64)]:
        for kind in ['flat','block','tree','trunk','bush','negative']:
            for angle_name,azimuth,altitude in angles:
                a,canopy,trunk,bush=scene(kind,dtype)
                # Fixture policy: nonnegative relief/canopy bound; this differs from
                # svf_calculator's a.max()/vegetation-height policy for negative DSM.
                # Retain the signed-negative bound separately below.
                bound=max(float((a.max()-a.min()).item()),float(canopy.max().item()))
                execute(f'{dtype_name}_{kind}_{angle_name}',module.shadow,
                        dict(amaxvalue=torch.tensor(bound,dtype=dtype),a=a,vegdem=canopy,
                             vegdem2=trunk,bush=bush,azimuth=azimuth,altitude=altitude,scale=1.),
                        ['sh','vegsh','vbshvegsh'],independent=kind=='flat')
        a,c,t,b=scene('negative',dtype)
        execute(f'{dtype_name}_negative_signed_bound',module.shadow,
                dict(amaxvalue=a.max(),a=a,vegdem=c,vegdem2=t,bush=b,azimuth=125.,altitude=35.,scale=1.),
                ['sh','vegsh','vbshvegsh'])
    categories={}
    for case in report['cases']:
        if case['function']=='shadow' and case['status']=='executed':
            for name,desc in case['outputs'].items():
                categories.setdefault(name,set()).update(desc['exact_values'] or [])
    report['observed_shadow_categories']={k:sorted(v) for k,v in categories.items()}
    report['counts']={k:sum(c['status']==k for c in report['cases']) for k in ['executed','failed']}
    args.report.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'counts':report['counts'],'categories':report['observed_shadow_categories']}))

if __name__=='__main__':
    main()
