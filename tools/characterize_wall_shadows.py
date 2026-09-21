"""Capture pinned, unmodified original CPU wall-shadow edge fixtures."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
COMMIT='0d7fe742abeeddd890dd58fc76ed7f78bd47faec'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from solweig_gpu import solweig as original
    source=ROOT/'.upstream/SOLWEIG-GPU'
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if revision!=COMMIT or subprocess.check_output(['git','-C',str(source),'diff','--name-only'],text=True).strip():
        raise RuntimeError('upstream pin/cleanliness failed')
    if sha(original.__file__)!=sha(source/'solweig_gpu/solweig.py') or torch.cuda.is_available():
        raise RuntimeError('original source or CPU isolation failed')
    torch.set_num_threads(1)
    report={'evidence_class':'original_upstream_cpu','upstream_commit':COMMIT,'upstream_patch_hash':None,'source_sha256':sha(original.__file__),'collector_sha256':sha(__file__), 'environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'torch_threads':1,'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}},'invocation':sys.argv,'cases':[]}
    rng=np.random.default_rng(291)
    a=rng.integers(0,8,(7,9)).astype(np.float32)
    v=a+rng.integers(0,8,a.shape).astype(np.float32)
    trunk=a+rng.integers(0,3,a.shape).astype(np.float32)
    walls=rng.integers(0,8,a.shape).astype(np.float32)
    aspect=rng.uniform(0,2*np.pi,a.shape).astype(np.float32)
    aspect[0,:5]=np.asarray([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],dtype=np.float32)
    bush=rng.choice([0.,1.,2.],a.shape).astype(np.float32)
    scenarios=[(f'angle-{i}',a,v,trunk,walls,aspect,bush,az,35.) for i,az in enumerate([0.,45.-1e-5,45.,45.+1e-5,90.,135.,180.,225.,270.,315.,360.,np.float64(45.),np.float64(270.),float('nan')])]
    scenarios += [(f'alt-{i}',a,v,trunk,walls,aspect,bush,123.,alt) for i,alt in enumerate([0.,.001,90.,-1.,float('nan')])]
    for label,field in [('negative','a'),('nan-dsm','a'),('nan-veg','v'),('nan-trunk','trunk'),('nan-walls','walls'),('nan-aspect','aspect'),('zero','a')]:
        arrays={k:x.copy() for k,x in [('a',a),('v',v),('trunk',trunk),('walls',walls),('aspect',aspect),('bush',bush)]}
        if label=='negative':arrays[field].fill(-3)
        elif label=='zero':arrays[field].fill(0)
        else:arrays[field][2,3]=np.nan
        scenarios.append((label,*(arrays[k] for k in ('a','v','trunk','walls','aspect','bush')),123.,35.))
    scenarios.append(('single',*(x[:1,:1] for x in (a,v,trunk,walls,aspect,bush)),0.,90.))
    for label,a,v,trunk,walls,aspect,bush,az,alt in scenarios:
        for family in (13,23):
            inputs={'a':torch.from_numpy(a.copy()),'azimuth':az,'altitude':alt,'scale':1.,'walls':torch.from_numpy(walls.copy()),'aspect':torch.from_numpy(aspect.copy())}
            if family==23:inputs.update(vegdem=torch.from_numpy(v.copy()),vegdem2=torch.from_numpy(trunk.copy()),amaxvalue=float(np.max(a)),bush=torch.from_numpy(bush.copy()))
            name=f'{label}-{family}'
            inp=args.output/(name+'-input.npz')
            np.savez_compressed(inp,**{k:x.numpy() if isinstance(x,torch.Tensor) else np.asarray(x) for k,x in inputs.items()})
            entry={'name':name,'family':family,'input':inp.name,'input_sha256':sha(inp),'scalar_types':{k:type(x).__name__ for k,x in inputs.items() if not isinstance(x,torch.Tensor)}}
            try:
                result=getattr(original,f'shadowingfunction_wallheight_{family}')(**inputs)
                out=args.output/(name+'-output.npz')
                np.savez_compressed(out,**{f'output_{i}':x.detach().numpy() for i,x in enumerate(result)})
                entry.update(output=out.name,output_sha256=sha(out),status='captured')
            except Exception as error:
                entry.update(status='original_failure',exception=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            report['cases'].append(entry)
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':len(report['cases']),'captured':sum(e['status']=='captured' for e in report['cases']),'original_failures':sum(e['status']=='original_failure' for e in report['cases'])}))

if __name__=='__main__':main()
