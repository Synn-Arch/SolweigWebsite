"""Original-only typed patch-radiation inputs, outputs, and edge failures.

Read-only call profiling captures reconstructable arguments before each body.
Instrumented execution is not performance evidence; no source repair is used.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
COMMIT='0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
FUNCTIONS=('Kside_veg_v2022a','Lcyl_v2022a','define_patch_characteristics')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    from solweig_gpu import solweig as original
    from solweig_gpu.shadow import create_patches
    source=ROOT/'.upstream/SOLWEIG-GPU'
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if revision!=COMMIT or subprocess.check_output(['git','-C',str(source),'diff','--name-only'],text=True).strip():raise RuntimeError('upstream pin/cleanliness failed')
    if sha(original.__file__)!=sha(source/'solweig_gpu/solweig.py') or torch.cuda.is_available():raise RuntimeError('original source or CPU isolation failed')
    torch.set_num_threads(1)
    report={'evidence_class':'original_upstream_cpu_instrumented','upstream_commit':COMMIT,'upstream_patch_hash':None,'source_sha256':sha(original.__file__),'collector_sha256':sha(__file__),'environment':{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'package_path':str(Path(original.__file__).parent),'torch_threads':1,'torch_default_dtype':str(torch.get_default_dtype()),'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}},'invocation':sys.argv,'cases':[],'failures':[]}
    label=None
    frames={}
    codes={getattr(original,name).__code__:name for name in FUNCTIONS}
    def profile(frame,event,returned):
        name=codes.get(frame.f_code)
        if name is None:return
        if event=='call':
            index=len(report['cases'])
            arrays={}
            fields={}
            for key in inspect.signature(getattr(original,name)).parameters:
                value=frame.f_locals[key]
                if isinstance(value,torch.Tensor):
                    arrays[key]=value.detach().cpu().numpy().copy()
                    fields[key]={'kind':'tensor','dtype':str(arrays[key].dtype),'shape':list(arrays[key].shape)}
                elif isinstance(value,np.ndarray):
                    arrays[key]=value.copy();fields[key]={'kind':'ndarray','dtype':str(value.dtype),'shape':list(value.shape)}
                elif isinstance(value,np.generic):
                    arrays[key]=np.asarray(value);fields[key]={'kind':'numpy_scalar','dtype':str(arrays[key].dtype)}
                elif value is None:fields[key]={'kind':'none'}
                else:
                    arrays[key]=np.asarray(value);fields[key]={'kind':type(value).__name__,'dtype':str(arrays[key].dtype)}
            path=args.output/f'{index:03d}-{name}-input.npz'
            np.savez_compressed(path,**arrays)
            entry={'function':name,'label':label,'input':path.name,'input_sha256':sha(path),'fields':fields,'status':'entered'}
            report['cases'].append(entry);frames[id(frame)]=entry
        elif event=='return':
            entry=frames.pop(id(frame))
            if returned is None:
                entry['status']='original_failure';return
            path=args.output/entry['input'].replace('-input.npz','-output.npz')
            np.savez_compressed(path,**{f'output_{i}':value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value) for i,value in enumerate(returned)})
            entry.update(output=path.name,output_sha256=sha(path),status='captured')
    def execute(name,kwargs,case_label):
        nonlocal label
        label=case_label
        sys.setprofile(profile)
        try:getattr(original,name)(**kwargs)
        except Exception as error:
            report['failures'].append({'label':label,'function':name,'exception':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        finally:sys.setprofile(None)
    # Replay a real 32x35 daytime argument packet, preserving Tensor vs NumPy kinds.
    boundaries=ROOT/'tests/reference/small_original_cpu/boundaries'
    main_manifest=json.loads((boundaries/'manifest.json').read_text())
    event=next(item for item in main_manifest['events'] if item['function']=='Solweig_2022a_calc' and item['boundary']=='input' and item['timestep']==12)
    if sha(boundaries/event['path'])!=event['sha256']:raise RuntimeError('frozen main input hash mismatch')
    with np.load(boundaries/event['path']) as data:
        kwargs={}
        for key,spec in event['fields'].items():
            if '/' in key:continue
            if spec['kind']=='array':kwargs[key]=torch.from_numpy(data[key].copy())
            elif spec['kind']=='dict':kwargs[key]={child:data[key+'/'+child].item() for child in spec['keys']}
            elif spec['kind']=='list':kwargs[key]=[]
            elif spec['kind']=='none':kwargs[key]=None
            elif key in ('altitude','azimuth','zen','dectime','altmax','jday'):kwargs[key]=data[key][()]
            else:kwargs[key]=data[key].item()
    execute('Solweig_2022a_calc',kwargs,'original-small-daytime-step12')
    report['main_input_provenance']={'manifest_sha256':sha(boundaries/'manifest.json'),'packet_sha256':event['sha256'],'restoration':'array kind is original Torch tensor; solar/time scalar types restored from driver source'}
    for option in (1,2,3,4):
        alts,azis,*_=create_patches(option)
        table=torch.stack((alts,azis,torch.ones_like(alts)/alts.numel()),dim=1)
        rows,cols=3,5
        p=alts.numel()
        rng=np.random.default_rng(122+option)
        for mode in ('binary','raw'):
            def visibility():
                values=[0.,1.] if mode=='binary' else [-0.,0.,1.,2.,.5,float('nan')]
                return torch.from_numpy(rng.choice(values,(rows,cols,p)).astype(np.float32))
            sh,vs,vb=visibility(),visibility(),visibility()
            field=torch.full((rows,cols),.3)
            asvf=torch.from_numpy(np.tile(np.asarray([0.,np.pi/2,np.nan,-.2,6*np.pi/180],dtype=np.float32),(rows,1)))
            radI,radD,radG=(torch.tensor(value) for value in (650.,120.,700.))
            diff=sh-(1-vs)*(1-torch.tensor(.03))
            common=dict(radI=radI,radD=radD,radG=radG,shadow=field,svfS=field,svfW=field,svfN=field,svfE=field,svfEveg=field,svfSveg=field,svfWveg=field,svfNveg=field,azimuth=180.,altitude=35.,psi=torch.tensor(.03),t=0.,albedo=torch.tensor(.2),F_sh=field,KupE=field,KupS=field,KupW=field,KupN=field,cyl=1,lv=table,anisotropic_diffuse=1,diffsh=diff,rows=rows,cols=cols,asvf=asvf,shmat=sh,vegshmat=vs,vbshvegshmat=vb)
            execute('Kside_veg_v2022a',common,f'option{option}-{mode}-cylinder')
            if mode=='binary':
                common['cyl']=0
                common['azimuth']=torch.tensor(180.)
                execute('Kside_veg_v2022a',common,f'option{option}-{mode}-box-tensor-azimuth')
                common['azimuth']=180.
                execute('Kside_veg_v2022a',common,f'option{option}-{mode}-box-python-azimuth')
            long=dict(esky=torch.tensor(.8),sky_patches=table,Ta=torch.tensor(24.),Tgwall=torch.tensor(10.),ewall=.9,Lup=field*1400,shmat=sh,vegshmat=vs,vbshvegshmat=vb,solar_altitude=35.,solar_azimuth=180.,rows=rows,cols=cols,asvf=asvf)
            execute('Lcyl_v2022a',long,f'option{option}-{mode}-longwave-python-solar')
            long.update(solar_altitude=torch.tensor(35.),solar_azimuth=torch.tensor(180.))
            execute('Lcyl_v2022a',long,f'option{option}-{mode}-longwave-tensor-solar')
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':len(report['cases']),'captured':sum(case['status']=='captured' for case in report['cases']),'failures':len(report['failures'])}))

if __name__=='__main__':main()
