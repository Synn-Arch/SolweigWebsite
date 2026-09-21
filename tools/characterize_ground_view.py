"""Original-only ground-view fixtures; never imports the candidate package."""
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PIN='0d7fe742abeeddd890dd58fc76ed7f78bd47faec'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    import numpy as np
    import torch
    original=importlib.import_module('solweig_gpu.solweig')
    upstream=ROOT/'.upstream/SOLWEIG-GPU'
    assert subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()==PIN
    assert not subprocess.check_output(['git','-C',str(upstream),'diff','--name-only'],text=True).strip()
    installed=Path(original.__file__)
    assert sha(installed)==sha(upstream/'solweig_gpu/solweig.py')
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    target=ROOT/'tests/reference/ground_view_original_cpu'
    target.mkdir(exist_ok=False)
    rng=np.random.default_rng(20260918)
    shape=(17,23)
    buildings=np.ones(shape,np.float32);buildings[6:10,8:12]=0
    walls=np.zeros(shape,np.float32);walls[5:11,7]=3;walls[5:11,12]=4
    shadow=rng.choice(np.array([0,.25,.75,1],np.float32),shape)
    sunwall=rng.choice(np.array([-.2,0,.2,3],np.float32),shape)
    Tg=rng.uniform(-2,17,shape).astype(np.float32)
    base=dict(scale=np.float32(1),buildings=buildings,shadow=shadow,sunwall=sunwall,
              first=np.float32(1),second=np.float32(14),
              aspect=rng.uniform(0,2*np.pi,shape).astype(np.float32),walls=walls,Tg=Tg,
              Tgwall=np.full(shape,8,np.float32),Ta=np.float32(25),emis_grid=np.full(shape,.95,np.float32),
              ewall=np.float32(.9),alb_grid=np.full(shape,.15,np.float32),SBC=np.float32(5.67051e-8),
              albedo_b=np.float32(.2),Twater=np.float32(18),lc_grid=np.full(shape,6,np.float32),landcover=0)
    cases=[]
    def execute(name,function,values):
        inputs={key:np.asarray(value).copy() for key,value in values.items()}
        tensors={key:torch.tensor(value) if value.dtype.kind=='f' or value.ndim>0 else int(value) for key,value in inputs.items()}
        inp=target/(name+'_inputs.npz');out=target/(name+'_outputs.npz');after=target/(name+'_after.npz')
        np.savez_compressed(inp,**inputs)
        entry=dict(name=name,function=function.__name__,input_artifact=dict(path=inp.name,sha256=sha(inp)))
        try:
            result=function(**tensors)
            np.savez_compressed(out,**{f'output_{i}':value.detach().numpy() for i,value in enumerate(result)})
            entry.update(status='executed',output_artifact=dict(path=out.name,sha256=sha(out)))
        except Exception as error:
            entry.update(status='failed',exception=type(error).__name__,message=str(error))
        np.savez_compressed(after,**{key:value.detach().numpy() if isinstance(value,torch.Tensor) else np.asarray(value) for key,value in tensors.items()})
        entry['after_artifact']=dict(path=after.name,sha256=sha(after))
        cases.append(entry)
    for direction in np.arange(5,359,20,dtype=np.float32):
        execute(f'sun_direction_{int(direction)}',original.sunonsurface_2018a,dict(base,azimuthA=direction))
    for name in ['water','nan_buildings','nan_shadow','nan_temperature','nonbinary','rounding_half','rounding_scale','beyond_raster','zero_distance','float64']:
        values={key:np.asarray(value).copy() for key,value in base.items()}
        if name=='water':values['landcover']=1;values['lc_grid'][::2,::2]=3
        if name=='nan_buildings':values['buildings'][8,9]=np.nan
        if name=='nan_shadow':values['shadow'][8,9]=np.nan
        if name=='nan_temperature':values['Tg'][8,9]=np.nan
        if name=='nonbinary':values['buildings'][3:7,4:9]=.5;values['buildings'][12,14]=-.25
        if name=='rounding_half':values['first']=np.float32(2.5);values['second']=np.float32(10.5)
        if name=='rounding_scale':values['scale']=np.float32(.5);values['first']=np.float32(3);values['second']=np.float32(15)
        if name=='beyond_raster':values['second']=np.float32(25)
        if name=='zero_distance':values['second']=np.float32(0)
        if name=='float64':
            for key,value in values.items():
                if np.asarray(value).dtype.kind=='f' and np.asarray(value).ndim>0:values[key]=value.astype(np.float64)
        execute('sun_'+name,original.sunonsurface_2018a,dict(values,azimuthA=np.float32(45)))
        gvf=dict(values)
        gvf.pop('sunwall');gvf.pop('aspect')
        gvf.update(wallsun=walls.copy(),dirwalls=rng.uniform(0,360,shape).astype(np.float32),rows=shape[0],cols=shape[1])
        execute('gvf_'+name,original.gvf_2018a,gvf)
    (target/'generator.py').write_bytes(Path(__file__).read_bytes())
    manifest=dict(evidence_class='original_upstream_cpu',source_commit=PIN,source_sha256=sha(installed),
                  patch=None,candidate_import=False,generator_sha256=sha(Path(__file__)),invocation=sys.argv,
                  python=sys.version,platform=platform.platform(),torch_threads=torch.get_num_threads(),cuda='not_available',
                  environment={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},cases=cases)
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print({status:sum(case['status']==status for case in cases) for status in ('executed','failed')})

if __name__=='__main__':main()
