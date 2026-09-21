"""Original full SVF option contracts, including unchanged upstream failures."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
PIN='0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
NAMES=('svf svfaveg svfE svfEaveg svfEveg svfN svfNaveg svfNveg svfS svfSaveg svfSveg svfveg svfW svfWaveg svfWveg vegshmat vbshvegshmat shmat svftotal').split()


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    import torch
    import solweig_gpu.shadow as original
    source=ROOT/'.upstream/SOLWEIG-GPU'
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==PIN
    assert not subprocess.check_output(['git','-C',str(source),'diff','--name-only'],text=True).strip()
    installed=Path(original.__file__)
    assert sha(installed)==sha(source/'solweig_gpu/shadow.py')
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    target=ROOT/'tests/reference/svf_original_cpu'
    target.mkdir(exist_ok=False)
    cases=[]
    for scene in ['open','building_vegetation','bush','negative']:
        a=np.zeros((9,13),np.float32)
        veg=np.zeros_like(a);trunk=np.zeros_like(a);bush=np.zeros_like(a)
        if scene=='building_vegetation':
            a[3:6,4:7]=8;veg[2:5,8:10]=12;trunk[2:5,8:10]=3
        elif scene=='bush':
            veg[2:5,3:7]=4;bush[2:5,3:7]=4
        elif scene=='negative':
            a[:]=-5;a[3:6,4:7]=-1
        maximum=np.float32(max(a.max(),veg.max())) if scene!='negative' else a.max()
        inputs=dict(a=a,vegdem=veg,vegdem2=trunk,bush=bush,amaxvalue=maximum,scale=np.float64(1.))
        for option in [1,2,3,4]:
            name=f'{scene}_option{option}'
            input_path=target/(name+'_inputs.npz')
            np.savez_compressed(input_path,**inputs,patch_option=np.int64(option))
            args={key:torch.from_numpy(np.asarray(value).copy()) for key,value in inputs.items() if key!='scale'}
            args['scale']=1.
            record=dict(name=name,option=option,input_file=input_path.name,input_sha256=sha(input_path))
            try:
                result=original.svf_calculator(option,**args)
                output_path=target/(name+'_outputs.npz')
                np.savez_compressed(output_path,**{key:value.detach().cpu().numpy() for key,value in zip(NAMES,result,strict=True)})
                record.update(status='executed',output_file=output_path.name,output_sha256=sha(output_path),
                    visibility_values={key:np.unique(value.detach().cpu().numpy()).tolist() for key,value in zip(NAMES,result) if key.endswith('mat')})
            except Exception as error:
                record.update(status='original_failure',exception_type=type(error).__name__,exception=str(error),traceback=traceback.format_exc())
            cases.append(record)
    (target/'generator.py').write_bytes(Path(__file__).read_bytes())
    manifest=dict(evidence_class='original_upstream_cpu',source_commit=PIN,source_sha256=sha(installed),
        oracle_patch_hash=None,generator_sha256=sha(Path(__file__)),
        command='.venv-oracle/bin/python tools/characterize_svf.py',python=sys.version,platform=platform.platform(),
        environment={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},cases=cases,
        option4_policy='Preserve original floating-annulus TypeError. Patch table supported, full SVF computation upstream-failing; no scientific repair.')
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print([(c['name'],c['status']) for c in cases])


if __name__=='__main__':main()
