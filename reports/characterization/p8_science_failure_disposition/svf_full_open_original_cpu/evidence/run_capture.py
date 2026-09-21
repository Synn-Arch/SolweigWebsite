import argparse, hashlib, importlib, json, os, platform, sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(); p.add_argument("--mode",choices=["original","candidate_dense","candidate_compact"],required=True); p.add_argument("--output",required=True); a=p.parse_args()
rows,cols=9,13
z=np.zeros((rows,cols),dtype=np.float32)
kwargs=dict(patch_option=2,amaxvalue=np.float32(0),a=z.copy(),vegdem=z.copy(),vegdem2=z.copy(),bush=z.copy(),scale=np.float32(1))
if a.mode=="original":
 import torch
 assert not torch.cuda.is_available(), "CPU isolation required"
 from solweig_gpu.shadow import svf_calculator
 tk={k:(torch.tensor(v,device="cpu") if isinstance(v,np.ndarray) or isinstance(v,np.floating) else v) for k,v in kwargs.items()}
 out=svf_calculator(**tk)
 arrays=[x.detach().cpu().numpy() for x in out]
 env={"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"torch_threads":torch.get_num_threads(),"torch_interop_threads":torch.get_num_interop_threads()}
else:
 from solweig_light.geometry.svf import svf_calculator,svf_calculator_compact
 f=svf_calculator_compact if a.mode.endswith("compact") else svf_calculator
 out=f(**kwargs)
 arrays=[x.to_dense() if hasattr(x,"to_dense") else np.asarray(x) for x in out]
 import numba
 env={"numpy":np.__version__,"numba":numba.__version__}
assert len(arrays)==19
payload={f"field_{i:02d}":np.asarray(x) for i,x in enumerate(arrays)}
np.savez_compressed(a.output,**payload)
meta={"mode":a.mode,"output_count":len(arrays),"shapes":[list(x.shape) for x in arrays],"dtypes":[str(x.dtype) for x in arrays],"mins":[float(np.min(x)) for x in arrays],"maxs":[float(np.max(x)) for x in arrays],"environment":env,"python":sys.version,"platform":platform.platform(),"fixture":{"rows":rows,"cols":cols,"patch_option":2,"all_inputs_float32_zero":True,"scale":1.0,"amaxvalue":0.0}}
Path(a.output).with_suffix(".json").write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n")
