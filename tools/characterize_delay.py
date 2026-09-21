"""Original-only thermal delay threshold and irregular-timestep fixtures."""
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import torch
from solweig_gpu import solweig as original

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'tests/reference/delay_original_cpu'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(original.__file__)==sha(ROOT/'.upstream/SOLWEIG-GPU/solweig_gpu/solweig.py')
assert not DEST.exists()
DEST.mkdir(parents=True)
torch.set_num_threads(1)
rng=np.random.default_rng(814)
current=rng.uniform(-10,600,(7,9)).astype('float32')
previous=rng.uniform(-10,600,(7,9)).astype('float32')
current[0,0]=np.nan
cases=[]
threshold=59/1440
for first in (0.,1.):
 for elapsed in (0.,np.nextafter(threshold,-np.inf),threshold,np.nextafter(threshold,np.inf),2*threshold):
  for step in (1/1440,30/1440,1/24,2/24):
   name=f'case-{len(cases):03d}'
   inputs=dict(gvfLup=current,Tgmap1=previous,firstdaytime=np.asarray(first),timeadd=np.asarray(elapsed),timestepdec=np.asarray(step))
   np.savez_compressed(DEST/(name+'-input.npz'),**inputs)
   output=original.TsWaveDelay_2015a(torch.from_numpy(current.copy()),first,elapsed,step,torch.from_numpy(previous.copy()))
   np.savez_compressed(DEST/(name+'-output.npz'),**{str(i):v.detach().numpy() if isinstance(v,torch.Tensor) else np.asarray(v) for i,v in enumerate(output)})
   cases.append(dict(name=name,input=name+'-input.npz',output=name+'-output.npz',input_sha256=sha(DEST/(name+'-input.npz')),output_sha256=sha(DEST/(name+'-output.npz'))))
manifest=dict(evidence_class='original_upstream_cpu',upstream_commit='0d7fe742abeeddd890dd58fc76ed7f78bd47faec',upstream_patch_hash=None,source_sha256=sha(original.__file__),collector_sha256=sha(__file__),python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),invocation=sys.argv,threads=1,cases=cases)
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(len(cases),'original cases captured')
