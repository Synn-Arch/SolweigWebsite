import json,sys
from pathlib import Path
import numpy as np,torch
root=Path(sys.argv[1]); z=np.load(root/'corpus.npz'); bits=z['input_bits']; x0=bits.view(np.float32)
base=np.load(root/'torch_reference.npz')['result_bits']
def evaluate(chunk,threads,offset):
 torch.set_num_threads(threads); outs=[]
 for lo in range(0,len(x0),chunk):
  x=torch.from_numpy(x0[lo:lo+chunk].copy())
  if offset:
   p=torch.empty(len(x)+offset,dtype=torch.float32); p[:offset]=0.25; p[offset:]=x; x=p[offset:]
  outs.append(torch.acos(torch.sqrt(x)).numpy().view(np.uint32).copy())
 out=np.concatenate(outs); d=out.astype(np.int64)-base.astype(np.int64); nz=np.flatnonzero(d)
 return {'chunk':chunk,'threads':threads,'offset_floats':offset,'mismatches_vs_full_default':int(nz.size),'first_indices':nz[:32].tolist(),'ulp_distribution':{str(int(k)):int(v) for k,v in zip(*np.unique(d,return_counts=True))}}
variants=[]
for chunk in (1,7,153,1024,4096,len(x0)):
 for threads in (1,torch.get_num_threads()):
  variants.append(evaluate(chunk,threads,0))
for off in (1,3,7,15): variants.append(evaluate(4096,1,off))
(root/'torch_layout_report.json').write_text(json.dumps({'variants':variants,'note':'full default reference was contiguous 1D; chunks exercise vector tails, offsets exercise starting alignment; torch thread counts affect inter-op partitioning only if kernel parallelizes'},indent=2,sort_keys=True)+'\n')
print(json.dumps([(v['chunk'],v['threads'],v['offset_floats'],v['mismatches_vs_full_default']) for v in variants]))
