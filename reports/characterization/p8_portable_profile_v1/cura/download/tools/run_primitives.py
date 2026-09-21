import hashlib,json,sys
from pathlib import Path
import numpy as np
from solweig_light.radiation._math_profile import asvf,profile_identity
from solweig_light.radiation._sleef_acos import asvf_fma
root=Path(__file__).resolve().parents[1]; oracle=root/'oracle'
def bits_report(actual,expected):
 f=np.isfinite(actual)&np.isfinite(expected)
 return {'finite_bit_mismatches':int(np.count_nonzero(actual[f].view(np.uint32)!=expected[f].view(np.uint32))),'nan_mask_mismatches':int(np.count_nonzero(np.isnan(actual)!=np.isnan(expected))),'posinf_mask_mismatches':int(np.count_nonzero(np.isposinf(actual)!=np.isposinf(expected))),'neginf_mask_mismatches':int(np.count_nonzero(np.isneginf(actual)!=np.isneginf(expected)))}
records=[]
z=np.load(oracle/'oracle.npz'); x=np.load(oracle/'inputs.npy'); y=asvf(x)
records.append({'name':'m1_411685_asvf','count':len(x),**bits_report(y,z['asvf'])})
c=np.load(oracle/'cura_189220_primitive_oracle.npz'); x=c['input_bits'].view(np.float32); sq=np.sqrt(x); y=asvf(x)
records.append({'name':'cura_189220_sqrt','count':len(x),**bits_report(sq,c['sqrt_bits'].view(np.float32))})
records.append({'name':'cura_189220_asvf','count':len(x),**bits_report(y,c['asvf_bits'].view(np.float32))})
llvm=str(asvf_fma.inspect_llvm(asvf_fma.signatures[0])); fma_count=llvm.count('llvm.fma.f32'); fast_count=llvm.count(' fast ')
out={'schema':'portable-profile-linux-primitives.v1','records':records,'profile':profile_identity(),'torch_imported':'torch' in sys.modules,'llvm':{'fma_call_text_count':fma_count,'fast_flag_text_count':fast_count,'signature':str(asvf_fma.signatures[0])},'passed':all(not sum(v for k,v in r.items() if k.endswith('mismatches')) for r in records)}
(root/'evidence/stage_c_linux_primitives.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
(root/'evidence/asvf_llvm.txt').write_text(llvm)
print(json.dumps({'passed':out['passed'],'records':records,'llvm':out['llvm']}))
raise SystemExit(0 if out['passed'] and fma_count and not fast_count else 2)
