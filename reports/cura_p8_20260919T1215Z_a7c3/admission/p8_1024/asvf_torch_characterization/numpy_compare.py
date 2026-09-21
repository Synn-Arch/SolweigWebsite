import hashlib,json,platform,sys,warnings
from pathlib import Path
import numpy as np
from solweig_light.radiation.engine import shaded_or_sunlit
root=Path(sys.argv[1]); z=np.load(root/'corpus.npz'); ref=np.load(root/'torch_reference.npz'); bits=z['input_bits']; x=bits.view(np.float32)
with warnings.catch_warnings(record=True) as ws:
 warnings.simplefilter('always')
 sqrt32=np.sqrt(x,dtype=np.float32)
 current=np.arccos(sqrt32,dtype=np.float32)
 proposed=np.arccos(sqrt32.astype(np.float64)).astype(np.float32)
 warning_text=[str(w.message) for w in ws]
rb=ref['result_bits']; cb=current.view(np.uint32); pb=proposed.view(np.uint32)
pa=z['patch_altitude']; pazi=z['patch_azimuth']; sa=np.asarray(z['solar_altitude']); saz=float(z['solar_azimuth'])
def classifications(values):
 ps=[]; ph=[]
 for lo in range(0,len(values),4096):
  v=values[lo:lo+4096,None]
  sun,shade=shaded_or_sunlit(sa,saz,pa,pazi,v)
  ps.append(np.packbits(sun,axis=1)); ph.append(np.packbits(shade,axis=1))
 return np.concatenate(ps),np.concatenate(ph)
cs,ch=classifications(current); ps,ph=classifications(proposed); rs=ref['sun_pack']; rh=ref['shade_pack']
def classify_counts(a,b):
 xor=np.bitwise_xor(a,b); return int(np.unpackbits(xor,axis=1,count=153).sum()),int(np.count_nonzero(np.any(xor,axis=1)))
def category(k):
 q=z[k+'_bits']; idx=np.searchsorted(bits,q); valid=(idx<len(bits))&(bits[np.minimum(idx,len(bits)-1)]==q); idx=idx[valid]
 def mm(v): return int(np.count_nonzero(v[idx]!=rb[idx]))
 def cl(s,h):
  sc,si=classify_counts(s[idx],rs[idx]); hc,hi=classify_counts(h[idx],rh[idx]); return {'sun_cell_mismatches':sc,'sun_input_mismatches':si,'shade_cell_mismatches':hc,'shade_input_mismatches':hi}
 return {'count':int(len(idx)),'current_bit_mismatches':mm(cb),'proposed_bit_mismatches':mm(pb),'current_classification':cl(cs,ch),'proposed_classification':cl(ps,ph)}
categories={k:category(k) for k in ['frozen','boundary','random','edge','hotspot']}
mismatch=np.flatnonzero(pb!=rb)
examples=[]
for i in mismatch[:64]: examples.append({'index':int(i),'input_bits':int(bits[i]),'input':float(x[i]),'sqrt_bits':int(sqrt32.view(np.uint32)[i]),'torch_bits':int(rb[i]),'current_bits':int(cb[i]),'proposed_bits':int(pb[i])})
np.savez_compressed(root/'comparison_bits.npz',input_bits=bits,torch_bits=rb,current_bits=cb,proposed_bits=pb,proposed_mismatch_indices=mismatch.astype(np.uint32))
report={'method':'np.arccos(np.sqrt(x,dtype=float32).astype(float64)).astype(float32)','preserves_sqrt_float32':True,'counts':{'inputs':int(len(bits)),'current_bit_mismatches':int(np.count_nonzero(cb!=rb)),'proposed_bit_mismatches':int(len(mismatch))},'classification':{'current':{'sun':classify_counts(cs,rs),'shade':classify_counts(ch,rh)},'proposed':{'sun':classify_counts(ps,rs),'shade':classify_counts(ph,rh)}},'categories':categories,'warnings':warning_text,'examples':examples,'numpy':np.__version__,'python':sys.version,'platform':platform.platform()}
(root/'comparison_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'counts':report['counts'],'classification':report['classification'],'categories':categories},indent=2))
