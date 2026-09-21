import ctypes, hashlib, json, sys
from pathlib import Path
import numpy as np

base=Path(sys.argv[1]); trace=np.load(base/'trace.npz'); corpus=np.load(base/'corpus.npz')
idx=np.searchsorted(corpus['input_bits'],corpus['boundary_bits']); assert np.array_equal(corpus['input_bits'][idx],corpus['boundary_bits'])
core=ctypes.CDLL(str((base/'libcore_probe.so').resolve())); core.eval.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
libs={}
for name,lib,symbol in [('svml_la','libsvml_tan_probe.so','eval_tan'),('sleef_u35_scalar','libsleef_tan_probe.so','eval_scalar'),('sleef_u35_avx512','libsleef_tan_probe.so','eval_vector')]:
 l=ctypes.CDLL(str((base/lib).resolve())); f=getattr(l,symbol); f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]; libs[name]=f
oracle={k:np.unpackbits(trace[k+'_pack'],axis=1,count=153).astype(bool) for k in ('sun','shade')}; raw={}; report={}
for profile,file in [('core_math','core_math_results.npz'),('sleef_fma','sleef_fma_results.npz'),('intel_igc_ha','intel_igc_ha_results.npz')]:
 z=np.load(base/file); asvf=np.ascontiguousarray(z['asvf_bits'][idx].view(np.float32)); report[profile]={}
 for tangent,fn in libs.items():
  hsvf=np.empty_like(asvf); fn(asvf.ctypes.data,hsvf.ctypes.data,hsvf.size)
  delta=np.ascontiguousarray(hsvf[:,None]+trace['yi_clamped'].astype(np.float32)[None,:]); atan=np.empty_like(delta); core.eval(3,delta.ctypes.data,atan.ctypes.data,delta.size); degrees=atan*np.float32(180/np.pi)
  sun=degrees<trace['patch_altitude']; shade=degrees>trace['patch_altitude']; key=profile+'__'+tangent; raw[key+'_hsvf_bits']=hsvf.view(np.uint32); raw[key+'_sun_pack']=np.packbits(sun,axis=1); raw[key+'_shade_pack']=np.packbits(shade,axis=1)
  report[profile][tangent]={'sun_mismatches':int(np.count_nonzero(sun!=oracle['sun'])),'shade_mismatches':int(np.count_nonzero(shade!=oracle['shade'])),'four_residual_rows_hsvf_bits':hsvf[63:67].view(np.uint32).tolist()}
np.savez_compressed(base/'results/generated_asvf_masks.npz',**raw)
(base/'results/generated_asvf_report.json').write_text(json.dumps({'qualification':'Generated ASVF end-to-end boundary diagnostic using declared profile ASVF, uniform native tangent, CORE-MATH atan, and original scalar coefficients; not a complete production profile.','results':report},indent=2,sort_keys=True)+'\n')
print(json.dumps(report,indent=2,sort_keys=True))
