"""Isolated portable primitive candidates against retained original Cura CPU data."""
import ctypes, json, sys, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(__file__).parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'reports/characterization/p8_asvf_parity_review')]
from solweig_light.radiation import engine as e
import sleef_acos_prototype as sa
import sleef_classifier_prototype as sc
B=ROOT/'reports/cura_p8_20260919T1215Z_a7c3/admission/p8_1024'
z=np.load(B/'classifier_mkl_diagnostic/corpus.npz'); bits=z['input_bits'];x=bits.view(np.float32)
r=np.load(B/'asvf_torch_characterization/torch_reference.npz')['result_bits'].view(np.float32)
sqref=np.load(B/'mkl_dispatch_diagnostic/mkl_probe_bits.npz')['torch_sqrt_bits'].view(np.float32)
maskref=np.load(B/'mkl_pipeline_experiment/remote/torch_actual_classifier.npz')
lib=ctypes.CDLL(str(OUT/'core_math/libprobe.dylib')); lib.eval.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
def core(op,x):
 x=np.ascontiguousarray(x,dtype=np.float32);y=np.empty_like(x);lib.eval(op,x.ctypes.data,y.ctypes.data,x.size);return y
igclib=ctypes.CDLL(str(OUT/'igc/libprobe.dylib'));igclib.eval.argtypes=lib.eval.argtypes
def igc(op,x):
 x=np.ascontiguousarray(x,dtype=np.float32);y=np.empty_like(x);igclib.eval(op,x.ctypes.data,y.ctypes.data,x.size);return y
def metrics(a,b):
 finite=np.isfinite(a)&np.isfinite(b); delta=np.abs(a[finite].astype(float)-b[finite].astype(float)); neq=a.view(np.uint32)!=b.view(np.uint32)
 return dict(finite_bit_mismatches=int((neq&finite).sum()),all_bit_mismatches=int(neq.sum()),nan_mask_equal=bool(np.array_equal(np.isnan(a),np.isnan(b))),max_abs=float(delta.max(initial=0)))
def classify(a,tan,atan):
 sun=[];shade=[];aa=a[:,None];rad=e._divide(np.pi,180.);deg=e._divide(180.,np.pi)
 for p in range(153):
  xi=np.cos(e._operate(np.multiply,np.abs(e._operate(np.subtract,np.asarray(z['solar_azimuth']),z['patch_azimuth'][p])),rad))
  yi=e._operate(np.multiply,e._operate(np.multiply,2,xi),np.tan(e._operate(np.multiply,np.asarray(z['solar_altitude']),rad)))
  delta=e._operate(np.add,tan(aa.ravel())[:,None],np.where(yi>0,0.,yi))
  degrees=e._operate(np.multiply,atan(delta.ravel())[:,None],deg)
  sun.append(degrees[:,0]<z['patch_altitude'][p]);shade.append(degrees[:,0]>z['patch_altitude'][p])
 return np.column_stack(sun),np.column_stack(shade)
idx=np.searchsorted(bits,z['boundary_bits']); result={'input_count':len(x),'boundary_inputs':len(idx),'patches':153,'qualification':'Actual per-patch original masks, not rejected vector-broadcast oracle; host NumPy float64 scalar cosine/tangent remains a qualified coefficient dependency.'}
with np.errstate(all='ignore'):
 cs=core(0,x); ca=core(1,cs);sf=sa.asvf_fma(x);da=np.arccos(cs.astype(np.float64)).astype(np.float32)
 result['sqrt_correct_rounded_vs_torch']=metrics(cs,sqref)
 result['profiles']={}
 for name,a,tan,atan in [('intel_igc_ha',igc(1,cs),lambda a:igc(2,a),lambda a:igc(3,a)),('core_math',ca,lambda a:core(2,a),lambda a:core(3,a)),('sleef_fma',sf,sc.tan_array,sc.atan_array),('float64_round',da,lambda a:np.tan(a.astype(float)).astype(np.float32),lambda a:np.arctan(a.astype(float)).astype(np.float32))]:
  su,sh=classify(a[idx],tan,atan)
  ms={k:int(np.count_nonzero(v!=np.unpackbits(maskref[k+'_pack'],axis=1,count=153))) for k,v in [('sun',su),('shade',sh)]}
  entry={'asvf':metrics(a,r),'actual_boundary_masks':ms,'categories':{k:metrics(a[np.searchsorted(bits,z[k+'_bits'])],r[np.searchsorted(bits,z[k+'_bits'])]) for k in ('frozen','boundary','random','edge','hotspot')}}
  full_sun,full_shade=su.copy(),sh.copy()
  su,sh=classify(r[idx],tan,atan);entry['oracle_asvf_isolating_classifier']={k:int(np.count_nonzero(v!=np.unpackbits(maskref[k+'_pack'],axis=1,count=153))) for k,v in [('sun',su),('shade',sh)]}
  result['profiles'][name]=entry
  np.savez_compressed(OUT/(name+'_results.npz'),asvf_bits=a.view(np.uint32),candidate_sun=full_sun,candidate_shade=full_shade,oracle_asvf_sun=su,oracle_asvf_shade=sh)
 result['core_acos_on_original_sqrt_diagnostic']=metrics(core(1,sqref),r)
(OUT/'primitive_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
