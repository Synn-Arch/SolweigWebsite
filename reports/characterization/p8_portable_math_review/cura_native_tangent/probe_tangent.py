"""Native source primitives on original ASVF; diagnostic stage-isolation gate."""
import argparse,ctypes,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--core-lib',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=np.load(a.trace);x=np.ascontiguousarray(r['asvf_bits'].view(np.float32));ref=r['hsvf'][:,0];core=ctypes.CDLL(str(a.core_lib.resolve()));core.eval.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];result={};raw={}
for name,lib,symbol in [('svml_la','libsvml_tan_probe.so','eval_tan'),('sleef_u35_scalar','libsleef_tan_probe.so','eval_scalar'),('sleef_u35_avx512','libsleef_tan_probe.so','eval_vector')]:
 l=ctypes.CDLL(str(Path(lib).resolve()));fn=getattr(l,symbol);fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];y=np.empty_like(x);fn(x.ctypes.data,y.ctypes.data,x.size);raw[name]=y
 # Preserve actual scalar yi wrapping to float32 before raster addition.
 delta=np.ascontiguousarray(y[:,None]+r['yi_clamped'].astype(np.float32)[None,:]);atan=np.empty_like(delta);core.eval(3,delta.ctypes.data,atan.ctypes.data,delta.size);deg=atan*np.float32(180/np.pi)
 result[name]={'finite_tan_bit_mismatches':int(np.count_nonzero(y.view(np.uint32)!=ref.view(np.uint32))),'max_abs':float(np.max(np.abs(y.astype(float)-ref.astype(float)))),'four_residual_rows_bits':y[63:67].view(np.uint32).tolist(),'masks_with_original_asvf_and_scalar_coefficients_and_core_atan':{k:int(np.count_nonzero(v!=np.unpackbits(r[k+'_pack'],axis=1,count=153))) for k,v in [('sun',deg<r['patch_altitude']),('shade',deg>r['patch_altitude'])]}}
a.output.mkdir(exist_ok=True,parents=True);np.savez_compressed(a.output/'tangent_results.npz',**raw);(a.output/'result.json').write_text(json.dumps({'results':result,'trace_sha256':hashlib.sha256(a.trace.read_bytes()).hexdigest(),'qualification':'Original ASVF and yi used for stage isolation only, not production lookup; no full-pipeline or complete portable profile pass implied.'},indent=2)+'\n');print(json.dumps(result,indent=2))
