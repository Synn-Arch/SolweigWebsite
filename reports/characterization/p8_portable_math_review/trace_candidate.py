"""Portable stage replay, using original ASVF only to localize classifier mismatch."""
import ctypes,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).parent;sys.path.insert(0,str(ROOT/'src'))
from solweig_light.radiation import engine as e
B=ROOT/'reports/cura_p8_20260919T1215Z_a7c3/admission/p8_1024';z=np.load(B/'classifier_mkl_diagnostic/corpus.npz');idx=np.searchsorted(z['input_bits'],z['boundary_bits']);a=np.load(B/'asvf_torch_characterization/torch_reference.npz')['result_bits'][idx].view(np.float32)
lib=ctypes.CDLL(str(OUT/'core_math/libprobe.dylib'));lib.eval.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
def fn(op,x):
 x=np.ascontiguousarray(x,dtype=np.float32);out=np.empty_like(x);lib.eval(op,x.ctypes.data,out.ctypes.data,x.size);return out
trace={'asvf':a};rad=e._divide(np.pi,180.);deg=e._divide(180.,np.pi);st=np.tan(e._operate(np.multiply,np.asarray(z['solar_altitude']),rad));trace['solar_tan']=st;trace['hsvf']=fn(2,a)
xi=[];yi=[];delta=[];atan=[];degrees=[]
for p in range(153):
 cx=np.cos(e._operate(np.multiply,np.abs(e._operate(np.subtract,np.asarray(z['solar_azimuth']),z['patch_azimuth'][p])),rad));cy=e._operate(np.multiply,e._operate(np.multiply,2,cx),st);td=e._operate(np.add,trace['hsvf'][:,None],np.where(cy>0,0.,cy));at=fn(3,td.ravel());dg=e._operate(np.multiply,at,deg)
 xi.append(cx);yi.append(cy);delta.append(td[:,0]);atan.append(at);degrees.append(dg)
trace.update(xi=np.asarray(xi),yi=np.asarray(yi),tan_delta=np.column_stack(delta),atan=np.column_stack(atan),degrees=np.column_stack(degrees))
np.savez_compressed(OUT/'core_actual_stage_trace.npz',**trace)
print(json.dumps({k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in trace.items()},indent=2))
