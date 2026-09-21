"""Uniform source-derived primitive combinations; no per-input tuning or oracle lookup."""
import ctypes,sys,json
from pathlib import Path
import numpy as np
P=Path(__file__).parent;ROOT=P.resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'reports/characterization/p8_asvf_parity_review')]
from solweig_light.radiation import engine as e
import sleef_classifier_prototype as s
R=ROOT/'reports/cura_p8_20260919T1215Z_a7c3/admission/p8_1024';r=np.load(R/'torch_patch_scalar_trace/results/trace.npz');z=np.load(R/'classifier_mkl_diagnostic/corpus.npz');idx=np.searchsorted(z['input_bits'],z['boundary_bits']);refs={k:np.unpackbits(r[k+'_pack'],axis=1,count=153).astype(bool) for k in ('sun','shade')}
libs={n:ctypes.CDLL(str((P/n/'libprobe.dylib').resolve())) for n in ('core_math','igc')}
for l in libs.values():l.eval.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
def fn(name,op,a):
 a=np.ascontiguousarray(a,dtype=np.float32)
 if name=='sleef':return (s.tan_array if op==2 else s.atan_array)(a)
 y=np.empty_like(a);libs[name].eval(op,a.ctypes.data,y.ctypes.data,a.size);return y
original=r['asvf_bits'].view(np.float32);rad=e._divide(np.pi,180.);deg=e._divide(180.,np.pi);report=[]
for asrc in ('oracle_diagnostic','core_math','intel_igc_ha','sleef_fma'):
 a=original if asrc=='oracle_diagnostic' else np.load(P/(asrc+'_results.npz'))['asvf_bits'].view(np.float32)[idx]
 for tn in ('core_math','igc','sleef'):
  hsvf=fn(tn,2,a)
  for an in ('core_math','igc','sleef'):
   cols=[]
   for p in range(153):
    xi=np.cos(e._operate(np.multiply,np.abs(e._operate(np.subtract,np.asarray(z['solar_azimuth']),z['patch_azimuth'][p])),rad));yi=e._operate(np.multiply,e._operate(np.multiply,2,xi),np.tan(e._operate(np.multiply,np.asarray(z['solar_altitude']),rad)))
    delta=e._operate(np.add,hsvf[:,None],np.where(yi>0,0.,yi));cols.append(e._operate(np.multiply,fn(an,3,delta[:,0]),deg))
   d=np.column_stack(cols);entry={'asvf':asrc,'tan':tn,'atan':an,'sun':int(np.count_nonzero((d<z['patch_altitude'])!=refs['sun'])),'shade':int(np.count_nonzero((d>z['patch_altitude'])!=refs['shade']))};report.append(entry)
(P/'uniform_source_combinations.json').write_text(json.dumps(report,indent=2)+'\n')
for v in report:
 if v['asvf']=='oracle_diagnostic' or v['sun']+v['shade']<10:print(v)
