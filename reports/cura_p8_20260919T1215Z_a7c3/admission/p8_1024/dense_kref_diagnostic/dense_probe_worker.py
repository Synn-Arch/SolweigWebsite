"""Diagnostic-only candidate tile worker; production source is untouched."""
import argparse,json,os,inspect
from pathlib import Path
import numpy as np
POINTS=((36,185),(37,184))
def val(x):
 a=np.asarray(x)
 if a.ndim>=2 and a.shape[0]>37 and a.shape[1]>185:return {f"{r},{c}":float(a[r,c]) for r,c in POINTS}
 if a.size==1:return float(a.reshape(-1)[0])
 return None
def main():
 p=argparse.ArgumentParser();p.add_argument("--job");p.add_argument("--options");a=p.parse_args()
 job=json.loads(Path(a.job).read_text()); opts=json.loads(a.options); log=Path(os.environ["SOLWEIG_DENSE_PROBE_LOG"]).open("w",buffering=1); step={"i":None}
 import solweig_light.pipeline as pipeline
 import solweig_light.radiation.engine as engine
 from solweig_light.runtime import RuntimeOptions
 def emit(kind,names,out):
  xs=out if isinstance(out,(tuple,list)) else (out,); rec={"kind":kind,"i":None if step["i"] is None else int(step["i"])}
  for n,x in zip(names,xs):rec[n]=val(x)
  log.write(json.dumps(rec,sort_keys=True)+"\n")
 helpers={"gvf_2018a":"gvfLup gvfalb gvfalbnosh gvfLupE gvfalbE gvfalbnoshE gvfLupS gvfalbS gvfalbnoshS gvfLupW gvfalbW gvfalbnoshW gvfLupN gvfalbN gvfalbnoshN gvfSum gvfNorm".split(),"Kside_veg_v2022a":"Keast Ksouth Kwest Knorth KsideI KsideD Kside".split(),"Lcyl_v2022a":"Ldown Lside Least_ Lwest_ Lnorth_ Lsouth_".split(),"Lside_veg_v2022a":"Least Lsouth Lwest Lnorth".split()}
 for name,names in helpers.items():
  orig=getattr(engine,name)
  def make(orig=orig,name=name,names=names):
   def wrap(*aa,**kk):out=orig(*aa,**kk);emit(name,names,out);return out
   return wrap
  setattr(engine,name,make())
 # Replace only the diagnostic wrapper around Kside to capture its exact step-7
 # inputs at the two failing pixels. Production functions remain unchanged.
 kside_orig=engine.Kside_veg_v2022a
 def kside_capture(*aa,**kk):
  out=kside_orig(*aa,**kk)
  emit("Kside_veg_v2022a",helpers["Kside_veg_v2022a"],out)
  if step["i"]==7:
   ref=engine._serial_Kside_veg_v2022a
   bound=inspect.signature(ref).bind(*aa,**kk).arguments
   saved={}
   for name,x in bound.items():
    if name in ("rows","cols"):
     saved[name]=np.array(1,dtype=np.int64);continue
    if name in ("shmat","vegshmat","vbshvegshmat","diffsh"):
     saved[name]=np.asarray([x[36,185,p] for p in range(x.shape[2])],dtype=np.float32).reshape(1,1,-1);continue
    a=np.asarray(x)
    saved[name]=a[36:37,185:186] if a.ndim>=2 and a.shape[0]>36 and a.shape[1]>185 else a
   saved["compiled_output_at_point"]=np.array([np.asarray(v)[36,185] for v in out],dtype=np.float32)
   np.savez_compressed(os.environ["SOLWEIG_DENSE_KREF_FIXTURE"],**saved)
  return out
 engine.Kside_veg_v2022a=kside_capture
 orig=engine.Solweig_2022a_calc; names="Tmrt Kdown Kup Ldown Lup Tg ea esky I0 CI shadow firstdaytime timestepdec timeadd Tgmap1 Tgmap1E Tgmap1S Tgmap1W Tgmap1N Keast Ksouth Kwest Knorth Least Lsouth Lwest Lnorth KsideI TgOut1 TgOut radI radD Lside L_patches CI_Tg CI_TgG KsideD dRad Kside".split()
 def calc(*aa,**kk):
  step["i"]=kk.get("i",aa[0] if aa else None);out=orig(*aa,**kk);emit("Solweig",names,out)
  if step["i"]==7: raise RuntimeError("DIAGNOSTIC_CAPTURE_COMPLETE_AFTER_TIMESTEP_7")
  return out
 pipeline.Solweig_2022a_calc=calc
 pipeline.run_tile(**job,runtime=RuntimeOptions(**opts));return 0
if __name__=="__main__":raise SystemExit(main())
