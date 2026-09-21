"""Diagnostic-only wrappers for dense-1024 directional radiation values."""
import argparse, json, os, sys, time
from pathlib import Path

POINTS = ((36, 185), (37, 184))

def value(x):
    import numpy as np
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    a = np.asarray(x)
    if a.ndim >= 2 and a.shape[0] > 37 and a.shape[1] > 185:
        return {f"{r},{c}": float(a[r, c]) for r, c in POINTS}
    if a.size == 1:
        return float(a.reshape(-1)[0])
    return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--backend",choices=("upstream","candidate")); ap.add_argument("--spec",type=Path); ap.add_argument("--log",type=Path)
    a=ap.parse_args(); spec=json.loads(a.spec.read_text()); log=a.log.open("w",buffering=1); step={"i":None}
    if a.backend == "candidate":
        import solweig_light, solweig_light.pipeline as pipeline
        import solweig_light.radiation.engine as engine
        import solweig_light.runtime as runtime
        call=solweig_light.thermal_comfort
        original_execute=runtime.execute_tiles
        runtime.execute_tiles=lambda jobs,options: original_execute(jobs,options,worker_module="dense_probe_worker")
        os.environ["SOLWEIG_DENSE_PROBE_LOG"]=str(a.log)
        os.environ["SOLWEIG_DENSE_KREF_FIXTURE"]=str(a.spec.parent/"kref_step7_candidate_input.npz")
    else:
        import torch; torch.set_num_threads(1)
        import solweig_gpu, solweig_gpu.solweig as engine
        import solweig_gpu.utci_process as upstream_pipeline
        call=solweig_gpu.thermal_comfort
        pipeline=None
    def emit(kind,names,result):
        vals=result if isinstance(result,(tuple,list)) else (result,)
        rec={"kind":kind,"i":None if step["i"] is None else int(step["i"])}
        for name,x in zip(names,vals): rec[name]=value(x)
        log.write(json.dumps(rec,sort_keys=True)+"\n")
    helpers={
      "gvf_2018a":("gvfLup gvfalb gvfalbnosh gvfLupE gvfalbE gvfalbnoshE gvfLupS gvfalbS gvfalbnoshS gvfLupW gvfalbW gvfalbnoshW gvfLupN gvfalbN gvfalbnoshN gvfSum gvfNorm".split()),
      "Kside_veg_v2022a":("Keast Ksouth Kwest Knorth KsideI KsideD Kside".split()),
      "Lcyl_v2022a":("Ldown Lside Least_ Lwest_ Lnorth_ Lsouth_".split()),
      "Lside_veg_v2022a":("Least Lsouth Lwest Lnorth".split()),
    }
    for fname,names in helpers.items():
        orig=getattr(engine,fname)
        def make(orig=orig,fname=fname,names=names):
            def wrapped(*args,**kwargs):
                out=orig(*args,**kwargs); emit(fname,names,out); return out
            return wrapped
        setattr(engine,fname,make())
    orig_calc=engine.Solweig_2022a_calc
    returns=("Tmrt Kdown Kup Ldown Lup Tg ea esky I0 CI shadow firstdaytime timestepdec timeadd Tgmap1 Tgmap1E Tgmap1S Tgmap1W Tgmap1N Keast Ksouth Kwest Knorth Least Lsouth Lwest Lnorth KsideI TgOut1 TgOut radI radD Lside L_patches CI_Tg CI_TgG KsideD dRad Kside".split())
    def calc(*args,**kwargs):
        step["i"]=kwargs.get("i",args[0] if args else None); out=orig_calc(*args,**kwargs); emit("Solweig",returns,out); return out
    engine.Solweig_2022a_calc=calc
    if pipeline is not None: pipeline.Solweig_2022a_calc=calc
    else: upstream_pipeline.Solweig_2022a_calc=calc
    start=time.perf_counter()
    if a.backend=="candidate":
        with solweig_light.runtime_options(solweig_light.RuntimeOptions(cpu_budget=1,workers=1,threads_per_worker=1,memory_budget_bytes=12*1024**3,block_pixels=128,cache_enabled=True,cache_dir=str(a.spec.parent/"runtime_cache"),checkpoint_interval=1,legacy_cache_policy="recompute",resume=False)):
            returned=call(**spec["kwargs"])
    else: returned=call(**spec["kwargs"])
    (a.spec.parent/"probe_outcome.json").write_text(json.dumps({"return":repr(returned),"seconds":time.perf_counter()-start})+"\n")

if __name__=="__main__": main()
