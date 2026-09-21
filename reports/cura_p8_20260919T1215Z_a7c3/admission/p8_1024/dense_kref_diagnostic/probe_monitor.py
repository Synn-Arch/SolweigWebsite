import json, os, signal, subprocess, sys, time
from pathlib import Path
import psutil

LIMIT=12*1024**3
backend, python, probe, spec, log = sys.argv[1:]
run=Path(spec).parent
env=dict(os.environ,PYTHONNOUSERSITE="1",PYTHONUNBUFFERED="1",CUDA_VISIBLE_DEVICES="",PYTHONHASHSEED="0",
         OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",NUMBA_NUM_THREADS="1")
root="/home/ssynn3/workspace/solweig-light-codex-p8-20260919T1215Z-a7c3"
env["PROJ_DATA"]=root+("/env/share/proj" if backend=="candidate" else "/envs/upstream-cpu/share/proj")
env.pop("PYTHONPATH",None)
if backend=="candidate":
    env["PYTHONPATH"]=str(Path(probe).resolve().parent)+":"+root+"/source/src"
    env["NUMBA_CACHE_DIR"]=str(run/"jit")
cmd=[python,probe,"--backend",backend,"--spec",spec,"--log",log]
start=time.perf_counter(); peak=0; samples=[]; aborted=False
with (run/"probe_stdout.log").open("w") as out,(run/"probe_stderr.log").open("w") as err:
    p=subprocess.Popen(cmd,cwd=run,env=env,stdout=out,stderr=err,start_new_session=True); root=psutil.Process(p.pid)
    while p.poll() is None:
        try: members=[root,*root.children(recursive=True)]
        except psutil.NoSuchProcess: members=[]
        rss=0
        for m in members:
            try: rss+=m.memory_info().rss
            except (psutil.NoSuchProcess,psutil.AccessDenied): pass
        samples.append([time.perf_counter()-start,rss]); peak=max(peak,rss)
        if rss>LIMIT:
            aborted=True; os.killpg(p.pid,signal.SIGKILL); break
        time.sleep(.02)
    code=p.wait()
(run/"probe_measurement.json").write_text(json.dumps({"command":cmd,"exit_code":code,"peak_process_tree_rss_bytes":peak,"memory_limit_bytes":LIMIT,"memory_limit_aborted":aborted,"elapsed_seconds":time.perf_counter()-start},indent=2)+"\n")
raise SystemExit(code)
