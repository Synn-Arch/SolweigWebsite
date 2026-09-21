"""Run P3 fixed configurations sequentially; preserve failures and source snapshots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();target=args.run.resolve();target.mkdir(parents=True,exist_ok=False)
    worker=ROOT/'tools/benchmark_p3_geometry.py';guard=sha(worker)
    sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/solweig_light').rglob('*.py')}
    jobs=[]
    for task in ['sky','wall13','wall23']:
        for bush in ([False,True] if task=='sky' else [False]):
            for threads in [1,4,10]:jobs.append((task,'upstream','step',False,threads,bush))
            for method in (['step','pixel'] if task=='sky' else ['step']):
                for parallel,threads in [(False,1),(True,1),(True,4),(True,10)]:jobs.append((task,'candidate',method,parallel,threads,bush))
    for threads in [1,4,10]:jobs.append(('svf','upstream','step',False,threads,False))
    for method in ['step','pixel']:jobs.append(('svf','candidate',method,False,1,False))
    jobs += [('load','upstream','step',False,1,False),('load','candidate','step',False,1,False)]
    failures=[]
    for task,backend,method,parallel,threads,bush in jobs:
        name=f'{task}_{backend}_{method}_{"parallel" if parallel else "serial"}_{threads}_{"bush" if bush else "clear"}'
        command=[str(ROOT/('.venv-oracle' if backend=='upstream' else '.venv-light')/'bin/python'),str(worker),
                 '--backend',backend,'--task',task,'--method',method,'--threads',str(threads),'--output',str(target/name)]
        if parallel:command.append('--parallel')
        if bush:command.append('--bush')
        if task=='load':command+=['--archive',str(target/'svf_upstream_step_serial_1_clear/visibility.npz')]
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT/'src') if backend=='candidate' else '',NUMBA_CACHE_DIR=str(target/(name+'_cache')))
        for key in ['NUMBA_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:environment[key]=str(threads)
        assert sha(worker)==guard and all(sha(ROOT/path)==value for path,value in sources.items())
        with (target/(name+'.log')).open('w') as log:result=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        assert sha(worker)==guard and all(sha(ROOT/path)==value for path,value in sources.items())
        print(name,result.returncode,flush=True)
        if result.returncode:failures.append(dict(name=name,returncode=result.returncode))
    for relative in sources:
        destination=target/'candidate_source'/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,destination)
    for relative in ['tools/benchmark_p3_geometry.py','tools/run_p3_benchmarks.py','benchmarks/protocols/p3_geometry_v1.json']:
        destination=target/'harness'/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,destination)
    (target/'execution_manifest.json').write_text(json.dumps(dict(worker_sha256=guard,source_sha256=sources,failures=failures,jobs=len(jobs)),indent=2)+'\n')
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
