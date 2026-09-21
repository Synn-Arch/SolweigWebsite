"""Execute frozen P2 kernel comparisons sequentially in isolated processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--kernels',nargs='+',choices=['walls','aspect','utci','utci_uniform'],default=['walls','aspect','utci','utci_uniform'])
    args=parser.parse_args()
    args.run.mkdir(parents=True,exist_ok=False)
    worker=ROOT/'tools/benchmark_p2_kernel.py'
    guard=hashlib.sha256(worker.read_bytes()).hexdigest()
    def candidate_hashes():
        return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((ROOT/'src/solweig_light').rglob('*.py'))}
    source_guard=candidate_hashes()
    jobs=[]
    for kernel in args.kernels:
        for threads in [1,4,10]:jobs.append(('upstream',kernel,'original',threads,None,{}))
        for mode,threads in [('serial',1),('parallel',1),('parallel',4),('parallel',10)]:
            if kernel=='walls':function='solweig_light.geometry.walls:findwalls_'+mode
            elif kernel=='aspect':function='solweig_light.geometry.walls:filter1Goodwin_as_aspect_v3_'+mode
            else:function='solweig_light.comfort.utci:'+('utci_calculator_uniform' if kernel=='utci_uniform' else 'utci_calculator_compiled')
            jobs.append(('candidate',kernel,mode,threads,function,{'parallel':mode=='parallel'} if kernel.startswith('utci') else {}))
    # Deterministic ordering fixed before observing results. Each configuration
    # gets a new process and unique, initially empty JIT cache directory.
    failures=[]
    for backend,kernel,mode,threads,function,kwargs in jobs:
        name=f'{kernel}_{backend}_{mode}_{threads}'
        report=args.run/(name+'.json')
        cache=args.run/(name+'_cache')
        assert not cache.exists()
        environment=dict(os.environ,NUMBA_CACHE_DIR=str(cache.resolve()),PYTHONPATH=str(ROOT/'src') if backend=='candidate' else '',CUDA_VISIBLE_DEVICES='')
        for variable in ['NUMBA_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
            environment[variable]=str(threads)
        python=ROOT/('.venv-oracle' if backend=='upstream' else '.venv-light')/'bin/python'
        command=[str(python),str(worker),'--backend',backend,'--kernel',kernel,'--threads',str(threads),'--report',str(report.resolve())]
        if function:command+=['--callable',function,'--kwargs',json.dumps(kwargs)]
        assert hashlib.sha256(worker.read_bytes()).hexdigest()==guard
        assert candidate_hashes()==source_guard, 'Candidate changed during frozen measurement batch'
        with (args.run/(name+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        assert hashlib.sha256(worker.read_bytes()).hexdigest()==guard
        assert candidate_hashes()==source_guard, 'Candidate changed during frozen measurement batch'
        print(name,result.returncode,flush=True)
        if result.returncode:failures.append(dict(name=name,returncode=result.returncode))
    (args.run/'execution_manifest.txt').write_text(json.dumps(dict(worker_sha256=guard,candidate_sources=source_guard,failures=failures),indent=2)+'\n')
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
