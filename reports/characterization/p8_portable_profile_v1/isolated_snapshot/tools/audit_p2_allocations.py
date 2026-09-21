"""Candidate warm-call allocation accounting; not cross-runtime peak-memory parity.

Run with NUMBA_NRT_STATS=1. Fixed shapes match the frozen P2 timing workloads.
Tracemalloc covers Python/NumPy-visible allocations; NRT counters count native
allocation events, not bytes. Neither is a claim of complete application memory.
"""
import gc
import hashlib
import json
from pathlib import Path
import tracemalloc
import numpy as np
from numba import set_num_threads
from numba.core.runtime import rtsys
from solweig_light.geometry import walls
from solweig_light.comfort.utci import utci_calculator_compiled, utci_calculator_uniform

ROOT=Path(__file__).resolve().parents[1]


def main():
    rng=np.random.default_rng(20260918)
    dsm=rng.integers(0,40,(512,512)).astype(np.float32)
    row,col=np.indices((64,80))
    block=(((row%16>=4)&(row%16<12))&((col%16>=4)&(col%16<12))).astype(np.float32)*12
    wallmap=walls.findwalls_serial(block,3.)
    inputs=tuple(rng.uniform(low,high,(512,512)).astype(np.float32)
                 for low,high in [(-30,45),(0,100),(-30,80),(.15,17)])
    results=[]
    for parallel,threads in [(False,1),(True,1),(True,4),(True,10)]:
        set_num_threads(threads)
        jobs=[('walls',walls.findwalls_parallel if parallel else walls.findwalls_serial,(dsm,3.),{}),
              ('aspect',walls.filter1Goodwin_as_aspect_v3_parallel if parallel else walls.filter1Goodwin_as_aspect_v3_serial,(wallmap,1.,block),{}),
              ('utci',utci_calculator_compiled,inputs,{'parallel':parallel}),
              ('utci_uniform',utci_calculator_uniform,(np.float32(25),np.float32(50),inputs[2],inputs[3]),{'parallel':parallel})]
        for name,function,args,kwargs in jobs:
            function(*args,**kwargs)
            gc.collect()
            before=rtsys.get_allocation_stats()
            tracemalloc.start()
            output=function(*args,**kwargs)
            current,peak=tracemalloc.get_traced_memory()
            live=rtsys.get_allocation_stats()
            output_bytes=output.nbytes
            del output
            gc.collect()
            after=rtsys.get_allocation_stats()
            tracemalloc.stop()
            differences=lambda value:{key:getattr(value,key)-getattr(before,key) for key in before._fields}
            results.append(dict(kernel=name,parallel=parallel,threads=threads,output_bytes=output_bytes,
                traced_live_bytes=current,traced_peak_bytes=peak,nrt_at_return=differences(live),nrt_after_release=differences(after)))
            delta=differences(after)
            assert delta['alloc']==delta['free'] and delta['mi_alloc']==delta['mi_free'],(name,delta)
    report=dict(command='NUMBA_NRT_STATS=1 PYTHONPATH=src .venv-light/bin/python tools/audit_p2_allocations.py',
        scope=__doc__,seed=20260918,source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/solweig_light').rglob('*.py')},results=results)
    (ROOT/'reports/p2_allocation_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'All {len(results)} warmed calls balanced NRT allocation/free counters after output release')


if __name__=='__main__':main()
