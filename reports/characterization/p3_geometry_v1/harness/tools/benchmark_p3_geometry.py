"""One frozen P3 geometry measurement, isolated by the outer runner."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NAMES=('svf svfaveg svfE svfEaveg svfEveg svfN svfNaveg svfNveg svfS svfSaveg svfSveg svfveg svfW svfWaveg svfWveg vegshmat vbshvegshmat shmat svftotal').split()


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def scene(shape,bushy):
    row,col=np.indices(shape)
    building=((row%16>=4)&(row%16<9)&(col%16>=4)&(col%16<9))
    canopy=((row%16>=10)&(row%16<13)&(col%16>=10)&(col%16<13))
    a=building.astype(np.float32)*8+3
    veg=np.where(canopy,a+12,0).astype(np.float32)
    trunk=np.where(canopy,a+3,0).astype(np.float32)
    bush=np.zeros(shape,np.float32)
    if bushy:bush[2:5,10:14]=4;veg[2:5,10:14]=7;trunk[2:5,10:14]=0
    walls=np.zeros(shape,np.float32);walls[building]=4
    aspect=(col.astype(np.float32)/np.float32(shape[1]))*np.float32(2*np.pi)
    return dict(a=a,vegdem=veg,vegdem2=trunk,bush=bush,walls=walls,aspect=aspect)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['upstream','candidate'],required=True)
    p.add_argument('--task',choices=['sky','wall13','wall23','svf','load'],required=True)
    p.add_argument('--method',default='step')
    p.add_argument('--parallel',action='store_true')
    p.add_argument('--bush',action='store_true')
    p.add_argument('--threads',type=int,required=True)
    p.add_argument('--archive',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    protocol=ROOT/'benchmarks/protocols/p3_geometry_v1.json'
    rules=json.loads(protocol.read_text());shape=tuple(rules['svf_shape'] if args.task=='svf' else rules['ray_shape'])
    fields=scene(shape,args.bush)
    input_sha=hashlib.sha256(b''.join(key.encode()+value.tobytes() for key,value in sorted(fields.items()))).hexdigest()
    if args.backend=='upstream':
        import torch
        torch.set_num_threads(args.threads)
        import solweig_gpu.shadow as sky
        import solweig_gpu.solweig as radiation
        for module in [sky,radiation]:
            path=Path(module.__file__)
            assert digest(path)==digest(ROOT/'.upstream/SOLWEIG-GPU/solweig_gpu'/path.name)
        tensor={k:torch.from_numpy(v.copy()) for k,v in fields.items()}
        maximum=torch.tensor(15,dtype=torch.float32)
        if args.task=='sky':call=lambda:sky.shadow(maximum,*(tensor[k] for k in ['a','vegdem','vegdem2','bush']),133.,12.,1.)
        elif args.task=='wall13':call=lambda:radiation.shadowingfunction_wallheight_13(tensor['a'],133.,12.,1.,tensor['walls'],tensor['aspect'])
        elif args.task=='wall23':call=lambda:radiation.shadowingfunction_wallheight_23(tensor['a'],tensor['vegdem'],tensor['vegdem2'],133.,12.,1.,15.,tensor['bush'],tensor['walls'],tensor['aspect'])
        elif args.task=='svf':call=lambda:sky.svf_calculator(2,maximum,tensor['a'],tensor['vegdem'],tensor['vegdem2'],tensor['bush'],1.)
        else:
            def call():
                with np.load(args.archive,allow_pickle=False) as f:return tuple(f[k] for k in ['shadowmat','vegshadowmat','vbshmat'])
    else:
        from numba import set_num_threads
        set_num_threads(args.threads)
        from solweig_light.geometry import sky_compiled,svf,visibility
        from solweig_light.radiation.wall_shadows import exact_13,exact_23
        function=getattr(sky_compiled,'shadow_'+('pixel_' if args.method=='pixel' else '')+('parallel' if args.parallel else 'serial'))
        if args.task=='sky':call=lambda:function(np.float32(15),*(fields[k] for k in ['a','vegdem','vegdem2','bush']),133.,12.,1.)
        elif args.task=='wall13':call=lambda:exact_13(fields['a'],133.,12.,1.,fields['walls'],fields['aspect'],parallel=args.parallel)
        elif args.task=='wall23':call=lambda:exact_23(fields['a'],fields['vegdem'],fields['vegdem2'],133.,12.,1.,15.,fields['bush'],fields['walls'],fields['aspect'],parallel=args.parallel)
        elif args.task=='svf':
            svf.shadow=function # Explicit actual-kernel alternative, no mocked numerical work.
            call=lambda:svf.svf_calculator_compact(2,np.float32(15),fields['a'],fields['vegdem'],fields['vegdem2'],fields['bush'],1.)
        else:
            def call():
                loaded=visibility.import_visibility_npz(args.archive)
                return tuple(loaded[k] for k in ['shadowmat','vegshadowmat','vbshmat'])
    if args.task=='load':input_sha=digest(args.archive)
    start=time.perf_counter();result=call();first=time.perf_counter()-start
    samples=[]
    repetitions=rules['warm_repetitions_ray'] if args.task in ['sky','wall13','wall23'] else rules['warm_repetitions_svf']
    for _ in range(repetitions):
        del result;gc.collect()
        start=time.perf_counter();result=call();samples.append(time.perf_counter()-start)
    def array(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else x
    if args.task=='svf':
        outputs=dict(zip(NAMES,result))
        np.savez_compressed(args.output/'fields.npz',**{k:array(v) for k,v in outputs.items() if not k.endswith('mat')})
        if args.backend=='upstream':
            np.savez_compressed(args.output/'visibility.npz',**{k:array(outputs[v]) for k,v in [('shadowmat','shmat'),('vegshadowmat','vegshmat'),('vbshmat','vbshvegshmat')]})
        else:visibility.export_visibility_npz(args.output/'visibility.npz',outputs['shmat'],outputs['vegshmat'],outputs['vbshvegshmat'])
        storage=sum(v.numel()*v.element_size() if hasattr(v,'numel') else v.nbytes for v in result[15:18])
    elif args.task=='load':
        storage=sum(v.nbytes for v in result)
        if args.backend=='candidate':visibility.export_visibility_npz(args.output/'visibility.npz',*result)
        else:np.savez_compressed(args.output/'visibility.npz',**dict(zip(['shadowmat','vegshadowmat','vbshmat'],result)))
    else:
        np.savez_compressed(args.output/'fields.npz',**{str(i):array(v) for i,v in enumerate(result)})
        storage=sum(array(v).nbytes for v in result)
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report=dict(backend=args.backend,task=args.task,method=args.method,parallel=args.parallel,bush=args.bush,threads=args.threads,
        input_sha256=input_sha,protocol_sha256=digest(protocol),first_call_seconds=first,warm_seconds=samples,
        process_lifetime_peak_rss_bytes=int(peak if sys.platform=='darwin' else peak*1024),retained_output_or_visibility_payload_bytes=storage,
        artifacts={p.name:digest(p) for p in args.output.glob('*.npz')},python=sys.version,platform=platform.platform(),invocation=sys.argv,
        environment={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},thread_environment={k:os.environ.get(k) for k in ['NUMBA_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},
        numba_cache_dir=os.environ.get('NUMBA_CACHE_DIR'),peak_memory_includes='imports,fixture,JIT,calls,finalexport; not process-tree RSS')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(args.output,flush=True)


if __name__=='__main__':main()
