#!/usr/bin/env python3
"""Small diagnostic for exact building-only SOLWEIG discrete-ray horizons."""
from __future__ import annotations
import argparse,json,time,importlib.util
from pathlib import Path
import numpy as np

def load_verified_shadow():
    path=Path(__file__).resolve().parents[2]/"src/solweig_light/geometry/shadows.py"
    spec=importlib.util.spec_from_file_location("p7_verified_shadows",path); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.shadow_numpy,path

def ray_samples(shape,amax,azimuth,altitude,scale=1.0):
    if azimuth==0: azimuth=1e-12
    az=np.float32(azimuth)*(np.pi/180); alt=np.float32(altitude)*(np.pi/180)
    sin,cos,tan=np.sin(az),np.cos(az),np.tan(az); ss,sc=np.sign(sin),np.sign(cos)
    ds_s,ds_c=abs(1/sin),abs(1/cos); tas=np.tan(alt)/scale
    sx,sy=shape; out=[]; dx=dy=dz=np.float32(0); index=1
    while amax>=dz and abs(dx)<sx and abs(dy)<sy:
        if np.pi/4<=az<3*np.pi/4 or 5*np.pi/4<=az<7*np.pi/4:
            dy=ss*index; dx=-sc*abs(np.round(index/tan)); ds=ds_s
        else:
            dy=ss*abs(np.round(index*tan)); dx=-sc*index; ds=ds_c
        dz=ds*index*tas; out.append((int(dx),int(dy),np.float32(dz))); index+=1
    return out

def exact_horizon(a,samples):
    sx,sy=a.shape; h=a.copy()
    for dx,dy,dz in samples:
        for r in range(sx):
            for c in range(sy):
                rr,cc=r+dx,c+dy
                if 0<=rr<sx and 0<=cc<sy: h[r,c]=max(h[r,c],np.float32(a[rr,cc]-dz))
    return h

def translated_scan(a,samples):
    """Proposed one-predecessor max-plus scan using the first ray translation."""
    dx,dy,dz=samples[0]; sx,sy=a.shape; h=a.copy()
    rows=range(sx-1,-1,-1) if dx>0 else range(sx)
    cols=range(sy-1,-1,-1) if dy>0 else range(sy)
    for r in rows:
        for c in cols:
            rr,cc=r+dx,c+dy
            if 0<=rr<sx and 0<=cc<sy: h[r,c]=max(h[r,c],np.float32(h[rr,cc]-dz))
    return h

def actual_shadow_mask(kernel,a,azimuth,altitude):
    zeros=np.zeros_like(a); sunlight,_,_=kernel(float(np.max(a)),a,zeros,zeros,zeros,azimuth,altitude,1.0)
    return sunlight==np.float32(0)

def run():
    kernel,kernel_path=load_verified_shadow()
    cases=[]
    # Minimal phase counterexample: at 30 degrees k=2 is (-2,+1), not twice k=1 (-2,+2).
    a=np.zeros((7,7),np.float32); receiver=(4,3); a[2,4]=np.float32(20)
    for name,grid,az,alt in [('oblique_phase_counterexample',a,30.,10.),('cardinal_tall_border',np.pad(np.array([[30]],np.float32),((0,6),(0,6))),0.,8.)]:
        samples=ray_samples(grid.shape,float(np.max(grid)),az,alt); e=exact_horizon(grid,samples); q=translated_scan(grid,samples); d=np.abs(e-q); where=np.argwhere(d>0); exact_mask=e>grid; actual_mask=actual_shadow_mask(kernel,grid,az,alt); assert np.array_equal(exact_mask,actual_mask)
        cases.append({'name':name,'shape':list(grid.shape),'azimuth':az,'altitude':alt,'samples':[(x,y,float(z)) for x,y,z in samples],'sample_increments':[(samples[i][0]-samples[i-1][0],samples[i][1]-samples[i-1][1]) for i in range(1,len(samples))],'equal':bool(np.array_equal(e,q)),'mismatch_count':int(len(where)),'shadow_mask_mismatch_count':int(np.count_nonzero(exact_mask!=(q>grid))),'actual_kernel_mask_equal':True,'max_abs':float(d.max()),'first_mismatch':where[0].tolist() if len(where) else None,'receiver':list(receiver) if name.startswith('oblique') else None,'receiver_exact_horizon':float(e[receiver]) if name.startswith('oblique') else None,'receiver_scan_horizon':float(q[receiver]) if name.startswith('oblique') else None})
    assert cases[0]['shadow_mask_mismatch_count'] > 0 and cases[0]['receiver_exact_horizon'] > cases[0]['receiver_scan_horizon']
    assert cases[1]['shadow_mask_mismatch_count'] == 0
    # Frozen deterministic small family, including borders and tall obstructions.
    rng=np.random.default_rng(20260919); grid=np.zeros((32,35),np.float32); grid[[0,-1],:]=25; grid[:,[0,-1]]=18; grid[5,7]=120; grid[19,23]=70; grid+=rng.integers(0,8,grid.shape).astype(np.float32)
    timings=[]
    for az in (0.,30.,45.,73.,90.,135.,225.,315.):
        s=ray_samples(grid.shape,float(grid.max()),az,12.); t=time.perf_counter(); e=exact_horizon(grid,s); te=time.perf_counter()-t; actual=actual_shadow_mask(kernel,grid,az,12.); assert np.array_equal(e>grid,actual); t=time.perf_counter(); q=translated_scan(grid,s); tq=time.perf_counter()-t; t=time.perf_counter(); checksum=0
        for _ in range(10000): checksum+=int(np.count_nonzero(e>grid))
        query=time.perf_counter()-t
        timings.append({'azimuth':az,'samples':len(s),'unique_increments':len(set((s[i][0]-s[i-1][0],s[i][1]-s[i-1][1]) for i in range(1,len(s)))),'equal':bool(np.array_equal(e,q)),'mismatch_count':int(np.count_nonzero(e!=q)),'shadow_mask_mismatch_count':int(np.count_nonzero(actual!=(q>grid))),'actual_kernel_mask_equal':True,'max_abs':float(np.max(np.abs(e-q))),'exact_construction_seconds':te,'scan_seconds':tq,'exact_storage_bytes':int(e.nbytes),'query_10000_seconds':query,'query_checksum':checksum})
    result={'decision':'reject_standard_single-translation_scan_for_general_oblique_rays','reason':'Rounded minor-axis offsets have phase-dependent increments; a horizon propagated from one fixed predecessor visits a different support than the frozen shifted-array traversal. Cardinal/diagonal cases showed sampled binary-mask agreement only; reusable float32 horizon equality remains unproved.','cases':cases,'family':timings,'reference_binding':{'kernel':str(kernel_path),'callable':'shadow_numpy (characterized serial recurrence)','mask_relation':'shadow == (sunlight_output == 0)','all_fixture_masks_equal':True},'operand_dtypes':{'DSM':'float32','python_azimuth_altitude':'converted to float32 before degree-to-radian multiply','dx_dy_initial':'float32','dz_assignment':'NumPy scalar from ds*index*tas; cast to float32 on DSM subtraction','rounding':'numpy.round ties-to-even'},'scope':'building-only diagnostic; vegetation, walls, and end-to-end speed are excluded'}
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--fixture',type=Path);a=p.parse_args();r=run();
    if a.fixture:
        g=np.zeros((7,7),np.float32);g[2,4]=np.float32(20);ss=ray_samples(g.shape,float(g.max()),30.,10.);np.savez_compressed(a.fixture,grid=g,samples=np.asarray(ss,dtype=np.float32),exact_horizon=exact_horizon(g,ss),translated_scan=translated_scan(g,ss),receiver=np.asarray([4,3],np.int64))
    a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({'decision':r['decision'],'family_equal':[x['equal'] for x in r['family']]}))
