#!/usr/bin/env python3
"""Diagnostic conservative building-only hierarchy over exact SOLWEIG ray samples."""
from __future__ import annotations
import argparse,importlib.util,json,time,sys
from pathlib import Path
import numpy as np

def load_shadow():
 p=Path(__file__).resolve().parents[2]/'src/solweig_light/geometry/shadows.py';s=importlib.util.spec_from_file_location('verified_shadow',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m.shadow_numpy,p

def samples(shape,amax,azimuth,altitude,scale=1.):
 if azimuth==0:azimuth=1e-12
 az=np.float32(azimuth)*(np.pi/180);al=np.float32(altitude)*(np.pi/180);sin,cos,tan=np.sin(az),np.cos(az),np.tan(az);ss,sc=np.sign(sin),np.sign(cos);dss,dsc=abs(1/sin),abs(1/cos);tas=np.tan(al)/scale;sx,sy=shape;out=[];dx=dy=dz=np.float32(0);i=1
 while amax>=dz and abs(dx)<sx and abs(dy)<sy:
  if np.pi/4<=az<3*np.pi/4 or 5*np.pi/4<=az<7*np.pi/4:dy=ss*i;dx=-sc*abs(np.round(i/tan));ds=dss
  else:dy=ss*abs(np.round(i*tan));dx=-sc*i;ds=dsc
  dz=ds*i*tas;out.append((int(dx),int(dy),np.float32(dz)));i+=1
 return out

class Quad:
 def __init__(self,a,r0=0,r1=None,c0=0,c1=None,nodes=None):
  r1=a.shape[0] if r1 is None else r1;c1=a.shape[1] if c1 is None else c1;self.box=(r0,r1,c0,c1);self.maximum=np.max(a[r0:r1,c0:c1]);self.children=[]
  if nodes is not None:nodes.append(self)
  if (r1-r0)>1 or (c1-c0)>1:
   rm=(r0+r1)//2;cm=(c0+c1)//2
   for x0,x1 in ((r0,rm),(rm,r1)):
    for y0,y1 in ((c0,cm),(cm,c1)):
     if x0<x1 and y0<y1:self.children.append(Quad(a,x0,x1,y0,y1,nodes))
 def rectmax(self,q,stats):
  stats[0]+=1;r0,r1,c0,c1=self.box;u0,u1,v0,v1=q
  if u1<=r0 or r1<=u0 or v1<=c0 or c1<=v0:return np.float32(-np.inf)
  if u0<=r0 and r1<=u1 and v0<=c0 and c1<=v1:return self.maximum
  return max((ch.rectmax(q,stats) for ch in self.children),default=self.maximum)

def baseline(a,ray):
 h=a.copy();visits=0;sx,sy=a.shape
 for r in range(sx):
  for c in range(sy):
   for dx,dy,dz in ray:
    rr,cc=r+dx,c+dy
    if 0<=rr<sx and 0<=cc<sy:h[r,c]=max(h[r,c],np.float32(a[rr,cc]-dz));visits+=1
 return h,visits

def hierarchical(a,ray,tree,block=4):
 h=a.copy();visited=skipped=bounds=0;node=[0];sx,sy=a.shape
 for r in range(sx):
  for c in range(sy):
   for lo in range(0,len(ray),block):
    seg=ray[lo:lo+block];valid=[(r+dx,c+dy,dz) for dx,dy,dz in seg if 0<=r+dx<sx and 0<=c+dy<sy]
    if not valid:continue
    r0=min(x[0] for x in valid);r1=max(x[0] for x in valid)+1;c0=min(x[1] for x in valid);c1=max(x[1] for x in valid)+1;mh=tree.rectmax((r0,r1,c0,c1),node);bounds+=1
    # monotone float32 upper bound: every exact source <= mh and every dz >= first valid dz.
    ub=np.float32(mh-np.float32(valid[0][2]))
    if ub<=h[r,c]:skipped+=len(valid);continue
    for rr,cc,dz in valid:h[r,c]=max(h[r,c],np.float32(a[rr,cc]-dz));visited+=1
 return h,{'visited_exact_samples':visited,'skipped_exact_samples':skipped,'bound_queries':bounds,'hierarchy_node_visits':node[0]}

def fixture(kind,seed=20260919):
 rng=np.random.default_rng(seed);a=np.zeros((32,35),np.float32)
 if kind=='sparse':a[5,7]=120;a[19,23]=70
 elif kind=='dense':a[:]=rng.integers(5,65,a.shape);a[5,7]=120
 elif kind=='border_tall':a[[0,-1],:]=80;a[:,[0,-1]]=55;a[16,17]=140
 return a

def run():
 shadow,path=load_shadow();records=[]
 for kind in ('sparse','dense','border_tall'):
  a=fixture(kind);nodes=[];t=time.perf_counter();tree=Quad(a,nodes=nodes);build=time.perf_counter()-t
  for az in (0.,30.,73.,90.,225.):
   ray=samples(a.shape,float(a.max()),az,12.);t=time.perf_counter();base,n=baseline(a,ray);bt=time.perf_counter()-t;t=time.perf_counter();got,stats=hierarchical(a,ray,tree);ht=time.perf_counter()-t
   zeros=np.zeros_like(a);sun,*_=shadow(float(a.max()),a,zeros,zeros,zeros,az,12.,1.);actual=sun==0
   assert np.array_equal(base,got);assert np.array_equal(base>a,actual)
   records.append({'fixture':kind,'construction_seconds':build,'node_count':len(nodes),'raw_maxima_bytes':len(nodes)*4,'python_node_storage_lower_bound_bytes':sum(sys.getsizeof(n)+sys.getsizeof(n.children)+sys.getsizeof(n.maximum) for n in nodes),'azimuth':az,'samples_per_full_ray':len(ray),'baseline_exact_samples':n,**stats,'skipped_fraction':stats['skipped_exact_samples']/n if n else 0.,'bit_exact_horizon':True,'actual_kernel_mask_equal':True,'baseline_seconds':bt,'hierarchical_seconds':ht})
 return {'decision':'reject_current_python_quadtree_for_promotion_despite_safe_bound','bound':'For each consecutive exact-sample segment, rectangle_max(source_height) - float32(first_valid_dz) upper-bounds every float32 source_height-dz candidate; skip only when <= current horizon.','block_samples':4,'hierarchy':{'kind':'quadtree rectangle maximum','construction_seconds_by_fixture':'recorded per fixture below','storage_model':'one float32 maximum plus Python node metadata per node'},'records':records,'fixtures':{k:{'shape':list(fixture(k).shape),'nonzero':int(np.count_nonzero(fixture(k)))} for k in ('sparse','dense','border_tall')},'reference_binding':{'kernel':str(path),'callable':'shadow_numpy','all_masks_equal':True},'scope':'building-only; no vegetation/wall extension: their canopy/trunk intervals, ordering, and wall outputs require separate simultaneous bounds. Diagnostic Python costs are not end-to-end speed evidence.','node_counts':{k:(lambda ns:(Quad(fixture(k),nodes=ns),len(ns)))([])[1] for k in ('sparse','dense','border_tall')}}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run();a.output.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print(json.dumps({'decision':r['decision'],'min_skip':min(x['skipped_fraction'] for x in r['records']),'max_skip':max(x['skipped_fraction'] for x in r['records'])}))
