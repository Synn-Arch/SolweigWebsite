"""Read-only annulus and boundary reconstruction; never candidate goldens."""
import numpy as np,ctypes,json
from pathlib import Path
from solweig_light.geometry.shadows import create_patches
from solweig_light.geometry.svf import annulus_weight

r=Path(__file__).resolve().parents[1]/'reports/characterization/p7_real_diagnosis'
weights=np.load(r/'svf_weights/original_annulus_weights.npz');gold={(int(a),float(b)):w for a,b,w in zip(weights['altitude'],weights['interval'],weights['weight'])}
_,_,ann,_,counts,_,_=create_patches(2);f=np.float32
lib=ctypes.CDLL(None);sinf=lib.sinf;sinf.argtypes=[ctypes.c_float];sinf.restype=ctypes.c_float
records=[]
def alternate(alt,interval,kind):
 n=f(90);an=f(91)-f(alt);step=(np.reciprocal(f(interval))*f(360))*f(np.pi/180);a=np.reciprocal(f(2)*n)*f(np.pi);b=(f(np.pi)*(f(2)*an-f(1)))/(f(2)*n)
 sin=(lambda x:np.sin(x)) if kind=='reciprocal_numpy_sin' else (lambda x:f(sinf(float(x))))
 return step*(f(1/(2*np.pi))*sin(a)*sin(b))
for scene,pixel in (('sparse',(250,40)),('vegetation_rich',(118,62))):
 scene_path=r.parents[1]/f'runs/p7_real_{scene}_original_cpu/scene/processed_inputs/SVF'
 with np.load(scene_path/'shadowmats_0_0.npz') as archive:masks=[archive[key][pixel].copy() for key in ('shadowmat','vegshadowmat')]
 for variant in ('current','reciprocal_numpy_sin','reciprocal_libc_sin','oracle_weights'):
  total=np.zeros(2,dtype=f);index=0
  for band,count in enumerate(counts):
   for patch in range(count):
    for alt in range(ann[band]+1,ann[band+1]+1):
     weight=annulus_weight(alt,count) if variant=='current' else gold[(alt,float(count))] if variant=='oracle_weights' else alternate(alt,count,variant)
     for c in range(2):total[c]=f(total[c]+f(weight*masks[c][index]))
    index+=1
  total=np.minimum(total,f(1));tmp=np.maximum(f(total[0]+total[1]-f(1)),f(0));alpha=np.arcsin(np.exp(np.log(f(1)-tmp)/f(2)));records.append(dict(scene=scene,variant=variant,svf=float(total[0]),svfveg=float(total[1]),tmp=float(tmp),svfalfa=float(alpha)))
print(json.dumps(records,indent=2));(r/'svf_weights/boundary_pixel_reconstruction.json').write_text(json.dumps(records,indent=2))
