from pathlib import Path
import json, hashlib, sys, zipfile
import numpy as np
from osgeo import gdal
patched, cpu, candidate, out = map(Path, sys.argv[1:])

def read(p):
 d=gdal.Open(str(p)); assert d, p
 a=d.ReadAsArray(); meta={'shape':list(a.shape),'projection':d.GetProjection(),'geotransform':list(d.GetGeoTransform()),'bands':d.RasterCount,'nodata':[d.GetRasterBand(i+1).GetNoDataValue() for i in range(d.RasterCount)]}; return a,meta
rules={'TMRT':('max',.01,0),'UTCI':('max',.02,0),'WBGT':('max',.02,0),'Ta':('exact',0,0),'Wind':('exact',0,0),'Shadow':('exact',0,0),'Kdown':('close',.05,1e-5),'Kup':('close',.05,1e-5),'Ldown':('close',.05,1e-5),'Lup':('close',.05,1e-5)}
res={'schema':'patched-upstream-cuda-workerlimit-admission.v1','comparisons':{},'geometry':{},'status':'passed'}
for label,ref in [('original_cpu',cpu),('candidate_cpu',candidate)]:
 rows=[]
 for name,(kind,atol,rtol) in rules.items():
  p=patched/'output_folder/0_0'/f'{name}_0_0.tif'; q=ref/'output_folder/0_0'/f'{name}_0_0.tif'; a,ma=read(p); b,mb=read(q)
  finite=np.isfinite(a)==np.isfinite(b); masks=bool(np.all(finite) and np.array_equal(np.isnan(a),np.isnan(b)) and np.array_equal(np.isposinf(a),np.isposinf(b)) and np.array_equal(np.isneginf(a),np.isneginf(b)))
  delta=np.abs(a.astype(np.float64)-b.astype(np.float64)); mx=float(np.nanmax(delta)) if delta.size else 0
  ok=ma==mb and masks and (np.array_equal(a,b) if kind=='exact' else (mx<=atol if kind=='max' else bool(np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True))))
  idx=list(np.unravel_index(np.nanargmax(delta),delta.shape)) if delta.size else []
  rows.append({'field':name,'passed':ok,'max_abs':mx,'worst_index':idx,'metadata_exact':ma==mb,'mask_exact':masks,'rule':{'kind':kind,'atol':atol,'rtol':rtol}})
  if not ok: res['status']='failed'
 res['comparisons'][label]=rows
# exact geometry against refs: three NPZ arrays + 15 zipped tiffs
for label,ref in [('original_cpu',cpu),('candidate_cpu',candidate)]:
 items=[]
 pa=patched/'processed_inputs/SVF/shadowmats_0_0.npz'; qa=ref/'processed_inputs/SVF/shadowmats_0_0.npz'
 with np.load(pa) as x,np.load(qa) as y:
  for k in sorted(x.files):
   ok=k in y.files and np.array_equal(x[k],y[k]); items.append({'field':k,'kind':'shadow_npz','passed':ok}); res['status']='failed' if not ok else res['status']
 pz=patched/'processed_inputs/SVF/svfs_0_0.zip'; qz=ref/'processed_inputs/SVF/svfs_0_0.zip'
 with zipfile.ZipFile(pz) as zp,zipfile.ZipFile(qz) as zq:
  names=sorted(n for n in zp.namelist() if n.endswith('.tif'))
  for n in names:
   a,_=read('/vsizip/'+str(pz)+'/'+n); b,_=read('/vsizip/'+str(qz)+'/'+n); mx=float(np.nanmax(np.abs(a.astype(float)-b.astype(float)))); ok=mx<=1e-6 and np.array_equal(np.isnan(a),np.isnan(b)); items.append({'field':n,'kind':'svf_zip','passed':ok,'max_abs':mx,'gate':1e-6}); res['status']='failed' if not ok else res['status']
 res['geometry'][label]=items
for base in [patched]:
 inv={}
 for p in sorted(base.rglob('*')):
  if p.is_file(): inv[str(p.relative_to(base))]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 res['patched_run_inventory']=inv
out.write_text(json.dumps(res,indent=2,sort_keys=True)+'\n')
print(res['status']); print({k:sum(not x['passed'] for x in v) for k,v in res['comparisons'].items()}); print({k:sum(not x['passed'] for x in v) for k,v in res['geometry'].items()})
raise SystemExit(0 if res['status']=='passed' else 2)
