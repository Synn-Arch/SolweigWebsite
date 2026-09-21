import json,sys,zipfile,tempfile
from pathlib import Path
import numpy as np
from osgeo import gdal
gdal.UseExceptions()
ref=Path(sys.argv[1]); cand=Path(sys.argv[2]); out=Path(sys.argv[3])
def schema(ds):
 return {'size':[ds.RasterXSize,ds.RasterYSize,ds.RasterCount],'projection':ds.GetProjection(),'geotransform':list(ds.GetGeoTransform()),'metadata':ds.GetMetadata(),'bands':[{'dtype':ds.GetRasterBand(i).DataType,'nodata':ds.GetRasterBand(i).GetNoDataValue(),'description':ds.GetRasterBand(i).GetDescription(),'metadata':ds.GetRasterBand(i).GetMetadata()} for i in range(1,ds.RasterCount+1)]}
def rule(name):
 if name=='TMRT':return ('max',.01,0)
 if name in ('UTCI','WBGT'):return ('max',.02,0)
 if name in ('Shadow','Ta','Wind'):return ('exact',0,0)
 return ('close',.05,1e-5)
def raster(a,b,name):
 da,db=gdal.Open(str(a)),gdal.Open(str(b)); se=schema(da)==schema(db); rec=[]; typ,atol,rtol=rule(name)
 for i in range(1,da.RasterCount+1):
  x=da.GetRasterBand(i).ReadAsArray();y=db.GetRasterBand(i).ReadAsArray(); masks=np.array_equal(np.isfinite(x),np.isfinite(y)) and np.array_equal(np.isnan(x),np.isnan(y)) and np.array_equal(np.isposinf(x),np.isposinf(y)) and np.array_equal(np.isneginf(x),np.isneginf(y)); finite=np.isfinite(x)&np.isfinite(y); d=np.abs(x.astype(np.float64)-y.astype(np.float64)); d[~finite]=0; j=np.unravel_index(np.argmax(d),d.shape); mx=float(d[j]); ok=masks and ((np.array_equal(x,y)) if typ=='exact' else (mx<=atol if typ=='max' else bool(np.allclose(x[finite],y[finite],atol=atol,rtol=rtol))))
  rec.append({'band':i,'passed':ok,'max_abs':mx,'worst_coordinate':list(map(int,j)),'masks_equal':masks})
 return {'path':a.name,'schema_equal':se,'passed':se and all(q['passed'] for q in rec),'bands':rec}
arts=[]
for p in sorted((ref/'output_folder/0_0').glob('*.tif')): arts.append(raster(p,cand/'output_folder/0_0'/p.name,p.name.split('_')[0]))
# geometry 15 zip TIFFs and 3 shadow matrices
geom=[]
with tempfile.TemporaryDirectory() as ta,tempfile.TemporaryDirectory() as tb:
 with zipfile.ZipFile(ref/'processed_inputs/SVF/svfs_0_0.zip') as z:z.extractall(ta)
 with zipfile.ZipFile(cand/'processed_inputs/SVF/svfs_0_0.zip') as z:z.extractall(tb)
 for p in sorted(Path(ta).glob('*.tif')):
  q=raster(p,Path(tb)/p.name,'SVF');
  # override SVF max gate1e-6
  for b in q['bands']: b['passed']=b['masks_equal'] and b['max_abs']<=1e-6
  q['passed']=q['schema_equal'] and all(b['passed'] for b in q['bands']);geom.append(q)
za=np.load(ref/'processed_inputs/SVF/shadowmats_0_0.npz');zb=np.load(cand/'processed_inputs/SVF/shadowmats_0_0.npz')
for k in za.files:
 x,y=za[k],zb[k];geom.append({'path':'shadowmats/'+k,'schema_equal':x.shape==y.shape and x.dtype==y.dtype,'passed':np.array_equal(x,y),'max_abs':float(np.max(np.abs(x.astype(float)-y.astype(float))))})
report={'passed':all(a['passed'] for a in arts) and all(a['passed'] for a in geom),'artifacts':arts,'geometry':geom,'summary':{'output_count':len(arts),'failed_outputs':[a['path'] for a in arts if not a['passed']],'geometry_count':len(geom),'failed_geometry':[a['path'] for a in geom if not a['passed']]}}
out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report['summary']|{'passed':report['passed']}))
