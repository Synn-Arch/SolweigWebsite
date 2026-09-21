import os,sys,json,time,hashlib,platform,resource
from pathlib import Path
import numpy as np
from solweig_light.geometry.visibility import PackedVisibility, LazyDiffVisibility
from solweig_light.radiation import patch_radiation
root=Path.cwd(); fixture=root/'tests/reference/p4_benchmark_inputs'
manifest=json.loads((fixture/'manifest.json').read_text()); rows=[]
for name in ('Kside_veg_v2022a','Lcyl_v2022a'):
 case=next(c for c in manifest['cases'] if c['function']==name)
 path=fixture/case['input']; assert hashlib.sha256(path.read_bytes()).hexdigest()==case['input_sha256']
 with np.load(path) as a: values={k:a[k].copy() for k in a.files}
 for k,v in list(values.items()):
  spec=case['fields'][k]; typ=spec['original_python_type']
  if typ.startswith('builtins.'): values[k]=v.item()
  elif typ.startswith('numpy.') and spec['kind']=='scalar': values[k]=v[()]
 start=time.perf_counter()
 for k in ('shmat','vegshmat','vbshvegshmat'): values[k]=PackedVisibility.from_dense(values[k])
 if 'diffsh' in values: values['diffsh']=LazyDiffVisibility(values['shmat'],values['vegshmat'])
 encode=time.perf_counter()-start
 fn=getattr(patch_radiation,name); start=time.perf_counter(); fn(**values,parallel=False,block_pixels=128); first=time.perf_counter()-start
 trials=[]
 for i in range(5):
  start=time.perf_counter(); fn(**values,parallel=False,block_pixels=128); trials.append(time.perf_counter()-start)
 descriptors=[]
 for k in ('shmat','vegshmat','vbshvegshmat'):
  ch=values[k]; start=time.perf_counter(); patch_radiation._block(ch,0,128,ch.shape[2]); duration=time.perf_counter()-start
  desc=getattr(ch,'_block_descriptor',None)
  descriptors.append({'channel':k,'encoded_bytes':ch.nbytes,'patches':ch.shape[2],'descriptor_numpy_bytes':0 if desc is None else desc[1].nbytes,'borrowed_payload_bytes':0 if desc is None else sum(x.nbytes for x in desc[0]),'encoded_copy_bytes':0,'block_seconds':duration})
 rows.append({'function':name,'trials_seconds':trials,'first_call_seconds':first,'encoding_seconds':encode,'descriptors':descriptors,'fixture_sha256':case['input_sha256']})
source=Path(patch_radiation.__file__).resolve().parents[1]
record={'scope':'P4 components only; serial reduction, default block128, five warm calls; no end-to-end claim','rows':rows,'source_sha256':{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.py')},'invocation':sys.argv,'pythonpath':os.environ['PYTHONPATH'],'platform':platform.platform(),'maxrss':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'maxrss_units':'bytes on macOS','threads':{k:os.environ.get(k) for k in ('NUMBA_NUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')}}
Path(sys.argv[1]).write_text(json.dumps(record,indent=2)+'\n')
