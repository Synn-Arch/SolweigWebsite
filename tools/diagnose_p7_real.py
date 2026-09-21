"""Read-only first-daytime boundary diagnostic; not timing evidence."""
import argparse, inspect, json, sys
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--step',type=int);p.add_argument('backend');p.add_argument('scene',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();a.scene=a.scene.resolve();a.destination.mkdir(parents=True,exist_ok=True)
if a.backend in ('original','freeze'):
 import torch
 torch.set_num_threads(1)
 from solweig_gpu import solweig as engine
 from solweig_gpu.utci_process import compute_utci
else:
 import solweig_light
 from solweig_light.radiation import engine
if a.backend=='freeze':
 import hashlib,importlib.metadata,shutil,subprocess,platform
 root=Path(__file__).resolve().parents[1];source=root/'.upstream/SOLWEIG-GPU';package=Path(engine.__file__).parent
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
 assert commit=='0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
 assert not subprocess.check_output(['git','-C',str(source),'diff','--name-only'],text=True).strip()
 source_hashes={p.name:sha(p) for p in (source/'solweig_gpu').iterdir() if p.suffix in ('.py','.txt')}
 assert all(sha(package/name)==digest for name,digest in source_hashes.items())
 captures=root/'reports/characterization/p7_real_diagnosis/component_original';cases=[]
 for entry in sorted(captures.glob('*sunonsurface*input.npz')):
  with np.load(entry) as packet: angle=int(packet['azimuthA'])
  if angle not in (65,145,245,325):continue
  count=int(entry.name.split('_')[0]);output=captures/f'{count+1:04d}_8_sunonsurface_2018a_output.npz'
  case=dict(angle=angle)
  for kind,path in (('inputs',entry),('outputs',output)):
   name=f'sun_{angle}_{kind}.npz';shutil.copyfile(path,a.destination/name);case[kind]=dict(path=name,sha256=sha(a.destination/name))
  local_path=captures/f'{count+2:04d}_8_sunonsurface_2018a_locals.npz'
  with np.load(local_path) as locals_packet:np.savez_compressed(a.destination/f'sun_{angle}_mask.npz',facesh=locals_packet['facesh'])
  case['mask']=dict(path=f'sun_{angle}_mask.npz',sha256=sha(a.destination/f'sun_{angle}_mask.npz'))
  cases.append(case)
  with np.load(entry) as packet:
   widened={k:(v.astype(np.float64) if v.ndim>0 and v.dtype==np.float32 else v.copy()) for k,v in packet.items()}
  kwargs={k:torch.from_numpy(v.copy()) for k,v in widened.items()}
  captured={}
  def mask_profile(frame,event,arg):
   if frame.f_code==engine.sunonsurface_2018a.__code__ and event=='return' and arg is not None:captured['facesh']=frame.f_locals['facesh'].detach().cpu().numpy().copy()
  assert sys.getprofile() is None
  try:
   sys.setprofile(mask_profile);result=engine.sunonsurface_2018a(**kwargs)
  finally:sys.setprofile(None)
  np.savez_compressed(a.destination/f'sun_{angle}_float64_mask.npz',**captured)
  float_case=dict(angle=angle,profile='float64_rasters_float32_angles')
  float_case['mask']=dict(path=f'sun_{angle}_float64_mask.npz',sha256=sha(a.destination/f'sun_{angle}_float64_mask.npz'))
  for kind,values in (('inputs',widened),('outputs',{'return/'+str(i):v.detach().cpu().numpy().copy() for i,v in enumerate(result)})):
   name=f'sun_{angle}_float64_{kind}.npz';np.savez_compressed(a.destination/name,**values);float_case[kind]=dict(path=name,sha256=sha(a.destination/name))
  cases.append(float_case)
 manifest=dict(evidence_class='original_upstream_cpu',source_commit=commit,patch=None,candidate_import=False,fixture_hashes={str(p.relative_to(a.scene)):sha(p) for p in a.scene.rglob('*') if p.is_file() and p.suffix in ('.tif','.txt') and 'output_folder' not in p.parts},source_hashes=source_hashes,package=str(package),environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),torch_threads=torch.get_num_threads(),packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}),invocation=sys.argv,harness_sha256=sha(__file__),origin='Captured original real dense urban first-daytime GVF component; reproduced all 17 original model-return channels bitwise',cases=cases)
 (a.destination/'manifest.json').write_text(json.dumps(manifest,indent=2));sys.exit(0)
names=('Solweig_2022a_calc','gvf_2018a','TsWaveDelay_2015a','Kup_veg_2015a','shadowingfunction_wallheight_13','shadowingfunction_wallheight_23','cylindric_wedge','Perez_v3','diffusefraction')
names=names+('sunonsurface_2018a',)
codes={getattr(engine,n).__code__:n for n in names}
if a.backend!='original':
 from solweig_light.radiation.ground_view import _sun,_gvf
 codes[_sun.__code__]='sunonsurface_2018a';codes[_gvf.__code__]='gvf_internal'
events=[];step=-1;day=False
class Done(Exception):pass
def encode(fields):
 out={}
 def add(k,v):
  if hasattr(v,'detach'):v=v.detach().cpu().numpy()
  if isinstance(v,dict):
   for child,x in v.items():add(k+'/'+child,x)
  elif isinstance(v,(tuple,list)):
   for i,x in enumerate(v):add(k+'/'+str(i),x)
  elif isinstance(v,(np.ndarray,np.generic,int,float,bool)):
   x=np.asarray(v)
   if not x.dtype.hasobject:out[k]=x.copy()
 for k,v in fields.items():add(k,v)
 return out
def save(name,boundary,fields):
 file=f'{len(events):04d}_{step}_{name}_{boundary}.npz';np.savez_compressed(a.destination/file,**encode(fields));events.append(dict(name=name,boundary=boundary,step=step,path=file))
def profile(f,event,arg):
 global step,day
 name=codes.get(f.f_code)
 if name is None:return
 if name=='Solweig_2022a_calc' and event=='call':
  step=int(f.f_locals['i']);day=float(f.f_locals['altitude'])>0 and (a.step is None or step==a.step);save(name,'input',f.f_locals)
 elif day and event=='call':save(name,'input',f.f_locals)
 elif day and event=='return' and arg is not None:
  save(name,'output',{'return':arg});save(name,'locals',f.f_locals)
  if name=='Solweig_2022a_calc':raise Done()
try:
 sys.setprofile(profile)
 if a.destination.name.startswith('component'):
  step=8;day=True
  packet=np.load(Path(__file__).resolve().parents[1]/'reports/characterization/p7_real_diagnosis/original/0012_8_gvf_2018a_input.npz')
  kwargs={k:(torch.from_numpy(v.copy()) if a.backend=='original' else v.copy()) for k in packet.files for v in [packet[k]]}
  engine.gvf_2018a(**kwargs)
  raise Done()
 pre=a.scene/'processed_inputs'
 if a.backend=='original':
  out=a.destination/'output';out.mkdir(exist_ok=True)
  paths=[str(pre/folder/(folder+'_0_0.tif')) for folder in ('Building_DSM','Trees','DEM','walls','aspect')]
  compute_utci(*paths,None,None,np.loadtxt(pre/'metfiles/metfile_0_0.txt',skiprows=1),str(out),'0_0','2020-07-18')
 else:
  with solweig_light.runtime_options(solweig_light.RuntimeOptions(cpu_budget=1,workers=1,threads_per_worker=1,cache_enabled=False,legacy_cache_policy="trust")):
   from solweig_light.pipeline import run_tile,files_by_key
   paths={name:files_by_key(pre/name)['0_0'] for name in ('Building_DSM','Trees','DEM','metfiles','walls','aspect')}
   flags={name:False for name in ('save_tmrt','save_svf','save_kup','save_kdown','save_lup','save_ldown','save_shadow','save_wbgt','save_ta','save_wind')}
   run_tile(str(a.destination),str(pre),'2020-07-18','0_0',paths,flags)
except Done:pass
finally:
 sys.setprofile(None);(a.destination/'manifest.json').write_text(json.dumps(dict(backend=a.backend,package=str(engine.__file__),events=events),indent=2))
