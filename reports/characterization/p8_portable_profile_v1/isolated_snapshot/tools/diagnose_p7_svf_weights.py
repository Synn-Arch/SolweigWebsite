"""Original-only scalar annulus oracle, not performance evidence."""
import argparse,hashlib,importlib.metadata,json,sys,subprocess
from pathlib import Path
import numpy as np
import torch
import solweig_gpu
from solweig_gpu.shadow import annulus_weight,create_patches

p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args();a.destination.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
_,_,ann,_,counts,_,_=create_patches(2)
records=[]
for band in range(len(counts)):
 for altitude in range(ann[band]+1,ann[band+1]+1):
  for interval in (counts[band],torch.ceil(counts[band]/2.0)):
   records.append((int(altitude),float(interval),float(annulus_weight(altitude,interval,torch.device('cpu')))))
path=a.destination/'original_annulus_weights.npz';np.savez_compressed(path,altitude=np.array([r[0] for r in records],dtype=np.int64),interval=np.array([r[1] for r in records],dtype=np.float32),weight=np.array([r[2] for r in records],dtype=np.float32))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=Path(solweig_gpu.__file__).parent
root=Path(__file__).resolve().parents[1];upstream=root/'.upstream/SOLWEIG-GPU'
assert subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()=='0d7fe742abeeddd890dd58fc76ed7f78bd47faec'
assert not subprocess.check_output(['git','-C',str(upstream),'diff','--name-only'],text=True).strip()
assert all(sha(p)==sha(source/p.name) for p in (upstream/'solweig_gpu').iterdir() if p.suffix in ('.py','.txt'))
manifest=dict(evidence_class='original_upstream_cpu',source_commit='0d7fe742abeeddd890dd58fc76ed7f78bd47faec',original_source_hashes={p.name:sha(p) for p in source.iterdir() if p.suffix in ('.py','.txt')},patch=None,package=str(source),python=sys.version,executable=sys.executable,invocation=sys.argv,torch_threads=1,artifact_sha256=sha(path),harness_sha256=sha(__file__),packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()})
edge_inputs=np.array([(1,0),(1,-0.0),(1,np.nan),(1,np.inf),(np.nan,31),(np.inf,31),(-np.inf,31)],dtype=np.float32)
edge_output=np.array([annulus_weight(torch.tensor(alt),torch.tensor(interval),torch.device('cpu')).item() for alt,interval in edge_inputs],dtype=np.float32)
edge_path=a.destination/'original_annulus_edges.npz';np.savez_compressed(edge_path,inputs=edge_inputs,outputs=edge_output)
manifest['edge_artifact']=dict(path=edge_path.name,sha256=sha(edge_path))
manifest['boundary_cases']=[]
for scene,label,pixel in (('sparse','sparse',(250,40)),('vegetation_rich','vegetation',(118,62))):
 archive_path=root/f'reports/runs/p7_real_{scene}_original_cpu/scene/processed_inputs/SVF/shadowmats_0_0.npz'
 with np.load(archive_path) as archive:mask=np.stack([archive[k][pixel].copy() for k in ('shadowmat','vegshadowmat')])
 main_path=sorted((root/f'reports/characterization/p7_real_diagnosis/{label}_original_14').glob('*Solweig*locals.npz'))[0]
 with np.load(main_path) as main:golden={k:np.asarray(main[k][pixel]).copy() for k in ('svf','svfveg','svfalfa','F_sh','Kdown')}
 packet=a.destination/(scene+'_boundary.npz');np.savez_compressed(packet,masks=mask,annulus=ann.cpu().numpy(),counts=counts.cpu().numpy(),**golden)
 manifest['boundary_cases'].append(dict(scene=scene,pixel=list(pixel),path=packet.name,sha256=sha(packet),original_visibility_sha256=sha(archive_path),original_main_boundary_sha256=sha(main_path),scope='Two original visibility receiver sequences and original full-scene scalar fields; not a reduced full simulation'))
(a.destination/'manifest.json').write_text(json.dumps(manifest,indent=2))
