import sys,numpy as np,torch
from pathlib import Path
B=Path(sys.argv[1]);z=np.load(B/'corpus.npz');bits=z['input_bits'];idx=np.searchsorted(bits,z['boundary_bits']);x=torch.from_numpy(bits.view(np.float32)[idx].copy());av=torch.acos(torch.sqrt(x))[:,None];pa=torch.from_numpy(z['patch_altitude']);pz=torch.from_numpy(z['patch_azimuth']);sa=torch.tensor(z['solar_altitude']);sz=torch.tensor(z['solar_azimuth']); xis=[];yis=[];hs=[];tds=[];ats=[]
for p in range(153):
 d=torch.abs(sz-pz[p]);xi=torch.cos(d*(torch.pi/180));yi=2*xi*torch.tan(sa*(torch.pi/180));h=torch.tan(av);td=h+torch.where(yi>0,0.,yi);at=torch.atan(td);xis.append(xi.numpy());yis.append(yi.numpy());hs.append(h.numpy()[:,0]);tds.append(td.numpy()[:,0]);ats.append(at.numpy()[:,0])
np.savez_compressed(B/'torch_actual_stages.npz',xi=np.asarray(xis),yi=np.asarray(yis),h=np.stack(hs,1),td=np.stack(tds,1),at=np.stack(ats,1))
