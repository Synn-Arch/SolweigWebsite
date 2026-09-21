import sys,numpy as np
from pathlib import Path
from solweig_light.radiation import math_profile,engine
B=Path(sys.argv[1]);z=np.load(B/'corpus.npz');bits=z['input_bits'];idx=np.searchsorted(bits,z['boundary_bits']);av=math_profile.sqrt_acos(bits.view(np.float32)[idx])[:,None];pa=z['patch_altitude'];pz=z['patch_azimuth'];sa=np.asarray(z['solar_altitude']);sz=float(z['solar_azimuth']);xis=[];yis=[];hs=[];tds=[];ats=[]
for p in range(153):
 d=np.abs(engine._operate(np.subtract,sz,pz[p]));deg=engine._divide(np.pi,180.);xi=math_profile.cos(engine._operate(np.multiply,d,deg));yi=engine._operate(np.multiply,engine._operate(np.multiply,2,xi),np.tan(engine._operate(np.multiply,sa,deg)));h=math_profile.tan(av);td=engine._operate(np.add,h,np.where(yi>0,0.,yi));at=math_profile.atan(td);xis.append(xi);yis.append(yi);hs.append(h[:,0]);tds.append(td[:,0]);ats.append(at[:,0])
np.savez_compressed(B/'candidate_actual_stages.npz',xi=np.asarray(xis),yi=np.asarray(yis),h=np.stack(hs,1),td=np.stack(tds,1),at=np.stack(ats,1))
