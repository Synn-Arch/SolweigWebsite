import json,sys
from pathlib import Path
import numpy as np,torch
B=Path(sys.argv[1]); z=np.load(B/'corpus.npz'); bits=z['input_bits']; idx=np.searchsorted(bits,z['boundary_bits']); x=torch.from_numpy(bits.view(np.float32)[idx].copy()); asvf=torch.acos(torch.sqrt(x))[:,None]; pa=torch.from_numpy(z['patch_altitude'].copy()); pazi=torch.from_numpy(z['patch_azimuth'].copy()); sa=torch.tensor(z['solar_altitude']); saz=torch.tensor(z['solar_azimuth']); sun=np.zeros((len(idx),153),bool); shade=np.zeros_like(sun)
for p in range(153):
 d=torch.abs(saz-pazi[p]); deg2=torch.pi/180.; rad2=180./torch.pi; xi=torch.cos(d*deg2); yi=2*xi*torch.tan(sa*deg2); yi_=torch.where(yi>0,0.,yi); deg=torch.atan(torch.tan(asvf)+yi_)*rad2; sun[:,p]=(deg<pa[p]).numpy()[:,0]; shade[:,p]=(deg>pa[p]).numpy()[:,0]
np.savez_compressed(B/'torch_actual_classifier.npz',sun_pack=np.packbits(sun,axis=1),shade_pack=np.packbits(shade,axis=1)); print(json.dumps({'sun':int(sun.sum()),'shade':int(shade.sum())}))
