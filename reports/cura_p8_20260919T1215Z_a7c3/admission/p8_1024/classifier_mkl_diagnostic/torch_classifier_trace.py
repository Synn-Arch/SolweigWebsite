import json,sys
from pathlib import Path
import numpy as np,torch
B=Path(sys.argv[1]); z=np.load(B/'corpus.npz'); allbits=z['input_bits']; q=z['boundary_bits']; idx=np.searchsorted(allbits,q); x=torch.from_numpy(allbits.view(np.float32)[idx].copy())
# Exact ASVF oracle is Torch/MKL unary result.
with torch.no_grad(): asvf=torch.acos(torch.sqrt(x))
pa=torch.from_numpy(z['patch_altitude'].copy()); pazi=torch.from_numpy(z['patch_azimuth'].copy()); sa=torch.tensor(z['solar_altitude']); saz=torch.tensor(z['solar_azimuth'])
d=torch.abs(saz-pazi); deg2rad=torch.pi/180.0; rad2deg=180.0/torch.pi; xi=torch.cos(d*deg2rad); solar_rad=sa*deg2rad; solar_tan=torch.tan(solar_rad); yi=2*xi*solar_tan; yi_=torch.where(yi>0,0.0,yi); hsvf=torch.tan(asvf[:,None]); td=hsvf+yi_; atan=torch.atan(td); sd=atan*rad2deg; sun=sd<pa; shade=sd>pa
arrays={'input_bits':allbits[idx],'asvf_bits':asvf.numpy().view(np.uint32),'patch_delta_bits':d.numpy().view(np.uint32),'xi_bits':xi.numpy().view(np.uint32),'solar_rad_bits':solar_rad.numpy().view(np.uint64 if solar_rad.dtype==torch.float64 else np.uint32),'solar_tan_bits':solar_tan.numpy().view(np.uint64 if solar_tan.dtype==torch.float64 else np.uint32),'yi_bits':yi.numpy().view(np.uint32),'yi_clamped_bits':yi_.numpy().view(np.uint32),'hsvf_bits':hsvf.numpy().view(np.uint32),'tan_delta_bits':td.numpy().view(np.uint32),'atan_bits':atan.numpy().view(np.uint32),'sun_pack':np.packbits(sun.numpy(),axis=1),'shade_pack':np.packbits(shade.numpy(),axis=1)}
np.savez_compressed(B/'torch_classifier_trace.npz',**arrays)
meta={'torch':torch.__version__,'git':torch.version.git_version,'dtypes':{k:str(v.dtype) for k,v in {'input':x,'asvf':asvf,'patch_altitude':pa,'patch_azimuth':pazi,'solar_altitude':sa,'solar_azimuth':saz,'patch_delta':d,'xi':xi,'solar_rad':solar_rad,'solar_tan':solar_tan,'yi':yi,'yi_clamped':yi_,'hsvf':hsvf,'tan_delta':td,'atan':atan,'sunlit_degrees':sd}.items()},'shapes':{k:list(v.shape) for k,v in {'asvf':asvf,'xi':xi,'yi':yi,'hsvf':hsvf,'tan_delta':td,'atan':atan}.items()},'scalar_values':{'solar_altitude':float(sa),'solar_azimuth':float(saz),'deg2rad':deg2rad,'rad2deg':rad2deg,'solar_rad':float(solar_rad),'solar_tan':float(solar_tan)}}
(B/'torch_classifier_trace_meta.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n'); print(json.dumps(meta,indent=2))
