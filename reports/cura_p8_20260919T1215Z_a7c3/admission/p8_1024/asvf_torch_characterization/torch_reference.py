import json,platform,sys,warnings
from pathlib import Path
import numpy as np, torch
root=Path(sys.argv[1]); z=np.load(root/'corpus.npz'); bits=z['input_bits']; x=torch.from_numpy(bits.view(np.float32).copy())
with torch.no_grad(): y=torch.acos(torch.sqrt(x))
ybits=y.numpy().view(np.uint32).copy()
pa=torch.from_numpy(z['patch_altitude'].copy()); pazi=torch.from_numpy(z['patch_azimuth'].copy())
sa=torch.tensor(z['solar_altitude']); saz=torch.tensor(z['solar_azimuth'])
packed_s=[]; packed_h=[]
for lo in range(0,len(bits),4096):
 a=y[lo:lo+4096,None]
 d=torch.abs(saz-pazi); xi=torch.cos(d*(torch.pi/180.0)); yi=2*xi*torch.tan(sa*(torch.pi/180.0)); yi_=torch.where(yi>0,0.0,yi)
 deg=torch.atan(torch.tan(a)+yi_)*(180.0/torch.pi)
 packed_s.append(np.packbits((deg<pa).numpy(),axis=1)); packed_h.append(np.packbits((deg>pa).numpy(),axis=1))
np.savez_compressed(root/'torch_reference.npz',result_bits=ybits,sun_pack=np.concatenate(packed_s),shade_pack=np.concatenate(packed_h))
meta={'torch':torch.__version__,'torch_config':torch.__config__.show(),'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),'torch_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'input_dtype':str(x.dtype),'result_dtype':str(y.dtype),'classifier_dtypes':{'patch':str(pa.dtype),'solar_altitude':str(sa.dtype),'solar_azimuth':str(saz.dtype)},'count':len(bits),'invalid_result_count':int((~torch.isfinite(y)).sum())}
(root/'torch_environment.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
