import hashlib,importlib.util,json,os,sys
from pathlib import Path
import numpy as np
from types import SimpleNamespace
from solweig_light.radiation import math_profile,engine,patch_radiation
B=Path(sys.argv[1]); z=np.load(B/'corpus.npz'); ref=np.load(B/'torch_reference.npz'); bits=z['input_bits']; x=bits.view(np.float32)
a=math_profile.sqrt_acos(x); expr=a.view(np.uint32)!=ref['result_bits']
q=z['boundary_bits']; idx=np.searchsorted(bits,q); av=a[idx,None]; pa=z['patch_altitude']; pazi=z['patch_azimuth']; sa=np.asarray(z['solar_altitude']); saz=np.asarray(z['solar_azimuth']); geom=SimpleNamespace(altitude=pa,azimuth=pazi)
# Actual serial engine patch calls.
sun=np.zeros((len(idx),153),bool); shade=np.zeros_like(sun)
for p in range(153): aa,bb=engine.shaded_or_sunlit(sa,saz,pa[p],pazi[p],av); sun[:,p],shade[:,p]=aa[:,0],bb[:,0]
# Actual prepared/compiled path.
prep=patch_radiation._class_coefficients(sa,saz,geom,av); cs,ch=patch_radiation._classes(sa,saz,geom,av,0,len(idx),prepared=prep)
def cmp(a,b):
 d=np.bitwise_xor(np.packbits(a,axis=1),b); return {'cells':int(np.unpackbits(d,axis=1,count=153).sum()),'inputs':int(np.any(d,axis=1).sum())}
report={'thread_env':{k:os.environ.get(k) for k in ['MKL_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS']},'asvf':{'all_bits':int(expr.sum()),'finite_bits':int((expr&np.isfinite(a)&np.isfinite(ref['result_bits'].view(np.float32))).sum()),'nan_masks_equal':bool(np.array_equal(np.isnan(a),np.isnan(ref['result_bits'].view(np.float32))))},'serial':{'sun':cmp(sun,ref['sun_pack'][idx]),'shade':cmp(shade,ref['shade_pack'][idx])},'compiled_prepared':{'sun':cmp(cs,ref['sun_pack'][idx]),'shade':cmp(ch,ref['shade_pack'][idx])},'serial_compiled_equal':bool(np.array_equal(sun,cs) and np.array_equal(shade,ch)),'profile':math_profile.profile_identity(),'torch_spec':str(importlib.util.find_spec('torch')),'torch_maps':any('torch' in q.lower() for q in open('/proc/self/maps',errors='replace'))}
np.savez_compressed(B/('preflight_t'+os.environ['MKL_NUM_THREADS']+'.npz'),asvf_bits=a.view(np.uint32),sun_pack=np.packbits(sun,axis=1),shade_pack=np.packbits(shade,axis=1),compiled_sun_pack=np.packbits(cs,axis=1),compiled_shade_pack=np.packbits(ch,axis=1)); open(B/('preflight_t'+os.environ['MKL_NUM_THREADS']+'.json'),'w').write(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report))
