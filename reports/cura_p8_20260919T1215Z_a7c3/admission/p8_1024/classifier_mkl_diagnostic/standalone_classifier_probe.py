import ctypes,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
B=Path(sys.argv[1]); E=Path(sys.argv[2]); label=sys.argv[3] if len(sys.argv)>3 else 'default'; libpath=E/'local/lib/libmkl_rt.so.2'; lib=ctypes.CDLL(str(libpath)); mode=0x140102
fp=ctypes.POINTER(ctypes.c_float); dp=ctypes.POINTER(ctypes.c_double)
for name in ['vmsCos','vmsTan','vmsAtan']:
 f=getattr(lib,name); f.argtypes=[ctypes.c_int,fp,fp,ctypes.c_longlong]
lib.vmdTan.argtypes=[ctypes.c_int,dp,dp,ctypes.c_longlong]
def vm(name,a):
 a=np.ascontiguousarray(a,dtype=np.float32); o=np.empty_like(a); getattr(lib,name)(a.size,a.ctypes.data_as(fp),o.ctypes.data_as(fp),mode); return o.reshape(a.shape)
def vd_tan(v):
 a=np.asarray([v],np.float64); o=np.empty_like(a); lib.vmdTan(1,a.ctypes.data_as(dp),o.ctypes.data_as(dp),mode); return o[0]
z=np.load(B/'corpus.npz'); t=np.load(B/'torch_classifier_trace.npz'); bits=z['input_bits']; idx=np.searchsorted(bits,z['boundary_bits']); asvf=t['asvf_bits'].view(np.float32); pa=z['patch_altitude']; pazi=z['patch_azimuth']; sa=np.asarray(z['solar_altitude']); saz=float(z['solar_azimuth']); deg2=np.pi/180.0; rad2=180.0/np.pi
# Candidate wrap semantics.
d=np.abs(np.subtract(np.float32(saz),pazi)); angle=np.multiply(d,np.float32(deg2)); xi_np=np.cos(angle); xi_mkl=vm('vmsCos',angle)
solar_rad=np.multiply(sa,deg2); stan_np=np.tan(solar_rad); stan_mkl=vd_tan(solar_rad)
def yi(xi,st): return np.multiply(np.multiply(np.float32(2),xi),np.float32(st))
def clamp(v): return np.where(v>0,0.0,v)
h_np=np.tan(asvf[:,None]); h_mkl=vm('vmsTan',asvf[:,None])
def finish(xi,st,h,use_mkl_atan):
 y=clamp(yi(xi,st)); td=np.add(h,y); at=vm('vmsAtan',td) if use_mkl_atan else np.arctan(td); sd=np.multiply(at,np.float32(rad2)); return y,td,at,np.packbits(sd<pa,axis=1),np.packbits(sd>pa,axis=1)
variants={}
for name,args in {'numpy':(xi_np,stan_np,h_np,False),'mkl_cos':(xi_mkl,stan_np,h_np,False),'mkl_cos_dtan':(xi_mkl,stan_mkl,h_np,False),'mkl_cos_dtan_tan':(xi_mkl,stan_mkl,h_mkl,False),'full_mkl_trig':(xi_mkl,stan_mkl,h_mkl,True)}.items(): variants[name]=finish(*args)
def bits(a): return np.asarray(a).view(np.uint32 if np.asarray(a).dtype==np.float32 else np.uint64)
def cmp(a,b):
 aa=bits(a); bb=np.asarray(b); d=aa!=bb; return {'mismatches':int(d.sum()),'first_indices':np.argwhere(d)[:12].tolist()}
def masks(s,h):
 def one(a,b):
  d=np.bitwise_xor(a,b); return {'cell_mismatches':int(np.unpackbits(d,axis=1,count=153).sum()),'input_mismatches':int(np.any(d,axis=1).sum())}
 return {'sun':one(s,t['sun_pack']),'shade':one(h,t['shade_pack'])}
stages={'patch_delta':cmp(d,t['patch_delta_bits']),'xi_numpy':cmp(xi_np,t['xi_bits']),'xi_mkl':cmp(xi_mkl,t['xi_bits']),'solar_rad':cmp(np.asarray(solar_rad),t['solar_rad_bits']),'solar_tan_numpy':cmp(np.asarray(stan_np),t['solar_tan_bits']),'solar_tan_mkl':cmp(np.asarray(stan_mkl),t['solar_tan_bits']),'hsvf_numpy':cmp(h_np,t['hsvf_bits']),'hsvf_mkl':cmp(h_mkl,t['hsvf_bits'])}
for name,(y,td,at,s,h) in variants.items(): stages[name]={'yi':cmp(y,t['yi_clamped_bits']),'tan_delta':cmp(td,t['tan_delta_bits']),'atan':cmp(at,t['atan_bits']),'masks':masks(s,h)}
# Exact mismatch fixture including every boundary mask divergence under NumPy and all corresponding Torch/MKL bits.
ns,nh=variants['numpy'][3:]; md=np.any(np.bitwise_xor(ns,t['sun_pack']),axis=1)|np.any(np.bitwise_xor(nh,t['shade_pack']),axis=1); mi=np.flatnonzero(md)
np.savez_compressed(B/'classifier_minimal_fixture.npz',corpus_indices=idx[mi],input_bits=t['input_bits'][mi],asvf_bits=t['asvf_bits'][mi],patch_altitude=pa,patch_azimuth=pazi,torch_xi_bits=t['xi_bits'],mkl_xi_bits=xi_mkl.view(np.uint32),torch_solar_tan_bits=t['solar_tan_bits'],mkl_solar_tan_bits=np.asarray(stan_mkl).view(np.uint64),torch_hsvf_bits=t['hsvf_bits'][mi],mkl_hsvf_bits=h_mkl[mi].view(np.uint32),torch_tan_delta_bits=t['tan_delta_bits'][mi],full_mkl_tan_delta_bits=variants['full_mkl_trig'][1][mi].view(np.uint32),torch_atan_bits=t['atan_bits'][mi],full_mkl_atan_bits=variants['full_mkl_trig'][2][mi].view(np.uint32),torch_sun_pack=t['sun_pack'][mi],torch_shade_pack=t['shade_pack'][mi],numpy_sun_pack=ns[mi],numpy_shade_pack=nh[mi])
report={'mode_hex':hex(mode),'stages':stages,'minimal_fixture_inputs':int(len(mi)),'environment':{'torch_import_spec':str(importlib.util.find_spec('torch')),'MKL_NUM_THREADS':__import__('os').environ.get('MKL_NUM_THREADS'),'OMP_NUM_THREADS':__import__('os').environ.get('OMP_NUM_THREADS'),'library_sha256':hashlib.sha256(libpath.read_bytes()).hexdigest()},'method':'Exact standalone MKL vmsCos/vmdTan/vmsTan/vmsAtan with float32 wrapped multiply/add and float64 solar scalar, matching original Torch dtypes.'}
np.savez_compressed(B/('classifier_prototype_'+label+'.npz'),atan_bits=variants['full_mkl_trig'][2].view(np.uint32),sun_pack=variants['full_mkl_trig'][3],shade_pack=variants['full_mkl_trig'][4]); (B/('standalone_classifier_report_'+label+'.json')).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report,indent=2))
