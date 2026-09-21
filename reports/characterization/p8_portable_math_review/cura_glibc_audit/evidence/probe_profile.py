"""Two predetermined coherent native profiles; no per-input algorithm selection."""
import argparse,ctypes,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--prefix',default='');p.add_argument('--corpus',type=Path,required=True);p.add_argument('--asvf-oracle',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);p.add_argument('--sqrt-oracle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
lib=ctypes.CDLL(str(a.library.resolve()),mode=ctypes.RTLD_LOCAL);loop=ctypes.CDLL(str(Path('libfunction_loop.so').resolve()))
for bits in (32,64):getattr(loop,'eval'+str(bits)).argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
symbols={}
class Info(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('fbase',ctypes.c_void_p),('sname',ctypes.c_char_p),('saddr',ctypes.c_void_p)]
dl=ctypes.CDLL(None);dl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(Info)];dl.dladdr.restype=ctypes.c_int
for fn in ('sqrtf','acosf','tanf','atanf','cos','tan'):
 symbol=getattr(lib,a.prefix+fn);info=Info();assert dl.dladdr(ctypes.cast(symbol,ctypes.c_void_p),ctypes.byref(info));origin=Path(info.fname.decode()).resolve();assert origin==a.library.resolve(),(fn,str(origin));symbols[fn]=symbol

def call(name,x):
 bits=32 if name.endswith('f') else 64;x=np.ascontiguousarray(x,dtype=np.float32 if bits==32 else np.float64);y=np.empty_like(x);getattr(loop,'eval'+str(bits))(ctypes.cast(symbols[name],ctypes.c_void_p),x.ctypes.data,y.ctypes.data,x.size);return y

def metric(x,y):
 f=np.isfinite(x)&np.isfinite(y);neq=x.view(np.uint32)!=y.view(np.uint32)
 return {'finite_bit_mismatches':int((neq&f).sum()),'all_bit_mismatches':int(neq.sum()),'nan_masks_equal':bool(np.array_equal(np.isnan(x),np.isnan(y))),'max_abs':float(np.max(np.abs(x[f].astype(float)-y[f].astype(float)),initial=0))}
z=np.load(a.corpus);r=np.load(a.asvf_oracle);t=np.load(a.trace);idx=np.searchsorted(z['input_bits'],z['boundary_bits']);x=z['input_bits'].view(np.float32);sq=call('sqrtf',x);asvf=call('acosf',sq);orig=r['result_bits'].view(np.float32);sqrtref=np.load(a.sqrt_oracle)['torch_sqrt_bits'].view(np.float32)
assert np.array_equal(z['input_bits'][idx],z['boundary_bits']);assert np.array_equal(t['boundary_bits'],z['boundary_bits']);assert np.array_equal(t['asvf_bits'],r['result_bits'][idx]);assert np.array_equal(np.load(a.sqrt_oracle)['input_bits'],z['input_bits'])
xi=call('cos',t['cos_input']);solar=call('tan',t['solar_tan_input']);yi=(2.0*xi)*solar;yi_clamped=np.where(yi>0,0.,yi).astype(np.float32)
profiles={};raw={'sqrt_bits':sq.view(np.uint32),'asvf_bits':asvf.view(np.uint32),'xi':xi,'solar_tan':solar}
for name,av in [('complete',asvf[idx]),('original_asvf_diagnostic',orig[idx])]:
 hs=call('tanf',av);delta=np.ascontiguousarray(hs[:,None]+yi_clamped[None,:]);at=call('atanf',delta);degrees=at*np.float32(180/np.pi)
 profiles[name]={'sun':int(np.count_nonzero((degrees<z['patch_altitude'])!=np.unpackbits(t['sun_pack'],axis=1,count=153))),'shade':int(np.count_nonzero((degrees>z['patch_altitude'])!=np.unpackbits(t['shade_pack'],axis=1,count=153)))};raw[name+'_hsvf']=hs;raw[name+'_degrees']=degrees
report={'sqrt':metric(sq,sqrtref),'acos_on_original_sqrt_diagnostic':metric(call('acosf',sqrtref),orig),'asvf':metric(asvf,orig),'boundary_masks':profiles,'library':str(a.library.resolve()),'library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),'prefix':a.prefix,'symbol_origins_verified':True,'qualification':'Uniform native library sqrt/acos/tan/atan and float64 scalar cos/tan; actual scalar wrapping/order. Original-ASVF branch is diagnostic only. Benign bits are reported, boundary/output gates unchanged.'}
a.output.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output/'results.npz',**raw);(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
