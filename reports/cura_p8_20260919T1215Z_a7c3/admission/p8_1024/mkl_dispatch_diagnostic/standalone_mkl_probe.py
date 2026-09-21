import ctypes,hashlib,importlib.util,json,os,platform,sys
from pathlib import Path
import numpy as np
B=Path(sys.argv[1]); E=Path(sys.argv[2]); libpath=E/'local/lib/libmkl_rt.so.2'; lib=ctypes.CDLL(str(libpath),mode=ctypes.RTLD_GLOBAL)
mode=0x00140102; fp=ctypes.POINTER(ctypes.c_float)
for n in ['vmsSqrt','vmsAcos']:
 f=getattr(lib,n); f.argtypes=[ctypes.c_int,fp,fp,ctypes.c_longlong]; f.restype=None
z=np.load(B/'corpus.npz'); ref=np.load(B/'torch_reference.npz'); x=z['input_bits'].view(np.float32).copy()
def vm(n,a):
 o=np.empty_like(a); getattr(lib,n)(len(a),a.ctypes.data_as(fp),o.ctypes.data_as(fp),mode); return o
s=vm('vmsSqrt',x); a=vm('vmsAcos',s); rb=ref['result_bits']; ab=a.view(np.uint32)
# Retained candidate classifier arithmetic, copied verbatim in behavior (no candidate import/deps).
def operands(a,b):
 aa,bb=np.asarray(a),np.asarray(b)
 if aa.dtype.kind=='f' and aa.ndim>0 and (bb.ndim==0 or bb.dtype.kind in 'biu'): b=np.asarray(b,dtype=aa.dtype)
 elif bb.dtype.kind=='f' and bb.ndim>0 and (aa.ndim==0 or aa.dtype.kind in 'biu'): a=np.asarray(a,dtype=bb.dtype)
 return a,b
def op(fn,a,b): return fn(*operands(a,b))
def div(a,b): return np.divide(*operands(a,b))
pa=z['patch_altitude']; pazi=z['patch_azimuth']; sa=np.asarray(z['solar_altitude']); saz=float(z['solar_azimuth'])
ps=[];ph=[]
for lo in range(0,len(a),4096):
 av=a[lo:lo+4096,None]; d=np.abs(op(np.subtract,saz,pazi)); deg2=div(np.pi,180.0); rad2=div(180.0,np.pi); xi=np.cos(op(np.multiply,d,deg2)); yi=op(np.multiply,op(np.multiply,2,xi),np.tan(op(np.multiply,sa,deg2))); yi_=np.where(yi>0,0.0,yi); sd=op(np.multiply,np.arctan(op(np.add,np.tan(av),yi_)),rad2); ps.append(np.packbits(sd<pa,axis=1)); ph.append(np.packbits(sd>pa,axis=1))
ps=np.concatenate(ps); ph=np.concatenate(ph)
def cc(q,r):
 d=np.bitwise_xor(q,r); return {'cell_mismatches':int(np.unpackbits(d,axis=1,count=153).sum()),'input_mismatches':int(np.any(d,axis=1).sum())}
def cat(k):
 q=z[k+'_bits']; idx=np.searchsorted(z['input_bits'],q); return {'count':len(idx),'sun':cc(ps[idx],ref['sun_pack'][idx]),'shade':cc(ph[idx],ref['shade_pack'][idx])}
maps=open('/proc/self/maps').read(); locks={}
for p in sorted(E.glob('local/lib/python*/dist-packages/*.dist-info/METADATA')):
 lines=p.read_text(errors='replace').splitlines(); name=next((q[6:] for q in lines if q.startswith('Name: ')),p.parent.name); ver=next((q[9:] for q in lines if q.startswith('Version: ')),'?'); locks[name]=ver
buf=ctypes.create_string_buffer(512); lib.MKL_Get_Version_String.argtypes=[ctypes.c_char_p,ctypes.c_int]; lib.MKL_Get_Version_String(buf,512)
finite=np.isfinite(a)&np.isfinite(rb.view(np.float32)); diff=ab!=rb
report={'inputs':len(x),'expression':{'all_bit_mismatches':int(diff.sum()),'finite_bit_mismatches':int((diff&finite).sum()),'nan_masks_equal':bool(np.array_equal(np.isnan(a),np.isnan(rb.view(np.float32))))},'classifications':{k:cat(k) for k in ['frozen','boundary','random','edge','hotspot']},'mode_hex':hex(mode),'mkl_version':buf.value.decode(),'library':{'path':str(libpath),'bytes':libpath.stat().st_size,'sha256':hashlib.sha256(libpath.read_bytes()).hexdigest()},'environment':{'python':sys.version,'platform':platform.platform(),'torch_import_spec':str(importlib.util.find_spec('torch')),'torch_in_proc_maps':'torch' in maps.lower(),'loaded_mkl_paths':sorted(set(line.split()[-1] for line in maps.splitlines() if 'mkl' in line.lower() and '/' in line)),'packages':locks}}
(B/'standalone_mkl_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); np.savez_compressed(B/'standalone_mkl_bits.npz',sqrt_bits=s.view(np.uint32),result_bits=ab); print(json.dumps(report,indent=2))
