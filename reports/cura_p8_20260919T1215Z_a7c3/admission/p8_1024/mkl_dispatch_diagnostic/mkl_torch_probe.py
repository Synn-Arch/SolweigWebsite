import ctypes,hashlib,json,platform,sys
from pathlib import Path
import numpy as np,torch
B=Path(sys.argv[1]); libpath=Path(torch.__file__).parent/'lib/libtorch_cpu.so'; lib=ctypes.CDLL(str(libpath))
mode=0x00140102; n=ctypes.c_int; fptr=ctypes.POINTER(ctypes.c_float); i64=ctypes.c_longlong
for name in ['vmsSqrt','vmsAcos']:
 fn=getattr(lib,name); fn.argtypes=[n,fptr,fptr,i64]; fn.restype=None
for name in ['vsSqrt','vsAcos']:
 fn=getattr(lib,name); fn.argtypes=[n,fptr,fptr]; fn.restype=None
z=np.load(B/'corpus.npz'); bits=z['input_bits']; x=bits.view(np.float32).copy(); tx=torch.from_numpy(x.copy())
with torch.no_grad(): ts=torch.sqrt(tx).numpy().copy(); oracle=torch.acos(torch.sqrt(tx)).numpy().copy()
def vm(name,a,withmode=True):
 out=np.empty_like(a); args=[len(a),a.ctypes.data_as(fptr),out.ctypes.data_as(fptr)];
 if withmode: args.append(mode)
 getattr(lib,name)(*args); return out
ms=vm('vmsSqrt',x); ma=vm('vmsAcos',ts); chain=vm('vmsAcos',ms); ds=vm('vsSqrt',x,False); da=vm('vsAcos',ts,False)
libm=ctypes.CDLL('libm.so.6'); libm.acosf.argtypes=[ctypes.c_float]; libm.acosf.restype=ctypes.c_float
la=np.fromiter((libm.acosf(float(q)) for q in ts),dtype=np.float32,count=len(ts))
np_s=np.sqrt(x,dtype=np.float32)
def cmp(a,r):
 ab=a.view(np.uint32); rb=r.view(np.uint32); finite=np.isfinite(a)&np.isfinite(r); d=ab!=rb
 return {'all_bit_mismatches':int(d.sum()),'finite_bit_mismatches':int((d&finite).sum()),'nan_masks_equal':bool(np.array_equal(np.isnan(a),np.isnan(r))),'max_finite_ulp':int(np.max(np.abs(ab[finite].astype(np.int64)-rb[finite].astype(np.int64)))) if finite.any() else None,'examples':[{'index':int(i),'input_bits':int(bits[i]),'actual_bits':int(ab[i]),'reference_bits':int(rb[i])} for i in np.flatnonzero(d&finite)[:16]]}
buf=ctypes.create_string_buffer(512); lib.MKL_Get_Version_String.argtypes=[ctypes.c_char_p,ctypes.c_int]; lib.MKL_Get_Version_String(buf,512)
report={'mode_hex':hex(mode),'mode_terms':{'VML_HA':'0x2','VML_FTZDAZ_OFF':'0x00140000','VML_ERRMODE_IGNORE':'0x100'},'mkl_version':buf.value.decode(),'torch':torch.__version__,'torch_git_version':torch.version.git_version,'platform':platform.platform(),'library':{'path':str(libpath),'bytes':libpath.stat().st_size,'sha256':hashlib.sha256(libpath.read_bytes()).hexdigest()},'comparisons':{'torch_sqrt_vs_vmsSqrt':cmp(ms,ts),'torch_sqrt_vs_vsSqrt_default':cmp(ds,ts),'torch_sqrt_vs_numpy_sqrt32':cmp(np_s,ts),'torch_full_vs_vmsAcos_torchsqrt':cmp(ma,oracle),'torch_full_vs_vmsAcos_vmsSqrt_chain':cmp(chain,oracle),'torch_full_vs_vsAcos_default_torchsqrt':cmp(da,oracle),'torch_full_vs_libc_acosf_torchsqrt':cmp(la,oracle)}}
(B/'mkl_probe_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); np.savez_compressed(B/'mkl_probe_bits.npz',input_bits=bits,torch_sqrt_bits=ts.view(np.uint32),vms_sqrt_bits=ms.view(np.uint32),torch_result_bits=oracle.view(np.uint32),vms_acos_bits=ma.view(np.uint32),chain_bits=chain.view(np.uint32),libc_acosf_bits=la.view(np.uint32)); print(json.dumps(report['comparisons'],indent=2))
