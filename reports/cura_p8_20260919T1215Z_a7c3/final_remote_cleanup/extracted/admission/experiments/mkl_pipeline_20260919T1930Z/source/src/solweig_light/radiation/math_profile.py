"""Diagnostic-only pinned oneMKL VML profile; never enabled implicitly."""
import ctypes, hashlib, os
from pathlib import Path
import numpy as np
MODE=0x00140102
EXPECTED_SHA256="0f3c805b85c36f3aaf6792930b054d80a945af032f0a97c4926d7ad94a7d61d9"
EXPECTED_VERSION="Intel(R) oneAPI Math Kernel Library Version 2024.2-Product Build 20240605 for Intel(R) 64 architecture applications"
_PROFILE="cura-mkl-vml-2024.2-ha-ftzdazoff-v1"
_lib=None

def _digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def _load():
 global _lib
 if _lib is not None:return _lib
 p=os.environ.get('SOLWEIG_DIAGNOSTIC_MKL_LIB')
 if os.environ.get('SOLWEIG_DIAGNOSTIC_MATH_PROFILE')!=_PROFILE or not p: raise RuntimeError('diagnostic MKL profile not explicitly configured')
 p=Path(p).resolve()
 if _digest(p)!=EXPECTED_SHA256: raise RuntimeError('diagnostic MKL library hash mismatch')
 L=ctypes.CDLL(str(p)); fp=ctypes.POINTER(ctypes.c_float)
 for n in ('vmsSqrt','vmsAcos','vmsCos','vmsTan','vmsAtan'):
  f=getattr(L,n); f.argtypes=[ctypes.c_int,fp,fp,ctypes.c_longlong]; f.restype=None
 dp=ctypes.POINTER(ctypes.c_double); L.vmdCos.argtypes=[ctypes.c_int,dp,dp,ctypes.c_longlong]; L.vmdCos.restype=None
 b=ctypes.create_string_buffer(512); L.MKL_Get_Version_String.argtypes=[ctypes.c_char_p,ctypes.c_int]; L.MKL_Get_Version_String(b,512)
 if b.value.decode()!=EXPECTED_VERSION: raise RuntimeError('diagnostic MKL version mismatch')
 if any('torch' in x.lower() for x in open('/proc/self/maps',errors='replace')): raise RuntimeError('Torch mapped in diagnostic candidate process')
 _lib=L; return L

def unary(name,value):
 shape=np.asarray(value).shape; a=np.ascontiguousarray(value,dtype=np.float32); o=np.empty_like(a); fp=ctypes.POINTER(ctypes.c_float); _load().__getattr__(name)(a.size,a.ctypes.data_as(fp),o.ctypes.data_as(fp),MODE); return o.reshape(shape)
def sqrt_acos(value): return unary('vmsAcos',unary('vmsSqrt',value))
def _unary64(name,value):
 shape=np.asarray(value).shape; a=np.ascontiguousarray(value,dtype=np.float64); o=np.empty_like(a); dp=ctypes.POINTER(ctypes.c_double); f=getattr(_load(),name); f.argtypes=[ctypes.c_int,dp,dp,ctypes.c_longlong]; f(a.size,a.ctypes.data_as(dp),o.ctypes.data_as(dp),MODE); return o.reshape(shape)
def cos(value):
 a=np.asarray(value); return _unary64("vmdCos",a) if a.dtype==np.float64 else unary("vmsCos",a)
def tan(value):
 a=np.asarray(value); return _unary64("vmdTan",a) if a.dtype==np.float64 else unary("vmsTan",a)
def atan(value): return unary('vmsAtan',value)
def profile_identity():
 _load(); return {'profile':_PROFILE,'mode_hex':'0x00140102','library_sha256':EXPECTED_SHA256,'version':EXPECTED_VERSION,'helper_sha256':_digest(__file__)}
