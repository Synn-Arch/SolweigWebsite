from pathlib import Path
import sys
S=Path(sys.argv[1]); pkg=S/'src/solweig_light'
helper='''"""Diagnostic-only pinned oneMKL VML profile; never enabled implicitly."""
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
 b=ctypes.create_string_buffer(512); L.MKL_Get_Version_String.argtypes=[ctypes.c_char_p,ctypes.c_int]; L.MKL_Get_Version_String(b,512)
 if b.value.decode()!=EXPECTED_VERSION: raise RuntimeError('diagnostic MKL version mismatch')
 if any('torch' in x.lower() for x in open('/proc/self/maps',errors='replace')): raise RuntimeError('Torch mapped in diagnostic candidate process')
 _lib=L; return L

def unary(name,value):
 a=np.ascontiguousarray(value,dtype=np.float32); o=np.empty_like(a); fp=ctypes.POINTER(ctypes.c_float); _load().__getattr__(name)(a.size,a.ctypes.data_as(fp),o.ctypes.data_as(fp),MODE); return o.reshape(a.shape)
def sqrt_acos(value): return unary('vmsAcos',unary('vmsSqrt',value))
def cos(value): return unary('vmsCos',value)
def tan(value): return unary('vmsTan',value)
def atan(value): return unary('vmsAtan',value)
def profile_identity():
 _load(); return {'profile':_PROFILE,'mode_hex':'0x00140102','library_sha256':EXPECTED_SHA256,'version':EXPECTED_VERSION,'helper_sha256':_digest(__file__)}
'''
(pkg/'radiation/math_profile.py').write_text(helper)
# pipeline
p=pkg/'pipeline.py'; s=p.read_text(); old='asvf = np.arccos(np.sqrt(svf))'; assert old in s; s=s.replace(old,"from .radiation.math_profile import sqrt_acos\n    asvf = sqrt_acos(svf)"); p.write_text(s)
# engine exact substitutions only classifier function slice
p=pkg/'radiation/engine.py'; s=p.read_text(); a=s.index('def shaded_or_sunlit('); b=s.index('\ndef ',a+10); q=s[a:b]; q=q.replace("    patch_to_sun_azi", "    from . import math_profile\n    patch_to_sun_azi",1).replace('xi = np.cos(', 'xi = math_profile.cos(').replace('hsvf = np.tan(asvf)','hsvf = math_profile.tan(asvf)').replace('np.arctan(tan_delta)','math_profile.atan(tan_delta)'); s=s[:a]+q+s[b:]; p.write_text(s)
# patch radiation funcs scoped substitutions
p=pkg/'radiation/patch_radiation.py'; s=p.read_text(); a=s.index('def _class_coefficients('); b=s.index('\n@njit',a); q=s[a:b]; q=q.replace("    from . import engine as e", "    from . import engine as e\n    from . import math_profile",1).replace('xi=np.cos(', 'xi=math_profile.cos(np.asarray(').replace('e._operate(np.multiply,difference,deg2rad))','e._operate(np.multiply,difference,deg2rad)))',1)
# above produces exact cos scalar array; tan/atan
q=q.replace('delta=np.add(np.tan(field),coefficients[None,:])','delta=np.add(math_profile.tan(field),coefficients[None,:])').replace('np.multiply(np.arctan(delta),','np.multiply(math_profile.atan(delta),')
s=s[:a]+q+s[b:]; p.write_text(s)
# identities include fail-closed profile and helper file
p=pkg/'identities.py'; s=p.read_text(); s=s.replace("def geometry_identity(paths, patch_option):\n    from .geometry.shadows import create_patches", "def geometry_identity(paths, patch_option):\n    from .geometry.shadows import create_patches\n    from .radiation.math_profile import profile_identity")
s=s.replace("'policy': 'legacy-logical-domain-geometry-v1', 'upstream_commit': BASELINE,", "'policy': 'legacy-logical-domain-geometry-v1', 'upstream_commit': BASELINE,\n        'diagnostic_math_profile': profile_identity(),",1)
s=s.replace("def simulation_identity(paths, wind_paths, selected_date, tile, flags, location, utc):\n    files", "def simulation_identity(paths, wind_paths, selected_date, tile, flags, location, utc):\n    from .radiation.math_profile import profile_identity\n    files")
s=s.replace("'policy': 'chronological-compatibility-v1', 'upstream_commit': BASELINE,", "'policy': 'chronological-compatibility-v1', 'upstream_commit': BASELINE,\n        'diagnostic_math_profile': profile_identity(),",1)
s=s.replace("'geometry/visibility.py', 'geometry/visibility_native.py'", "'geometry/visibility.py', 'geometry/visibility_native.py', 'radiation/math_profile.py'")
p.write_text(s)
