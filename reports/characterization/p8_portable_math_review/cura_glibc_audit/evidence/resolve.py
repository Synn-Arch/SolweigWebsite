import ctypes, hashlib, json, os
lib=ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libm.so.6", mode=ctypes.RTLD_LOCAL)
libdl=ctypes.CDLL(None)
class Dl_info(ctypes.Structure):
 _fields_=[("dli_fname",ctypes.c_char_p),("dli_fbase",ctypes.c_void_p),("dli_sname",ctypes.c_char_p),("dli_saddr",ctypes.c_void_p)]
libdl.dladdr.argtypes=[ctypes.c_void_p,ctypes.POINTER(Dl_info)];libdl.dladdr.restype=ctypes.c_int
out={}
for n in ("sqrtf","acosf","tanf","atanf","cos","tan"):
 f=getattr(lib,n); a=ctypes.cast(f,ctypes.c_void_p).value; i=Dl_info(); assert libdl.dladdr(ctypes.c_void_p(a),ctypes.byref(i))
 b=ctypes.string_at(a,128)
 out[n]={"address":hex(a),"library":i.dli_fname.decode(),"base":hex(i.dli_fbase),"offset":hex(a-i.dli_fbase),"dl_symbol":i.dli_sname.decode() if i.dli_sname else None,"first128_sha256":hashlib.sha256(b).hexdigest(),"first128_hex":b.hex()}
print(json.dumps(out,indent=2,sort_keys=True))
