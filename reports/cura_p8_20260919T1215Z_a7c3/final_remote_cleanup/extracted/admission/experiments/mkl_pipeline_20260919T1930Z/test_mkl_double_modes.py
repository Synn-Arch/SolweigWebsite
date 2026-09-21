import ctypes,sys,numpy as np
B=sys.argv[1];E=sys.argv[2];z=np.load(B+'/corpus.npz');t=np.load(B+'/torch_actual_stages.npz');pz=z['patch_azimuth'];sz=float(z['solar_azimuth']);ang=np.asarray([np.abs(np.subtract(np.float64(sz),q))*(np.pi/180) for q in pz],np.float64);L=ctypes.CDLL(E+'/local/lib/libmkl_rt.so.2');dp=ctypes.POINTER(ctypes.c_double);L.vmdCos.argtypes=[ctypes.c_int,dp,dp,ctypes.c_longlong]
for m in [0x140100,0x140101,0x140102,0x140103,0x2,0x1,0x3]:
 o=np.empty_like(ang);L.vmdCos(len(o),ang.ctypes.data_as(dp),o.ctypes.data_as(dp),m);d=o.view(np.uint64)!=t['xi'].view(np.uint64);print(hex(m),d.sum(),np.max(np.abs(o.view(np.uint64).astype(np.int64)-t['xi'].view(np.uint64).astype(np.int64))))
