import sys
import numpy as np
from solweig_light.radiation.patch_radiation import _shortwave, _shortwave_serial, _longwave, _longwave_serial
p, q = 2, 3
a = np.zeros((p,q), np.float32); one=np.ones_like(a); gates=np.ones((q,4),bool); vec=np.ones(q,np.float32); dirs=np.ones((q,4),np.float32)
short=(a,a,a,one,one,one,vec,vec,vec,dirs,gates,gates,gates,np.float32(1),np.float32(1),False)
long=(a,a,a,one,one,vec,vec,vec,dirs,gates,np.ones(q,bool),vec,vec,np.float32(1),np.float32(1),np.zeros(p,np.float32),np.float32(1))
if sys.argv[1] == 'serial':
    f, g, args = _shortwave_serial, _longwave_serial, (short,long)
else:
    f, g, args = _shortwave, _longwave, (short,long)
print(sys.argv[1], f(*args[0]).shape, g(*args[1]).shape)
print('short_overloads', f.overloads)
print('long_overloads', g.overloads)
