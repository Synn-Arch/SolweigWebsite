import sys
import numpy as np
from solweig_light.radiation.patch_radiation import _shortwave_serial, _shortwave
p, q = 2, 3
a = np.zeros((p, q), np.float32)
args = (a, a, a, np.ones_like(a), np.ones_like(a), np.ones_like(a),
        np.ones(q, np.float32), np.ones(q, np.float32), np.ones(q, np.float32),
        np.ones((q, 4), np.float32), np.ones((q, 4), bool), np.ones((q, 4), bool),
        np.ones((q, 4), bool), np.float32(1), np.float32(1), False)
fn = _shortwave_serial if sys.argv[1] == 'serial' else _shortwave
out = fn(*args)
print(sys.argv[1], out.shape, float(out.sum()))
print(fn.overloads)
