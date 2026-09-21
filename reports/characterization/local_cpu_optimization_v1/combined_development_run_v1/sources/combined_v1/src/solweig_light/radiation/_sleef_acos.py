"""Diagnostic-only Numba port of SLEEF xacosf_u1 with FMA enabled.

Derived from SLEEF commit 5a1d179df9cf652951b59010a2d2075372d67f68,
src/libm/sleefsimdsp.c and src/common/df.h. Copyright Naoki Shibata
and contributors 2010-2024. Distributed under the Boost Software License,
Version 1.0; the complete license accompanies this file in SLEEF_LICENSE.txt.
This is an operation-order port, not a new fitted approximation.
"""
import numpy as np
from numba import njit, types
from numba.extending import intrinsic
from llvmlite import ir
from ._jit_cache import bind_cache_identity

F = np.float32


@intrinsic
def fma(typingctx, a, b, c):
    if (a, b, c) != (types.float32,) * 3:
        raise TypeError('Explicit float32 FMA operands required')
    def codegen(context, builder, signature, args):
        fn = builder.module.globals.get('llvm.fma.f32')
        if fn is None:
            fn = ir.Function(builder.module, ir.FunctionType(ir.FloatType(), [ir.FloatType()] * 3), name='llvm.fma.f32')
        return builder.call(fn, args)
    return types.float32(a, b, c), codegen


@njit(inline='always', fastmath=False, error_model='numpy')
def dfmul(x, y):
    s = F(x[0] * y[0])
    return s, fma(x[0], y[1], fma(x[1], y[0], fma(x[0], y[0], -s)))


@njit(inline='always', fastmath=False, error_model='numpy')
def dfsqrt(d):
    t = F(np.sqrt(d))
    tt = F(t * t)
    lo = fma(t, t, -tt)
    # dfadd2_vf2_vf_vf2(d, dfmul_vf2_vf_vf(t,t))
    s = F(d + tt)
    v = F(s - d)
    e = F(F(F(d - F(s - v)) + F(tt - v)) + lo)
    rec = F(F(1) / t)
    rec_lo = F(rec * fma(-t, rec, F(1)))
    hi, low = dfmul((s, e), (rec, rec_lo))
    return F(hi * F(.5)), F(low * F(.5))


@njit(inline='always', fastmath=False, error_model='numpy')
def dfsub(x, y):
    s = F(x[0] - y[0])
    t = F(x[0] - s)
    t = F(t - y[0])
    t = F(t + x[1])
    return s, F(t - y[1])


@njit(inline='always', fastmath=False, error_model='numpy')
def acos_fma(d):
    a = abs(d)
    small = a < F(.5)
    x2 = F(d * d) if small else F(F(F(1) - a) * F(.5))
    x = (a, F(0)) if small else dfsqrt(x2)
    if a == F(1):
        x = (F(0), F(0))
    u = F(.4197454825e-1)
    u = fma(u, x2, F(.2424046025e-1))
    u = fma(u, x2, F(.4547423869e-1))
    u = fma(u, x2, F(.7495029271e-1))
    u = fma(u, x2, F(.1666677296e+0))
    u = F(u * F(x2 * x[0]))
    if small:
        sx = F(np.copysign(x[0], d))
        su = F(np.copysign(u, d))
        s = F(sx + su)
        err = F(F(sx - s) + su)
        y = dfsub((F(3.1415927410125732422 / 2), F(-8.7422776573475857731e-8 / 2)), (s, err))
    else:
        s = F(x[0] + u)
        err = F(F(F(x[0] - s) + u) + x[1])
        y = (F(s * F(2)), F(err * F(2)))
        if d < F(0):
            y = dfsub((F(3.1415927410125732422), F(-8.7422776573475857731e-8)), y)
    return F(y[0] + y[1])


@njit(cache=True, fastmath=False, error_model='numpy')
@bind_cache_identity
def asvf_fma(values):
    result = np.empty(values.size, dtype=np.float32)
    for i in range(values.size):
        result[i] = acos_fma(F(np.sqrt(values[i])))
    return result


