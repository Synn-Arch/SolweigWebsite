"""Bounded SLEEF float32 tan/atan port, diagnostic only.

Derived from SLEEF 5a1d179df9cf652951b59010a2d2075372d67f68
sleefsimdsp.c xtanf_u1/atan2kf_u1/xatanf_u1 and df.h.
Copyright Naoki Shibata and contributors 2010-2024; Boost Software License
1.0, full text in SLEEF_LICENSE.txt. No production integration.
"""
import numpy as np
from numba import njit
from ._sleef_acos import fma, dfmul, F

opts = dict(inline='always', fastmath=False, error_model='numpy')

@njit(**opts)
def add_ff(a,b):
    s=F(a+b)
    return s,F(F(a-s)+b)

@njit(**opts)
def add_ff2(a,b):
    s=F(a+b[0])
    return s,F(F(F(a-s)+b[0])+b[1])

@njit(**opts)
def add_f2f(a,b):
    s=F(a[0]+b)
    return s,F(F(F(a[0]-s)+b)+a[1])

@njit(**opts)
def add_f2f2(a,b):
    s=F(a[0]+b[0])
    return s,F(F(F(F(a[0]-s)+b[0])+a[1])+b[1])

@njit(**opts)
def normalize(a):
    s=F(a[0]+a[1])
    return s,F(F(a[0]-s)+a[1])

@njit(**opts)
def square(a):
    s=F(a[0]*a[0])
    return s,fma(F(a[0]+a[0]),a[1],fma(a[0],a[0],-s))

@njit(**opts)
def reciprocal(a):
    s=F(F(1)/a[0])
    return s,F(s*fma(-a[1],s,fma(-a[0],s,F(1))))

@njit(**opts)
def divide(a,b):
    t=F(F(1)/b[0]);s=F(a[0]*t)
    u=fma(t,a[0],-s)
    v=fma(-b[1],t,fma(-b[0],t,F(1)))
    return s,fma(s,v,fma(a[1],t,u))

@njit(**opts)
def tan_fma(d):
    # This source-derived fast reduction is valid on |d| < 125.  The public
    # wrapper dispatches other accepted inputs to the explicitly identified
    # legacy NumPy path; this scalar is deliberately not called outside it.
    if not np.isfinite(d) or abs(d)>=F(125):
        return F(np.nan)
    u=F(np.rint(F(d*F(2/np.pi))))
    q=int(u)
    v=fma(u,F(-3.1414794921875*.5),d)
    w=F(u*F(-.00011315941810607910156*.5))
    s0=F(v+w);sv=F(s0-v)
    s=(s0,F(F(v-F(s0-sv))+F(w-sv)))
    s=add_f2f(s,F(u*F(-1.9841872589410058936e-9*.5)))
    odd=(q&1)==1
    if odd:s=(-s[0],-s[1])
    t=s;s=normalize(square(s))
    u=F(.00446636462584137916564941)
    u=fma(u,s[0],F(-8.3920182078145444393158e-5))
    u=fma(u,s[0],F(.0109639242291450500488281))
    u=fma(u,s[0],F(.0212360303848981857299805))
    u=fma(u,s[0],F(.0540687143802642822265625))
    x=add_ff(F(.133325666189193725585938),F(u*s[0]))
    x=add_ff2(F(1),dfmul(add_ff2(F(.33333361148834228515625),dfmul(s,x)),s))
    x=dfmul(t,x)
    if odd:x=reciprocal(x)
    out=F(x[0]+x[1])
    return d if d==F(0) and np.signbit(d) else out

@njit(**opts)
def atan_fma(d):
    if np.isnan(d):return F(np.nan)
    if np.isinf(d):return F(np.copysign(F(1.5707963267948966),d))
    a=abs(d);q=1 if F(1)<a else 0
    s=divide((F(-1),F(0)),(a,F(0))) if q else divide((a,F(0)),(F(1),F(0)))
    t=normalize(square(s))
    u=F(-.00176397908944636583328247)
    u=fma(u,t[0],F(.0107900900766253471374512))
    u=fma(u,t[0],F(-.0309564601629972457885742))
    u=fma(u,t[0],F(.0577365085482597351074219))
    u=fma(u,t[0],F(-.0838950723409652709960938))
    u=fma(u,t[0],F(.109463557600975036621094))
    u=fma(u,t[0],F(-.142626821994781494140625))
    u=fma(u,t[0],F(.199983194470405578613281))
    t=dfmul(t,add_ff(F(-.333332866430282592773438),F(u*t[0])))
    t=dfmul(s,add_ff2(F(1),t))
    # dfmul_vf2_vf2_vf(pi/2 pair, float(q)) is exact for q=0 or1.
    t=add_f2f2((F(F(1.5707963705062866211)*F(q)),F(F(-4.3711388286737928865e-8)*F(q))),t)
    return F(np.copysign(F(t[0]+t[1]),d))

@njit(fastmath=False,error_model='numpy')
def tan_array(a):
    """SLEEF u10 tangent for finite float32 inputs with ``abs(x) < 125``.

    Complete-domain dispatch belongs to :func:`_math_profile.tan32`; callers
    must mask this private compiled helper to its source-validated domain.
    """
    out=np.empty(a.size,dtype=np.float32)
    for i in range(a.size):
        out[i]=tan_fma(a[i])
    return out

@njit(fastmath=False,error_model='numpy')
def atan_array(a):
    out=np.empty(a.size,dtype=np.float32)
    for i in range(a.size):out[i]=atan_fma(a[i])
    return out
