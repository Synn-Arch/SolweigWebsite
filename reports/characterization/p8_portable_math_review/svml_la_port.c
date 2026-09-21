/* Diagnostic scalar operation-order port of Intel SVML acosf16 LA.
 * Derived from numpy/SVML commit 9a43e744b15006d63e8d1b57626dfbe6b984f5ee,
 * linux/avx512/svml_z0_acos_s_la.s, Copyright (C) 2023 Intel Corporation,
 * SPDX-License-Identifier: BSD-3-Clause. Original source and LICENSE retained.
 * Uses Intel's published BSD-licensed RECIP14 reference, not proprietary binaries.
 * Special-case exception flags/NaN payloads are not modeled by this prototype.
 */
#include <math.h>
#include <stdint.h>
#include <stddef.h>
typedef union {unsigned int u;float f;} type32;
extern void RSQRT14S(unsigned int,type32*,type32);
static float bits(uint32_t u){type32 x={.u=u};return x.f;}
static float one(float arg){
 if(isnan(arg)||fabsf(arg)>1) return NAN;
 float x=-fabsf(arg),y=fmaf(x,.5f,.5f),xx=x*x;
 type32 src={.f=y},dst;RSQRT14S(0,&dst,src);float r=dst.f;
 if(y<bits(0x2f800000))r=0;
 float u=fminf(y,xx),yy=y+y,uu=u*u,rr=r*r,s=r*yy;
 int k1=!(u<y),k3=arg<u;
 float err=fmaf(yy,rr,-2.f);
 float p0=fmaf(u,bits(0x3d3a9ab4),bits(0x3d997c12));
 float q=fmaf(err,bits(0xbdc00004),bits(0x3e800001));
 float p1=fmaf(u,bits(0x3d2edc07),bits(0x3cc32a6b));
 float es=err*s;
 p1=fmaf(p1,uu,p0);
 q=fmaf(-q,es,s);
 p1=fmaf(p1,u,bits(0x3e2aaaff));
 float a=k1?q:x;
 if(signbit(arg))a=-a;
 float off=k1?0:bits(0x3fc90fdb);
 if(k1&&k3)off=bits(0x40490fdb);
 return fmaf(u*p1,a,a)+off;
}
void svml_la_acos(const float*x,float*y,size_t n){for(size_t i=0;i<n;i++)y[i]=one(x[i]);}
