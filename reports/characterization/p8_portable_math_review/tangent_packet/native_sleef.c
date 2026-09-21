/* Diagnostic wrappers around unmodified pinned Boost-licensed SLEEF. */
#include <immintrin.h>
#include <stddef.h>
#include <sleef.h>
void eval_scalar(const float*x,float*y,size_t n){for(size_t i=0;i<n;i++)y[i]=Sleef_tanf_u35(x[i]);}
void eval_vector(const float*x,float*y,size_t n){
 for(size_t i=0;i<n;i+=16){float a[16],b[16];size_t count=n-i<16?n-i:16;
  for(size_t j=0;j<16;j++)a[j]=x[i+(j<count?j:count-1)];
  _mm512_storeu_ps(b,Sleef_tanf16_u35avx512f(_mm512_loadu_ps(a)));
  for(size_t j=0;j<count;j++)y[i+j]=b[j];
 }
}
