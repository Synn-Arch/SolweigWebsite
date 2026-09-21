/* Diagnostic ABI wrapper only. Original BSD-3-Clause Intel SVML assembly is
 * compiled unchanged; copyright and license retained beside this file. */
#include <immintrin.h>
#include <stddef.h>
extern __m512 __svml_tanf16(__m512);
void eval_tan(const float*x,float*y,size_t n){
 for(size_t i=0;i<n;i+=16){float a[16],b[16];size_t count=n-i<16?n-i:16;
  for(size_t j=0;j<16;j++)a[j]=x[i+(j<count?j:count-1)];
  _mm512_storeu_ps(b,__svml_tanf16(_mm512_loadu_ps(a)));
  for(size_t j=0;j<count;j++)y[i+j]=b[j];
 }
}
