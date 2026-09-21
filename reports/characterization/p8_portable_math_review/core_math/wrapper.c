#include <math.h>
#include <stddef.h>
float cr_acosf(float); float cr_tanf(float); float cr_atanf(float);
void eval(int op,const float *x,float *y,size_t n){for(size_t i=0;i<n;i++){float a=x[i]; y[i]=op==0?sqrtf(a):op==1?cr_acosf(a):op==2?cr_tanf(a):cr_atanf(a);}}
