#include <stddef.h>
float __ocl_svml_acosf_ha(float);float __ocl_svml_tanf_ha(float);float __ocl_svml_atanf_ha(float);
void eval(int op,const float*x,float*y,size_t n){for(size_t i=0;i<n;i++)y[i]=op==1?__ocl_svml_acosf_ha(x[i]):op==2?__ocl_svml_tanf_ha(x[i]):__ocl_svml_atanf_ha(x[i]);}
