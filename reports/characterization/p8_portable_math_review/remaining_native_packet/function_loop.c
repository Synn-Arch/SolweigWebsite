/* Diagnostic ABI loop: explicit function pointer prevents compiler replacement
 * of named mathematical functions with builtin host implementations. */
#include <stddef.h>
void eval32(float (*fn)(float),const float*x,float*y,size_t n){for(size_t i=0;i<n;i++)y[i]=fn(x[i]);}
void eval64(double (*fn)(double),const double*x,double*y,size_t n){for(size_t i=0;i<n;i++)y[i]=fn(x[i]);}
