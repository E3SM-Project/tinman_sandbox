#include "common.hpp"

void saxpby( const double a
           , const double b
           , double * x
           , const double * y
           )
{
  #pragma omp parallel for simd
  for (int i=0; i<I1; ++i) {
  for (int j=0; j<I2; ++j) {
  for (int k=0; k<I3; ++k) {
    x[IDX(i,j,k)] = a * x[IDX(i,j,k)] + b * y[IDX(i,j,k)];
  }}}
}

