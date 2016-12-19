#include "common.hpp"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <config.h>

int I1 = I1_MACRO;

int main( int argc, char * argv[] )
{
  if (argc > 1) {
    I1 = std::atoi( argv[1] );
  }

  double * x;
  double * y;

  posix_memalign( reinterpret_cast<void**>(&x), 128, sizeof(double)*I1*I2*I3);
  posix_memalign( reinterpret_cast<void**>(&y), 128, sizeof(double)*I1*I2*I3);

  std::cout << I1 << "    " << I2 << "    " << I3 << std::endl;

  {
    Timer t("Init");
    #pragma omp parallel for
    for (int i=0; i<I1; ++i) {
    for (int j=0; j<I2; ++j) {
    for (int k=0; k<I3; ++k) {
      x[IDX(i,j,k)] = i*j*k;
      y[IDX(i,j,k)] = i*i*j*j*k*k;
    }}}
  }

  {
    Timer t("saxpby");
    for (int iteration = 0; iteration < 100; ++iteration) {
      saxpby(3.0,5.0,x,y);
    }
  }


  free(x);
  free(y);

  return 0;
}



