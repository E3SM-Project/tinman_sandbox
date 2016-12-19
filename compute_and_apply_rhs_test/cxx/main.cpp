#include <iostream>

//#include <mpi.h>

#include "data_structures.hpp"
#include "compute_and_apply_rhs.hpp"

int main (int argc, char** argv)
{
  using namespace Homme;

  //MPI_Init (argc, argv);

  TestData data;

  std::cout << " --- Initializing data...\n";
  data.init_data();

  std::cout << " --- Performing computations...\n";
  compute_and_apply_rhs(data);

  std::cout << " --- Cleaning up data...\n";
  data.cleanup_data ();

  //MPI_Finalize ();

  return 0;
}
