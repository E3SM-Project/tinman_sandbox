#include <iostream>
#include <cstdlib>


//#include <mpi.h>

#include "dimensions.hpp"
#include "data_structures.hpp"
#include "compute_and_apply_rhs.hpp"

namespace Homme {

int nelems = 10;

} // ugly hack


int main (int argc, char** argv)
{
  using namespace Homme;

  //MPI_Init (argc, argv);

  if (argc > 1) {
    Homme::nelems = std::atoi(argv[1]);
  }

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
