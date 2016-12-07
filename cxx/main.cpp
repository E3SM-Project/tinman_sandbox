#include <cstdlib>

//#include <mpi.h>

#include "data_structures.hpp"
#include "compute_and_apply_rhs.hpp"

int main (int argc, char** argv)
{
  using namespace Homme;

  int status = EXIT_SUCCESS;

  //MPI_Init (argc, argv);

  TestData data;

  status += init_test_data(data);

  status += compute_and_apply_rhs(data);

  status += cleanup_data (data);

  //MPI_Finalize ();

  return status;
}
