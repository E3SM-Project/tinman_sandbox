#include <cstdlib>

//#include <mpi.h>

#include "data_structures.hpp"

int compute_and_apply_rhs (TestData& data);

int main (int argc, char** argv)
{
  int status = EXIT_SUCCESS;

  //MPI_Init (argc, argv);

  TestData data;

  status += init_test_data(data);

  status += compute_and_apply_rhs(data);

  status += cleanup_data (data);

  //MPI_Finalize ();

  return status;
}
