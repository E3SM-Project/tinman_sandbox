#include <iostream>
#include <sys/time.h>

#include "data_structures.hpp"
#include "compute_and_apply_rhs.hpp"

int main (int argc, char** argv)
{
  using namespace Homme;

  struct timeval start, end;

  TestData data;

  std::cout << " --- Initializing data...\n";
  data.init_data();

  print_results_2norm (data);

  std::cout << " --- Performing computations...\n";
  gettimeofday(&start, NULL);
  compute_and_apply_rhs(data);
  gettimeofday(&end, NULL);
  double delta = ((end.tv_sec  - start.tv_sec) * 1000000u +
                   end.tv_usec - start.tv_usec) / 1.e6;
  std::cout << "   ---> compute_and_apply_rhs execution time: " << delta << " seconds.\n";

  print_results_2norm (data);

  std::cout << " --- Dumping results to file...\n";
  dump_results_to_file (data);

  std::cout << " --- Cleaning up data...\n";
  data.cleanup_data ();

  return 0;
}
