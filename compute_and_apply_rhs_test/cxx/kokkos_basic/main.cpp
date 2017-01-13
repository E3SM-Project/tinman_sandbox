#include "compute_and_apply_rhs.hpp"
#include "Region.hpp"
#include "TestData.hpp"
#include "timer.hpp"
#include "Kokkos_Core.hpp"

#include <iostream>
#include <cstring>
#include <vector>

bool is_unsigned_int(const char* str)
{
  const size_t len = strlen (str);
  for (size_t i = 0; i < len; ++i) {
    if (! isdigit (str[i])) {
      return false;
    }
  }
  return true;
}

int main (int argc, char** argv)
{
  int num_elems = 10;
  bool dump_res = false;
  int num_exec = 1;

  if (argc > 1) {
    int iarg = 1;
    while (iarg<argc)
    {
      if (strncmp(argv[iarg],"--tinman-num-elems=",18) == 0)
      {
        char* number = strchr(argv[iarg],'=')+1;
        if (!is_unsigned_int(number))
        {
          std::cerr << "Expecting an unsigned integer after '--tinman-num-elems='.\n";
          std::exit(1);
        }

        num_elems = std::atoi(number);

        ++iarg;
        continue;
      }
      else if (strncmp(argv[iarg],"--tinman-num-exec=",18) == 0)
      {
        char* val = strchr(argv[iarg],'=')+1;
        num_exec = std::stoi(val);

        ++iarg;
        continue;
      }
      else if (strncmp(argv[iarg],"--tinman-dump-res=",18) == 0)
      {
        char* val = strchr(argv[iarg],'=')+1;
        dump_res = std::stoi(val);

        ++iarg;
        continue;
      }
      else if (strncmp(argv[iarg],"--tinman-help",13) == 0)
      {
        std::cout << "+------------------------------------------------------------------+\n"
                  << "|                   TinMan command line arguments                  |\n"
                  << "+------------------------------------------------------------------+\n"
                  << "|  --tinman-num-elems  : the number of elements (def=10)           |\n"
                  << "|  --tinman-dump-res   : whether to dump results to file (def=NO)  |\n"
                  << "|  --tinman-num-exec   : number of times to execute (def=1)        |\n"
                  << "|  --tinman-help       : prints this message                       |\n"
                  << "|  --kokkos-help       : prints kokkos help                        |\n"
                  << "+------------------------------------------------------------------+\n";

        std::exit(0);
      }

      ++iarg;
    }
  }

  if (num_elems < 1) {
    std::cerr << "Invalid number of elements: " << num_elems << std::endl;
    std::exit(1);
  }

  Kokkos::initialize (argc, argv);

  // Kokkos::OpenMP::print_configuration(std::cout,true);
  // std::cout << " --- Initializing data...\n";
  TinMan::TestData data(num_elems);
  TinMan::Region* region = new TinMan::Region(num_elems); // A pointer, so the views are destroyed before Kokkos::finalize

  // Burn in before timing to reduce cache effect
  TinMan::compute_and_apply_rhs(data,*region);

  std::vector<Timer::Timer> timers(num_exec);
  for (int i=0; i<num_exec; ++i)
  {
    timers[i].startTimer();
    TinMan::compute_and_apply_rhs(data,*region);
    timers[i].stopTimer();
  }

  for(int i = 0; i < num_exec; ++i) {
    std::cout << timers[i] << std::endl;
  }

  delete region;

  Kokkos::finalize ();
  return 0;
}
