#include "compute_and_apply_rhs.hpp"
#include "Region.hpp"
#include "TestData.hpp"
#include "timer.hpp"
#include "Kokkos_Core.hpp"

#include <iostream>
#include <cstring>
#include <memory>

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

void parse_args(int argc, char** argv, int &num_elems, int &num_exec, bool &dump_results) {
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
        if (strcmp(val,"yes")==0 || strcmp(val,"YES")==0)
          dump_results = true;
        else if (strcmp(val,"no")==0 || strcmp(val,"NO")==0)
          dump_results = false;
        else
        {
          std::cout << " ERROR! Unrecognized command line option '" << argv[iarg] << "'.\n"
                    << "        Run with '--tinman-help' to see the available options.\n";

          std::exit(1);
        }

        ++iarg;
        continue;
      }
      else if (strncmp(argv[iarg],"--tinman-help",13) == 0)
      {
        std::cout << "+------------------------------------------------------------------------+\n"
                  << "|                      TinMan command line arguments                     |\n"
                  << "+------------------------------------------------------------------------+\n"
                  << "|  --tinman-num-elems=N  : the number of elements (default=10)           |\n"
                  << "|  --tinman-dump-res=val : whether to dump results to file (default=no)  |\n"
                  << "|  --tinman-num-exec=N   : number of times to execute (default=1)        |\n"
                  << "|  --tinman-help         : prints this message                           |\n"
                  << "|  --kokkos-help         : prints kokkos help                            |\n"
                  << "+------------------------------------------------------------------------+\n";

        std::exit(0);
      }

      ++iarg;
    }
  }
}

void run_simulation(int num_elems, int num_exec, bool dump_results) {
  TinMan::Control data(num_elems);
  TinMan::Region region(num_elems);

  // Print norm of initial states, to check we are using same data in all tests
  print_results_2norm(data, region);

  // Burn in before timing to reduce cache effect
  //TinMan::compute_and_apply_rhs(data, region);

  std::unique_ptr<Timer::Timer[]> timers(new Timer::Timer[num_exec]);
  for (int i=0; i<num_exec; ++i)
  {
    timers[i].startTimer();
    TinMan::compute_and_apply_rhs(data, region);
//    data.update_time_levels();
    timers[i].stopTimer();
  }

#ifdef KOKKOS_HAVE_DEFAULT_DEVICE_TYPE_OPENMP
  Kokkos::OpenMP::fence();
#endif

  for(int i = 0; i < num_exec; ++i) {
    std::cout << timers[i] << std::endl;
  }

  print_results_2norm (data,region);

  // if (dump_results)
  // {
  //   dump_results_to_file (data,region);
  // }
}

int main (int argc, char** argv)
{
  int num_elems = 10;
  int num_exec = 1;
  bool dump_results = false;
  parse_args(argc, argv, num_elems, num_exec, dump_results);

  if (num_elems < 1) {
    std::cerr << "Invalid number of elements: " << num_elems << std::endl;
    std::exit(1);
  }

  Kokkos::initialize (argc, argv);

#ifdef KOKKOS_HAVE_DEFAULT_DEVICE_TYPE_OPENMP
  Kokkos::OpenMP::print_configuration(std::cout,true);
#endif

  run_simulation(num_elems, num_exec, dump_results);

  Kokkos::finalize ();
  return 0;
}
