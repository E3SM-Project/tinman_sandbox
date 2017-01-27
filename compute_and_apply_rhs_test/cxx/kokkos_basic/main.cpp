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
        if (strcmp(val,"yes")==0 || strcmp(val,"YES")==0)
          dump_res = true;
        else if (strcmp(val,"no")==0 || strcmp(val,"NO")==0)
          dump_res = false;
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

  if (num_elems < 1) {
    std::cerr << "Invalid number of elements: " << num_elems << std::endl;
    std::exit(1);
  }

  Kokkos::initialize (argc, argv);
  Kokkos::OpenMP::print_configuration(std::cout,true);

  std::cout << " --- Initializing data...\n";
  TinMan::TestData data(num_elems);
  TinMan::Region* region = new TinMan::Region(num_elems); // A pointer, so the views are destryed before Kokkos::finalize

  // Print norm of initial states, to check we are using same data in all tests
  print_results_2norm (data, *region);

  // Burn in before timing to reduce cache effect
  TinMan::compute_and_apply_rhs(data,*region);

  std::cout << " --- Performing computations... (" << num_exec << " executions of the main loop on " << num_elems << " elements)\n";

  std::unique_ptr<Timer::Timer[]> timers(new Timer::Timer[num_exec]);
  Timer::Timer global_timer;
  for (int i=0; i<num_exec; ++i)
  {
    global_timer.startTimer();
    timers[i].startTimer();
    TinMan::compute_and_apply_rhs(data,*region);
    timers[i].stopTimer();
    global_timer.stopTimer();
  }

  std::cout << "   ---> individual executions times:\n";
  for(int i = 0; i < num_exec; ++i) {
    std::cout << timers[i] << std::endl;
  }
  std::cout << "   ---> compute_and_apply_rhs execution total time: " << global_timer << "\n";

  print_results_2norm (data,*region);

  if (dump_res)
  {
    std::cout << " --- Dumping results to file...\n";
    dump_results_to_file (data,*region);
  }

  std::cout << " --- Cleaning up data...\n";
  delete region;

  Kokkos::finalize ();
  return 0;
}
