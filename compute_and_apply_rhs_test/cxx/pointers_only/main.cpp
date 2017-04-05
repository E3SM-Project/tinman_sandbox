#include "data_structures.hpp"
#include "compute_and_apply_rhs.hpp"
#include "timer.hpp"

#include <iostream>
#include <cstring>
#include <vector>

namespace Homme
{
int num_elems = 10;
}

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
  using namespace Homme;

  bool dump_res = false;
  int num_exec = 1;

  if (argc > 1) {
    int iarg = 1;
    while (iarg<argc)
    {
      if (strncmp(argv[iarg],"--tinman-num-elems=",18) == 0)
      {
        char* number =  strchr(argv[iarg],'=')+1;
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

  TestData data;

  data.init_data();

  // Burn in to avoid cache effects
  compute_and_apply_rhs(data);

  std::vector<Timer::Timer> timers(num_exec);
  for (int i=0; i<num_exec; ++i)
  {
    timers[i].startTimer();
    compute_and_apply_rhs(data);
    //data.update_time_levels();
    timers[i].stopTimer();
  }

  for(int i = 0; i < num_exec; ++i) {
    std::cout << i << "  " << timers[i] << std::endl;
  }

  data.cleanup_data ();

  return 0;
}
