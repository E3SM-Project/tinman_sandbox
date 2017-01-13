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

  TestData data;

  data.init_data();

  // Burn in to avoid cache effects
  compute_and_apply_rhs(data);

  std::vector<Timer::Timer> timers(num_exec);
  for (int i=0; i<num_exec; ++i)
  {
    timers[i].startTimer();
    compute_and_apply_rhs(data);
    timers[i].stopTimer();
  }

  for(int i = 0; i < num_exec; ++i) {
    std::cout << timers[i] << std::endl;
  }

  if (dump_res)
  {
    dump_results_to_file (data);
  }

  data.cleanup_data ();

  return 0;
}
