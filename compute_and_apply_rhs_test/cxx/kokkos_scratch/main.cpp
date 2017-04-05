#include "compute_and_apply_rhs.hpp"
#include "Region.hpp"
#include "TestData.hpp"
#include "timer.hpp"
#include "Kokkos_Core.hpp"
#include "Types.hpp"

#include <iostream>
#include <cstring>
#include <memory>

#include <cuda_profiler_api.h>

bool is_unsigned_int(const char *str) {
  const size_t len = strlen(str);
  for (size_t i = 0; i < len; ++i) {
    if (!isdigit(str[i])) {
      return false;
    }
  }
  return true;
}

void parse_args(int argc, char **argv, int &num_elems, int &num_exec,
                bool &dump_results, int &threads, int &vectors) {
  if (argc > 1) {
    int iarg = 1;
    while (iarg < argc) {
      if (strncmp(argv[iarg], "--tinman-num-elems=", 18) == 0) {
        char *number = strchr(argv[iarg], '=') + 1;
        if (!is_unsigned_int(number)) {
          std::cerr
              << "Expecting an unsigned integer after '--tinman-num-elems='.\n";
          std::exit(1);
        }

        num_elems = std::atoi(number);

        ++iarg;
        continue;
      } else if (strncmp(argv[iarg], "--tinman-num-exec=", 18) == 0) {
        char *val = strchr(argv[iarg], '=') + 1;
        num_exec = std::stoi(val);

        ++iarg;
        continue;
      } else if (strncmp(argv[iarg], "--tinman-threads=", 17) == 0) {
        char *val = strchr(argv[iarg], '=') + 1;
        threads = std::stoi(val);

        ++iarg;
        continue;
      } else if (strncmp(argv[iarg], "--tinman-vectors=", 17) == 0) {
        char *val = strchr(argv[iarg], '=') + 1;
        vectors = std::stoi(val);

        ++iarg;
        continue;
      } else if (strncmp(argv[iarg], "--tinman-dump-res=", 18) == 0) {
        char *val = strchr(argv[iarg], '=') + 1;
        if (strcmp(val, "yes") == 0 || strcmp(val, "YES") == 0)
          dump_results = true;
        else if (strcmp(val, "no") == 0 || strcmp(val, "NO") == 0)
          dump_results = false;
        else {
          std::cout << " ERROR! Unrecognized command line option '"
                    << argv[iarg] << "'.\n"
                    << "        Run with '--tinman-help' to see the available "
                       "options.\n";

          std::exit(1);
        }

        ++iarg;
        continue;
      } else if (strncmp(argv[iarg], "--tinman-help", 13) == 0) {
        std::cout << "+--------------------------------------------------------"
                     "----------------+\n"
                  << "|                      TinMan command line arguments     "
                     "                |\n"
                  << "+--------------------------------------------------------"
                     "----------------+\n"
                  << "|  --tinman-num-elems=N  : the number of elements "
                     "(default=10)           |\n"
                  << "|  --tinman-dump-res=val : whether to dump results to "
                     "file (default=no)  |\n"
                  << "|  --tinman-num-exec=N   : number of times to execute "
                     "(default=1)        |\n"
                  << "|  --tinman-help         : prints this message           "
                     "                |\n"
                  << "|  --kokkos-help         : prints kokkos help            "
                     "                |\n"
                  << "+--------------------------------------------------------"
                     "----------------+\n";

        std::exit(0);
      }

      ++iarg;
    }
  }
}

void run_simulation(int num_elems, int num_exec, bool dump_results, int threads,
                    int vectors) {
  TinMan::Control data(num_elems);
  TinMan::Region region(num_elems);

  // Burn in before timing to reduce cache effect
  TinMan::compute_and_apply_rhs(data, region, threads, vectors);
  TinMan::ExecSpace::fence();

  std::vector<Timer::Timer> timers(num_exec);
  for (Timer::Timer t : timers) {
    // region.next_compute_apply_rhs();
    t.startTimer();
    TinMan::compute_and_apply_rhs(data, region, threads, vectors);
    t.stopTimer();
  }

  int id = 0;
  for (Timer::Timer t : timers) {
    std::cout << id << "  " << t << "\n";
    id++;
  }
}

int main(int argc, char **argv) {
  int num_elems = 10;
  int num_exec = 1;
  bool dump_results = false;
  int threads = TinMan::Default_Threads_Per_Team;
  int vectors = TinMan::Default_Vectors_Per_Thread;
  parse_args(argc, argv, num_elems, num_exec, dump_results, threads, vectors);

  Kokkos::initialize(argc, argv);

  run_simulation(num_elems, num_exec, dump_results, threads, vectors);

  Kokkos::finalize();
  return 0;
}
