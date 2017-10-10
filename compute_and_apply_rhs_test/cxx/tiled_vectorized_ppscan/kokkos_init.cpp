
#include "Types.hpp"
#include "Control.hpp"
#include "Elements.hpp"
#include "Derivative.hpp"
#include "CaarFunctor.hpp"

#include "profiling.hpp"

#include <iostream>
#include <chrono>

using namespace Homme;

using clock_type = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;

// See https://youtu.be/nXaxk27zwlk?t=2478 for an in depth explanation
// Use escape to force the compiler to pin down a specific piece of memory,
// and force the compiler to allocate it
static __inline__ void escape(void *p) {
  // The volatile tells the compiler some unknowable side-effect is occuring
  // This is used for benchmarking, as this disables the optimizer,
  // preventing it from getting rid of the evaluation entirely
  // This tells the compiler we have touched all of the memory in the program
  // We use the pointer to ensure the compiler has created an address for the data we're concerned about,
  // and that it's not just in registers
  asm volatile("" : : "g"(p) : "memory");
}

static __inline__ void clobber() {
  // Tell the compiler we've magically written to all memory in the program
  asm volatile("" : : : "memory");
}

void init_kokkos(const bool print_configuration = true) {
  /* Make certain profiling is only done for code we're working on */
  profiling_pause();

  /* Set OpenMP Environment variables to control how many
   * threads/processors Kokkos uses */
  Kokkos::initialize();

  ExecSpace::print_configuration(std::cout, print_configuration);
}

template <int v>
void instruction_cache_filler() {
  clobber();
  if(v > 0) {
    instruction_cache_filler<v - 1>();
  }
}

template <>
void instruction_cache_filler<0>() {
  clobber();
}

void flush_cache(std::mt19937_64 &rng, HostViewManaged<Real *> &trash) {
  std::uniform_real_distribution<Real> dist(0.0, 1.0);
  for(int i = 0; i < trash.extent(0); ++i) {
    trash(i) = dist(rng);
  }
  instruction_cache_filler<512>();
}

void finalize_kokkos() { Kokkos::finalize(); }

int main(int argc, char **argv) {
  constexpr int tstep = 600;

  init_kokkos();

  constexpr const int threads_per_team = 4;
  constexpr const int vectors_per_thread = 1;

  std::random_device rd;
  std::mt19937_64 rng(rd());

  Control data;
  data.nm1 = 0;
  data.n0 = 1;
  data.np1 = 2;
  data.qn0 = -1;
  data.dt = tstep;
  data.ps0 = 1.0;
  data.eta_ave_w = 1.0;
  data.hybrid_a = ExecViewManaged<Real[NUM_LEV_P]>("Hybrid coordinates; translates between pressure and velocity");
  HostViewManaged<Real[NUM_LEV_P]> hybrid_a_host = Kokkos::create_mirror_view(data.hybrid_a);
  std::uniform_real_distribution<Real> dist(1.0, 2.0);
  for(int i = 0; i < NUM_LEV_P; ++i) {
    hybrid_a_host(i) = dist(rng);
  }
  Kokkos::deep_copy(data.hybrid_a, hybrid_a_host);

  int num_elems = 32;
  if(argc > 1) {
    num_elems = atoi(argv[1]);
  }

  Elements elem;
  elem.random_init(num_elems, rng);

  Derivative deriv;
  deriv.random_init(rng);

  constexpr int seconds_per_day = 24 * 3600;
  constexpr int rk_stages = 5;
  int num_exec = (seconds_per_day / tstep) * rk_stages;

  if(argc > 2) {
    num_exec = atoi(argv[2]);
  }

  // Create the functor
  CaarFunctor func(data, elem, deriv);

  constexpr int kb_size = 1024;
  constexpr int doubles_per_kb = kb_size / sizeof(double);
  constexpr int doubles_per_mb = doubles_per_kb * 1024;

  HostViewManaged<Real *> trash("trash cache filler", 4 * doubles_per_mb);

  {
    // Setup the policy
    Kokkos::TeamPolicy<ExecSpace> policy(num_elems, threads_per_team, vectors_per_thread);
    policy.set_chunk_size(1);

    std::vector<clock_type::time_point> start_times(num_exec);
    std::vector<clock_type::time_point> end_times(num_exec);

    for(int exec = 0; exec < num_exec; ++exec) {
      auto start = clock_type::now();
      ExecSpace::fence();
      Kokkos::parallel_for(policy, func);
      escape(elem.m_u.data());
      escape(elem.m_v.data());
      escape(elem.m_t.data());
      escape(elem.m_omega_p.data());
      ExecSpace::fence();
      auto end = clock_type::now();
      start_times[exec] = start;
      end_times[exec] = end;
      flush_cache(rng, trash);
    }

    clobber();

    clock_type::duration total_time = end_times[0] - start_times[0];
    for(int exec = 1; exec < num_exec; ++exec) {
      total_time += end_times[exec] - start_times[exec];
    }

    auto count = std::chrono::duration_cast<ns>(total_time).count();
    std::cout << "Seconds " << count * 1e-9 << " to evaluate " << num_elems << " elements " << num_exec << " times\n";
  }

  {
    // Setup the policy
    Kokkos::TeamPolicy<ExecSpace, CaarFunctor::EmptyTag> policy(num_elems, threads_per_team, vectors_per_thread);
    policy.set_chunk_size(1);

    auto start = clock_type::now();
    for(int exec = 0; exec < num_exec; ++exec) {
      ExecSpace::fence();
      Kokkos::parallel_for(policy, func);
    }
    auto end = clock_type::now();
  
    auto count = std::chrono::duration_cast<ns>(end - start).count();
    std::cout << "Seconds " << count * 1e-9 << " to fake evaluate " << num_elems << " elements " << num_exec << " times\n";
  }

  {
    auto start = clock_type::now();
    for(int exec = 0; exec < num_exec; ++exec) {
#pragma omp parallel for
      for(int i = 0; i < num_elems; ++i) {
        if(exec > 100000) {
          printf("argh");
        }
      }
    }
    auto end = clock_type::now();
  
    auto count = std::chrono::duration_cast<ns>(end - start).count();
    std::cout << "Seconds " << count * 1e-9 << " to fake evaluate " << num_elems << " elements " << num_exec << " times with pure OpenMP\n";
    
  }

  finalize_kokkos();
}

