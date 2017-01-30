#include "TestData.hpp"

namespace TinMan {

void Control::Control(int num_elems, int num_threads)
    : thread_control("Thread control information", num_threads) {
  constexpr const int elems_per_thread = num_elems / num_threads;
  Kokkos::parallel_for(num_threads, KOKKOS_LAMBDA(int i) {
    thread_control(i).nets = elems_per_thread * i;
    thread_control(i).nete = elems_per_thread * (i + 1);
    thread_control(i).n0 = 0;
    thread_control(i).np1 = 1;
    thread_control(i).nm1 = 2;
    thread_control(i).qn0 = 0;
    thread_control(i).dt2 = 1.0;
  });
}

void Derivative::Derivative() {
  Kokkos::parallel_for(NP * NP, KOKKOS_LAMBDA(int idx) {
    static constexpr const Real values[NP * NP] = {
      -3.0000000000000000,  -0.80901699437494745,  0.30901699437494745,
      -0.50000000000000000,  4.0450849718747373,   0.00000000000000000,
      -1.11803398874989490,  1.54508497187473700, -1.5450849718747370,
       1.11803398874989490,  0.00000000000000000, -4.04508497187473730,
       0.5000000000000000,  -0.30901699437494745,  0.80901699437494745,
       3.000000000000000000
    };
    const int i = idx / NP;
    const int j = idx % NP;
    Dvv_host(i, j) = values[idx];
  });
}

TestData::TestData(const int num_elems) {
  m_control.init_data(num_elems);
  m_hvcoord.init_data();
  m_deriv.init_data();
}

void TestData::update_time_levels() {
  int tmp = m_control.np1;
  m_control.np1 = m_control.nm1;
  m_control.nm1 = m_control.n0;
  m_control.n0 = tmp;
}

} // Namespace TinMan
