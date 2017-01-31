#include "TestData.hpp"

namespace TinMan {

Control::Control(int num_elems)
    : m_device_mem("Control information"), m_host_num_elems(num_elems) {
  ExecViewManaged<Real[NP][NP]>::HostMirror dvv_host = Kokkos::create_mirror(m_dvv);
  for (int i = 0; i < NP; ++i) {
    for (int j = 0; j < NP; ++j) {
      static constexpr const Real values[NP][NP] = {
        { -3.0000000000000000, -0.80901699437494745,
          0.30901699437494745, -0.50000000000000000 },
        { 4.0450849718747373,   0.00000000000000000,
          -1.11803398874989490, 1.54508497187473700 },
        { -1.5450849718747370, 1.11803398874989490,
          0.00000000000000000, -4.04508497187473730 },
        { 0.5000000000000000,  -0.30901699437494745,
          0.80901699437494745, 3.000000000000000000 }
      };
      dvv_host(i, j) = values[i][j];
    }
  }

  Kokkos::deep_copy(m_dvv, dvv_host);

  ExecViewManaged<Control_Data[1]>::HostMirror host_mem = Kokkos::create_mirror(m_device_mem);
  host_mem(0).num_elems = num_elems;
  host_mem(0).n0 = 0;
  host_mem(0).np1 = 1;
  host_mem(0).nm1 = 2;
  host_mem(0).qn0 = 0;
  host_mem(0).dt2 = 1.0;
  host_mem(0).ps0 = 1.0;

  Kokkos::deep_copy(m_device_mem, host_mem);

  ExecViewManaged<Real[NUM_LEV_P]>::HostMirror host_hybrid_a = Kokkos::create_mirror(m_hybrid_a);
  for(int i = 0; i < NUM_LEV_P; ++i) {
    host_hybrid_a(i) = 1.0;
  }
  Kokkos::deep_copy(m_hybrid_a, host_hybrid_a);
}

void Control::update_time_levels() {
  int tmp = m_device_mem(0).np1;
  m_device_mem(0).np1 = m_device_mem(0).nm1;
  m_device_mem(0).nm1 = m_device_mem(0).n0;
  m_device_mem(0).n0 = tmp;
}

} // Namespace TinMan
