#include "TestData.hpp"

namespace TinMan {

Control::Control(int num_elems)
    : m_num_elems(num_elems), m_qn0(0), m_dt2(1.0), m_ps0(10.0),
      m_hybrid_a(
          "Hybrid coordinates; translates between pressure and velocity"),
      m_dvv("Laplacian"), m_pressure("Pressure", num_elems), m_T_v("Vertical Temperature", num_elems), m_div_vdp("Global array scratch", num_elems), m_vector_buf("Vector buffer for vectors on each level", num_elems) {
  ExecViewManaged<Real[NP][NP]>::HostMirror dvv_host =
      Kokkos::create_mirror_view(m_dvv);
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
      dvv_host(i, j) = values[j][i];
    }
  }

  Kokkos::deep_copy(m_dvv, dvv_host);

  ExecViewManaged<Real[NUM_LEV_P]>::HostMirror host_hybrid_a =
      Kokkos::create_mirror_view(m_hybrid_a);
  for (int i = 0; i < NUM_LEV_P; ++i) {
    host_hybrid_a(i) = NUM_LEV + 1.0 - i;
  }
  Kokkos::deep_copy(m_hybrid_a, host_hybrid_a);
}

} // Namespace TinMan
