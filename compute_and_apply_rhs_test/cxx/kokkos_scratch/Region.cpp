#include "Region.hpp"

namespace TinMan {

Region::Region(int num_elems)
    : m_2d_scalars("2d scalars", num_elems),
      m_2d_tensors("2d tensors", num_elems),
      m_3d_scalars("3d scalars", num_elems),
      m_4d_scalars("4d scalars", num_elems), m_Qdp("qdp", num_elems),
      m_eta_dot_dpdn("eta_dot_dpdn", num_elems) {
  ExecViewManaged<Real *[NUM_2D_SCALARS][NP][NP]>::HostMirror h_2d_scalars =
      Kokkos::create_mirror_view(m_2d_scalars);
  ExecViewManaged<Real *[NUM_2D_TENSORS][2][2][NP][NP]>::HostMirror
  h_2d_tensors = Kokkos::create_mirror_view(m_2d_tensors);

  ExecViewManaged<Real *[NUM_3D_SCALARS][NUM_LEV][NP][NP]>::HostMirror
  h_3d_scalars = Kokkos::create_mirror_view(m_3d_scalars);

  ExecViewManaged<Real *[NUM_TIME_LEVELS][NUM_4D_SCALARS][NUM_LEV][NP]
                                         [NP]>::HostMirror h_4d_scalars =
      Kokkos::create_mirror_view(m_4d_scalars);

  ExecViewManaged<Real *[QSIZE_D][2][NUM_LEV][NP][NP]>::HostMirror h_Qdp =
      Kokkos::create_mirror_view(m_Qdp);

  ExecViewManaged<Real *[NUM_LEV_P][NP][NP]>::HostMirror h_eta_dot_dpdn =
      Kokkos::create_mirror_view(m_eta_dot_dpdn);

  // Now fill all the arrays
  for (int ie = 0; ie < num_elems; ++ie) {
    for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {

        double iie = ie + 1;
        double iip = igp + 1;
        double jjp = jgp + 1;

        // Initializing h_2d_tensors and h_2d_scalars
        h_2d_tensors(ie, IDX_D, 0, 0, igp, jgp) = 1.0;
        h_2d_tensors(ie, IDX_D, 0, 1, igp, jgp) = 0.0;
        h_2d_tensors(ie, IDX_D, 1, 0, igp, jgp) = 0.0;
        h_2d_tensors(ie, IDX_D, 1, 1, igp, jgp) = 2.0;

        h_2d_tensors(ie, IDX_DINV, 0, 0, igp, jgp) = 1.0;
        h_2d_tensors(ie, IDX_DINV, 0, 1, igp, jgp) = 0.0;
        h_2d_tensors(ie, IDX_DINV, 1, 0, igp, jgp) = 0.0;
        h_2d_tensors(ie, IDX_DINV, 1, 1, igp, jgp) = 0.5;

        h_2d_scalars(ie, IDX_FCOR, igp, jgp)     = sin(iip + jjp);
        h_2d_scalars(ie, IDX_METDET, igp, jgp)   = iip*jjp;
        h_2d_scalars(ie, IDX_PHIS, igp, jgp)     = iip + jjp;
        h_2d_scalars(ie, IDX_SPHEREMP, igp, jgp) = 2*iip;

        // Initializing arrays that contain [NUM_LEV]
        for (int il = 0; il < NUM_LEV; ++il) {
          double iil = il + 1;

          // h_3d_scalars
          h_3d_scalars(ie, IDX_PHI, il, igp, jgp)     = cos(iip + 3*jjp) + iil;
          h_3d_scalars(ie, IDX_UN0, il, igp, jgp)     = 1.0;
          h_3d_scalars(ie, IDX_VN0, il, igp, jgp)     = 1.0;
          h_3d_scalars(ie, IDX_PECND, il, igp, jgp)   = 1.0;
          h_3d_scalars(ie, IDX_OMEGA_P, il, igp, jgp) = jjp*jjp;

          // Initializing h_Qdp
          h_Qdp(ie, 0, 0, il, igp, jgp) = 1.0 + sin(iip*jjp*iil);

          // Initializing arrays that contain [NUM_TIME_LEVELS]
          for (int it = 0; it < NUM_TIME_LEVELS; ++it) {
            double iit = it + 1;
            // Initializing h_element_states
            h_4d_scalars(ie, it, IDX_DP3D, il, igp, jgp) = 10.0*iil + iie + iip + jjp + iit;
            h_4d_scalars(ie, it, IDX_U, il, igp, jgp)    = 1.0 + 0.5*iil + iip + jjp + 0.2*iie + 2.0*iit;
            h_4d_scalars(ie, it, IDX_V, il, igp, jgp)    = 1.0 + 0.5*iil + iip + jjp + 0.2*iie + 3.0*iit;
            h_4d_scalars(ie, it, IDX_T, il, igp, jgp)    = 1000.0 - iil - iip - jjp +0.1*iie + iit;
          }
        }

        // Initializing h_eta_dot_dpdn
        for (int il = 0; il < NUM_LEV_P; ++il) {
          h_eta_dot_dpdn(ie, il, igp, jgp) = 0.0;
        }
      }
    }
  }

  Kokkos::deep_copy(m_2d_scalars, h_2d_scalars);
  Kokkos::deep_copy(m_2d_tensors, h_2d_tensors);
  Kokkos::deep_copy(m_3d_scalars, h_3d_scalars);
  Kokkos::deep_copy(m_4d_scalars, h_4d_scalars);
  Kokkos::deep_copy(m_Qdp, m_Qdp);
  Kokkos::deep_copy(m_eta_dot_dpdn, m_eta_dot_dpdn);
}

} // namespace TinMan
