#include "Region.hpp"
#include "TestData.hpp"

namespace TinMan {

double init_map(double x, int n) { return std::pow(sin(n * x), 2); }

Region::Region(int num_elems)
    : m_2d_scalars("2d scalars", num_elems),
      m_2d_tensors("2d tensors", num_elems),
      m_3d_scalars("3d scalars", num_elems),
      m_4d_scalars("4d scalars", num_elems), m_Qdp("qdp", num_elems),
      m_eta_dot_dpdn("eta_dot_dpdn", num_elems),
      m_2d_scalars_update("2d scalars_update", num_elems),
      m_2d_tensors_update("2d tensors_update", num_elems),
      m_3d_scalars_update("3d scalars_update", num_elems),
      m_Qdp_update("qdp", num_elems),
      m_eta_dot_dpdn_update("eta_dot_dpdn", num_elems),
      m_timelevels({ 0, 1, 2 }) {
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

  // Initialize arrays using sin^2(n*x) map.
  // This is easily portable across different platforms and/or
  // languages without relying on implementation details

  // Set seed for the init map
  constexpr Real x = 0.123456789;

  int n = 1;
  // Now fill all the arrays
  for (int ie = 0; ie < num_elems; ++ie) {
    for (int igp = 0; igp < NP; ++igp) {
      for (int jgp = 0; jgp < NP; ++jgp) {
        // Initializing h_2d_tensors and h_2d_scalars
        h_2d_tensors(ie, IDX_D, 0, 0, igp, jgp) = init_map(x, n++);
        h_2d_tensors(ie, IDX_D, 0, 1, igp, jgp) = init_map(x, n++);
        h_2d_tensors(ie, IDX_D, 1, 0, igp, jgp) = init_map(x, n++);
        h_2d_tensors(ie, IDX_D, 1, 1, igp, jgp) = init_map(x, n++);

        Real detD = h_2d_tensors(ie, IDX_D, 0, 0, igp, jgp) *
                        h_2d_tensors(ie, IDX_D, 1, 0, igp, jgp) -
                    h_2d_tensors(ie, IDX_D, 0, 1, igp, jgp) *
                        h_2d_tensors(ie, IDX_D, 1, 1, igp, jgp);

        h_2d_tensors(ie, IDX_DINV, 0, 0, igp, jgp) =
            h_2d_tensors(ie, IDX_D, 1, 1, igp, jgp) / detD;
        h_2d_tensors(ie, IDX_DINV, 0, 1, igp, jgp) =
            -h_2d_tensors(ie, IDX_D, 0, 1, igp, jgp) / detD;
        h_2d_tensors(ie, IDX_DINV, 1, 0, igp, jgp) =
            -h_2d_tensors(ie, IDX_D, 1, 0, igp, jgp) / detD;
        h_2d_tensors(ie, IDX_DINV, 1, 1, igp, jgp) =
            h_2d_tensors(ie, IDX_D, 0, 0, igp, jgp) / detD;

        h_2d_scalars(ie, IDX_FCOR, igp, jgp) = init_map(x, n++);
        h_2d_scalars(ie, IDX_SPHEREMP, igp, jgp) = init_map(x, n++);
        h_2d_scalars(ie, IDX_METDET, igp, jgp) = init_map(x, n++);
        h_2d_scalars(ie, IDX_PHIS, igp, jgp) = init_map(x, n++);

        // Initializing arrays that contain [NUM_LEV]
        for (int il = 0; il < NUM_LEV; ++il) {
          // h_3d_scalars
          h_3d_scalars(ie, IDX_OMEGA_P, il, igp, jgp) = init_map(x, n++);
          h_3d_scalars(ie, IDX_PECND, il, igp, jgp) = init_map(x, n++);
          h_3d_scalars(ie, IDX_PHI, il, igp, jgp) = init_map(x, n++);

          h_3d_scalars(ie, IDX_DERIVED_UN0, il, igp, jgp) = init_map(x, n++);
          h_3d_scalars(ie, IDX_DERIVED_VN0, il, igp, jgp) = init_map(x, n++);

          // Initializing h_Qdp
          for (int iq = 0; iq < QSIZE_D; ++iq) {
            // 0 <= ie < num_elems
            // 0 <= il < NUM_LEV
            // 0 <= iq < QSIZE_D
            // 0 <= igp < NP
            // 0 <= jgp < NP
            // NUM_ELEMS, QSIZE_D, 2, NUM_LEV, NP, NP
            h_Qdp(ie, iq, 0, il, igp, jgp) = init_map(x, n++);
            h_Qdp(ie, iq, 1, il, igp, jgp) = init_map(x, n++);
          }

          // Initializing arrays that contain [NUM_TIME_LEVELS]
          for (int it = 0; it < NUM_TIME_LEVELS; ++it) {
            // Initializing h_element_states
            h_4d_scalars(ie, it, IDX_U, il, igp, jgp) = init_map(x, n++);
            h_4d_scalars(ie, it, IDX_V, il, igp, jgp) = init_map(x, n++);
            h_4d_scalars(ie, it, IDX_T, il, igp, jgp) = init_map(x, n++);
            h_4d_scalars(ie, it, IDX_DP3D, il, igp, jgp) = init_map(x, n++);
          }
        }

        // Initializing h_eta_dot_dpdn
        for (int il = 0; il < NUM_LEV_P; ++il) {
          h_eta_dot_dpdn(ie, il, igp, jgp) = init_map(x, n++);
        }
      }
    }
  }

  Kokkos::deep_copy(m_2d_scalars, h_2d_scalars);
  Kokkos::deep_copy(m_2d_scalars_update, m_2d_scalars);
  Kokkos::deep_copy(m_2d_tensors, h_2d_tensors);
  Kokkos::deep_copy(m_2d_tensors_update, m_2d_tensors);
  Kokkos::deep_copy(m_3d_scalars, h_3d_scalars);
  Kokkos::deep_copy(m_3d_scalars_update, m_3d_scalars);
  Kokkos::deep_copy(m_4d_scalars, h_4d_scalars);
  Kokkos::deep_copy(m_Qdp, h_Qdp);
  Kokkos::deep_copy(m_Qdp_update, m_Qdp);
  Kokkos::deep_copy(m_eta_dot_dpdn, h_eta_dot_dpdn);
  Kokkos::deep_copy(m_eta_dot_dpdn_update, m_eta_dot_dpdn);
}

void Region::save_state(const Control &data) const {
  // Input parameters

  auto file_opener = [&](std::ofstream &file, const char *fname) {
    file.open(fname);
    if (!file.is_open()) {
      std::cout << "Error! Cannot open '" << fname << "'.\n";
      std::abort();
    }
    file.precision(17);
  };

  std::ofstream vxfile, vyfile, tfile, dpfile;
  file_opener(vxfile, "elem_state_vx.txt");
  file_opener(vyfile, "elem_state_vy.txt");
  file_opener(tfile, "elem_state_t.txt");
  file_opener(dpfile, "elem_state_dp3d.txt");

  for (int ie = 0; ie < data.num_elems(); ++ie) {
    for (int ilev = 0; ilev < NUM_LEV; ++ilev) {
      vxfile << "[" << ie << ", " << ilev << "]\n";
      vyfile << "[" << ie << ", " << ilev << "]\n";
      tfile << "[" << ie << ", " << ilev << "]\n";
      dpfile << "[" << ie << ", " << ilev << "]\n";

      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {
          vxfile << " " << U_future(ie)(ilev, igp, jgp);
          vyfile << " " << V_future(ie)(ilev, igp, jgp);
          tfile << " " << T_future(ie)(ilev, igp, jgp);
          dpfile << " " << DP3D_future(ie)(ilev, igp, jgp);
        }
        vxfile << "\n";
        vyfile << "\n";
        tfile << "\n";
        dpfile << "\n";
      }
    }
  }

  // Closing files
  vxfile.close();
  vyfile.close();
  tfile.close();
  dpfile.close();
}

} // namespace TinMan
