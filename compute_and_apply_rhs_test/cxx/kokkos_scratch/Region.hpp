#ifndef TINMAN_REGION_HPP
#define TINMAN_REGION_HPP

#include "config.h"

#include "Types.hpp"

#include <Kokkos_Core.hpp>

namespace TinMan {

namespace {
// The number of fields for each dimension
constexpr int NUM_4D_SCALARS = 4;
constexpr int NUM_3D_SCALARS = 5;
constexpr int NUM_2D_SCALARS = 4;
constexpr int NUM_2D_TENSORS = 2;

// Some constexpr for the index of different variables in the views
// 4D Scalars
constexpr int IDX_U    = 0;
constexpr int IDX_V    = 1;
constexpr int IDX_T    = 2;
constexpr int IDX_DP3D = 3;

// 3D Scalars
constexpr int IDX_OMEGA_P     = 0;
constexpr int IDX_PECND       = 1;
constexpr int IDX_PHI         = 2;
constexpr int IDX_DERIVED_UN0 = 3;
constexpr int IDX_DERIVED_VN0 = 4;

// 2D Scalars
constexpr int IDX_FCOR     = 0;
constexpr int IDX_SPHEREMP = 1;
constexpr int IDX_METDET   = 2;
constexpr int IDX_PHIS     = 3;

// 2D Tensors
constexpr int IDX_D    = 0;
constexpr int IDX_DINV = 1;
}

/* Per element data - specific velocity, temperature, pressure, etc. */
class Region {
private:
  /* Contains U, V, T, DP3D */
  ExecViewManaged<Real * [NUM_TIME_LEVELS][NUM_4D_SCALARS][NUM_LEV][NP][NP]>
  m_4d_scalars;
  /* Contains OMEGA_P, PECND, PHI, DERIVED_UN0, DERIVED_VN0, QDP, ETA_DPDN */
  ExecViewManaged<Real * [NUM_3D_SCALARS][NUM_LEV][NP][NP]> m_3d_scalars,
      m_3d_scalars_update;
  /* Contains FCOR, SPHEREMP, METDET, PHIS */
  ExecViewManaged<Real * [NUM_2D_SCALARS][NP][NP]> m_2d_scalars,
      m_2d_scalars_update;
  /* Contains D, DINV */
  ExecViewManaged<Real * [NUM_2D_TENSORS][2][2][NP][NP]> m_2d_tensors,
      m_2d_tensors_update;

  ExecViewManaged<Real * [QSIZE_D][Q_NUM_TIME_LEVELS][NUM_LEV][NP][NP]> m_Qdp,
      m_Qdp_update;

  ExecViewManaged<Real * [NUM_LEV_P][NP][NP]> m_eta_dot_dpdn,
      m_eta_dot_dpdn_update;

  struct TimeLevel_Indices {
    // Current Timelevel
    int n0;
    // Future Timelevel
    int np1;
    // Previous Timelevel
    int nm1;
  } m_timelevels;

public:
  explicit Region(int num_elems);

  KOKKOS_INLINE_FUNCTION
  void next_compute_apply_rhs() {
    swap_views(m_3d_scalars, m_3d_scalars_update);
    swap_views(m_2d_scalars, m_2d_scalars_update);
    swap_views(m_2d_tensors, m_2d_tensors_update);
    swap_views(m_eta_dot_dpdn, m_eta_dot_dpdn_update);

    // Make certain the timelevels are updated as well
    int tmp = m_timelevels.n0;
    m_timelevels.n0 = m_timelevels.np1;
    m_timelevels.np1 = m_timelevels.nm1;
    m_timelevels.nm1 = tmp;
  }

  KOKKOS_INLINE_FUNCTION
  void next_QDP() { swap_views(m_Qdp, m_Qdp_update); }

  void save_state(const struct Control &data) const;

  // v is the tracer we're working with, 0 <= v < QSIZE_D
  // qn0 is the timelevel, 0 <= qn0 < Q_NUM_TIME_LEVELS
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> QDP(int ie, int v, int qn0) const {
    return Kokkos::subview(m_Qdp, ie, v, qn0, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> QDP_update(int ie, int v,
                                                      int qn0) const {
    return Kokkos::subview(m_Qdp_update, ie, v, qn0, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV_P][NP][NP]> ETA_DPDN(int ie) const {
    return Kokkos::subview(m_eta_dot_dpdn, ie, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV_P][NP][NP]> ETA_DPDN_update(int ie) const {
    return Kokkos::subview(m_eta_dot_dpdn_update, ie, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  /* 4D Scalars */
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> U_current(int ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.n0, IDX_U,
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> U_current(int ie, int ilev) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.n0, IDX_U, ilev,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> V_current(int ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.n0, IDX_V,
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> V_current(int ie, int ilev) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.n0, IDX_V, ilev,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> T_current(int ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.n0, IDX_T,
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> DP3D_current(int ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.n0, IDX_DP3D,
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> U_previous(int ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.nm1, IDX_U,
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> U_previous(int ie, int ilev) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.nm1, IDX_U, ilev,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> V_previous(int ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.nm1, IDX_V,
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> V_previous(int ie, int ilev) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.nm1, IDX_V, ilev,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> T_previous(int ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.nm1, IDX_T,
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> DP3D_previous(int ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.nm1, IDX_DP3D,
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> U_future(int ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.np1, IDX_U,
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> U_future(int ie, int ilev) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.np1, IDX_U, ilev,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> V_future(int ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.np1, IDX_V,
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> V_future(int ie, int ilev) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.np1, IDX_V, ilev,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_future(int ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.np1, IDX_T,
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> DP3D_future(int ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.np1, IDX_DP3D,
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

  /* 3D Scalars */
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> OMEGA_P(int ie, int level) const {
    return Kokkos::subview(m_3d_scalars, ie, IDX_OMEGA_P, level, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> PECND(int ie, int level) const {
    return Kokkos::subview(m_3d_scalars, ie, IDX_PECND, level, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> PHI(int ie) const {
    return Kokkos::subview(m_3d_scalars, ie, IDX_PHI, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> DERIVED_UN0(int ie, int level) const {
    return Kokkos::subview(m_3d_scalars, ie, IDX_DERIVED_UN0, level,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> DERIVED_VN0(int ie, int level) const {
    return Kokkos::subview(m_3d_scalars, ie, IDX_DERIVED_VN0, level,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> OMEGA_P_update(int ie, int level) const {
    return Kokkos::subview(m_3d_scalars_update, ie, IDX_OMEGA_P, level,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> PECND_update(int ie, int level) const {
    return Kokkos::subview(m_3d_scalars_update, ie, IDX_PECND, level,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> PHI_update(int ie) const {
    return Kokkos::subview(m_3d_scalars_update, ie, IDX_PHI, Kokkos::ALL,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> DERIVED_UN0_update(int ie, int level) const {
    return Kokkos::subview(m_3d_scalars_update, ie, IDX_DERIVED_UN0, level,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> DERIVED_VN0_update(int ie, int level) const {
    return Kokkos::subview(m_3d_scalars_update, ie, IDX_DERIVED_VN0, level,
                           Kokkos::ALL, Kokkos::ALL);
  }

  /* 2D Scalars */
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> FCOR(int ie) const {
    return Kokkos::subview(m_2d_scalars, ie, IDX_FCOR, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> SPHEREMP(int ie) const {
    return Kokkos::subview(m_2d_scalars, ie, IDX_SPHEREMP, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> METDET(int ie) const {
    return Kokkos::subview(m_2d_scalars, ie, IDX_METDET, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> PHIS(int ie) const {
    return Kokkos::subview(m_2d_scalars, ie, IDX_PHIS, Kokkos::ALL,
                           Kokkos::ALL);
  }

  /* 2D Tensors */
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[2][2][NP][NP]> D(int ie) const {
    return Kokkos::subview(m_2d_tensors, ie, IDX_D, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[2][2][NP][NP]> DINV(int ie) const {
    return Kokkos::subview(m_2d_tensors, ie, IDX_DINV, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> FCOR_update(int ie) const {
    return Kokkos::subview(m_2d_scalars_update, ie, IDX_FCOR, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> SPHEREMP_update(int ie) const {
    return Kokkos::subview(m_2d_scalars_update, ie, IDX_SPHEREMP, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> METDET_update(int ie) const {
    return Kokkos::subview(m_2d_scalars_update, ie, IDX_METDET, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> PHIS_update(int ie) const {
    return Kokkos::subview(m_2d_scalars_update, ie, IDX_PHIS, Kokkos::ALL,
                           Kokkos::ALL);
  }

  /* 2D Tensors */
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[2][2][NP][NP]> D_update(int ie) const {
    return Kokkos::subview(m_2d_tensors_update, ie, IDX_D, Kokkos::ALL,
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[2][2][NP][NP]> DINV_update(int ie) const {
    return Kokkos::subview(m_2d_tensors_update, ie, IDX_DINV, Kokkos::ALL,
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

private:
  template <typename T> KOKKOS_INLINE_FUNCTION void swap_views(T &v1, T &v2) {
    T tmp = v1;
    v1 = v2;
    v2 = tmp;
  }
};

} // TinMan

#endif // TINMAN_REGION_HPP
