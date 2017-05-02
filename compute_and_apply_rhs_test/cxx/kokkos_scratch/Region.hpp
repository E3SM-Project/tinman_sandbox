#ifndef TINMAN_REGION_HPP
#define TINMAN_REGION_HPP

#include "Dimensions.hpp"

#include "Types.hpp"

#include <Kokkos_Core.hpp>

namespace TinMan {

/* Per element data - specific velocity, temperature, pressure, etc. */
class Region {
private:
  enum {
    // The number of fields for each dimension
    NUM_4D_SCALARS = 4,
    NUM_3D_SCALARS = 5,
    NUM_2D_SCALARS = 4,
    NUM_2D_TENSORS = 2,

    // Some constexpr for the index of different variables in the views
    // 4D Scalars
    IDX_U = 0,
    IDX_V = 1,
    IDX_T = 2,
    IDX_DP3D = 3,

    // 3D Scalars
    IDX_OMEGA_P = 0,
    IDX_PECND = 1,
    IDX_PHI = 2,
    IDX_DERIVED_UN0 = 3,
    IDX_DERIVED_VN0 = 4,

    // 2D Scalars
    IDX_FCOR = 0,
    IDX_SPHEREMP = 1,
    IDX_METDET = 2,
    IDX_PHIS = 3,

    // 2D Tensors
    IDX_D = 0,
    IDX_DINV = 1,
  };

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
  ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> QDP(const int &ie, int v,
                                                     int qn0) const {
    return Kokkos::subview(m_Qdp, ie, v, qn0, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]>
  QDP_update(const int &ie, const int &v, const int &qn0) const {
    return Kokkos::subview(m_Qdp_update, ie, v, qn0, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real QDP(const int &ie, const int &v, const int &qn0, const int &ilev,
           const int &igp, const int &jgp) const {
    return m_Qdp(ie, v, qn0, ilev, igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  Real &QDP_update(const int &ie, const int &v, const int &qn0, const int &ilev,
                   const int &igp, const int &jgp) const {
    return m_Qdp_update(ie, v, qn0, ilev, igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV_P][NP][NP]>
  ETA_DPDN(const int &ie) const {
    return Kokkos::subview(m_eta_dot_dpdn, ie, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV_P][NP][NP]>
  ETA_DPDN_update(const int &ie) const {
    return Kokkos::subview(m_eta_dot_dpdn_update, ie, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real ETA_DPDN(const int &ie, const int &ilev, const int &igp,
                const int &jgp) const {
    return m_eta_dot_dpdn(ie, ilev, igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  Real &ETA_DPDN_update(const int &ie, const int &ilev, const int &igp,
                        const int &jgp) const {
    return m_eta_dot_dpdn_update(ie, ilev, igp, jgp);
  }

  /* 4D Scalars */
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>
  U_current(const int &ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.n0,
                           static_cast<int>(IDX_U), Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> U_current(const int &ie,
                                                  const int &ilev) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.n0,
                           static_cast<int>(IDX_U), ilev, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real U_current(const int &ie, const int &ilev, const int &igp,
                 const int &jgp) const {
    return m_4d_scalars(ie, m_timelevels.n0, static_cast<int>(IDX_U), ilev, igp,
                        jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>
  V_current(const int &ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.n0,
                           static_cast<int>(IDX_V), Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> V_current(const int &ie,
                                                  const int &ilev) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.n0,
                           static_cast<int>(IDX_V), ilev, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real V_current(const int &ie, const int &ilev, const int &igp,
                 const int &jgp) const {
    return m_4d_scalars(ie, m_timelevels.n0, static_cast<int>(IDX_V), ilev, igp,
                        jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>
  T_current(const int &ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.n0,
                           static_cast<int>(IDX_T), Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real T_current(const int &ie, const int &ilev, const int &igp,
                 const int &jgp) const {
    return m_4d_scalars(ie, m_timelevels.n0, static_cast<int>(IDX_T), ilev, igp,
                        jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>
  DP3D_current(const int &ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.n0,
                           static_cast<int>(IDX_DP3D), Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real DP3D_current(const int &ie, const int &ilev, const int &igp,
                    const int &jgp) const {
    return m_4d_scalars(ie, m_timelevels.n0, static_cast<int>(IDX_DP3D), ilev,
                        igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>
  U_previous(const int &ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.nm1,
                           static_cast<int>(IDX_U), Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real U_previous(const int &ie, const int &ilev, const int &igp,
                  const int &jgp) const {
    return m_4d_scalars(ie, m_timelevels.nm1, static_cast<int>(IDX_U), ilev,
                        igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> U_previous(const int &ie,
                                                   const int &ilev) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.nm1,
                           static_cast<int>(IDX_U), ilev, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>
  V_previous(const int &ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.nm1,
                           static_cast<int>(IDX_V), Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> V_previous(const int &ie,
                                                   const int &ilev) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.nm1,
                           static_cast<int>(IDX_V), ilev, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real V_previous(const int &ie, const int &ilev, const int &igp,
                  const int &jgp) const {
    return m_4d_scalars(ie, m_timelevels.nm1, static_cast<int>(IDX_V), ilev,
                        igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>
  T_previous(const int &ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.nm1,
                           static_cast<int>(IDX_T), Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real T_previous(const int &ie, const int &ilev, const int &igp,
                  const int &jgp) const {
    return m_4d_scalars(ie, m_timelevels.nm1, static_cast<int>(IDX_T), ilev,
                        igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]>
  DP3D_previous(const int &ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.nm1,
                           static_cast<int>(IDX_DP3D), Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real DP3D_previous(const int &ie, const int &ilev, const int &igp,
                     const int &jgp) const {
    return m_4d_scalars(ie, m_timelevels.nm1, static_cast<int>(IDX_DP3D), ilev,
                        igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> U_future(const int &ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.np1,
                           static_cast<int>(IDX_U), Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> U_future(const int &ie,
                                           const int &ilev) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.np1,
                           static_cast<int>(IDX_U), ilev, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real &U_future(const int &ie, const int &ilev, const int &igp,
                 const int &jgp) const {
    return m_4d_scalars(ie, m_timelevels.np1, static_cast<int>(IDX_U), ilev,
                        igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> V_future(const int &ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.np1,
                           static_cast<int>(IDX_V), Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> V_future(const int &ie,
                                           const int &ilev) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.np1,
                           static_cast<int>(IDX_V), ilev, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real &V_future(const int &ie, const int &ilev, const int &igp,
                 const int &jgp) const {
    return m_4d_scalars(ie, m_timelevels.np1, static_cast<int>(IDX_V), ilev,
                        igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_future(const int &ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.np1,
                           static_cast<int>(IDX_T), Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real &T_future(const int &ie, const int &ilev, const int &igp,
                 const int &jgp) const {
    return m_4d_scalars(ie, m_timelevels.np1, static_cast<int>(IDX_T), ilev,
                        igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> DP3D_future(const int &ie) const {
    return Kokkos::subview(m_4d_scalars, ie, m_timelevels.np1,
                           static_cast<int>(IDX_DP3D), Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real &DP3D_future(const int &ie, const int &ilev, const int &igp,
                    const int &jgp) const {
    return m_4d_scalars(ie, m_timelevels.np1, static_cast<int>(IDX_DP3D), ilev,
                        igp, jgp);
  }

  /* 3D Scalars */
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> OMEGA_P(const int &ie,
                                                const int &ilev) const {
    return Kokkos::subview(m_3d_scalars, ie, static_cast<int>(IDX_OMEGA_P),
                           ilev, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real &OMEGA_P(const int &ie, const int &ilev, const int &igp,
                const int &jgp) const {
    return m_3d_scalars(ie, static_cast<int>(IDX_OMEGA_P), ilev, igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> PECND(const int &ie,
                                              const int &ilev) const {
    return Kokkos::subview(m_3d_scalars, ie, static_cast<int>(IDX_PECND), ilev,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real &PECND(const int &ie, const int &ilev, const int &igp,
              const int &jgp) const {
    return m_3d_scalars(ie, static_cast<int>(IDX_PECND), ilev, igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> PHI(const int &ie) const {
    return Kokkos::subview(m_3d_scalars, ie, static_cast<int>(IDX_PHI),
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real PHI(const int &ie, const int &ilev, const int &igp,
           const int &jgp) const {
    return m_3d_scalars(ie, static_cast<int>(IDX_PHI), ilev, igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> DERIVED_UN0(const int &ie,
                                                    const int &level) const {
    return Kokkos::subview(m_3d_scalars, ie, static_cast<int>(IDX_DERIVED_UN0),
                           level, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real DERIVED_UN0(const int &ie, const int &level, const int &igp,
                   const int &jgp) const {
    return m_3d_scalars(ie, static_cast<int>(IDX_DERIVED_UN0), level, igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> DERIVED_VN0(const int &ie,
                                                    const int &level) const {
    return Kokkos::subview(m_3d_scalars, ie, static_cast<int>(IDX_DERIVED_VN0),
                           level, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real DERIVED_VN0(const int &ie, const int &level, const int &igp,
                   const int &jgp) const {
    return m_3d_scalars(ie, static_cast<int>(IDX_DERIVED_VN0), level, igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  Real &DERIVED_UN0_update(const int &ie, const int &level, const int &igp,
                           const int &jgp) const {
    return m_3d_scalars_update(ie, static_cast<int>(IDX_DERIVED_UN0), level,
                               igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> DERIVED_UN0_update(const int &ie,
                                                     const int &level) const {
    return Kokkos::subview(m_3d_scalars_update, ie,
                           static_cast<int>(IDX_DERIVED_UN0), level,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real &DERIVED_VN0_update(const int &ie, const int &level, const int &igp,
                           const int &jgp) const {
    return m_3d_scalars_update(ie, static_cast<int>(IDX_DERIVED_VN0), level,
                               igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> DERIVED_VN0_update(const int &ie,
                                                     const int &level) const {
    return Kokkos::subview(m_3d_scalars_update, ie,
                           static_cast<int>(IDX_DERIVED_VN0), level,
                           Kokkos::ALL, Kokkos::ALL);
  }

  /* 2D Scalars */
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> FCOR(const int &ie) const {
    return Kokkos::subview(m_2d_scalars, ie, static_cast<int>(IDX_FCOR),
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real FCOR(const int &ie, const int &igp, const int &jgp) const {
    return m_2d_scalars(ie, static_cast<int>(IDX_FCOR), igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> SPHEREMP(const int &ie) const {
    return Kokkos::subview(m_2d_scalars, ie, static_cast<int>(IDX_SPHEREMP),
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real SPHEREMP(const int &ie, const int &igp, const int &jgp) const {
    return m_2d_scalars(ie, static_cast<int>(IDX_SPHEREMP), igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> METDET(const int &ie) const {
    return Kokkos::subview(m_2d_scalars, ie, static_cast<int>(IDX_METDET),
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[NP][NP]> PHIS(const int &ie) const {
    return Kokkos::subview(m_2d_scalars, ie, static_cast<int>(IDX_PHIS),
                           Kokkos::ALL, Kokkos::ALL);
  }

  /* 2D Tensors */
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[2][2][NP][NP]> D(const int &ie) const {
    return Kokkos::subview(m_2d_tensors, ie, static_cast<int>(IDX_D),
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<const Real[2][2][NP][NP]> DINV(const int &ie) const {
    return Kokkos::subview(m_2d_tensors, ie, static_cast<int>(IDX_DINV),
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> FCOR_update(const int &ie) const {
    return Kokkos::subview(m_2d_scalars_update, ie, static_cast<int>(IDX_FCOR),
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  Real &FCOR_update(const int &ie, const int &igp, const int &jgp) const {
    return m_2d_scalars_update(ie, static_cast<int>(IDX_FCOR), igp, jgp);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> SPHEREMP_update(const int &ie) const {
    return Kokkos::subview(m_2d_scalars_update, ie,
                           static_cast<int>(IDX_SPHEREMP), Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> METDET_update(const int &ie) const {
    return Kokkos::subview(m_2d_scalars_update, ie,
                           static_cast<int>(IDX_METDET), Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> PHIS_update(const int &ie) const {
    return Kokkos::subview(m_2d_scalars_update, ie, static_cast<int>(IDX_PHIS),
                           Kokkos::ALL, Kokkos::ALL);
  }

  /* 2D Tensors */
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[2][2][NP][NP]> D_update(const int &ie) const {
    return Kokkos::subview(m_2d_tensors_update, ie, static_cast<int>(IDX_D),
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[2][2][NP][NP]> DINV_update(const int &ie) const {
    return Kokkos::subview(m_2d_tensors_update, ie, static_cast<int>(IDX_DINV),
                           Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
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
