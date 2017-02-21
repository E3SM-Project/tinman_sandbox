#ifndef TINMAN_REGION_HPP
#define TINMAN_REGION_HPP

#include "config.h"

#include "Types.hpp"

#include <Kokkos_Core.hpp>

namespace TinMan {

// The number of fields for each dimension
constexpr const int NUM_4D_SCALARS = 4;
constexpr const int NUM_3D_SCALARS = 5;
constexpr const int NUM_2D_SCALARS = 4;
constexpr const int NUM_2D_TENSORS = 2;

// Some constexpr for the index of different variables in the views
// 4D Scalars
constexpr const int IDX_U = 0;
constexpr const int IDX_V = 1;
constexpr const int IDX_T = 2;
constexpr const int IDX_DP3D = 3;

// 3D Scalars
constexpr const int IDX_OMEGA_P = 0;
constexpr const int IDX_PECND = 1;
constexpr const int IDX_PHI = 2;
constexpr const int IDX_UN0 = 3;
constexpr const int IDX_VN0 = 4;

// 2D Scalars
constexpr const int IDX_FCOR = 0;
constexpr const int IDX_SPHEREMP = 1;
constexpr const int IDX_METDET = 2;
constexpr const int IDX_PHIS = 3;

// 2D Tensors
constexpr const int IDX_D = 0;
constexpr const int IDX_DINV = 1;

/* Per element data - specific velocity, temperature, pressure, etc. */
class Region {
private:

  /* Contains U, V, T, DP3D */
  ExecViewManaged<Real * [NUM_TIME_LEVELS][NUM_4D_SCALARS][NUM_LEV][NP][NP]>
  m_4d_scalars;
  /* Contains OMEGA_P, PECND, PHI, UN0, VN0, QDP, ETA_DPDN */
  ExecViewManaged<Real * [NUM_3D_SCALARS][NUM_LEV][NP][NP]> m_3d_scalars;
  /* Contains FCOR, SPHEREMP, METDET, PHIS */
  ExecViewManaged<Real * [NUM_2D_SCALARS][NP][NP]> m_2d_scalars;
  /* Contains D, DINV */
  ExecViewManaged<Real * [NUM_2D_TENSORS][2][2][NP][NP]> m_2d_tensors;

  // TODO: should this be divided into components and put into 3d scalars?
  ExecViewManaged<Real * [QSIZE_D][2][NUM_LEV][NP][NP]> m_Qdp;

  ExecViewManaged<Real * [NUM_LEV_P][NP][NP]> m_eta_dot_dpdn;

public:
  explicit Region(int num_elems);

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> QDP(int ie, int qn0, int v) const {
    return Kokkos::subview(m_Qdp, ie, qn0, v, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV_P][NP][NP]> ETA_DPDN(int ie) const {
    return Kokkos::subview(m_eta_dot_dpdn, ie, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  /* 4D Scalars */
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> U(int ie, int timelevel) const {
    return Kokkos::subview(m_4d_scalars, ie, timelevel, IDX_U, Kokkos::ALL,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> V(int ie, int timelevel) const {
    return Kokkos::subview(m_4d_scalars, ie, timelevel, IDX_V, Kokkos::ALL,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T(int ie, int timelevel) const {
    return Kokkos::subview(m_4d_scalars, ie, timelevel, IDX_T, Kokkos::ALL,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> DP3D(int ie, int timelevel) const {
    return Kokkos::subview(m_4d_scalars, ie, timelevel, IDX_DP3D, Kokkos::ALL,
                           Kokkos::ALL, Kokkos::ALL);
  }

  /* 3D Scalars */
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> OMEGA_P(int ie, int level) const {
    return Kokkos::subview(m_3d_scalars, ie, IDX_OMEGA_P, level, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> PECND(int ie, int level) const {
    return Kokkos::subview(m_3d_scalars, ie, IDX_PECND, level, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> PHI(int ie) const {
    return Kokkos::subview(m_3d_scalars, ie, IDX_PHI, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> UN0(int ie, int level) const {
    return Kokkos::subview(m_3d_scalars, ie, IDX_UN0, level, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> VN0(int ie, int level) const {
    return Kokkos::subview(m_3d_scalars, ie, IDX_VN0, level, Kokkos::ALL,
                           Kokkos::ALL);
  }

  /* 2D Scalars */
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> FCOR(int ie) const {
    return Kokkos::subview(m_2d_scalars, ie, IDX_FCOR, Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> SPHEREMP(int ie) const {
    return Kokkos::subview(m_2d_scalars, ie, IDX_SPHEREMP, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> METDET(int ie) const {
    return Kokkos::subview(m_2d_scalars, ie, IDX_METDET, Kokkos::ALL,
                           Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[NP][NP]> PHIS(int ie) const {
    return Kokkos::subview(m_2d_scalars, ie, IDX_PHIS, Kokkos::ALL,
                           Kokkos::ALL);
  }

  /* 2D Tensors */
  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[2][2][NP][NP]> D(int ie) const {
    return Kokkos::subview(m_2d_tensors, ie, IDX_D, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL, Kokkos::ALL);
  }

  KOKKOS_INLINE_FUNCTION
  ExecViewUnmanaged<Real[2][2][NP][NP]> DINV(int ie) const {
    return Kokkos::subview(m_2d_tensors, ie, IDX_DINV, Kokkos::ALL, Kokkos::ALL,
                           Kokkos::ALL, Kokkos::ALL);
  }
};

} // TinMan

#endif // TINMAN_REGION_HPP
