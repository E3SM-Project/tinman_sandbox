#ifndef TINMAN_REGION_HPP
#define TINMAN_REGION_HPP

#include "config.h"

#include "Types.hpp"

#include <Kokkos_Core.hpp>

namespace TinMan {

// The number of fields for each dimension
constexpr int NUM_4D_SCALARS = 4;
constexpr int NUM_3D_SCALARS = 5;
constexpr int NUM_2D_SCALARS = 4;
constexpr int NUM_2D_TENSORS = 2;

// Some constexpr for the index of different variables in the views
constexpr int U        = 0;
constexpr int V        = 1;
constexpr int T        = 2;
constexpr int DP3D     = 3;
constexpr int OMEGA_P  = 0;
constexpr int PECND    = 1;
constexpr int PHI      = 2;
constexpr int UN0      = 3;
constexpr int VN0      = 4;
constexpr int FCOR     = 0;
constexpr int SPHEREMP = 1;
constexpr int METDET   = 2;
constexpr int PHIS     = 3;
constexpr int D        = 0;
constexpr int DINV     = 1;

class Region
{
private:

  int m_nelems;
  ViewManaged< Real*[NUM_TIME_LEVELS][NUM_4D_SCALARS][NUM_LEV][NP][NP] > m_4d_scalars;
  ViewManaged< Real*[NUM_3D_SCALARS][NUM_LEV][NP][NP] >                  m_3d_scalars;
  ViewManaged< Real*[NUM_2D_SCALARS][NP][NP] >                           m_2d_scalars;
  ViewManaged< Real*[NUM_2D_TENSORS][2][2][NP][NP] >                     m_2d_tensors;

  // TODO: should this be divided into components and put into 3d scalars?
  ViewManaged< Real *[NUM_LEV][QSIZE_D][2][NP][NP]> m_Qdp;

  ViewManaged< Real *[NUM_LEV_P][NP][NP] > m_eta_dot_dpdn;

public:

  explicit
  Region( int num_elems );

  // Getters for all internal views
  ViewUnmanaged< Real*[NUM_2D_SCALARS][NP][NP] > get_2d_scalars () const
  {
    return m_2d_scalars;
  }

  ViewUnmanaged< Real*[NUM_2D_TENSORS][2][2][NP][NP] > get_2d_tensors () const
  {
    return m_2d_tensors;
  }

  ViewUnmanaged< Real*[NUM_3D_SCALARS][NUM_LEV][NP][NP] > get_3d_scalars () const
  {
    return m_3d_scalars;
  }

  ViewUnmanaged< Real*[NUM_TIME_LEVELS][NUM_4D_SCALARS][NUM_LEV][NP][NP] > get_4d_scalars () const
  {
    return m_4d_scalars;
  }

  ViewUnmanaged< Real*[NUM_LEV][QSIZE_D][2][NP][NP] > get_Qdp () const
  {
    return m_Qdp;
  }

  ViewUnmanaged< Real*[NUM_LEV_P][NP][NP] > get_eta_dot_dpdn () const
  {
    return m_eta_dot_dpdn;
  }

/*
  template <typename iElem>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<Real[NP][NP]> get_2d_scalarDInv(iElem ie) const

  // Getters for m_2d_tensors
  template <typename iElem>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<Real[2][2][NP][NP]> DInv(iElem ie) const
  {
    return Kokkos::subview(m_2d_tensors, ie, E2Int(EVar2Idx::DINV), Kokkos::ALL(), Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }

  template <typename iElem>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<Real[2][2][NP][NP]> D(iElem ie) const
  {
    return Kokkos::subview(m_2d_tensors, ie, E2Int(EVar2Idx::D), Kokkos::ALL(), Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }

  // Getters for m_2d_scalars
  template <typename iElem>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<Real[NP][NP]> metDet(iElem ie) const
  {
    return Kokkos::subview(m_2d_scalars, ie, E2Int(EVar2Idx::METDET),Kokkos::ALL(),Kokkos::ALL());
  }

  template <typename iElem>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<Real[NP][NP]> phis(iElem ie) const
  {
    return Kokkos::subview(m_2d_scalars, ie, E2Int(EVar2Idx::PHIS), Kokkos::ALL(), Kokkos::ALL());
  }

  // Getters for m_4d_scalars
  template <typename iElem, typename iTimeLevel>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<Real[NUM_LEV][NP][NP]> dp3d(iElem ie, iTimeLevel itl) const
  {
    return Kokkos::subview(m_4d_scalars, ie, itl, E2Int(EVar2Idx::DP3D), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
  }

  template <typename iElem, typename iTimeLevel, typename iLevel>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<Real[NP][NP]> T(iElem ie, iTimeLevel itl, iLevel il) const
  {
    return Kokkos::subview (m_4d_scalars, ie, itl, E2Int(EVar2Idx::T), il, Kokkos::ALL(), Kokkos::ALL());
  }

  template <typename iElem>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<Real[NUM_LEV][NP][NP]> phi(iElem ie) const
  {
    return Kokkos::subview(m_3d_scalars, ie, E2Int(EVar2Idx::PHI), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
  }

  // Getters for Qdp
  template <typename I0, typename I1, typename I2, typename I3, typename I4, typename I5>
  KOKKOS_FORCEINLINE_FUNCTION
  auto Qdp(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5) const -> decltype( m_Qdp(i0,i1,i2,i3,i4,i5) )
  {
    return m_Qdp(i0,i1,i2,i3,i4,i5);
  }

  // Getters for eta_dot_dpdn
  template <typename I0, typename I1, typename I2, typename I3>
  KOKKOS_FORCEINLINE_FUNCTION
  auto eta_dot_dpdn(I0 i0, I1 i1, I2 i2, I3 i3) const -> decltype( m_eta_dot_dpdn(i0,i1,i2,i3) )
  {
    return m_eta_dot_dpdn(i0,i1,i2,i3);
  }
*/
  // TODO other accessors

};

} // TinMan

#endif // TINMAN_REGION_HPP
