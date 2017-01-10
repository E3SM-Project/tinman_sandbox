#ifndef TINMAN_REGION_HPP
#define TINMAN_REGION_HPP

#include "config.h"

#include "Types.hpp"

#include <Kokkos_Core.hpp>

namespace TinMan {

constexpr int NUM_4D_SCALARS = 4;
constexpr int NUM_3D_SCALARS = 3;
constexpr int NUM_2D_SCALARS = 4;
constexpr int NUM_2D_TENSORS = 2;

enum class EVar2Idx
{
  U        = 0,
  V        = 1,
  T        = 2,
  DP3D     = 3,
  OMEGA_P  = 0,
  PECND    = 1,
  PHI      = 2,
  FCOR     = 0,
  SPHEREMP = 1,
  METDET   = 2,
  PHIS     = 3,
  D        = 0,
  DINV     = 1
};

template<typename EnumType>
KOKKOS_FORCEINLINE_FUNCTION
constexpr int E2Int(EnumType e)
{
  return static_cast<int>(e);
}

class Region
{
private:

  int m_nelems;
  ViewManaged< Real*[NUM_TIME_LEVELS][NUM_4D_SCALARS][NUM_LEV][NP][NP] > m_4d_scalars;
  ViewManaged< Real*[NUM_3D_SCALARS][NUM_LEV][NP][NP] >                  m_3d_scalars;
  ViewManaged< Real*[NUM_2D_SCALARS][NP][NP] >                           m_2d_scalars;
  ViewManaged< Real*[NUM_2D_TENSORS][2][2][NP][NP] >                     m_2d_tensors;

  ViewManaged< Real*[NUM_LEV][2][NP][NP] > m_Vn0;

  // TODO: should this be divided into components and put into 3d scalars?
  ViewManaged< Real *[NUM_LEV][QSIZE_D][2][NP][NP]> m_Qdp;

  ViewManaged< Real *[NUM_LEV_P][NP][NP] > m_eta_dot_dpdn;

public:

  explicit
  Region( int num_elems );

  // Getters for m_2d_tensors
  template <typename iElem>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<Real[2][2][NP][NP]> DInv(iElem ie) const
  {
    return Kokkos::subview(m_2d_tensors, ie, E2Int(EVar2Idx::DINV), Kokkos::ALL(), Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4>
  KOKKOS_FORCEINLINE_FUNCTION
  auto DInv(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) const -> decltype( m_2d_tensors(i0,EVar2Idx::DINV,i1,i2,i3,i4) )
  {
    return m_2d_tensors(i0,EVar2Idx::DINV,i1,i2,i3,i4);
  }

  template <typename iElem>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<double[2][2][NP][NP]> D(iElem ie) const
  {
    return Kokkos::subview(m_2d_tensors, ie, E2Int(EVar2Idx::D), Kokkos::ALL(), Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4>
  KOKKOS_FORCEINLINE_FUNCTION
  auto D(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) const -> decltype( m_2d_tensors(i0,EVar2Idx::D,i1,i2,i3,i4) )
  {
    return m_2d_tensors(i0,EVar2Idx::D,i1,i2,i3,i4);
  }

  // Getters for m_2d_scalars
  template <typename I0, typename I1, typename I2>
  KOKKOS_FORCEINLINE_FUNCTION
  auto fcor(I0 i0, I1 i1, I2 i2) const -> decltype( m_2d_scalars(i0,EVar2Idx::FCOR,i1,i2) )
  {
    return m_2d_scalars(i0,E2Int(EVar2Idx::FCOR),i1,i2);
  }

  template <typename I0, typename I1, typename I2>
  KOKKOS_FORCEINLINE_FUNCTION
  auto spheremp(I0 i0, I1 i1, I2 i2) const -> decltype( m_2d_scalars(i0,EVar2Idx::SPHEREMP,i1,i2) )
  {
    return m_2d_scalars(i0,E2Int(EVar2Idx::SPHEREMP),i1,i2);
  }

  template <typename iElem>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<double[NP][NP]> metDet(iElem ie) const
  {
    return Kokkos::subview(m_2d_scalars, ie, E2Int(EVar2Idx::METDET),Kokkos::ALL(),Kokkos::ALL());
  }

  template <typename iElem>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<double[NP][NP]> phis(iElem ie) const
  {
    return Kokkos::subview(m_2d_scalars, ie, E2Int(EVar2Idx::PHIS), Kokkos::ALL(), Kokkos::ALL());
  }

  // Getters for m_4d_scalars
  template <typename iElem, typename iTimeLevel>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<double[NUM_LEV][NP][NP]> dp3d(iElem ie, iTimeLevel itl) const
  {
    return Kokkos::subview(m_4d_scalars, ie, itl, E2Int(EVar2Idx::DP3D), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4>
  KOKKOS_FORCEINLINE_FUNCTION
  auto dp3d(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) const -> decltype( m_4d_scalars(i0,i1,EVar2Idx::DP3D,i2,i3,i4) )
  {
    return m_4d_scalars(i0,i1,E2Int(EVar2Idx::DP3D),i2,i3,i4);
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4>
  KOKKOS_FORCEINLINE_FUNCTION
  auto U(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) const -> decltype( m_4d_scalars(i0,i1,EVar2Idx::U,i2,i3,i4) )
  {
    return m_4d_scalars(i0,i1,E2Int(EVar2Idx::U),i2,i3,i4) ;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4>
  KOKKOS_FORCEINLINE_FUNCTION
  auto V(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) const -> decltype( m_4d_scalars(i0,i1,EVar2Idx::V,i2,i3,i4)  )
  {
    return m_4d_scalars(i0,i1,E2Int(EVar2Idx::V),i2,i3,i4);
  }

  template <typename iElem, typename iTimeLevel, typename iLevel>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<Real[NP][NP]> T(iElem ie, iTimeLevel itl, iLevel il) const
  {
    return Kokkos::subview (m_4d_scalars, ie, itl, E2Int(EVar2Idx::T), il, Kokkos::ALL(), Kokkos::ALL());
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4>
  KOKKOS_FORCEINLINE_FUNCTION
  auto T(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) const -> decltype( m_4d_scalars(i0,i1,EVar2Idx::T,i2,i3,i4)  )
  {
    return m_4d_scalars(i0,i1,E2Int(EVar2Idx::T),i2,i3,i4) ;
  }

  // Getters for m_3d_scalars
  template <typename I0, typename I1, typename I2, typename I3>
  KOKKOS_FORCEINLINE_FUNCTION
  auto omega_p(I0 i0, I1 i1, I2 i2, I3 i3) const -> decltype( m_3d_scalars(i0,EVar2Idx::OMEGA_P,i1,i2,i3) )
  {
    return m_3d_scalars(i0,E2Int(EVar2Idx::OMEGA_P),i1,i2,i3) ;
  }

  template <typename I0, typename I1, typename I2, typename I3>
  KOKKOS_FORCEINLINE_FUNCTION
  auto pecnd(I0 i0, I1 i1, I2 i2, I3 i3) const -> decltype( m_3d_scalars(i0,EVar2Idx::PECND,i1,i2,i3) )
  {
    return m_3d_scalars(i0,E2Int(EVar2Idx::PECND),i1,i2,i3);
  }

  template <typename iElem, typename iLevel>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<Real[2][NP][NP]> Vn0(iElem ie, iLevel il) const
  {
    return Kokkos::subview (m_Vn0, ie, il, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
  }

  template <typename I0, typename I1, typename I2, typename I3>
  KOKKOS_FORCEINLINE_FUNCTION
  auto Un0(I0 i0, I1 i1, I2 i2, I3 i3) const -> decltype( m_Vn0(i0,i1,0,i2,i3) )
  {
    return m_Vn0(i0,i1,0,i2,i3);
  }

  template <typename I0, typename I1, typename I2, typename I3>
  KOKKOS_FORCEINLINE_FUNCTION
  auto Vn0(I0 i0, I1 i1, I2 i2, I3 i3) const -> decltype( m_Vn0(i0,i1,1,i2,i3) )
  {
    return m_Vn0(i0,i1,1,i2,i3);
  }

  template <typename iElem>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<Real[NUM_LEV][NP][NP]> phi(iElem ie) const
  {
    return Kokkos::subview(m_3d_scalars, ie, E2Int(EVar2Idx::PHI), Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());
  }

  template <typename I0, typename I1, typename I2, typename I3>
  KOKKOS_FORCEINLINE_FUNCTION
  auto phi(I0 i0, I1 i1, I2 i2, I3 i3) const -> decltype( m_3d_scalars(i0,EVar2Idx::PHI,i1,i2,i3) )
  {
    return m_3d_scalars(i0,E2Int(EVar2Idx::PHI),i1,i2,i3);
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

  // TODO other accessors

};

} // TinMan

#endif // TINMAN_REGION_HPP
