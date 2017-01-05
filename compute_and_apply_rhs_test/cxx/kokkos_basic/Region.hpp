#ifndef TINMAN_REGION_HPP
#define TINMAN_REGION_HPP

#include <config.h>

#include <Types.hpp>

#include <Kokkos_Core.hpp>

namespace TinMan {

namespace Impl {

struct Element {
  Real D[2][2][NP][NP];
  Real DInv[2][2][NP][NP]; // TODO Should D and DInv be in the same structure???

  Real Fcor[NP][NP];
  Real Spheremp[NP][NP];
  Real MetDet[NP][NP];

  Real State_Phis[NP][NP]; // TODO moved from ElemState...does this belong here???
};

struct ElementState {
  Real V[2][NP][NP];
  Real T[NP][NP];
};

struct ElementDerived {
  Real OmegaP[NP][NP];
  Real Pecnd[NP][NP];
  Real Vn0[2][NP][NP];
};

} // namespace Impl

class Region
{
private:

  int m_nelems;
  ViewManaged< Impl::Element* >                                  m_elements;
  ViewManaged< Impl::ElementState*[NUM_TIME_LEVELS][NUM_LEV] >   m_element_states;
  ViewManaged< Impl::ElementDerived*[NUM_LEV] >                  m_element_deriveds;

  // TODO: should this be a member of ElementDerived???
  ViewManaged< Real *[NUM_LEV][QSIZE_D][2][NP][NP]> m_Qdp;

  ViewManaged< Real *[NUM_LEV_P][NP][NP] > m_eta_dot_dpdn;

  // TODO: pulled out from ElementDerived since I often need to pass around phi(ie,:,:,:)
  //       Is this design reasonable? Should I keep it in ElementDerived and change the
  //       signature of the functions that need phi(ie,:,:,:)?
  ViewManaged< Real *[NUM_LEV][NP][NP] > m_phi;

  // TODO: pulled out from ElementState since I need to pass around dp3d(ie,it,:,:,:)
  //       Is this design reasonable? Should I keep it in ElementState and change the
  //       signature of the functions that need dp3d(ie,it,:,:,:)?
  ViewManaged< Real *[NUM_TIME_LEVELS][NUM_LEV][NP][NP] > m_dp3d;

public:

  explicit
  Region( int num_elems );

  // Getters for m_elements sub arrays
  template <typename iElem>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<double[2][2][NP][NP]> DInv(iElem ie) const
  {
    ViewUnmanaged<double[2][2][NP][NP]> DInv_ie(&(m_elements(ie).DInv[0][0][0][0]));
    return DInv_ie;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4>
  KOKKOS_FORCEINLINE_FUNCTION
  auto DInv(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) const -> decltype( m_elements(i0).DInv[i1][i2][i3][i4] )
  {
    return m_elements(i0).DInv[i1][i2][i3][i4];
  }

  template <typename iElem>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<double[2][2][NP][NP]> D(iElem ie) const
  {
    ViewUnmanaged<double[2][2][NP][NP]> D_ie(&(m_elements(ie).D[0][0][0][0]));
    return D_ie;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4>
  KOKKOS_FORCEINLINE_FUNCTION
  auto D(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) const -> decltype( m_elements(i0).D[i1][i2][i3][i4] )
  {
    return m_elements(i0).D[i1][i2][i3][i4];
  }

  template <typename I0, typename I1, typename I2>
  KOKKOS_FORCEINLINE_FUNCTION
  auto fcor(I0 i0, I1 i1, I2 i2) const -> decltype( m_elements(i0).Fcor[i1][i2] )
  {
    return m_elements(i0).Fcor[i1][i2];
  }

  template <typename I0, typename I1, typename I2>
  KOKKOS_FORCEINLINE_FUNCTION
  auto spheremp(I0 i0, I1 i1, I2 i2) const -> decltype( m_elements(i0).Spheremp[i1][i2] )
  {
    return m_elements(i0).Spheremp[i1][i2];
  }

  template <typename iElem>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<double[NP][NP]> metDet(iElem ie) const
  {
    ViewUnmanaged<double[NP][NP]> metDet_ie(&(m_elements(ie).MetDet[0][0]));
    return metDet_ie;
  }

  template <typename iElem>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<double[NP][NP]> phis(iElem ie) const
  {
    ViewUnmanaged<double[NP][NP]> phis_ie(&(m_elements(ie).State_Phis[0][0]));
    return phis_ie;
  }

  // Getters for m_element_states sub arrays
  template <typename iElem, typename iTimeLevel>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<double[NUM_LEV][NP][NP]> dp3d(iElem ie, iTimeLevel itl) const
  {
    return Kokkos::subview(m_dp3d,ie,itl,Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4>
  KOKKOS_FORCEINLINE_FUNCTION
  auto dp3d(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) const -> decltype( m_dp3d(i0,i1,i2,i3,i4) )
  {
    return m_dp3d(i0,i1,i2,i3,i4);
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4, typename I5>
  KOKKOS_FORCEINLINE_FUNCTION
  auto V(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5) const -> decltype( m_element_states(i0,i1,i2).V[i3][i4][i5] )
  {
    return m_element_states(i0,i1,i2).V[i3][i4][i5];
  }

  template <typename iElem, typename iTimeLevel, typename iLevel>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<Real[NP][NP]> T(iElem ie, iTimeLevel itl, iLevel il) const
  {
    ViewUnmanaged<Real[NP][NP]> T_ie_itl_il ( &(m_element_states(ie,itl,il).T[0][0]) );
    return T_ie_itl_il;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4>
  KOKKOS_FORCEINLINE_FUNCTION
  auto T(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) const -> decltype( m_element_states(i0,i1,i2).T[i3][i4] )
  {
    return m_element_states(i0,i1,i2).T[i3][i4];
  }

  // Getters for m_element_deriveds sub arrays
  template <typename I0, typename I1, typename I2, typename I3>
  KOKKOS_FORCEINLINE_FUNCTION
  auto omega_p(I0 i0, I1 i1, I2 i2, I3 i3) const -> decltype( m_element_deriveds(i0,i1).OmegaP[i2][i3] )
  {
    return m_element_deriveds(i0,i1).OmegaP[i2][i3];
  }

  template <typename I0, typename I1, typename I2, typename I3>
  KOKKOS_FORCEINLINE_FUNCTION
  auto pecnd(I0 i0, I1 i1, I2 i2, I3 i3) const -> decltype( m_element_deriveds(i0,i1).Pecnd[i2][i3] )
  {
    return m_element_deriveds(i0,i1).Pecnd[i2][i3];
  }

  template <typename iElem, typename iLevel>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<Real[2][NP][NP]> Vn0(iElem ie, iLevel il) const
  {
    ViewUnmanaged<Real[2][NP][NP]> Vn0_ie_il( &(m_element_deriveds(ie,il).Vn0[0][0][0]) );
    return Vn0_ie_il;
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4>
  KOKKOS_FORCEINLINE_FUNCTION
  auto Vn0(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) const -> decltype( m_element_deriveds(i0,i1).Vn0[i2][i3][i4] )
  {
    return m_element_deriveds(i0,i1).Vn0[i2][i3][i4];
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

  // Getters for phi
  template <typename iElem>
  KOKKOS_FORCEINLINE_FUNCTION
  ViewUnmanaged<Real[NUM_LEV][NP][NP]> phi(iElem ie) const
  {
    return Kokkos::subview(m_phi,ie,Kokkos::ALL(),Kokkos::ALL(),Kokkos::ALL());
  }

  template <typename I0, typename I1, typename I2, typename I3>
  KOKKOS_FORCEINLINE_FUNCTION
  auto phi(I0 i0, I1 i1, I2 i2, I3 i3) const -> decltype( m_phi(i0,i1,i2,i3) )
  {
    return m_phi(i0,i1,i2,i3);
  }


  // TODO other accessors

};

} // TinMan

#endif // TINMAN_REGION_HPP
