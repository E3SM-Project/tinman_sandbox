#ifndef TINMAN_KOKKOS_TYPES_HPP
#define TINMAN_KOKKOS_TYPES_HPP

#include <config.h>

#include <Kokkos_Core.hpp>

namespace Homme {

namespace Impl {

struct Element {
  double D[2][2][NP][NP];
  double DInv[2][2][NP][NP]; // TODO Should D and DInv be in the same structure???

  double Fcor[NP][NP];
  double Spheremp[NP][NP];
  double MetDet[NP][NP];

  double State_Phis[NP][NP]; // TODO moved from ElemState...does this belong here???
};

struct ElementState {
  double DP3D[NP][NP];
  double V[2][NP][NP];
  double T[NP][NP];
};

struct ElementDerived {
  double OmegaP[NP][NP];
  double Phi[NP][NP];
  double Pecnd[NP][NP];
  double Vn0[2][NP][NP];
};

} // namespace Impl

class Region
{
public:

public:

  explicit
  Region( int num_elems )
    : m_nelems( num_elems )
    , m_elements( "elements", num_elems )
    , m_element_states( "element_states", num_elems )
    , m_element_deriveds( "element_deriveds", num_elems )
    , m_Qdp( "qdp", num_elems )
    , m_eta_dot_dpdn( "eta_dot_dpdn", num_elems )
  {}


  template <typename I0, typename I1, typename I2, typename I3, typename I4>
  KOKKOS_FORCEINLINE_FUNCTION
  auto D(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) const -> decltype( m_elements(i0).D[i1][i2][i3][i4] )
  {
    return  m_elements(i0).D[i1][i2][i3][i4];
  }


  // TODO other accessors

private:

  int m_nelems;
  Kokkos::View< Impl::Element* > m_elements;
  Kokkos::View< Impl::ElementState*[NUM_LEV][NUM_TIME_LEVELS] > m_element_states;
  Kokkos::View< Impl::ElementDerived*[NUM_LEV] > m_element_deriveds;

  // TODO: should this be a member of ElementDerived???
  Kokkos::View< double *[NUM_LEV][QSIZE_D][2][NP][NP], Kokkos::LayoutRight> m_Qdp;

  Kokkos::View< double *[NUM_LEV_P][NP][NP], Kokkos::LayoutRight > m_eta_dot_dpdn;


};

} // Homme

#endif // TINMAN_KOKKOS_TYPES_HPP
