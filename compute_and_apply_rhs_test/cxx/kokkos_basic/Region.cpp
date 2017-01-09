#include "Region.hpp"

namespace TinMan
{

double init_map(double x, int n)
{
  return std::pow(sin(n*x),2);
}

Region::Region( int num_elems )
    : m_nelems( num_elems )
    , m_elements( "elements", num_elems )
    , m_element_states( "element_states", num_elems )
    , m_element_deriveds( "element_deriveds", num_elems )
    , m_Qdp( "qdp", num_elems )
    , m_eta_dot_dpdn( "eta_dot_dpdn", num_elems )
    , m_phi ( "phi", num_elems )
    , m_dp3d ( "dp3d", num_elems )
{
  // Initialize arrays using sin^2(n*x) map.
  // This iss easily portable across different platforms and/or
  // languages without relying on implementation details

  // Set seed for the init map
  constexpr Real x = 0.123456789;

  int n = 1;
  // Now fill all the arrays
  for (int ie=0; ie<num_elems; ++ie)
  {
    for (int igp=0; igp<NP; ++igp)
    {
      for (int jgp=0; jgp<NP; ++jgp)
      {
        // Initializing Real*[NP][NP] arrays
        m_elements(ie).D[0][0][igp][jgp] = init_map(x,n++);
        m_elements(ie).D[0][1][igp][jgp] = init_map(x,n++);
        m_elements(ie).D[1][0][igp][jgp] = init_map(x,n++);
        m_elements(ie).D[1][1][igp][jgp] = init_map(x,n++);
        Real detD = m_elements(ie).D[0][0][igp][jgp]*m_elements(ie).D[1][1][igp][jgp]
                  - m_elements(ie).D[0][1][igp][jgp]*m_elements(ie).D[1][0][igp][jgp];

        m_elements(ie).DInv[0][0][igp][jgp] =  m_elements(ie).D[1][1][igp][jgp] / detD;
        m_elements(ie).DInv[0][1][igp][jgp] = -m_elements(ie).D[0][1][igp][jgp] / detD;
        m_elements(ie).DInv[1][0][igp][jgp] = -m_elements(ie).D[1][0][igp][jgp] / detD;
        m_elements(ie).DInv[1][1][igp][jgp] =  m_elements(ie).D[0][0][igp][jgp] / detD;

        m_elements(ie).Fcor[igp][jgp]       = init_map(x,n++);
        m_elements(ie).Spheremp[igp][jgp]   = init_map(x,n++);
        m_elements(ie).MetDet[igp][jgp]     = init_map(x,n++);
        m_elements(ie).State_Phis[igp][jgp] = init_map(x,n++);

        // Initializing arrays that contain [NUM_LEV]
        for (int il=0; il<NUM_LEV; ++il)
        {
          // m_element_deriveds
          m_element_deriveds(ie,il).OmegaP[igp][jgp] = init_map(x,n++);
          m_element_deriveds(ie,il).Pecnd[igp][jgp]  = init_map(x,n++);
          m_element_deriveds(ie,il).Vn0[0][igp][jgp] = init_map(x,n++);
          m_element_deriveds(ie,il).Vn0[1][igp][jgp] = init_map(x,n++);

          // Initializing m_phi
          m_phi(ie,il,igp,jgp) = init_map(x,n++);

          // Initializing m_Qdp
          for (int iq=0; iq<QSIZE_D; ++iq)
          {
            m_Qdp(ie,il,iq,0,igp,jgp) = init_map(x,n++);
            m_Qdp(ie,il,iq,1,igp,jgp) = init_map(x,n++);
          }

          // Initializing arrays that contain [NUM_TIME_LEVELS]
          for (int it=0; it<NUM_TIME_LEVELS; ++it)
          {
            // Initializing m_dp3d
            m_dp3d(ie,it,il,igp,jgp) = init_map(x,n++);

            // Initializing m_element_states
            m_element_states(ie,it,il).T[igp][jgp]    = init_map(x,n++);
            m_element_states(ie,it,il).V[0][igp][jgp] = init_map(x,n++);
            m_element_states(ie,it,il).V[1][igp][jgp] = init_map(x,n++);
          }
        }

        // Initializing m_eta_dot_dpdn
        for (int il=0; il<NUM_LEV_P; ++il)
        {
          m_eta_dot_dpdn(ie,il,igp,jgp) = init_map(x,n++);
        }
      }
    }
  }
}

} // namespace TinMan
