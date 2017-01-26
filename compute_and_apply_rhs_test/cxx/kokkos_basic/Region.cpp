#include "Region.hpp"

namespace TinMan
{

double init_map(double x, int n)
{
  return std::pow(sin(n*x),2);
}

Region::Region( int num_elems )
    : m_nelems( num_elems )
    , m_2d_scalars( "2d scalars", num_elems )
    , m_2d_tensors( "2d tensors", num_elems )
    , m_3d_scalars( "3d scalars", num_elems )
    , m_4d_scalars( "4d scalars", num_elems )
    , m_Qdp( "qdp", num_elems )
    , m_eta_dot_dpdn( "eta_dot_dpdn", num_elems )
{
  // Initialize arrays using sin^2(n*x) map.
  // This is easily portable across different platforms and/or
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
        // Initializing m_2d_tensors and m_2d_scalars
        m_2d_tensors(ie,D,0,0,igp,jgp) = init_map(x,n++);
        m_2d_tensors(ie,D,0,1,igp,jgp) = init_map(x,n++);
        m_2d_tensors(ie,D,1,0,igp,jgp) = init_map(x,n++);
        m_2d_tensors(ie,D,1,1,igp,jgp) = init_map(x,n++);

        Real detD = m_2d_tensors(ie,D,0,0,igp,jgp)*m_2d_tensors(ie,D,1,0,igp,jgp)
                  - m_2d_tensors(ie,D,0,1,igp,jgp)*m_2d_tensors(ie,D,1,1,igp,jgp);

        m_2d_tensors(ie,DINV,0,0,igp,jgp) =  m_2d_tensors(ie,D,1,1,igp,jgp) / detD;
        m_2d_tensors(ie,DINV,0,1,igp,jgp) = -m_2d_tensors(ie,D,0,1,igp,jgp) / detD;
        m_2d_tensors(ie,DINV,1,0,igp,jgp) = -m_2d_tensors(ie,D,1,0,igp,jgp) / detD;
        m_2d_tensors(ie,DINV,1,1,igp,jgp) =  m_2d_tensors(ie,D,0,0,igp,jgp) / detD;

        m_2d_scalars(ie,FCOR,    igp,jgp) = init_map(x,n++);
        m_2d_scalars(ie,SPHEREMP,igp,jgp) = init_map(x,n++);
        m_2d_scalars(ie,METDET,  igp,jgp) = init_map(x,n++);
        m_2d_scalars(ie,PHIS,    igp,jgp) = init_map(x,n++);

        // Initializing arrays that contain [NUM_LEV]
        for (int il=0; il<NUM_LEV; ++il)
        {
          // m_3d_scalars
          m_3d_scalars(ie,OMEGA_P,il,igp,jgp) = init_map(x,n++);
          m_3d_scalars(ie,PECND  ,il,igp,jgp) = init_map(x,n++);
          m_3d_scalars(ie,PHI    ,il,igp,jgp) = init_map(x,n++);
          m_3d_scalars(ie,UN0    ,il,igp,jgp) = init_map(x,n++);
          m_3d_scalars(ie,VN0    ,il,igp,jgp) = init_map(x,n++);

          // Initializing m_Qdp
          for (int iq=0; iq<QSIZE_D; ++iq)
          {
            m_Qdp(ie,il,iq,0,igp,jgp) = init_map(x,n++);
            m_Qdp(ie,il,iq,1,igp,jgp) = init_map(x,n++);
          }

          // Initializing arrays that contain [NUM_TIME_LEVELS]
          for (int it=0; it<NUM_TIME_LEVELS; ++it)
          {
            // Initializing m_element_states
            m_4d_scalars(ie,it,U,   il,igp,jgp) = init_map(x,n++);
            m_4d_scalars(ie,it,V,   il,igp,jgp) = init_map(x,n++);
            m_4d_scalars(ie,it,T,   il,igp,jgp) = init_map(x,n++);
            m_4d_scalars(ie,it,DP3D,il,igp,jgp) = init_map(x,n++);
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
