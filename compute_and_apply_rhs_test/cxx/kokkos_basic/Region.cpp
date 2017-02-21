#include "Region.hpp"

namespace TinMan
{

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
    for (int ip=0; ip<NP; ++ip)
    {
      for (int jp=0; jp<NP; ++jp)
      {
        double iie = ie + 1;
        double iip = ip + 1;
        double jjp = jp + 1;

        // Initializing m_2d_tensors and m_2d_scalars
        m_2d_scalars(ie,IDX_FCOR,    ip,jp) = sin(iip + jjp);
        m_2d_scalars(ie,IDX_METDET,  ip,jp) = iip*jjp;
        m_2d_scalars(ie,IDX_SPHEREMP,ip,jp) = 2*iip;
        m_2d_scalars(ie,IDX_PHIS,    ip,jp) = iip + jjp;

        m_2d_tensors(ie,IDX_D,0,0,ip,jp) = 1.0;
        m_2d_tensors(ie,IDX_D,0,1,ip,jp) = 0.0;
        m_2d_tensors(ie,IDX_D,1,0,ip,jp) = 0.0;
        m_2d_tensors(ie,IDX_D,1,1,ip,jp) = 2.0;

        m_2d_tensors(ie,IDX_DINV,0,0,ip,jp) = 1.0;
        m_2d_tensors(ie,IDX_DINV,0,1,ip,jp) = 0.0;
        m_2d_tensors(ie,IDX_DINV,1,0,ip,jp) = 0.0;
        m_2d_tensors(ie,IDX_DINV,1,1,ip,jp) = 0.5;

        // Initializing arrays that contain [NUM_LEV]
        for (int il=0; il<NUM_LEV; ++il)
        {
          double iil = il + 1;

          // m_3d_scalars
          m_3d_scalars(ie,IDX_PHI    ,il,ip,jp) = cos(iip + 3*jjp) + iil;
          m_3d_scalars(ie,IDX_UN0    ,il,ip,jp) = 1.0;
          m_3d_scalars(ie,IDX_VN0    ,il,ip,jp) = 1.0;
          m_3d_scalars(ie,IDX_PECND  ,il,ip,jp) = 1.0;
          m_3d_scalars(ie,IDX_OMEGA_P,il,ip,jp) = jjp*jjp;

          // Initializing m_Qdp
          m_Qdp(ie,il,0,0,ip,jp) = 1.0 + sin(iip*jjp*iil);

          // Initializing arrays that contain [NUM_TIME_LEVELS]
          for (int it=0; it<NUM_TIME_LEVELS; ++it)
          {
            double iit = it + 1;

            // Initializing m_element_states
            m_4d_scalars(ie,it,IDX_DP3D,il,ip,jp) = 10.0*iil + iie + iip + jjp + iit;
            m_4d_scalars(ie,it,IDX_U,   il,ip,jp) = 1.0 + 0.5*iil + iip + jjp + 0.2*iie + 2.0*iit;
            m_4d_scalars(ie,it,IDX_V,   il,ip,jp) = 1.0 + 0.5*iil + iip + jjp + 0.2*iie + 3.0*iit;
            m_4d_scalars(ie,it,IDX_T,   il,ip,jp) = 1000.0 + sin(iip + jjp + iil);
          }
        }

        // Initializing m_eta_dot_dpdn
        for (int il=0; il<NUM_LEV_P; ++il)
        {
          m_eta_dot_dpdn(ie,il,ip,jp) = 0;
        }
      }
    }
  }
}

} // namespace TinMan
