#include "dimensions.hpp"
#include "data_structures.hpp"

#include "test_macros.hpp"
#include <random>

namespace Homme
{

void Arrays::init_data ()
{
  elem_D                    = new real[nelems*np*np*2*2] {};
  elem_Dinv                 = new real[nelems*np*np*2*2] {};
  elem_fcor                 = new real[nelems*np*np]     {};
  elem_spheremp             = new real[nelems*np*np]     {};
  elem_metdet               = new real[nelems*np*np]     {};
  elem_rmetdet              = new real[nelems*np*np]     {};

  elem_state_dp3d           = new real[nelems*timelevels*np*np*nlev]   {};
  elem_state_v              = new real[nelems*timelevels*np*np*2*nlev] {};
  elem_state_T              = new real[nelems*timelevels*np*np*nlev]   {};
  elem_state_phis           = new real[nelems*np*np]                   {};
  elem_state_Qdp            = new real[nelems*nlev*np*np*qsize_d*2]    {};

  elem_derived_eta_dot_dpdn = new real[nelems*nlevp*np*np]  {};
  elem_derived_omega_p      = new real[nelems*nlev*np*np]   {};
  elem_derived_phi          = new real[nelems*nlev*np*np]   {};
  elem_derived_pecnd        = new real[nelems*nlev*np*np]   {};
  elem_derived_vn0          = new real[nelems*nlev*np*np*2] {};

  if (true)
  {
    std::uniform_real_distribution<double> unid(1.0,2.0);
    std::default_random_engine re;
    for (int ie=0; ie<nelems; ++ie)
    {
      for (int ip=0; ip<np; ++ip)
      {
        for (int jp=0; jp<np; ++jp)
        {
          AT_3D(elem_fcor,ie,ip,jp,np,np) = unid(re);
          AT_3D(elem_spheremp,ie,ip,jp,np,np) = unid(re);
          AT_3D(elem_metdet,ie,ip,jp,np,np) = unid(re);
          AT_3D(elem_rmetdet,ie,ip,jp,np,np) = unid(re);
          AT_3D(elem_state_phis,ie,ip,jp,np,np) = unid(re);
          AT_3D(elem_fcor,ie,ip,jp,np,np) = unid(re);

          AT_5D(elem_D,ie,ip,jp,0,0,np,np,2,2) = unid(re);
          AT_5D(elem_D,ie,ip,jp,0,1,np,np,2,2) = unid(re);
          AT_5D(elem_D,ie,ip,jp,1,0,np,np,2,2) = unid(re);
          AT_5D(elem_D,ie,ip,jp,1,1,np,np,2,2) = unid(re);

          AT_5D(elem_Dinv,ie,ip,jp,0,0,np,np,2,2) = unid(re);
          AT_5D(elem_Dinv,ie,ip,jp,0,1,np,np,2,2) = unid(re);
          AT_5D(elem_Dinv,ie,ip,jp,1,0,np,np,2,2) = unid(re);
          AT_5D(elem_Dinv,ie,ip,jp,1,1,np,np,2,2) = unid(re);

          for (int il=0; il<nlev; ++il)
          {
            AT_4D (elem_derived_omega_p,ie,il,ip,jp,nlev,np,np) = unid(re);
            AT_4D (elem_derived_phi,ie,il,ip,jp,nlev,np,np) = unid(re);
            AT_4D (elem_derived_pecnd,ie,il,ip,jp,nlev,np,np) = unid(re);
            AT_5D (elem_derived_vn0,ie,il,ip,jp,0,nlev,np,np,2) = unid(re);
            AT_5D (elem_derived_vn0,ie,il,ip,jp,1,nlev,np,np,2) = unid(re);

            for (int iq=0; iq<qsize_d; ++iq)
            {
              AT_6D(elem_state_Qdp,ie,il,ip,jp,iq,0,nlev,np,np,qsize_d,2) = unid(re);
              AT_6D(elem_state_Qdp,ie,il,ip,jp,iq,1,nlev,np,np,qsize_d,2) = unid(re);
            }

            for (int it=0; it<timelevels; ++it)
            {

              AT_5D(elem_state_dp3d,ie,it,il,ip,jp,timelevels,nlev,np,np) = unid(re);
              AT_5D(elem_state_v,ie,it,il,ip,jp,timelevels,nlev,np,np) = unid(re);
              AT_5D(elem_state_T,ie,it,il,ip,jp,timelevels,nlev,np,np) = unid(re);
            }
          }

          for (int il=0; il<nlevp; ++il)
          {
            AT_4D (elem_derived_eta_dot_dpdn,ie,il,ip,jp,nlevp,np,np) = unid(re);
          }
        }
      }
    }
  }
}

void Arrays::cleanup_data ()
{
  delete[] elem_D;
  delete[] elem_Dinv;
  delete[] elem_fcor;
  delete[] elem_spheremp;
  delete[] elem_metdet;
  delete[] elem_rmetdet;

  delete[] elem_state_dp3d;
  delete[] elem_state_v;
  delete[] elem_state_T;
  delete[] elem_state_phis;
  delete[] elem_state_Qdp;

  delete[] elem_derived_eta_dot_dpdn;
  delete[] elem_derived_omega_p;
  delete[] elem_derived_phi;
  delete[] elem_derived_pecnd;
  delete[] elem_derived_vn0;
}

void Constants::init_data ()
{
  if (true)
  {
    std::uniform_real_distribution<double> unid(1.0,2.0);
    std::default_random_engine re;

    rrearth      = unid(re);
    eta_ave_w    = unid(re);
    Rwater_vapor = unid(re);
    Rgas         = unid(re);
    kappa        = unid(re);
  }
}

void Control::init_data ()
{
  nets = 0;
  nete = nelems;
  n0 = 0;
  np1 = 1;
  nm1 = 2;
  qn0 = 0;

  real dt2 = 0.1;
}

void HVCoord::init_data ()
{
  if (true)
  {
    std::uniform_real_distribution<double> unid(1.0,2.0);
    std::default_random_engine re;

    ps0 = unid(re);

    for (int i=0; i<nlevp; ++i)
    {
      hyai[i] = unid(re);
    }
  }
}

void Derivative::init_data ()
{
  if (true)
  {
    std::uniform_real_distribution<double> unid(1.0,2.0);
    std::default_random_engine re;

    for (int i=0; i<np; ++i)
    {
      for (int j=0; j<np; ++j)
      {
        Dvv[i][j] = unid(re);
      }
    }
  }
}

void TestData::init_data ()
{
  arrays.init_data();
  constants.init_data();
  control.init_data();
  hvcoord.init_data();
  deriv.init_data();
}

void TestData::cleanup_data ()
{
  arrays.cleanup_data();
}

} // Namespace Homme
