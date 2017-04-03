#include "dimensions.hpp"
#include "data_structures.hpp"

#include "test_macros.hpp"
#include <random>

namespace Homme
{

extern int num_elems;

void Arrays::init_data ()
{
  elem_D                    = new real[num_elems*np*np*2*2] {};
  elem_Dinv                 = new real[num_elems*np*np*2*2] {};
  elem_fcor                 = new real[num_elems*np*np]     {};
  elem_spheremp             = new real[num_elems*np*np]     {};
  elem_metdet               = new real[num_elems*np*np]     {};
  elem_rmetdet              = new real[num_elems*np*np]     {};

  elem_state_dp3d           = new real[num_elems*timelevels*nlev*np*np]   {};
  elem_state_v              = new real[num_elems*timelevels*nlev*np*np*2] {};
  elem_state_T              = new real[num_elems*timelevels*nlev*np*np]   {};
  elem_state_phis           = new real[num_elems*np*np]                   {};
  elem_state_Qdp            = new real[num_elems*qsize_d*2*nlev*np*np]    {};

  elem_derived_eta_dot_dpdn = new real[num_elems*nlevp*np*np]  {};
  elem_derived_omega_p      = new real[num_elems*nlev*np*np]   {};
  elem_derived_phi          = new real[num_elems*nlev*np*np]   {};
  elem_derived_pecnd        = new real[num_elems*nlev*np*np]   {};
  elem_derived_vn0          = new real[num_elems*nlev*np*np*2] {};

  scratch_2d_0              = new real[np*np] {};
  scratch_2d_1              = new real[np*np] {};

  // Initialize arrays using sin^2(n*x) map.
  // This is easily portable across different platforms and/or
  // languages without relying on implementation details

  // Set seed for the init map
  constexpr double x = 0.123456789;

  int n = 1;
  // Now fiil all the arrays
  for (int ie=0; ie<num_elems; ++ie)
  {
    for (int ip=0; ip<np; ++ip)
    {
      for (int jp=0; jp<np; ++jp)
      {
        double iie = ie + 1;
        double iip = ip + 1;
        double jjp = jp + 1;

        AT_3D(elem_fcor,      ie,ip,jp,np,np)  = sin(iip + jjp);
        AT_3D(elem_metdet,    ie,ip,jp,np,np)  = iip*jjp;
        AT_3D(elem_rmetdet,   ie,ip,jp,np,np)  = 1./ AT_3D(elem_metdet,ie,ip,jp,np,np);
        AT_3D(elem_spheremp,  ie,ip,jp,np,np)  = 2*iip;
        AT_3D(elem_state_phis,ie,ip,jp,np,np)  = iip + jjp;

        AT_5D(elem_D,ie,ip,jp,0,0,np,np,2,2)  = 1.0;
        AT_5D(elem_D,ie,ip,jp,0,1,np,np,2,2)  = 0.0;
        AT_5D(elem_D,ie,ip,jp,1,0,np,np,2,2)  = 0.0;
        AT_5D(elem_D,ie,ip,jp,1,1,np,np,2,2)  = 2.0;

        AT_5D(elem_Dinv,ie,ip,jp,0,0,np,np,2,2) = 1.0;
        AT_5D(elem_Dinv,ie,ip,jp,0,1,np,np,2,2) = 0.0;
        AT_5D(elem_Dinv,ie,ip,jp,1,0,np,np,2,2) = 0.0;
        AT_5D(elem_Dinv,ie,ip,jp,1,1,np,np,2,2) = 0.5;

        for (int il=0; il<nlev; ++il)
        {
          double iil = il + 1;

          AT_4D (elem_derived_phi,    ie,il,ip,jp,  nlev,np,np  ) = cos(iip + 3*jjp) + iil;
          AT_5D (elem_derived_vn0,    ie,il,ip,jp,0,nlev,np,np,2) = 1.0;
          AT_5D (elem_derived_vn0,    ie,il,ip,jp,1,nlev,np,np,2) = 1.0;
          AT_4D (elem_derived_pecnd,  ie,il,ip,jp,  nlev,np,np  ) = 1.0;
          AT_4D (elem_derived_omega_p,ie,il,ip,jp,  nlev,np,np  ) = jjp*jjp;

          AT_6D(elem_state_Qdp,ie,0,0,il,ip,jp,qsize_d,2,nlev,np,np) = 1.0 + sin(iip*jjp*iil);

          for (int it=0; it<timelevels; ++it)
          {
            double iit = it + 1;

            AT_5D(elem_state_dp3d,ie,it,il,ip,jp,  timelevels,nlev,np,np)   = 10.0*iil + iie + iip + jjp + iit;
            AT_6D(elem_state_v,   ie,it,il,ip,jp,0,timelevels,nlev,np,np,2) = 1.0 + 0.5*iil + iip + jjp + 0.2*iie + 2.0*iit;
            AT_6D(elem_state_v,   ie,it,il,ip,jp,1,timelevels,nlev,np,np,2) = 1.0 + 0.5*iil + iip + jjp + 0.2*iie + 3.0*iit;
            AT_5D(elem_state_T,   ie,it,il,ip,jp,  timelevels,nlev,np,np)   = 1000.0 - iil - iip - jjp + 0.1*iie + iit;
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

  delete[] scratch_2d_0;
  delete[] scratch_2d_1;
}

void Constants::init_data ()
{
  Rwater_vapor = 461.5;
  Rgas         = 287.04;
  cp           = 1005.0;
  kappa        = Rgas/cp;
  rrearth      = 1.0/6.376e6;

  eta_ave_w    = 1.0;
}

void Control::init_data ()
{
  nets = 0;
  nete = num_elems;
  n0   = 0;
  np1  = 1;
  nm1  = 2;
  qn0  = 0;

  dt2  = 1.0;
}

void HVCoord::init_data ()
{
  ps0 = 10.0;

  for (int i=0; i<nlevp; ++i)
  {
    hyai[i] = nlev + 1 - i;
  }
}

void Derivative::init_data ()
{
  real values[16] = { -3.0000000000000000, -0.80901699437494745,  0.30901699437494745, -0.50000000000000000,
                       4.0450849718747373,  0.00000000000000000, -1.11803398874989490,  1.54508497187473700,
                      -1.5450849718747370,  1.11803398874989490,  0.00000000000000000, -4.04508497187473730,
                       0.5000000000000000, -0.30901699437494745,  0.80901699437494745, 3.000000000000000000  };
  for (int i=0; i<np; ++i)
  {
    for (int j=0; j<np; ++j)
    {
      Dvv[i][j] = values[j*np + i];
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

void TestData::update_time_levels ()
{
  int tmp = control.np1;
  control.np1 = control.nm1;
  control.nm1 = control.n0;
  control.n0  = tmp;
}

void TestData::cleanup_data ()
{
  arrays.cleanup_data();
}

} // Namespace Homme
