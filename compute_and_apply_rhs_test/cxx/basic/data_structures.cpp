#include "dimensions.hpp"
#include "data_structures.hpp"

#include "test_macros.hpp"
#include <random>

namespace
{
double logistic_map(double x, const double r)
{
  return r*x*(1-x);
}

} // Anonymous namespace

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
  elem_state_Qdp            = new real[num_elems*nlev*qsize_d*2*np*np]    {};

  elem_derived_eta_dot_dpdn = new real[num_elems*nlevp*np*np]  {};
  elem_derived_omega_p      = new real[num_elems*nlev*np*np]   {};
  elem_derived_phi          = new real[num_elems*nlev*np*np]   {};
  elem_derived_pecnd        = new real[num_elems*nlev*np*np]   {};
  elem_derived_vn0          = new real[num_elems*nlev*np*np*2] {};

  // Starting point for the logistic map
  double x = 0.123456789;

  // Logistic parameter: 4.0 gives the whole (0,1) as an attractor
  constexpr double r = 3.987654321;

  // Initialize arrays using logistic map. Although techically deterministic, if
  // r/gtrsim 3.57, it is a chaotic system. And it's easily portable across
  // different platforms and/or languages without relying on implementation details

  for (int ie=0; ie<num_elems; ++ie)
  {
    for (int ip=0; ip<np; ++ip)
    {
      for (int jp=0; jp<np; ++jp)
      {
        AT_5D(elem_D,ie,ip,jp,0,0,np,np,2,2) = x = logistic_map(x,r);
        AT_5D(elem_D,ie,ip,jp,0,1,np,np,2,2) = x = logistic_map(x,r);
        AT_5D(elem_D,ie,ip,jp,1,0,np,np,2,2) = x = logistic_map(x,r);
        AT_5D(elem_D,ie,ip,jp,1,1,np,np,2,2) = x = logistic_map(x,r);

        double detD = AT_5D(elem_D,ie,ip,jp,0,0,np,np,2,2)*AT_5D(elem_D,ie,ip,jp,1,1,np,np,2,2)
                    - AT_5D(elem_D,ie,ip,jp,0,1,np,np,2,2)*AT_5D(elem_D,ie,ip,jp,1,0,np,np,2,2);

        AT_5D(elem_Dinv,ie,ip,jp,0,0,np,np,2,2) =  AT_5D(elem_D,ie,ip,jp,1,1,np,np,2,2) / detD;
        AT_5D(elem_Dinv,ie,ip,jp,0,1,np,np,2,2) = -AT_5D(elem_D,ie,ip,jp,0,1,np,np,2,2) / detD;
        AT_5D(elem_Dinv,ie,ip,jp,1,0,np,np,2,2) = -AT_5D(elem_D,ie,ip,jp,1,0,np,np,2,2) / detD;
        AT_5D(elem_Dinv,ie,ip,jp,1,1,np,np,2,2) =  AT_5D(elem_D,ie,ip,jp,0,0,np,np,2,2) / detD;

        AT_3D(elem_fcor,      ie,ip,jp,np,np) = x = logistic_map(x,r);
        AT_3D(elem_spheremp,  ie,ip,jp,np,np) = x = logistic_map(x,r);
        AT_3D(elem_metdet,    ie,ip,jp,np,np) = x = logistic_map(x,r);
        AT_3D(elem_state_phis,ie,ip,jp,np,np) = x = logistic_map(x,r);

        AT_3D(elem_rmetdet,   ie,ip,jp,np,np) = 1./ AT_3D(elem_metdet,ie,ip,jp,np,np);

        for (int il=0; il<nlev; ++il)
        {
          AT_4D (elem_derived_omega_p,ie,il,ip,jp,  nlev,np,np  ) = x = logistic_map(x,r);
          AT_4D (elem_derived_pecnd,  ie,il,ip,jp,  nlev,np,np  ) = x = logistic_map(x,r);
          AT_5D (elem_derived_vn0,    ie,il,ip,jp,0,nlev,np,np,2) = x = logistic_map(x,r);
          AT_5D (elem_derived_vn0,    ie,il,ip,jp,1,nlev,np,np,2) = x = logistic_map(x,r);
          AT_4D (elem_derived_phi,    ie,il,ip,jp,  nlev,np,np  ) = x = logistic_map(x,r);

          for (int iq=0; iq<qsize_d; ++iq)
          {
            AT_6D(elem_state_Qdp,ie,il,ip,jp,iq,0,nlev,np,np,qsize_d,2) = x = logistic_map(x,r);
            AT_6D(elem_state_Qdp,ie,il,ip,jp,iq,1,nlev,np,np,qsize_d,2) = x = logistic_map(x,r);
          }

          for (int it=0; it<timelevels; ++it)
          {

            AT_5D(elem_state_dp3d,ie,it,il,ip,jp,  timelevels,nlev,np,np)   = x = logistic_map(x,r);
            AT_5D(elem_state_T,   ie,it,il,ip,jp,  timelevels,nlev,np,np)   = x = logistic_map(x,r);
            AT_6D(elem_state_v,   ie,it,il,ip,jp,0,timelevels,nlev,np,np,2) = x = logistic_map(x,r);
            AT_6D(elem_state_v,   ie,it,il,ip,jp,1,timelevels,nlev,np,np,2) = x = logistic_map(x,r);
          }
        }

        for (int il=0; il<nlevp; ++il)
        {
          AT_4D (elem_derived_eta_dot_dpdn,ie,il,ip,jp,nlevp,np,np) = x = logistic_map(x,r);
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
  kappa        = 1.0;
  Rwater_vapor = 1.0;
  Rgas         = 10.0;
  eta_ave_w    = 1.0;
}

void Control::init_data ()
{
  nets = 0;
  nete = num_elems;
  n0 = 0;
  np1 = 1;
  nm1 = 2;
  qn0 = 0;

  real dt2 = 1.0;
}

void HVCoord::init_data ()
{
  ps0 = 1.0;

  for (int i=0; i<nlevp; ++i)
  {
    hyai[i] = 1.0;
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
      Dvv[i][j] = values[(j-1)*np + i];
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
