#include <cstdlib>

#include "dimensions.hpp"
#include "data_structures.hpp"

namespace Homme
{

int init_test_data (TestData& data)
{
  data.arrays.elem_D                    = new real[nelems*np*np*2*2];
  data.arrays.elem_Dinv                 = new real[nelems*np*np*2*2];
  data.arrays.elem_fcor                 = new real[nelems*np*np];
  data.arrays.elem_spheremp             = new real[nelems*np*np];
  data.arrays.elem_metdet               = new real[nelems*np*np];
  data.arrays.elem_rmetdet              = new real[nelems*np*np];
  data.arrays.elem_sub_elem_mass_flux   = new real[nlev*nc*nc*4];

  data.arrays.elem_state_ps_v           = new real[nelems*timelevels*np*np];
  data.arrays.elem_state_dp3d           = new real[nelems*timelevels*np*np*nlev];
  data.arrays.elem_state_v              = new real[nelems*timelevels*np*np*2*nlev];
  data.arrays.elem_state_T              = new real[nelems*timelevels*np*np*nlev];
  data.arrays.elem_state_phis           = new real[nelems*np*np];
  data.arrays.elem_state_Qdp            = new real[nelems*np*np*nlev*qsize_d*2];

  data.arrays.elem_derived_eta_dot_dpdn = new real[nelems*np*np*nlevp];
  data.arrays.elem_derived_omega_p      = new real[nelems*nlev*np*np];
  data.arrays.elem_derived_phi          = new real[nelems*nlev*np*np];
  data.arrays.elem_derived_pecnd        = new real[nelems*nlev*np*np];
  data.arrays.elem_derived_ps_met       = new real[nelems*np*np];
  data.arrays.elem_derived_dpsdt_met    = new real[nelems*np*np];
  data.arrays.elem_derived_vn0          = new real[nelems*nlev*np*np*2];

  return EXIT_SUCCESS;
}

int cleanup_data (TestData& data)
{
  delete[] data.arrays.elem_D;
  delete[] data.arrays.elem_Dinv;
  delete[] data.arrays.elem_fcor;
  delete[] data.arrays.elem_spheremp;
  delete[] data.arrays.elem_metdet;
  delete[] data.arrays.elem_rmetdet;
  delete[] data.arrays.elem_sub_elem_mass_flux;

  delete[] data.arrays.elem_state_ps_v;
  delete[] data.arrays.elem_state_dp3d;
  delete[] data.arrays.elem_state_v;
  delete[] data.arrays.elem_state_T;
  delete[] data.arrays.elem_state_phis;
  delete[] data.arrays.elem_state_Qdp;

  delete[] data.arrays.elem_derived_eta_dot_dpdn;
  delete[] data.arrays.elem_derived_omega_p;
  delete[] data.arrays.elem_derived_phi;
  delete[] data.arrays.elem_derived_pecnd;
  delete[] data.arrays.elem_derived_ps_met;
  delete[] data.arrays.elem_derived_dpsdt_met;
  delete[] data.arrays.elem_derived_vn0;

  return EXIT_SUCCESS;
}

} // Namespace Homme
