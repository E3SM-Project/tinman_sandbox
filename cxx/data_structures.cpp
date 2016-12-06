#include <cstdlib>

#include "test_macros.hpp"

#include "data_structures.hpp"

int init_test_data (TestData& data)
{
  constexpr int np          = NUM_POINTS;
  constexpr int nc          = NUM_CORNERS;
  constexpr int nlev        = NUM_LEVELS;
  constexpr int nlevp       = NUM_LEVELS_P;
  constexpr int nelems      = NUM_ELEMENTS;
  constexpr int timelevels  = NUM_TIME_LEVELS;
  constexpr int qsize_d     = QSIZE_D;

  data.arrays.elem_Dinv                 = new real[np*np*2*2*nelems];
  data.arrays.elem_fcor                 = new real[np*np*nelems];
  data.arrays.elem_spheremp             = new real[np*np*nelems];
  data.arrays.elem_metdet               = new real[np*np*nelems];
  data.arrays.elem_sub_elem_mass_flux   = new real[nc*nc*4*nlev];

  data.arrays.elem_state_ps_v           = new real[np*np*timelevels*nelems];
  data.arrays.elem_state_dp3d           = new real[np*np*nlev*timelevels*nelems];
  data.arrays.elem_state_v              = new real[np*np*2*nlev*timelevels*nelems];
  data.arrays.elem_state_T              = new real[np*np*nlev*timelevels*nelems];
  data.arrays.elem_state_phis           = new real[np*np*nelems];
  data.arrays.elem_state_Qdp            = new real[np*np*nlev*qsize_d*2*nelems];

  data.arrays.elem_derived_eta_dot_dpdn = new real[np*np*nlevp*nelems];
  data.arrays.elem_derived_omega_p      = new real[np*np*nlev*nelems];
  data.arrays.elem_derived_phi          = new real[np*np*nlev*nelems];
  data.arrays.elem_derived_pecnd        = new real[np*np*nlev*nelems];
  data.arrays.elem_derived_ps_met       = new real[np*np*nelems];
  data.arrays.elem_derived_dpsdt_met    = new real[np*np*nelems];
  data.arrays.elem_derived_vn0          = new real[np*np*2*nlev*nelems];

  return EXIT_SUCCESS;
}

int cleanup_data (TestData& data)
{
  delete[] data.arrays.elem_Dinv;
  delete[] data.arrays.elem_fcor;
  delete[] data.arrays.elem_spheremp;
  delete[] data.arrays.elem_metdet;
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
