#ifndef DATA_STRUCTURES_HPP
#define DATA_STRUCTURES_HPP

#include "kinds.hpp"
#include "dimensions.hpp"

namespace Homme
{

struct HVCoord
{
  real ps0;         // base state surface-pressure for level definitions
  real hyai[nlevp]; // ps0 component of hybrid coordinate - interfaces
  real hyam[nlev];  // ps0 component of hybrid coordinate - midpoints
  real hybi[nlevp]; // ps  component of hybrid coordinate - interfaces
  real hybm[nlev];  // ps  component of hybrid coordinate - midpoints
  real hybd[nlev];  // difference in b (hybi) across layers
  real prsfac;      // log pressure extrapolation factor (time, space independent)
  real etam[nlev];  // eta-levels at midpoints
  real etai[nlevp]; // eta-levels at interfaces

  int nprlev;       // number of pure pressure levels at top
  int pad;
};

struct Arrays
{
  // Members of elem
  real* elem_D;
  real* elem_Dinv;
  real* elem_fcor;
  real* elem_spheremp;
  real* elem_metdet;
  real* elem_rmetdet;
  real* elem_sub_elem_mass_flux;

  // Members of elem%state
  real* elem_state_ps_v;
  real* elem_state_dp3d;
  real* elem_state_v;
  real* elem_state_T;
  real* elem_state_phis;
  real* elem_state_Qdp;

  // Members of elem%derived
  real* elem_derived_eta_dot_dpdn;
  real* elem_derived_omega_p;
  real* elem_derived_phi;
  real* elem_derived_pecnd;
  real* elem_derived_ps_met;
  real* elem_derived_dpsdt_met;
  real* elem_derived_vn0;
};

struct Constants
{
  real  rrearth;
  real  eta_ave_w;
};

struct Control
{
  int     rsplit;
};

struct Derivative
{
  real Dvv[np][np];
};

struct TestData
{
  Arrays      arrays;
  Constants   constants;
  Control     control;
  Derivative  deriv;
  HVCoord     hvcoord;
};

int init_test_data (TestData& data);

int cleanup_data (TestData& data);

} // Namespace Homme

#endif // DATA_STRUCTURES_HPP
