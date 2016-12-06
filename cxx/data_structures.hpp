#ifndef DATA_STRUCTURES_HPP
#define DATA_STRUCTURES_HPP

#include "test_macros.hpp"

struct Arrays
{
  // Members of elem
  real* elem_Dinv;
  real* elem_fcor;
  real* elem_spheremp;
  real* elem_metdet;
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

};

struct Control
{
  int     rsplit;
};

struct Derivative
{

};

struct TestData
{
  Arrays      arrays;
  Constants   constants;
  Control     control;
  Derivative  deriv;
};

int init_test_data (TestData& data);

int cleanup_data (TestData& data);

#endif // DATA_STRUCTURES_HPP
