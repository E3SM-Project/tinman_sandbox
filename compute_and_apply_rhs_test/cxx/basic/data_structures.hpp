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

  void init_data ();
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

  // Members of elem%state
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
  real* elem_derived_vn0;

  void init_data ();
  void cleanup_data ();
};

struct Constants
{
  static constexpr const real rrearth = 1.0;
  static constexpr const real eta_ave_w = 1.0;
  static constexpr const real Rwater_vapor = 1.0;
  static constexpr const real Rgas = 10.0;
  static constexpr const real kappa = 1.0;
};

struct Control
{
  int  nets;
  int  nete;
  int  n0;
  int  np1;
  int  nm1;
  int  qn0;
  real dt2;

  void init_data ();
};

struct Derivative
{
  real Dvv[np][np];

  void init_data ();
};

struct TestData
{
  Arrays      arrays    = {};
  Control     control   = {};
  Derivative  deriv     = {};
  HVCoord     hvcoord   = {};

  void init_data ();
  void cleanup_data ();
};

} // Namespace Homme

#endif // DATA_STRUCTURES_HPP
