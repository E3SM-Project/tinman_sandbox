#ifndef DATA_STRUCTURES_HPP
#define DATA_STRUCTURES_HPP

#include "config.h"

#include "Types.hpp"

namespace TinMan
{

struct HVCoord
{
  Real ps0;         // base state surface-pressure for level definitions
  Real hyai[NUM_LEV_P]; // ps0 component of hybrid coordinate - interfaces

  void init_data ();
};

struct Constants
{
  Real  rrearth;
  Real  eta_ave_w;
  Real  Rwater_vapor;
  Real  Rgas;
  Real  kappa;

  void init_data ();
};

struct Control
{
  int  nets;
  int  nete;
  int  n0;
  int  np1;
  int  nm1;
  int  qn0;
  Real dt2;

  void init_data (const int num_elems);
};

struct Derivative
{
  Real Dvv[NP][NP];

  void init_data ();
};

class TestData
{
public:
  TestData (const int num_elems);

  const Constants&  constants() const { return m_constants; }
  const Control&    control()   const { return m_control;   }
  const Derivative& deriv()     const { return m_deriv;     }
  const HVCoord&    hvcoord()   const { return m_hvcoord;   }

private:
  Constants   m_constants = {};
  Control     m_control   = {};
  Derivative  m_deriv     = {};
  HVCoord     m_hvcoord   = {};
};

} // Namespace TinMan

#endif // DATA_STRUCTURES_HPP
