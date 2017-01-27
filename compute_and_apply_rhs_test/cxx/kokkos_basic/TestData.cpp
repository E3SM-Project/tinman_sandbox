#include "TestData.hpp"

namespace TinMan
{

void Constants::init_data ()
{
  kappa        = 1.0;
  Rwater_vapor = 1.0;
  Rgas         = 10.0;
  eta_ave_w    = 1.0;
}

void Control::init_data (const int num_elems)
{
  nets = 0;
  nete = num_elems;
  n0 = 0;
  np1 = 1;
  nm1 = 2;
  qn0 = 0;

  Real dt2 = 1.0;
}

void HVCoord::init_data ()
{
  ps0 = 1.0;

  for (int i=0; i<NUM_LEV_P; ++i)
  {
    hyai[i] = 1.0;
  }
}

void Derivative::init_data ()
{
  Real values[16] = { -3.0000000000000000, -0.80901699437494745,  0.30901699437494745, -0.50000000000000000,
                       4.0450849718747373,  0.00000000000000000, -1.11803398874989490,  1.54508497187473700,
                      -1.5450849718747370,  1.11803398874989490,  0.00000000000000000, -4.04508497187473730,
                       0.5000000000000000, -0.30901699437494745,  0.80901699437494745, 3.000000000000000000  };
  for (int i=0; i<NP; ++i)
  {
    for (int j=0; j<NP; ++j)
    {
      Dvv[i][j] = values[j*NP + i];
    }
  }
}

TestData::TestData (const int num_elems)
{
  m_constants.init_data();
  m_control.init_data(num_elems);
  m_hvcoord.init_data();
  m_deriv.init_data();
}

void TestData::update_time_levels()
{
  int tmp = m_control.np1;
  m_control.np1 = m_control.nm1;
  m_control.nm1 = m_control.n0;
  m_control.n0  = tmp;
}

} // Namespace TinMan
