#ifndef DATA_STRUCTURES_HPP
#define DATA_STRUCTURES_HPP

#include "config.h"

#include "Types.hpp"

namespace TinMan
{

struct Constants
{
  static constexpr const Real rrearth = 1.0;
  static constexpr const Real eta_ave_w = 1.0;
  static constexpr const Real Rwater_vapor = 1.0;
  static constexpr const Real Rgas = 10.0;
  static constexpr const Real kappa = 1.0;
};

struct HVCoord
{
  Real ps0;         // base state surface-pressure for level definitions
  Real hyai[NUM_LEV_P]; // ps0 component of hybrid coordinate - interfaces
};

class Control
{
public:
  // num_elems is te number of elements in the simulation
  // num_workers is the number of threads used to simulate
  // thread_id is between 0 and num_elems-1, inclusive
  // thread_id should be unique between threads
  Control(int num_elems, int num_threads);

  int nets(int thread_id) const;
  int nete(int thread_id) const;
  int n0(int thread_id) const;
  int np1(int thread_id) const;
  int nm1(int thread_id) const;
  int qn0(int thread_id) const;
  int dt2(int thread_id) const;
private:
  struct Control_Data {
    int  nets;
    int  nete;
    int  n0;
    int  np1;
    int  nm1;
    int  qn0;
    Real dt2;
  };

  // Use this "struct" to ensure each Control_Data object fills a cache line
  // This is necessary to prevent threads on different cores from causing false-sharing cache issues
  template <typename cache_filler>
  struct Cache_Wrapper : public cache_filler
  {
  private:
    // x86 cache line size is 64
    // CUDA cache line size may vary, but Fermi and Kepler are 128
    constexpr const int cache_line_size = 128;
    constexpr const int leftover_line = cache_line_size - (sizeof(cache_filler) % cache_line_size);
    char filler[leftover_line];
  };
  ExecViewManaged<Cache_Wrapper<Control_Data> *> thread_control;
};

struct Derivative
{
public:
  Derivative();
  ExecViewManaged<Real[NP][NP]> Dvv;
};

class TestData
{
public:
  TestData (const int num_elems);

  const Control&    control()   const { return m_control;   }
  const Derivative& deriv()     const { return m_deriv;     }
  const HVCoord&    hvcoord()   const { return m_hvcoord;   }

  void update_time_levels();

private:
  Control     m_control   = {};
  Derivative  m_deriv     = {};
  HVCoord     m_hvcoord   = {};
};

} // Namespace TinMan

#endif // DATA_STRUCTURES_HPP
