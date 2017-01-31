#ifndef DATA_STRUCTURES_HPP
#define DATA_STRUCTURES_HPP

#include "config.h"

#include "Types.hpp"

namespace TinMan {

struct PhysicalConstants {
  static constexpr const Real rrearth = 1.0;
  static constexpr const Real eta_ave_w = 1.0;
  static constexpr const Real Rwater_vapor = 1.0;
  static constexpr const Real Rgas = 10.0;
  static constexpr const Real kappa = 1.0;
};

class Control {
public:
  // num_elems is te number of elements in the simulation
  // num_workers is the number of threads used to simulate
  // thread_id is between 0 and num_elems-1, inclusive
  // thread_id should be unique between threads
  Control(int num_elems);

  int host_num_elems() const { return m_host_num_elems; }

  /* These functions must be called from device code */
  KOKKOS_INLINE_FUNCTION int num_elems() const {
    return m_device_mem(0).num_elems;
  }
  /* Timelevels 1-3, respectively */
  KOKKOS_INLINE_FUNCTION int n0() const { return m_device_mem(0).n0; }
  KOKKOS_INLINE_FUNCTION int np1() const { return m_device_mem(0).np1; }
  KOKKOS_INLINE_FUNCTION int nm1() const { return m_device_mem(0).nm1; }
  /* Tracer timelevel */
  KOKKOS_INLINE_FUNCTION int qn0() const { return m_device_mem(0).qn0; }
  /* dt * 2 */
  KOKKOS_INLINE_FUNCTION int dt2() const { return m_device_mem(0).dt2; }

  /* ps0 component of hybrid coordinate-interfaces
   * The A part of the pressure at a level
   * Complemented by hybrid_b, which is not needed in this code
   */
  KOKKOS_INLINE_FUNCTION Real hybrid_a(int level) const {
    return m_hybrid_a(level);
  }
  /* Global constant - boundary condition at the top of the atmosphere
   * This is used to ensure that the top level has pressure ptop
   * The pressure at a level k is approximately:
   * p(k) = A(k) ps0 + B(k) ps_v
   */
  KOKKOS_INLINE_FUNCTION Real ps0() const { return m_device_mem(0).ps0; }

  KOKKOS_INLINE_FUNCTION Real dvv(int x, int y) const { return m_dvv(x, y); }

  void update_time_levels();

  struct Control_Data {
    int num_elems;
    int n0;
    int np1;
    int nm1;
    int qn0;
    Real dt2;

    Real ps0; // base state surface-pressure for level definitions
  };
private:

  const int m_host_num_elems;

  /* Device objects */
  ExecViewManaged<Control_Data[1]> m_device_mem;
  ExecViewManaged<Real[NUM_LEV_P]> m_hybrid_a;
  ExecViewManaged<Real[NP][NP]> m_dvv; // Laplacian
};

} // Namespace TinMan

#endif // DATA_STRUCTURES_HPP
