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

  // This constructor should only be used by the host
  Control(int num_elems);

  /* These functions must be called from device code */
  KOKKOS_INLINE_FUNCTION int num_elems() const {
    return m_num_elems;
  }
  /* Tracer timelevel */
  KOKKOS_INLINE_FUNCTION int qn0() const { return m_qn0; }
  /* dt * 2 */
  KOKKOS_INLINE_FUNCTION int dt2() const { return m_dt2; }

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
  KOKKOS_INLINE_FUNCTION Real ps0() const { return m_ps0; }

  KOKKOS_INLINE_FUNCTION Real dvv(int x, int y) const { return m_dvv(x, y); }

private:
  const int m_num_elems;

  // tracer timelevel?
  const int m_qn0;

  // dt
  const int m_dt2;

  const int m_ps0;

  /* Device objects, to reduce the memory transfer required */
  ExecViewManaged<Real[NUM_LEV_P]> m_hybrid_a;
  ExecViewManaged<Real[NP][NP]> m_dvv; // Laplacian
};

} // Namespace TinMan

#endif // DATA_STRUCTURES_HPP
