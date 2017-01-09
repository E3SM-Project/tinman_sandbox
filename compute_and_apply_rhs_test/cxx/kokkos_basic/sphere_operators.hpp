#ifndef SPHERE_OPERATORS_HPP
#define SPHERE_OPERATORS_HPP

#include "config.h"
#include "Types.hpp"

#include <Kokkos_Core.hpp>

namespace TinMan
{

class TestData;

void gradient_sphere (const ViewUnmanaged<Real[NP][NP]> s, const TestData& data,
                      const ViewUnmanaged<Real[2][2][NP][NP]> DInv,
                      ViewUnmanaged<Real[2][NP][NP]> grad_s);

void gradient_sphere (Kokkos::TeamPolicy<>::member_type &team,
                      const ViewUnmanaged<Real[NP][NP]> s, const TestData& data,
                      const ViewUnmanaged<Real[2][2][NP][NP]> DInv,
                      ViewUnmanaged<Real[2][NP][NP]> grad_s);

void divergence_sphere (const ViewUnmanaged<Real[2][NP][NP]> v, const TestData& data,
                        const ViewUnmanaged<Real[NP][NP]> metDet,
                        const ViewUnmanaged<Real[2][2][NP][NP]> DInv,
                        ViewUnmanaged<Real[NP][NP]> div_v);

void divergence_sphere (Kokkos::TeamPolicy<>::member_type &team,
                        const ViewUnmanaged<Real[2][NP][NP]> v, const TestData& data,
                        const ViewUnmanaged<Real[NP][NP]> metDet,
                        const ViewUnmanaged<Real[2][2][NP][NP]> DInv,
                        ViewUnmanaged<Real[NP][NP]> div_v);

void vorticity_sphere (const ViewUnmanaged<Real[2][NP][NP]> v, const TestData& data,
                       const ViewUnmanaged<Real[NP][NP]> metDet,
                       const ViewUnmanaged<Real[2][2][NP][NP]> D,
                       ViewUnmanaged<Real[NP][NP]> vort);

void vorticity_sphere (Kokkos::TeamPolicy<>::member_type &team,
                       const ViewUnmanaged<Real[2][NP][NP]> v, const TestData& data,
                       const ViewUnmanaged<Real[NP][NP]> metDet,
                       const ViewUnmanaged<Real[2][2][NP][NP]> D,
                       ViewUnmanaged<Real[NP][NP]> vort);

} // Namespace TinMan

#endif // SPHERE_OPERATORS_HPP
