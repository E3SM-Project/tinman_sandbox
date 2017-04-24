#ifndef SPHERE_OPERATORS_HPP
#define SPHERE_OPERATORS_HPP

#include "Dimensions.hpp"
#include "Types.hpp"

#include <Kokkos_Core.hpp>

namespace TinMan {

class Control;

KOKKOS_INLINE_FUNCTION void
gradient_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                const ExecViewUnmanaged<const Real[NP][NP]> scalar,
                const Control &data,
                const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
                ExecViewUnmanaged<Real[2][NP][NP]> buffer,
                ExecViewUnmanaged<Real[2][NP][NP]> grad_s);

KOKKOS_INLINE_FUNCTION void
gradient_sphere_update(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                       const ExecViewUnmanaged<const Real[NP][NP]> scalar,
                       const Control &data,
                       const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
                       ExecViewUnmanaged<Real[2][NP][NP]> buffer,
                       ExecViewUnmanaged<Real[2][NP][NP]> grad_s);

KOKKOS_INLINE_FUNCTION void
divergence_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                  const ExecViewUnmanaged<const Real[2][NP][NP]> v,
                  const Control &data,
                  const ExecViewUnmanaged<const Real[NP][NP]> metDet,
                  const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
                  ExecViewUnmanaged<Real[2][NP][NP]> gv,
                  ExecViewUnmanaged<Real[NP][NP]> div_v);

KOKKOS_FUNCTION void
vorticity_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                 const ExecViewUnmanaged<const Real[NP][NP]> u,
                 const ExecViewUnmanaged<const Real[NP][NP]> v,
                 const Control &data,
                 const ExecViewUnmanaged<const Real[NP][NP]> metDet,
                 const ExecViewUnmanaged<const Real[2][2][NP][NP]> D,
                 ExecViewUnmanaged<Real[2][NP][NP]> vcov,
                 ExecViewUnmanaged<Real[NP][NP]> vort);

// IMPLEMENTATION

// Note that gradient_sphere requires scratch space of 2 x NP x NP Reals
// This must be called from the device space
KOKKOS_INLINE_FUNCTION void
gradient_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                const ExecViewUnmanaged<const Real[NP][NP]> scalar,
                const Control &data,
                const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
                ExecViewUnmanaged<Real[2][NP][NP]> v_buf,
                ExecViewUnmanaged<Real[2][NP][NP]> grad_s) {
  constexpr int contra_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, contra_iters),
                       KOKKOS_LAMBDA(const int loop_idx) {
    const int j = loop_idx / NP;
    const int l = loop_idx % NP;
    Real dsdx(0), dsdy(0);
    for (int i = 0; i < NP; ++i) {
      dsdx += data.dvv(i, l) * scalar(i, j);
      dsdy += data.dvv(i, l) * scalar(j, i);
    }

    v_buf(0, l, j) = dsdx * PhysicalConstants::rrearth;
    v_buf(1, j, l) = dsdy * PhysicalConstants::rrearth;
  });

  constexpr int grad_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, grad_iters),
                       KOKKOS_LAMBDA(const int loop_idx) {
    const int i = loop_idx / NP;
    const int j = loop_idx % NP;
    Real tmp = DInv(0, 0, i, j) * v_buf(0, i, j);
    grad_s(0, i, j) = tmp + DInv(1, 0, i, j) * v_buf(1, i, j);
    tmp = DInv(0, 1, i, j) * v_buf(0, i, j);
    grad_s(1, i, j) = tmp + DInv(1, 1, i, j) * v_buf(1, i, j);
  });
}

KOKKOS_INLINE_FUNCTION void
gradient_sphere_update(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                       const ExecViewUnmanaged<const Real[NP][NP]> scalar,
                       const Control &data,
                       const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
                       ExecViewUnmanaged<Real[2][NP][NP]> v_buf,
                       ExecViewUnmanaged<Real[2][NP][NP]> grad_s) {
  constexpr int contra_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, contra_iters),
                       KOKKOS_LAMBDA(const int loop_idx) {
    const int j = loop_idx / NP;
    const int l = loop_idx % NP;
    Real dsdx(0.0), dsdy(0.0);
    for (int i = 0; i < NP; ++i) {
      dsdx += data.dvv(i, l) * scalar(i, j);
      dsdy += data.dvv(i, l) * scalar(j, i);
    }
    v_buf(0, l, j) = dsdx * PhysicalConstants::rrearth;
    v_buf(1, j, l) = dsdy * PhysicalConstants::rrearth;
  });

  constexpr int grad_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, grad_iters),
                       KOKKOS_LAMBDA(const int loop_idx) {
    const int i = loop_idx / NP;
    const int j = loop_idx % NP;
    Real tmp = DInv(0, 0, i, j) * v_buf(0, i, j);
    tmp += DInv(1, 0, i, j) * v_buf(1, i, j);
    grad_s(0, i, j) += tmp;

    tmp = DInv(0, 1, i, j) * v_buf(0, i, j);
    tmp += DInv(1, 1, i, j) * v_buf(1, i, j);
    grad_s(1, i, j) += tmp;
  });
}

// Note that divergence_sphere requires scratch space of 2 x NP x NP Reals
// This must be called from the device space
KOKKOS_INLINE_FUNCTION void
divergence_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                  const ExecViewUnmanaged<const Real[2][NP][NP]> v,
                  const Control &data,
                  const ExecViewUnmanaged<const Real[NP][NP]> metDet,
                  const ExecViewUnmanaged<const Real[2][2][NP][NP]> DInv,
                  ExecViewUnmanaged<Real[2][NP][NP]> gv,
                  ExecViewUnmanaged<Real[NP][NP]> div_v) {
  constexpr int contra_iters = NP * NP * 2;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, contra_iters),
                       KOKKOS_LAMBDA(const int loop_idx) {
    const int igp = loop_idx / 2 / NP;
    const int jgp = (loop_idx / 2) % NP;
    const int kgp = loop_idx % 2;
    Real cur_gv = DInv(kgp, 0, igp, jgp) * v(0, igp, jgp);
    cur_gv += DInv(kgp, 1, igp, jgp) * v(1, igp, jgp);
    gv(kgp, igp, jgp) = cur_gv * metDet(igp, jgp);
  });

  constexpr int div_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, div_iters),
                       KOKKOS_LAMBDA(const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Real tmp_sum = 0.0;
    for (int kgp = 0; kgp < NP; ++kgp) {
      tmp_sum += data.dvv(kgp, igp) * gv(0, kgp, jgp);
      tmp_sum += data.dvv(kgp, jgp) * gv(1, igp, kgp);
    }
    tmp_sum /= metDet(igp, jgp);
    div_v(igp, jgp) = PhysicalConstants::rrearth * tmp_sum;
  });
}

// Note that divergence_sphere requires scratch space of 3 x NP x NP Reals
// This must be called from the device space
KOKKOS_INLINE_FUNCTION void
vorticity_sphere(const Kokkos::TeamPolicy<ExecSpace>::member_type &team,
                 const ExecViewUnmanaged<const Real[NP][NP]> u,
                 const ExecViewUnmanaged<const Real[NP][NP]> v,
                 const Control &data,
                 const ExecViewUnmanaged<const Real[NP][NP]> metDet,
                 const ExecViewUnmanaged<const Real[2][2][NP][NP]> D,
                 ExecViewUnmanaged<Real[2][NP][NP]> vcov,
                 ExecViewUnmanaged<Real[NP][NP]> vort) {
  constexpr int covar_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, covar_iters),
                       KOKKOS_LAMBDA(const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;

    Real vcov_part1 = D(0, 0, igp, jgp) * u(igp, jgp);
    vcov(0, igp, jgp) = vcov_part1 + D(1, 0, igp, jgp) * v(igp, jgp);

    vcov_part1 = D(0, 1, igp, jgp) * u(igp, jgp);
    vcov(1, igp, jgp) = vcov_part1 + D(1, 1, igp, jgp) * v(igp, jgp);
  });

  constexpr int vort_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, vort_iters),
                       KOKKOS_LAMBDA(const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    Real dudy = 0.0;
    Real dvdx = 0.0;
    for (int kgp = 0; kgp < NP; ++kgp) {
      dudy += data.dvv(kgp, jgp) * vcov(0, igp, kgp);
      dvdx += data.dvv(kgp, igp) * vcov(1, kgp, jgp);
    }

    vort(igp, jgp) =
        PhysicalConstants::rrearth * (dvdx - dudy) / metDet(igp, jgp);
  });
}

} // Namespace TinMan

#endif // SPHERE_OPERATORS_HPP
