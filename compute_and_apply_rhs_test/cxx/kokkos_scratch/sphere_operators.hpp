#ifndef SPHERE_OPERATORS_HPP
#define SPHERE_OPERATORS_HPP

#include "config.h"
#include "Types.hpp"

#include <Kokkos_Core.hpp>

namespace TinMan
{

class Control;

template<typename MemSpaceIn, typename MemSpaceOut>
void gradient_sphere (const ViewType<Real[NP][NP],MemSpaceIn,Kokkos::MemoryUnmanaged> s,
                      const Control& data,
                      const ExecViewUnmanaged<Real[2][2][NP][NP]> DInv,
                      ViewType<Real[2][NP][NP],MemSpaceOut,Kokkos::MemoryUnmanaged> grad_s);

template<typename MemSpaceIn, typename MemSpaceOut>
KOKKOS_INLINE_FUNCTION
void gradient_sphere (const Kokkos::TeamPolicy<>::member_type &team,
                      const ViewType<Real[NP][NP],MemSpaceIn,Kokkos::MemoryUnmanaged> s,
                      const Control& data,
                      const ExecViewUnmanaged<Real[2][2][NP][NP]> DInv,
                      ViewType<Real[2][NP][NP],MemSpaceOut,Kokkos::MemoryUnmanaged> grad_s);

template<typename MemSpaceIn, typename MemSpaceOut>
void divergence_sphere (const ExecViewUnmanaged<Real[2][NP][NP]> v,
                        const Control& data,
                        const ExecViewUnmanaged<Real[NP][NP]> metDet,
                        const ExecViewUnmanaged<Real[2][2][NP][NP]> DInv,
                        ExecViewUnmanaged<Real[NP][NP]> div_v);

template<typename MemSpaceIn, typename MemSpaceOut>
KOKKOS_INLINE_FUNCTION
void divergence_sphere (const Kokkos::TeamPolicy<>::member_type &team,
                        const ViewType<Real[2][NP][NP],MemSpaceIn,Kokkos::MemoryUnmanaged> v,
                        const Control& data,
                        const ExecViewUnmanaged<Real[NP][NP]> metDet,
                        const ExecViewUnmanaged<Real[2][2][NP][NP]> DInv,
                        ViewType<Real[NP][NP],MemSpaceOut,Kokkos::MemoryUnmanaged> div_v);

template<typename MemSpaceIn, typename MemSpaceOut>
void vorticity_sphere (const ViewType<Real[NP][NP],MemSpaceIn,Kokkos::MemoryUnmanaged> u,
                       const ViewType<Real[NP][NP],MemSpaceIn,Kokkos::MemoryUnmanaged> v,
                       const Control& data,
                       const ExecViewUnmanaged<Real[NP][NP]> metDet,
                       const ExecViewUnmanaged<Real[2][2][NP][NP]> D,
                       ViewType<Real[NP][NP],MemSpaceOut,Kokkos::MemoryUnmanaged> vort);

template<typename MemSpaceIn, typename MemSpaceOut>
KOKKOS_INLINE_FUNCTION
void vorticity_sphere (const Kokkos::TeamPolicy<>::member_type &team,
                       const ViewType<Real[NP][NP],MemSpaceIn,Kokkos::MemoryUnmanaged> u,
                       const ViewType<Real[NP][NP],MemSpaceIn,Kokkos::MemoryUnmanaged> v,
                       const Control& data,
                       const ExecViewUnmanaged<Real[NP][NP]> metDet,
                       const ExecViewUnmanaged<Real[2][2][NP][NP]> D,
                       ViewType<Real[NP][NP],MemSpaceOut,Kokkos::MemoryUnmanaged> vort);

// ==================================== IMPLEMENTATION =================================== //

template<typename MemSpaceIn, typename MemSpaceOut>
void gradient_sphere (const ViewType<Real[NP][NP],MemSpaceIn,Kokkos::MemoryUnmanaged> s,
                      const Control& data,
                      const ExecViewUnmanaged<Real[2][2][NP][NP]> DInv,
                      ViewType<Real[2][NP][NP],MemSpaceOut,Kokkos::MemoryUnmanaged> grad_s)
{
  Real rrearth = PhysicalConstants::rrearth;

  Real dsdx, dsdy;
  Real v1[NP][NP];
  Real v2[NP][NP];
  for (int j=0; j<NP; ++j)
  {
    for (int l=0; l<NP; ++l)
    {
      dsdx = dsdy = 0;
      for (int i=0; i<NP; ++i)
      {
        dsdx += data.dvv(i, l)*s(i,j);
        dsdy += data.dvv(i, l)*s(j,i);
      }

      v1[l][j] = dsdx * rrearth;
      v2[j][l] = dsdy * rrearth;
    }
  }

  for (int j=0; j<NP; ++j)
  {
    for (int i=0; i<NP; ++i)
    {
      grad_s(0, i, j) = DInv(0,0,i,j)*v1[i][j]+ DInv(1,0,i,j)*v2[i][j];
      grad_s(1, i, j) = DInv(0,1,i,j)*v1[i][j]+ DInv(1,1,i,j)*v2[i][j];
    }
  }
}

// Note that gradient_sphere requires scratch space of 2 x NP x NP Reals
// This must be called from the device space
template<typename MemSpaceIn, typename MemSpaceOut>
KOKKOS_INLINE_FUNCTION
void gradient_sphere (const Kokkos::TeamPolicy<>::member_type &team,
                      const ViewType<Real[NP][NP],MemSpaceIn,Kokkos::MemoryUnmanaged> s,
                      const Control& data,
                      const ExecViewUnmanaged<Real[2][2][NP][NP]> DInv,
                      ViewType<Real[2][NP][NP],MemSpaceOut,Kokkos::MemoryUnmanaged> grad_s)
{
  Real rrearth = PhysicalConstants::rrearth;

  ScratchView<Real[2][NP][NP]> v(team.team_scratch(0));
  constexpr const int contra_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, contra_iters),
                       [&](const int loop_idx) {
    const int j = loop_idx / NP;
    const int l = loop_idx % NP;
    Real dsdx, dsdy;
    dsdx = dsdy = 0;
    for (int i=0; i<NP; ++i)
    {
      dsdx += data.dvv(i, l)*s(i,j);
      dsdy += data.dvv(i, l)*s(j,i);
    }

    v(0, l, j) = dsdx * rrearth;
    v(1, j, l) = dsdy * rrearth;
  });

  constexpr const int grad_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, grad_iters),
                       KOKKOS_LAMBDA(const int loop_idx) {
    const int j = loop_idx % NP;
    const int i = loop_idx / NP;
    grad_s(0, i, j) = DInv(0,0,i,j) * v(0, i, j) + DInv(1,0,i,j) * v(1, i, j); // Poor performance here
    grad_s(1, i, j) = DInv(0,1,i,j) * v(0, i, j) + DInv(1,1,i,j) * v(1, i, j); // Poor performance here
  });
}

template<typename MemSpaceIn, typename MemSpaceOut>
void divergence_sphere (const ViewType<Real[2][NP][NP],MemSpaceIn,Kokkos::MemoryUnmanaged> v,
                        const Control& data,
                        const ExecViewUnmanaged<Real[NP][NP]> metDet,
                        const ExecViewUnmanaged<Real[2][2][NP][NP]> DInv,
                        ViewType<Real[NP][NP],MemSpaceOut,Kokkos::MemoryUnmanaged> div_v)
{
  Real rrearth = PhysicalConstants::rrearth;

  Real gv[2][NP][NP];
  for (int igp=0; igp<NP; ++igp)
  {
    for (int jgp=0; jgp<NP; ++jgp)
    {
      gv[0][igp][jgp] = metDet(igp,jgp) * ( DInv(0,0,igp,jgp)*v(0,igp,jgp) + DInv(0,1,igp,jgp)*v(1,igp,jgp) );
      gv[1][igp][jgp] = metDet(igp,jgp) * ( DInv(1,0,igp,jgp)*v(0,igp,jgp) + DInv(1,1,igp,jgp)*v(1,igp,jgp) );
    }
  }

  Real dudx, dvdy;
  for (int igp=0; igp<NP; ++igp)
  {
    for (int jgp=0; jgp<NP; ++jgp)
    {
      dudx = dvdy = 0.;
      for (int kgp=0; kgp<NP; ++kgp)
      {
        dudx += data.dvv(kgp, igp) * gv[0][kgp][jgp];
        dvdy += data.dvv(kgp, jgp) * gv[1][igp][kgp];
      }

      div_v(igp,jgp) = rrearth * (dudx + dvdy) / metDet(igp,jgp);
    }
  }
}

// Note that divergence_sphere requires scratch space of NP x NP Reals
// This must be called from the device space
template<typename MemSpaceIn, typename MemSpaceOut>
KOKKOS_INLINE_FUNCTION
void divergence_sphere (const Kokkos::TeamPolicy<>::member_type &team,
                        const ViewType<Real[2][NP][NP],MemSpaceIn,Kokkos::MemoryUnmanaged> v,
                        const Control& data,
                        const ExecViewUnmanaged<Real[NP][NP]> metDet,
                        const ExecViewUnmanaged<Real[2][2][NP][NP]> DInv,
                        ViewType<Real[NP][NP],MemSpaceOut,Kokkos::MemoryUnmanaged> div_v)
{
  Real rrearth = PhysicalConstants::rrearth;

  Real gv[2][NP][NP];
  constexpr const int contra_iters = NP * NP * 2;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, contra_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / 2 / NP;
    const int jgp = (loop_idx / 2) % NP;
    const int kgp = loop_idx % 2;
    gv[kgp][igp][jgp] = metDet(igp,jgp) * ( DInv(kgp,0,igp,jgp)*v(0,igp,jgp) + DInv(kgp,1,igp,jgp)*v(1,igp,jgp) );
  });

  Real dudx, dvdy;
  constexpr const int div_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, div_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    dudx = dvdy = 0.;
    for (int kgp=0; kgp<NP; ++kgp)
    {
      dudx += data.dvv(kgp, igp) * gv[0][kgp][jgp];
      dvdy += data.dvv(kgp, jgp) * gv[1][igp][kgp];
    }

    div_v(igp,jgp) = rrearth * (dudx + dvdy) / metDet(igp,jgp);
  });
}

template<typename MemSpaceIn, typename MemSpaceOut>
void vorticity_sphere (const ViewType<Real[NP][NP],MemSpaceIn,Kokkos::MemoryUnmanaged> u,
                       const ViewType<Real[NP][NP],MemSpaceIn,Kokkos::MemoryUnmanaged> v,
                       const Control& data,
                       const ExecViewUnmanaged<Real[NP][NP]> metDet,
                       const ExecViewUnmanaged<Real[2][2][NP][NP]> D,
                       ViewType<Real[NP][NP],MemSpaceOut,Kokkos::MemoryUnmanaged> vort)
{
  Real rrearth = PhysicalConstants::rrearth;

  Real vcov[2][NP][NP];
  for (int igp=0; igp<NP; ++igp)
  {
    for (int jgp=0; jgp<NP; ++jgp)
    {
      vcov[0][igp][jgp] = D(0,0,igp,jgp)*u(igp,jgp) + D(1,0,igp,jgp)*v(igp,jgp);
      vcov[1][igp][jgp] = D(0,1,igp,jgp)*u(igp,jgp) + D(1,1,igp,jgp)*v(igp,jgp);
    }
  }

  Real dudy, dvdx;
  for (int igp=0; igp<NP; ++igp)
  {
    for (int jgp=0; jgp<NP; ++jgp)
    {
      dudy = dvdx = 0.;
      for (int kgp=0; kgp<NP; ++kgp)
      {
        dudy += data.dvv(kgp, jgp) * vcov[1][igp][kgp];
        dvdx += data.dvv(kgp, igp) * vcov[0][kgp][jgp];
      }

      vort(igp,jgp) = rrearth * (dvdx - dudy) / metDet(igp,jgp);
    }
  }
}

// Note that divergence_sphere requires scratch space of 3 x NP x NP Reals
// This must be called from the device space
template<typename MemSpaceIn, typename MemSpaceOut>
KOKKOS_INLINE_FUNCTION
void vorticity_sphere (const Kokkos::TeamPolicy<>::member_type &team,
                       const ViewType<Real[NP][NP],MemSpaceIn,Kokkos::MemoryUnmanaged> u,
                       const ViewType<Real[NP][NP],MemSpaceIn,Kokkos::MemoryUnmanaged> v,
                       const Control& data,
                       const ExecViewUnmanaged<Real[NP][NP]> metDet,
                       const ExecViewUnmanaged<Real[2][2][NP][NP]> D,
                       ViewType<Real[NP][NP],MemSpaceOut,Kokkos::MemoryUnmanaged> vort)
{
  Real rrearth = PhysicalConstants::rrearth;

  Real vcov[2][NP][NP];
  constexpr const int covar_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, covar_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    vcov[0][igp][jgp] = D(0,0,igp,jgp)*u(igp,jgp) + D(1,0,igp,jgp)*v(igp,jgp);
    vcov[1][igp][jgp] = D(0,1,igp,jgp)*u(igp,jgp) + D(1,1,igp,jgp)*v(igp,jgp);
  });

  Real dudy, dvdx;
  constexpr const int vort_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, vort_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    dudy = dvdx = 0.;
    for (int kgp=0; kgp<NP; ++kgp)
    {
      dudy += data.dvv(kgp, jgp) * vcov[1][igp][kgp];
      dvdx += data.dvv(kgp, igp) * vcov[0][kgp][jgp];
    }

    vort(igp,jgp) = rrearth * (dvdx - dudy) / metDet(igp,jgp);
  });
}

} // Namespace TinMan

#endif // SPHERE_OPERATORS_HPP
