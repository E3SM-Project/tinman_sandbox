#include "sphere_operators.hpp"
#include "TestData.hpp"

namespace TinMan
{

void gradient_sphere (const ViewUnmanaged<Real[NP][NP]> s, const TestData& data,
                      const ViewUnmanaged<Real[2][2][NP][NP]> DInv,
                      ViewUnmanaged<Real[2][NP][NP]> grad_s)
{
  typedef Real Dvv_type[NP][NP];

  const Dvv_type& Dvv = data.deriv().Dvv;
  Real rrearth = data.constants().rrearth;

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
        dsdx += Dvv[i][l]*s(i,j);
        dsdy += Dvv[i][l]*s(j,i);
      }

      v1[l][j] = dsdx * rrearth;
      v2[j][l] - dsdy * rrearth;
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

void gradient_sphere (Kokkos::TeamPolicy<>::member_type &team,
                      const ViewUnmanaged<Real[NP][NP]> s, const TestData& data,
                      const ViewUnmanaged<Real[2][2][NP][NP]> DInv,
                      ViewUnmanaged<Real[2][NP][NP]> grad_s)
{
  typedef Real Dvv_type[NP][NP];

  const Dvv_type& Dvv = data.deriv().Dvv;
  Real rrearth = data.constants().rrearth;

  Real dsdx, dsdy;
  Real v1[NP][NP];
  Real v2[NP][NP];
  constexpr const int contra_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, contra_iters),
                       [&](const int loop_idx) {
    const int j = loop_idx / NP;
    const int l = loop_idx % NP;
    dsdx = dsdy = 0;
    for (int i=0; i<NP; ++i)
    {
      dsdx += Dvv[i][l]*s(i,j);
      dsdy += Dvv[i][l]*s(j,i);
    }

    v1[l][j] = dsdx * rrearth;
    v2[j][l] - dsdy * rrearth;
  });

  constexpr const int grad_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, grad_iters),
                       [&](const int loop_idx) {
    const int j = loop_idx / NP;
    const int i = loop_idx % NP;
    grad_s(0, i, j) = DInv(0,0,i,j)*v1[i][j]+ DInv(1,0,i,j)*v2[i][j];
    grad_s(1, i, j) = DInv(0,1,i,j)*v1[i][j]+ DInv(1,1,i,j)*v2[i][j];
  });
}

void divergence_sphere (const ViewUnmanaged<Real[2][NP][NP]> v, const TestData& data,
                        const ViewUnmanaged<Real[NP][NP]> metDet,
                        const ViewUnmanaged<Real[2][2][NP][NP]> DInv,
                        ViewUnmanaged<Real[NP][NP]> div_v)
{
  typedef Real Dvv_type[NP][NP];

  const Dvv_type& Dvv = data.deriv().Dvv;
  Real rrearth = data.constants().rrearth;

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
        dudx += Dvv[kgp][igp] * gv[0][kgp][jgp];
        dvdy += Dvv[kgp][jgp] * gv[1][igp][kgp];
      }

      div_v(igp,jgp) = rrearth * (dudx + dvdy) / metDet(igp,jgp);
    }
  }
}

void divergence_sphere (Kokkos::TeamPolicy<>::member_type &team,
                        const ViewUnmanaged<Real[2][NP][NP]> v, const TestData& data,
                        const ViewUnmanaged<Real[NP][NP]> metDet,
                        const ViewUnmanaged<Real[2][2][NP][NP]> DInv,
                        ViewUnmanaged<Real[NP][NP]> div_v)
{
  typedef Real Dvv_type[NP][NP];

  const Dvv_type& Dvv = data.deriv().Dvv;
  Real rrearth = data.constants().rrearth;

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
      dudx += Dvv[kgp][igp] * gv[0][kgp][jgp];
      dvdy += Dvv[kgp][jgp] * gv[1][igp][kgp];
    }

    div_v(igp,jgp) = rrearth * (dudx + dvdy) / metDet(igp,jgp);
  });
}

void vorticity_sphere (const ViewUnmanaged<Real[2][NP][NP]> v, const TestData& data,
                       const ViewUnmanaged<Real[NP][NP]> metDet,
                       const ViewUnmanaged<Real[2][2][NP][NP]> D,
                       ViewUnmanaged<Real[NP][NP]> vort)
{
  typedef Real Dvv_type[NP][NP];

  const Dvv_type& Dvv = data.deriv().Dvv;
  Real rrearth = data.constants().rrearth;

  Real vcov[2][NP][NP];
  for (int igp=0; igp<NP; ++igp)
  {
    for (int jgp=0; jgp<NP; ++jgp)
    {
      vcov[0][igp][jgp] = D(0,0,igp,jgp)*v(0,igp,jgp) + D(1,0,igp,jgp)*v(1,igp,jgp);
      vcov[1][igp][jgp] = D(0,1,igp,jgp)*v(0,igp,jgp) + D(1,1,igp,jgp)*v(1,igp,jgp);
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
        dudy += Dvv[kgp][jgp] * vcov[1][igp][kgp];
        dvdx += Dvv[kgp][igp] * vcov[0][kgp][jgp];
      }

      vort(igp,jgp) = rrearth * (dvdx - dudy) / metDet(igp,jgp);
    }
  }
}

void vorticity_sphere (Kokkos::TeamPolicy<>::member_type &team,
                       const ViewUnmanaged<Real[2][NP][NP]> v, const TestData& data,
                       const ViewUnmanaged<Real[NP][NP]> metDet,
                       const ViewUnmanaged<Real[2][2][NP][NP]> D,
                       ViewUnmanaged<Real[NP][NP]> vort)
{
  typedef Real Dvv_type[NP][NP];

  const Dvv_type& Dvv = data.deriv().Dvv;
  Real rrearth = data.constants().rrearth;

  Real vcov[2][NP][NP];
  constexpr const int covar_iters = NP * NP;
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, covar_iters),
                       [&](const int loop_idx) {
    const int igp = loop_idx / NP;
    const int jgp = loop_idx % NP;
    vcov[0][igp][jgp] = D(0,0,igp,jgp)*v(0,igp,jgp) + D(1,0,igp,jgp)*v(1,igp,jgp);
    vcov[1][igp][jgp] = D(0,1,igp,jgp)*v(0,igp,jgp) + D(1,1,igp,jgp)*v(1,igp,jgp);
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
      dudy += Dvv[kgp][jgp] * vcov[1][igp][kgp];
      dvdx += Dvv[kgp][igp] * vcov[0][kgp][jgp];
    }

    vort(igp,jgp) = rrearth * (dvdx - dudy) / metDet(igp,jgp);
  });
}

} // Namespace TinMan
