#include "compute_and_apply_rhs.hpp"

#include "Types.hpp"
#include "Region.hpp"
#include "TestData.hpp"
#include "sphere_operators.hpp"
#include "ScratchManager.hpp"
#include "ScratchMemoryDefs.hpp"

#include <fstream>
#include <iomanip>

namespace TinMan
{

KOKKOS_INLINE_FUNCTION
void preq_hydrostatic(const ExecViewUnmanaged<Real[NP][NP]> phis,
                      const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_v,
                      const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> p,
                      const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> dp,
                      Real Rgas,
                      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> phi);

KOKKOS_INLINE_FUNCTION
void preq_hydrostatic(const Kokkos::TeamPolicy<>::member_type &team,
                      const ExecViewUnmanaged<Real[NP][NP]> phis,
                      const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_v,
                      const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> p,
                      const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> dp,
                      Real Rgas,
                      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> phi);

KOKKOS_INLINE_FUNCTION
void preq_omega_ps(const Kokkos::TeamPolicy<>::member_type &team,
                   const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> p,
                   const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> vgrad_p,
                   const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> div_vdp,
                   ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> omega_p);

KOKKOS_INLINE_FUNCTION
void preq_omega_ps(const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> p,
                   const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> vgrad_p,
                   const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> div_vdp,
                   ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> omega_p);


void compute_and_apply_rhs (const Control& data, Region& region)
{
  using Kokkos::subview;
  using Kokkos::ALL;

  Kokkos::TeamPolicy<> policy(data.host_num_elems(), Kokkos::AUTO);

  const int mem_needed = ScratchMemoryDefs::CAARS_ScratchManager::memory_needed(policy.team_size());

  policy = policy.set_scratch_size(0, Kokkos::PerTeam(mem_needed));

  Kokkos::parallel_for(policy,
                       KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type &team) {

    // The manager for scratch memory
    ScratchMemoryDefs::CAARS_ScratchManager scratch_manager;

    // We only need to get a pointer out of the scratch space, doesn't really matter
    // how much we ask for, as long as it fits. 1 byte should do
    scratch_manager.set_scratch_memory(reinterpret_cast<Real*>(team.team_scratch(0).get_shmem(1)));

    const int ie = team.league_rank();
    const int team_rank = team.team_rank(); // This is the thread id

    // Input parameters
    const int n0   = data.n0();
    const int np1  = data.np1();
    const int nm1  = data.nm1();
    const int qn0  = data.qn0();
    const Real dt2 = data.dt2();

    // 3d scalars:
    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> div_vdp    (scratch_manager.get_team_scratch<ID_3D_SCALAR,0>()); //
    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure   (scratch_manager.get_team_scratch<ID_3D_SCALAR,1>()); //
    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> vort       (scratch_manager.get_team_scratch<ID_3D_SCALAR,2>());

    // 3d vectors:
    ExecViewUnmanaged<Real[NUM_LEV][2][NP][NP]> grad_p(scratch_manager.get_team_scratch<ID_3D_VECTOR,0>());

    if(ie < data.num_elems()) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
        const int igp = idx / NP;
        const int jgp = idx % NP;
        pressure(0,igp,jgp) = data.hybrid_a(0)*data.ps0() + 0.5*region.DP3D(ie, n0)(0,igp,jgp);
      });

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
        for(int ilev = 1; ilev < NUM_LEV; ilev++) {
          const int igp = idx / NP;
          const int jgp = idx % NP;
          pressure(ilev, igp, jgp) = pressure(ilev - 1, igp, jgp)
            + 0.5*region.DP3D(ie, n0)(ilev - 1, igp, jgp)
            + 0.5*region.DP3D(ie, n0)(ilev, igp, jgp);
        }
      });

      team.team_barrier();

      ExecViewUnmanaged<Real[2][NP][NP]> vdp_ilev(scratch_manager.get_thread_scratch<ID_2D_VECTOR,0>(team_rank));
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> vgrad_p    (scratch_manager.get_team_scratch<ID_3D_SCALAR,3>());

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV), [&](const int ilev) {

        // Create subviews to explicitly have static dimensions
        ExecViewUnmanaged<Real[NP][NP]> p_ilev = subview(pressure, ilev, ALL(), ALL());
        ExecViewUnmanaged<Real[2][NP][NP]> grad_p_ilev = subview(grad_p, ilev, ALL(), ALL(), ALL());
        gradient_sphere(team, p_ilev, data, scratch_manager, region.DINV(ie), grad_p_ilev);

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
          const int igp = idx / NP;
          const int jgp = idx % NP;
          Real v1 = region.U(ie, n0)(ilev,igp,jgp);
          Real v2 = region.V(ie, n0)(ilev,igp,jgp);
          vgrad_p(ilev, igp, jgp) = v1*grad_p(ilev, 0, igp, jgp) + v2 * grad_p(ilev, 1, igp, jgp);

          vdp_ilev(0, igp, jgp) = v1 * region.DP3D(ie, n0)(ilev, igp, jgp);
          vdp_ilev(1, igp, jgp) = v2 * region.DP3D(ie, n0)(ilev, igp, jgp);

          (region.UN0(ie, ilev))(igp, jgp) += PhysicalConstants::eta_ave_w * vdp_ilev(0, igp, jgp);
          (region.VN0(ie, ilev))(igp, jgp) += PhysicalConstants::eta_ave_w * vdp_ilev(1, igp, jgp);
        });

        ExecViewUnmanaged<Real[NP][NP]> div_vdp_ilev = subview(div_vdp, ilev, ALL(), ALL());
        divergence_sphere(team, vdp_ilev, data, region.METDET(ie), region.DINV(ie), div_vdp_ilev);

        // Create subviews to explicitly have static dimensions
        ExecViewUnmanaged<Real[NP][NP]> vort_ilev = subview(vort, ilev, ALL(), ALL());

        vorticity_sphere(team, region.UN0(ie, ilev), region.VN0(ie, ilev), data, region.METDET(ie), region.D(ie), vort_ilev);
      });

      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> kappa_star (scratch_manager.get_team_scratch<ID_3D_SCALAR,4>()); //
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_v        (scratch_manager.get_team_scratch<ID_3D_SCALAR,5>());
      if (qn0==-1)
      {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV), [&](const int ilev) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
            const int igp = idx / NP;
            const int jgp = idx % NP;
            T_v(ilev,igp,jgp) = region.T(ie, n0)(ilev,igp,jgp);
            kappa_star(ilev,igp,jgp) = PhysicalConstants::kappa;
          });
        });
      }
      else
      {
        ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> qdp = region.QDP(ie, qn0, 1);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV), [&](const int ilev) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
            const int igp = idx / NP;
            const int jgp = idx % NP;
            Real Qt = qdp(ilev,igp,jgp) / region.DP3D(ie, n0)(ilev,igp,jgp);
            T_v(ilev,igp,jgp) = region.T(ie, n0)(ilev,igp,jgp)*(1.0+ (PhysicalConstants::Rwater_vapor/PhysicalConstants::Rgas - 1.0)*Qt);
            kappa_star(ilev,igp,jgp) = PhysicalConstants::kappa;
          });
        });
      }

      team.team_barrier();

      ExecViewUnmanaged<Real[NP][NP]>          phis_ie = region.PHIS(ie);
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> phi_ie  = region.PHI(ie);
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> omega_p (scratch_manager.get_team_scratch<ID_3D_SCALAR,6>());

      preq_hydrostatic(team, phis_ie, T_v, pressure, region.DP3D(ie, n0), PhysicalConstants::Rgas, phi_ie);
      preq_omega_ps(pressure, vgrad_p, div_vdp, omega_p);

      ExecViewUnmanaged<Real[NUM_LEV_P][NP][NP]>  eta_dot_dpdn_ie(scratch_manager.get_team_scratch<ID_3D_P_SCALAR,0>());
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV_P), [&](const int ilev) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
          const int igp = idx / NP;
          const int jgp = idx % NP;
          auto etadpdn = region.ETA_DPDN(ie);
          etadpdn(ilev, igp, jgp) += PhysicalConstants::eta_ave_w * eta_dot_dpdn_ie(ilev,igp,jgp);
          if (ilev<NUM_LEV)
            (region.OMEGA_P(ie, ilev))(igp, jgp) += PhysicalConstants::eta_ave_w * omega_p(ilev,igp,jgp);
        });
      });

      team.team_barrier();

      // Note: the only purpose of T_vadv is to be stuffed (with other terms) into ttens. By making ttens share
      //       the same ptr of T_vadv, we save memory and flops. The same holds for vtens and v_vadv
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]>    T_vadv (scratch_manager.get_team_scratch<ID_3D_SCALAR,7>());
      ExecViewUnmanaged<Real[NUM_LEV][2][NP][NP]> v_vadv (scratch_manager.get_team_scratch<ID_3D_VECTOR,1>());
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]>    ttens  (T_vadv.data());//scratch_manager.get_team_scratch<ID_3D_SCALAR_8>());
      ExecViewUnmanaged<Real[NUM_LEV][2][NP][NP]> vtens  (v_vadv.data());//scratch_manager.get_team_scratch<ID_3D_SCALAR_8>());

      ExecViewUnmanaged<Real[NP][NP]> Ephi              (scratch_manager.get_thread_scratch<ID_2D_SCALAR,0>(team_rank));
      ExecViewUnmanaged<Real[NP][NP]> vgrad_T           (scratch_manager.get_thread_scratch<ID_2D_SCALAR,1>(team_rank));
      ExecViewUnmanaged<Real[2][NP][NP]> grad_tmp       (scratch_manager.get_thread_scratch<ID_2D_VECTOR,0>(team_rank));

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV), [&](const int ilev) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
          const int igp = idx / NP;
          const int jgp = idx % NP;
          // T_vadv initialized
          T_vadv(ilev, igp, jgp) = 0.0;
          // v_vadv initialized
          v_vadv(ilev, 0, igp, jgp) = v_vadv(ilev, 1, igp, jgp) = 0.0;
        });
      });

      team.team_barrier();

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV), [&](const int ilev) {
        // Create subviews to explicitly have static dimensions
        ExecViewUnmanaged<Real[NP][NP]> T_ie_n0_ilev = subview(region.T(ie, n0), ilev, ALL(), ALL());
        gradient_sphere(team, T_ie_n0_ilev, data, scratch_manager, region.DINV(ie), grad_tmp);

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
          const int igp = idx / NP;
          const int jgp = idx % NP;
          Real v1 = region.U(ie, n0)(ilev,igp,jgp);
          Real v2 = region.V(ie, n0)(ilev,igp,jgp);

          vgrad_T(igp, jgp) = v1*grad_tmp(0,igp,jgp) + v2*grad_tmp(1,igp,jgp);
        });

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
          const int igp = idx / NP;
          const int jgp = idx % NP;
          Real v1 = region.U(ie, n0)(ilev,igp,jgp);
          Real v2 = region.V(ie, n0)(ilev,igp,jgp);

          Ephi(igp,jgp) = 0.5 * (v1*v1 + v2*v2) + (region.PHI(ie))(ilev,igp,jgp) + (region.PECND(ie, ilev))(igp,jgp);
        });
        gradient_sphere(team, Ephi, data, scratch_manager, region.DINV(ie), grad_tmp);

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
          const int igp = idx / NP;
          const int jgp = idx % NP;
          Real gpterm = T_v(ilev,igp,jgp) / pressure(ilev,igp,jgp);

          Real glnps1 = PhysicalConstants::Rgas*gpterm*grad_p(ilev,0,igp,jgp);
          Real glnps2 = PhysicalConstants::Rgas*gpterm*grad_p(ilev,1,igp,jgp);

          Real v1 = region.U(ie, n0)(ilev,igp,jgp);
          Real v2 = region.V(ie, n0)(ilev,igp,jgp);

          vtens(ilev, 0, igp, jgp) = v_vadv(ilev, 0, igp, jgp) + v2 * ((region.FCOR(ie))(igp, jgp) + vort(ilev,igp,jgp)) - grad_tmp(0,igp,jgp) - glnps1;
          vtens(ilev, 1, igp, jgp) = v_vadv(ilev, 1, igp, jgp) - v1 * ((region.FCOR(ie))(igp, jgp) + vort(ilev,igp,jgp)) - grad_tmp(1,igp,jgp) - glnps2;

          ttens(ilev, igp, jgp)  = T_vadv(ilev, igp, jgp) - vgrad_T(igp, jgp) + kappa_star(ilev,igp,jgp)*T_v(ilev,igp,jgp)*omega_p(ilev,igp,jgp);
        });
      });

      team.team_barrier();

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV), [&](const int ilev) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
          const int igp = idx / NP;
          const int jgp = idx % NP;
          (region.U(ie, np1))(ilev, igp, jgp) = region.SPHEREMP(ie)(igp, jgp) * ((region.U(ie, nm1))(ilev, igp, jgp) + dt2 * vtens(ilev, 0, igp, jgp));
          (region.V(ie, np1))(ilev, igp, jgp) = region.SPHEREMP(ie)(igp, jgp) * ((region.V(ie, nm1))(ilev, igp, jgp) + dt2 * vtens(ilev, 1, igp, jgp));
          (region.T(ie, np1))(ilev, igp, jgp) = region.SPHEREMP(ie)(igp, jgp) * ((region.T(ie, nm1))(ilev, igp, jgp) + dt2 * ttens(ilev, igp, jgp));
          (region.DP3D(ie, np1))(ilev, igp, jgp) = region.SPHEREMP(ie)(igp, jgp) * ((region.DP3D(ie, nm1))(ilev, igp, jgp) - dt2 * div_vdp(ilev, igp, jgp));
        });
      });

    }
  });
}

KOKKOS_INLINE_FUNCTION
void preq_hydrostatic (const Kokkos::TeamPolicy<>::member_type &team,
                       const ExecViewUnmanaged<Real[NP][NP]> phis,
                       const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_v,
                       const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> p,
                       const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> dp,
                       Real Rgas,
                       ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> phi)
{
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NP * NP),
                       KOKKOS_LAMBDA(const int loop_idx) {
    Real hkk, hkl;
    Real phii[NUM_LEV][NP][NP];
    const int jgp = loop_idx / NP;
    const int igp = loop_idx % NP;
    hkk = 0.5 * dp(NUM_LEV-1, igp, jgp) / p(NUM_LEV - 1, igp, jgp);
    hkl = 2.0 * hkk;
    phii[NUM_LEV - 1][igp][jgp] = Rgas * T_v(NUM_LEV - 1, igp, jgp) * hkl;
    phi(NUM_LEV - 1, igp, jgp) = phis(igp, jgp) + Rgas * T_v(NUM_LEV - 1, igp, jgp) * hkk;

    for(int ilev = NUM_LEV - 2; ilev > 1; --ilev) {
      hkk = 0.5 * dp(ilev,igp,jgp) / p(ilev,igp,jgp);
      hkl = 2.0 * hkk;
      phii[ilev][igp][jgp] = phii[ilev + 1][igp][jgp] + Rgas * T_v(ilev, igp, jgp)*hkl;
      phi(ilev, igp, jgp) = phis(igp, jgp) + phii[ilev + 1][igp][jgp] + Rgas*T_v(ilev, igp, jgp)*hkk;
    }

    hkk = 0.5 * dp(0, igp, jgp) / p(0, igp, jgp);
    phi(0, igp, jgp) = phis(igp, jgp) + phii[1][igp][jgp] + Rgas * T_v(0, igp, jgp) * hkk;
  });
  team.team_barrier();
}

KOKKOS_INLINE_FUNCTION
void preq_hydrostatic (const ExecViewUnmanaged<Real[NP][NP]> phis,
                       const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_v,
                       const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> p,
                       const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> dp,
                       Real Rgas,
                       ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> phi)
{
  Real hkk, hkl;
  Real phii[NUM_LEV][NP][NP];
  for (int jgp=0; jgp<NP; ++jgp)
  {
    for (int igp=0; igp<NP; ++igp)
    {
      hkk = 0.5 * dp(NUM_LEV-1,igp,jgp) / p(NUM_LEV-1,igp,jgp);
      hkl = 2.0*hkk;
      phii[NUM_LEV-1][igp][jgp] = Rgas*T_v(NUM_LEV-1,igp,jgp)*hkl;
      phi(NUM_LEV-1,igp,jgp) = phis(igp,jgp) + Rgas*T_v(NUM_LEV-1,igp,jgp)*hkk;
    }
    for (int ilev=NUM_LEV-2; ilev>1; --ilev)
    {
      for (int igp=0; igp<NP; ++igp)
      {
        hkk = 0.5 * dp(ilev,igp,jgp) / p(ilev,igp,jgp);
        hkl = 2.0*hkk;
        phii[ilev][igp][jgp] = phii[ilev+1][igp][jgp] + Rgas*T_v(ilev,igp,jgp)*hkl;
        phi(ilev,igp,jgp) = phis(igp,jgp) + phii[ilev+1][igp][jgp] + Rgas*T_v(ilev,igp,jgp)*hkk;
      }
    }
    for (int igp=0; igp<NP; ++igp)
    {
      hkk = 0.5 * dp(0,igp,jgp) / p(0,igp,jgp);
      phi(0,igp,jgp) = phis(igp,jgp) + phii[1][igp][jgp] + Rgas*T_v(0,igp,jgp)*hkk;
    }
  }
}

KOKKOS_INLINE_FUNCTION
void preq_omega_ps(const Kokkos::TeamPolicy<>::member_type &team,
                   const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> p,
                   const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> vgrad_p,
                   const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> div_vdp,
                   ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> omega_p)
{
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NP * NP),
                       KOKKOS_LAMBDA(const int loop_idx) {
    Real ckk, ckl, term;
    Real suml[NP][NP];
    const int jgp = loop_idx / NP;
    const int igp = loop_idx % NP;

    ckk = 0.5 / p(0, igp, jgp);
    term  = div_vdp(0, igp, jgp);
    omega_p(0, igp, jgp) = vgrad_p(0, igp, jgp) / p(0, igp, jgp) - ckk * term;
    suml[igp][jgp] = term;
    for(int ilev = 1; ilev < NUM_LEV - 1; ++ilev)
    {
      ckk = 0.5 / p(ilev, igp, jgp);
      ckl = 2.0 * ckk;
      term  = div_vdp(ilev, igp, jgp);
      omega_p(ilev, igp, jgp) = vgrad_p(ilev, igp, jgp) / p(ilev, igp, jgp) - ckl * suml[igp][jgp] - ckk * term;

      suml[igp][jgp] += term;
    }

    ckk = 0.5 / p(NUM_LEV - 1, igp, jgp);
    ckl = 2.0 * ckk;
    term = div_vdp(NUM_LEV - 1, igp, jgp);
    omega_p(NUM_LEV - 1, igp, jgp) = vgrad_p(NUM_LEV - 1, igp, jgp) / p(NUM_LEV - 1, igp, jgp) - ckl * suml[igp][jgp] - ckk * term;
  });
  team.team_barrier();
}

KOKKOS_INLINE_FUNCTION
void preq_omega_ps(const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> p,
                   const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> vgrad_p,
                   const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> div_vdp,
                   ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> omega_p)
{
  Real ckk, ckl, term;
  Real suml[NP][NP];
  for (int jgp=0; jgp<NP; ++jgp)
  {
    for (int igp=0; igp<NP; ++igp)
    {
      ckk = 0.5 / p(0,igp,jgp);
      term  = div_vdp(0,igp,jgp);
      omega_p(0,igp,jgp) = vgrad_p(0,igp,jgp) / p(0,igp,jgp) - ckk*term;
      suml[igp][jgp] = term;
    }
    for (int ilev=1; ilev<NUM_LEV-1; ++ilev)
    {
      for (int igp=0; igp<NP; ++igp)
      {
        ckk = 0.5 / p(ilev,igp,jgp);
        ckl = 2.0 * ckk;
        term  = div_vdp(ilev,igp,jgp);
        omega_p(ilev,igp,jgp) = vgrad_p(ilev,igp,jgp) / p(ilev,igp,jgp) - ckl*suml[igp][jgp] - ckk*term;

        suml[igp][jgp] += term;
      }
    }
    for (int igp=0; igp<NP; ++igp)
    {
      ckk = 0.5 / p(NUM_LEV-1,igp,jgp);
      ckl = 2.0 * ckk;
      term  = div_vdp(NUM_LEV-1,igp,jgp);
      omega_p(NUM_LEV-1,igp,jgp) = vgrad_p(NUM_LEV-1,igp,jgp) / p(NUM_LEV-1,igp,jgp) - ckl*suml[igp][jgp] - ckk*term;
    }
  }
}

void print_results_2norm (const Control& data, const Region& region)
{
  using Kokkos::subview;
  using Kokkos::ALL;

  // Input parameters
  const int num_elems = data.host_num_elems();
  const int np1  = data.np1();

  typedef ExecViewManaged<Real[NUM_LEV][NP][NP]>::HostMirror MirrorView;

  Real vnorm(0.), tnorm(0.), dpnorm(0.);
  for (int ie=0; ie<num_elems; ++ie)
  {
    MirrorView U_host = Kokkos::create_mirror_view(region.U(ie, np1));
    MirrorView V_host = Kokkos::create_mirror_view(region.V(ie, np1));
    MirrorView T_host = Kokkos::create_mirror_view(region.T(ie, np1));
    MirrorView P_host = Kokkos::create_mirror_view(region.DP3D(ie, np1));

    Kokkos::deep_copy (U_host,region.U(ie, np1));
    Kokkos::deep_copy (V_host,region.V(ie, np1));
    Kokkos::deep_copy (T_host,region.T(ie, np1));
    Kokkos::deep_copy (P_host,region.DP3D(ie, np1));

    vnorm  += std::pow( compute_norm( U_host ), 2 );
    vnorm  += std::pow( compute_norm( V_host ), 2 );
    tnorm  += std::pow( compute_norm( T_host ), 2 );
    dpnorm += std::pow( compute_norm( P_host ), 2 );
  }

  std::cout << "   ---> Norms:\n"
            << "          ||v||_2  = " << std::setprecision(18) << std::sqrt (vnorm) << "\n"
            << "          ||T||_2  = " << std::setprecision(18) << std::sqrt (tnorm) << "\n"
            << "          ||dp||_2 = " << std::setprecision(18) << std::sqrt (dpnorm) << "\n";
}

//   std::cout << "   ---> Norms:\n"
//             << "          ||v||_2  = " << std::sqrt(vnorm) << "\n"
//             << "          ||T||_2  = " << std::sqrt(tnorm) << "\n"
//             << "          ||dp||_2 = " << std::sqrt(dpnorm) << "\n";
// }

// void dump_results_to_file (const Control& data, const Region& region)
// {
//   // Input parameters
//   const int nets = data.control().nets;
//   const int nete = data.control().nete;
//   const int np1  = data.control().np1;

//   std::ofstream vxfile, vyfile, tfile, dpfile;
//   vxfile.open("elem_state_vx.txt");
//   if (!vxfile.is_open())
//   {
//     std::cout << "Error! Cannot open 'elem_state_vx.txt'.\n";
//     std::abort();
//   }

//   vyfile.open("elem_state_vy.txt");
//   if (!vyfile.is_open())
//   {
//     vxfile.close();
//     std::cout << "Error! Cannot open 'elem_state_vy.txt'.\n";
//     std::abort();
//   }

//   tfile.open("elem_state_t.txt");
//   if (!tfile.is_open())
//   {
//     std::cout << "Error! Cannot open 'elem_state_t.txt'.\n";
//     vxfile.close();
//     vyfile.close();
//     std::abort();
//   }

//   dpfile.open("elem_state_dp3d.txt");
//   if (!dpfile.is_open())
//   {
//     std::cout << "Error! Cannot open 'elem_state_dp3d.txt'.\n";
//     vxfile.close();
//     vyfile.close();
//     tfile.close();
//     std::abort();
//   }

//   vxfile.precision(6);
//   vyfile.precision(6);
//   tfile.precision(6);
//   dpfile.precision(6);

//   auto scalars_4d = region.get_4d_scalars();

//   for (int ie=nets; ie<nete; ++ie)
//   {
//     for (int ilev=0; ilev<NUM_LEV; ++ilev)
//     {
//       vxfile << "[" << ie << ", " << ilev << "]\n";
//       vyfile << "[" << ie << ", " << ilev << "]\n";
//       tfile  << "[" << ie << ", " << ilev << "]\n";
//       dpfile << "[" << ie << ", " << ilev << "]\n";

//       for (int igp=0; igp<NP; ++igp)
//       {
//         for (int jgp=0; jgp<NP; ++jgp)
//         {
//           vxfile << " " << scalars_4d(ie,IDX_U,np1,ilev,igp,jgp)   ;
//           vyfile << " " << scalars_4d(ie,IDX_V,np1,ilev,igp,jgp)   ;
//           tfile  << " " << scalars_4d(ie,IDX_T,np1,ilev,igp,jgp)   ;
//           dpfile << " " << scalars_4d(ie,IDX_DP3D,np1,ilev,igp,jgp);
//         }
//         vxfile << "\n";
//         vyfile << "\n";
//         tfile  << "\n";
//         dpfile << "\n";
//       }
//     }
//   }

//   // Closing files
//   vxfile.close();
//   vyfile.close();
//   tfile.close();
//   dpfile.close();
// }

} // Namespace TinMan
