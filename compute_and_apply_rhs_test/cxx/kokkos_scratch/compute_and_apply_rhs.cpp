#include "compute_and_apply_rhs.hpp"

#include "Types.hpp"
#include "Region.hpp"
#include "TestData.hpp"
#include "sphere_operators.hpp"

#include <fstream>

namespace TinMan
{

KOKKOS_INLINE_FUNCTION
void preq_hydrostatic(const ExecViewUnmanaged<Real[NP][NP]> phis,
                      const ScratchView<Real[NUM_LEV][NP][NP]> T_v,
                      const ScratchView<Real[NUM_LEV][NP][NP]> p,
                      const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> dp,
                      Real Rgas,
                      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> phi);

KOKKOS_INLINE_FUNCTION
void preq_hydrostatic(const Kokkos::TeamPolicy<>::member_type &team,
                      const ExecViewUnmanaged<Real[NP][NP]> phis,
                      const ScratchView<Real[NUM_LEV][NP][NP]> T_v,
                      const ScratchView<Real[NUM_LEV][NP][NP]> p,
                      const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> dp,
                      Real Rgas,
                      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> phi);

KOKKOS_INLINE_FUNCTION
void preq_omega_ps(const Kokkos::TeamPolicy<>::member_type &team,
                   const ScratchView<Real[NUM_LEV][NP][NP]> p,
                   const ScratchView<Real[NUM_LEV][NP][NP]> vgrad_p,
                   const ScratchView<Real[NUM_LEV][NP][NP]> div_vdp,
                   ScratchView<Real[NUM_LEV][NP][NP]> omega_p);

KOKKOS_INLINE_FUNCTION
void preq_omega_ps(const ScratchView<Real[NUM_LEV][NP][NP]> p,
                   const ScratchView<Real[NUM_LEV][NP][NP]> vgrad_p,
                   const ScratchView<Real[NUM_LEV][NP][NP]> div_vdp,
                   ScratchView<Real[NUM_LEV][NP][NP]> omega_p);


struct compute_and_apply_rhs {
  const int m_n0;
  const int m_np1;
  const int m_nm1;
  const int m_qn0;
  const Real m_dt2;

  Region &m_region;

  KOKKOS_INLINE_FUNCTION
  compute_and_apply_rhs(const Control &data, Region &region)
    : m_n0(data.n0()), m_np1(data.np1()), m_nm1(data.nm1()),
      m_qn0(data.qn0()), m_dt2(data.dt2()), m_region(region)
  {}

  // Requires 3 x NP x NP memory
  // The main purpose of this method is to reduce the scope of Ephi,
  // allowing it's memory to be reused
  template <typename Grad_View>
  KOKKOS_INLINE_FUNCTION
  void compute_energy_grad(Kokkos::TeamPolicy<>::member_type &team, Grad_View Ephi_grad) {
    ScratchView<Real[NP][NP]> Ephi(team.team_scratch(0));
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      Real v1 = m_region.U(ie, m_n0)(ilev,igp,jgp);
      Real v2 = m_region.V(ie, m_n0)(ilev,igp,jgp);
      // Kinetic energy + PHI (thermal energy?) + PECND (potential energy?)
      Ephi(igp,jgp) = 0.5 * (v1*v1 + v2*v2) + m_region.PHI(ie)(ilev,igp,jgp)
        + m_region.PECND(ie, ilev)(igp,jgp);
    });
    gradient_sphere(team, Ephi, data, m_region.DINV(ie), Ephi_grad);
    // We shouldn't need a block here, as the parallel loops were vector level, not thread level
  }

  // For each level, requires NP x NP x 6 Scratch Memory
  KOKKOS_INLINE_FUNCTION
  void compute_velocity(Kokkos::TeamPolicy<>::member_type &team) {
    const int ie = team.league_rank();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV), [&](const int ilev) {
      ScratchView<Real[2][NP][NP]> Ephi_grad(team.team_scratch(0));
      // Memory is reused, so no increase in memory requirement here
      compute_energy_grad(team, Ephi_grad);

      ScratchView<Real[NP][NP][2]> v_vadv(team.team_scratch(0));
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP * 2), [&](const int idx) {
        const int igp = idx / 2 / NP;
        const int jgp = (idx / 2) % NP;
        const int kgp = idx % 2;
        v_vadv(igp, jgp, kgp) = 0.0;
      });

      ScratchView<Real[NP][NP]> vtens1(team.team_scratch(0));
      ScratchView<Real[NP][NP]> vtens2(team.team_scratch(0));
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
        const int igp = idx / NP;
        const int jgp = idx % NP;

        Real glnps1 = PhysicalConstants::Rgas*gpterm*grad_p(ilev,0,igp,jgp);
        vtens1(igp, jgp) = v_vadv(igp, jgp, 0)
          + v2 * (m_region.FCOR(ie)(igp, jgp) + vort(ilev,igp,jgp))
          - Ephi_grad(0,igp,jgp) - glnps1;

        Real glnps2 = PhysicalConstants::Rgas*gpterm*grad_p(ilev,1,igp,jgp);
        vtens2(igp, jgp) = v_vadv(igp, jgp, 1)
          - v1 * (m_region.FCOR(ie)(igp, jgp) + vort(ilev,igp,jgp))
          - Ephi_grad(0,igp,jgp) - glnps2;
      });
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
        m_region.U(ie, m_np1)(ilev, igp, jgp) = m_region.SPHEREMP(ie)(igp, jgp)
          * (m_region.U(ie, m_nm1)(ilev, igp, jgp) + m_dt2 * vtens1(igp, jgp));
        m_region.V(ie, m_np1)(ilev, igp, jgp) = m_region.SPHEREMP(ie)(igp, jgp)
          * (m_region.V(ie, m_nm1)(ilev, igp, jgp) + m_dt2 * vtens2(igp, jgp));
      });
    });
    team.team_barrier();
  }

  // Requires NUM_LEV x NP x NP Scratch Memory
  KOKKOS_INLINE_FUNCTION
  void compute_eta_dpdn(Kokkos::TeamPolicy<>::member_type &team) {
    const int ie = team.league_rank();
    ScratchView<Real[NUM_LEV_P][NP][NP]> eta_dot_dpdn_ie(team.team_scratch(0));
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV_P), [&](const int ilev) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
        const int igp = idx / NP;
        const int jgp = idx % NP;
        eta_dot_dpdn_ie(ilev, igp, jgp) = 0.0;
      });

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
        const int igp = idx / NP;
        const int jgp = idx % NP;
        m_region.ETA_DPDN(ie)(ilev, igp, jgp) += PhysicalConstants::eta_ave_w * eta_dot_dpdn_ie(ilev,igp,jgp);
      });
    });
    team.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(Kokkos::TeamPolicy<>::member_type &team) {
    using Kokkos::subview;
    using Kokkos::ALL;
    const int ie = team.league_rank();

    // 3d scalars:
    ScratchView<Real[NUM_LEV][NP][NP]> div_vdp(team.team_scratch(0));
    ScratchView<Real[NUM_LEV][NP][NP]> kappa_star(team.team_scratch(0));
    ScratchView<Real[NUM_LEV][NP][NP]> omega_p(team.team_scratch(0));
    ScratchView<Real[NUM_LEV][NP][NP]> pressure(team.team_scratch(0));
    ScratchView<Real[NUM_LEV][NP][NP]> T_v(team.team_scratch(0));
    ScratchView<Real[NUM_LEV][NP][NP]> vgrad_p(team.team_scratch(0));
    ScratchView<Real[NUM_LEV][NP][NP]> vort(team.team_scratch(0));

    // 3d vectors:
    ScratchView<Real[NUM_LEV][2][NP][NP]> grad_p(team.team_scratch(0));

    if(ie < data.num_elems()) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
        const int igp = idx / NP;
        const int jgp = idx % NP;
        pressure(0,igp,jgp) = data.hybrid_a(0)*data.ps0() + 0.5*m_region.DP3D(ie, m_n0)(0,igp,jgp);
      });

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
        for(int ilev = 1; ilev < NUM_LEV; ilev++) {
          const int igp = idx / NP;
          const int jgp = idx % NP;
          pressure(ilev, igp, jgp) = pressure(ilev - 1, igp, jgp)
            + 0.5*m_region.DP3D(ie, m_n0)(ilev - 1, igp, jgp)
            + 0.5*m_region.DP3D(ie, m_n0)(ilev, igp, jgp);
        }
      });

      team.team_barrier();

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV), [&](const int ilev) {

        // Create subviews to explicitly have static dimensions
        ScratchView<Real[NP][NP]> p_ilev = subview(pressure, ilev, ALL(), ALL());
        ScratchView<Real[2][NP][NP]> grad_p_ilev = subview(grad_p, ilev, ALL(), ALL(), ALL());
        gradient_sphere(team, p_ilev, data, m_region.DINV(ie), grad_p_ilev);

        ScratchView<Real[2][NP][NP]> vdp_ilev(team.team_scratch(0));
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
          const int igp = idx / NP;
          const int jgp = idx % NP;
          Real v1 = m_region.U(ie, m_n0)(ilev,igp,jgp);
          Real v2 = m_region.V(ie, m_n0)(ilev,igp,jgp);
          vgrad_p(ilev, igp, jgp) = v1*grad_p(ilev, 0, igp, jgp) + v2 * grad_p(ilev, 1, igp, jgp);

          vdp_ilev(0, igp, jgp) = v1 * m_region.DP3D(ie, m_n0)(ilev, igp, jgp);
          vdp_ilev(1, igp, jgp) = v2 * m_region.DP3D(ie, m_n0)(ilev, igp, jgp);

          m_region.UN0(ie, ilev)(igp, jgp) += PhysicalConstants::eta_ave_w * vdp_ilev(0, igp, jgp);
          m_region.VN0(ie, ilev)(igp, jgp) += PhysicalConstants::eta_ave_w * vdp_ilev(1, igp, jgp);
        });

        ScratchView<Real[NP][NP]> div_vdp_ilev = subview(div_vdp, ilev, ALL(), ALL());
        divergence_sphere(team, vdp_ilev, data, m_region.METDET(ie), m_region.DINV(ie), div_vdp_ilev);

        // Create subviews to explicitly have static dimensions
        ScratchView<Real[NP][NP]> vort_ilev = subview(vort, ilev, ALL(), ALL());

        vorticity_sphere(team, m_region.UN0(ie, ilev), m_region.VN0(ie, ilev), data, m_region.METDET(ie), m_region.D(ie), vort_ilev);
      });

      if (qn0==-1)
      {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV), [&](const int ilev) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
            const int igp = idx / NP;
            const int jgp = idx % NP;
            T_v(ilev,igp,jgp) = m_region.T(ie, m_n0)(ilev,igp,jgp);
            kappa_star(ilev,igp,jgp) = PhysicalConstants::kappa;
          });
        });
      }
      else
      {
        ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> qdp = m_region.QDP(ie, m_qn0, 1);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV), [&](const int ilev) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
            const int igp = idx / NP;
            const int jgp = idx % NP;
            Real Qt = qdp(ilev,igp,jgp) / m_region.DP3D(ie, m_n0)(ilev,igp,jgp);
            T_v(ilev,igp,jgp) = m_region.T(ie, m_n0)(ilev,igp,jgp)
              * (1.0 + (PhysicalConstants::Rwater_vapor / PhysicalConstants::Rgas - 1.0) * Qt);
            kappa_star(ilev,igp,jgp) = PhysicalConstants::kappa;
          });
        });
      }

      team.team_barrier();

      ExecViewUnmanaged<Real[NP][NP]> phis_ie = m_region.PHIS(ie);
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> phi_ie = m_region.PHI(ie);
      preq_hydrostatic(team, phis_ie, T_v, pressure, m_region.DP3D(ie, m_n0), PhysicalConstants::Rgas, phi_ie);
      preq_omega_ps(pressure, vgrad_p, div_vdp, omega_p);

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV), [&](const int ilev) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
          const int igp = idx / NP;
          const int jgp = idx % NP;
          m_region.OMEGA_P(ie, ilev)(igp, jgp) += PhysicalConstants::eta_ave_w * omega_p(ilev,igp,jgp);
        });
      });

      team.team_barrier();

      {
        /* Reduce the scope of these scratch views so they can be cleaned up when no longer needed */
        ScratchView<Real[NUM_LEV][NP][NP]> ttens(team.team_scratch(0));
        {
          ScratchView<Real[NP][NP]> vgrad_T(team.team_scratch(0));
          ScratchView<Real[2][NP][NP]> grad_tmp(team.team_scratch(0));
          ScratchView<Real[NUM_LEV][NP][NP]> T_vadv(team.team_scratch(0));
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV), [&](const int ilev) {
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
              const int igp = idx / NP;
              const int jgp = idx % NP;
              T_vadv(ilev, igp, jgp) = 0.0;
            });
          });
          team.team_barrier();

          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV_P), [&](const int ilev) {
            // Create subviews to explicitly have static dimensions
            ExecViewUnmanaged<Real[NP][NP]> T_ie_n0_ilev = subview(m_region.T(ie, m_n0), ilev, ALL(), ALL());
            gradient_sphere(team, T_ie_n0_ilev, data, m_region.DINV(ie), grad_tmp);

            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
              const int igp = idx / NP;
              const int jgp = idx % NP;
              Real v1 = m_region.U(ie, m_n0)(ilev,igp,jgp);
              Real v2 = m_region.V(ie, m_n0)(ilev,igp,jgp);

              vgrad_T(igp, jgp) = v1*grad_tmp(0,igp,jgp) + v2*grad_tmp(1,igp,jgp);
            });

            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
              const int igp = idx / NP;
              const int jgp = idx % NP;
              Real gpterm = T_v(ilev,igp,jgp) / pressure(ilev,igp,jgp);

              Real v1 = m_region.U(ie, m_n0)(ilev,igp,jgp);
              Real v2 = m_region.V(ie, m_n0)(ilev,igp,jgp);


              ttens(ilev, igp, jgp) = T_vadv(ilev, igp, jgp) - vgrad_T(igp, jgp)
                + kappa_star(ilev,igp,jgp)*T_v(ilev,igp,jgp)*omega_p(ilev,igp,jgp);
            });
          });
        }

        team.team_barrier();

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV), [&](const int ilev) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP), [&](const int idx) {
            const int igp = idx / NP;
            const int jgp = idx % NP;
            m_region.T(ie, m_np1)(ilev, igp, jgp) = m_region.SPHEREMP(ie)(igp, jgp)
              * (m_region.T(ie, m_nm1)(ilev, igp, jgp) + m_dt2 * ttens(ilev, igp, jgp));
            m_region.DP3D(ie, m_np1)(ilev, igp, jgp) = m_region.SPHEREMP(ie)(igp, jgp)
              * (m_region.DP3D(ie, m_nm1)(ilev, igp, jgp) + m_dt2 * div_vdp(ilev, igp, jgp));
          });
        });

        compute_velocity(team);
        compute_eta_dpdn(team);
      }
    }
  }

  size_t shmem_size(const int team_size) const {
    const int mem_2d_scalar = ScratchView<Real[NP][NP]>::shmem_size();
    const int mem_2d_vector = ScratchView<Real[2][NP][NP]>::shmem_size();
    const int mem_3d_scalar = ScratchView<Real[NUM_LEV][NP][NP]>::shmem_size();
    const int mem_3d_p_scalar = ScratchView<Real[NUM_LEV_P][NP][NP]>::shmem_size();
    const int mem_3d_vector = ScratchView<Real[NUM_LEV][2][NP][NP]>::shmem_size();
  
    constexpr const int num_2d_tmp_scalars = 5;
    constexpr const int num_2d_tmp_vectors = 2;
    constexpr const int num_3d_tmp_scalars = 11;
    constexpr const int num_3d_tmp_vectors = 3;
    constexpr const int num_3d_p_tmp_scalars = 1;

    const int mem_needed = num_2d_tmp_scalars * mem_2d_scalar
                         + num_2d_tmp_vectors * mem_2d_vector
                         + num_3d_tmp_scalars * mem_3d_scalar
                         + num_3d_tmp_vectors * mem_3d_vector
                         + num_3d_p_tmp_scalars * mem_3d_p_scalar;

    return mem_needed;
  }
};

KOKKOS_INLINE_FUNCTION
void preq_hydrostatic (const Kokkos::TeamPolicy<>::member_type &team,
                       const ExecViewUnmanaged<Real[NP][NP]> phis,
                       const ScratchView<Real[NUM_LEV][NP][NP]> T_v,
                       const ScratchView<Real[NUM_LEV][NP][NP]> p,
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
                       const ScratchView<Real[NUM_LEV][NP][NP]> T_v,
                       const ScratchView<Real[NUM_LEV][NP][NP]> p,
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
                   const ScratchView<Real[NUM_LEV][NP][NP]> p,
                   const ScratchView<Real[NUM_LEV][NP][NP]> vgrad_p,
                   const ScratchView<Real[NUM_LEV][NP][NP]> div_vdp,
                   ScratchView<Real[NUM_LEV][NP][NP]> omega_p)
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
void preq_omega_ps(const ScratchView<Real[NUM_LEV][NP][NP]> p,
                   const ScratchView<Real[NUM_LEV][NP][NP]> vgrad_p,
                   const ScratchView<Real[NUM_LEV][NP][NP]> div_vdp,
                   ScratchView<Real[NUM_LEV][NP][NP]> omega_p)
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

// void print_results_2norm (const Control& data, const Region& region)
// {
//   // Input parameters
//   const int np1  = data.np1();

//   auto scalars_4d = region.get_4d_scalars();

//   Real vnorm(0.), tnorm(0.), dpnorm(0.);
//   for (int ie=0; ie<; ++ie)
//   {
//     for (int ilev=0; ilev<NUM_LEV; ++ilev)
//     {
//       for (int igp=0; igp<NP; ++igp)
//       {
//         for (int jgp=0; jgp<NP; ++jgp)
//         {
//           vnorm  += std::pow( scalars_4d(ie,np1,IDX_U,ilev,igp,jgp)   , 2 );
//           vnorm  += std::pow( scalars_4d(ie,np1,IDX_V,ilev,igp,jgp)   , 2 );
//           tnorm  += std::pow( scalars_4d(ie,np1,IDX_T,ilev,igp,jgp)   , 2 );
//           dpnorm += std::pow( scalars_4d(ie,np1,IDX_DP3D,ilev,igp,jgp), 2 );
//         }
//       }
//     }
//   }

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
