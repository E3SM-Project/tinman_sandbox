#include "compute_and_apply_rhs.hpp"

#include "Types.hpp"
#include "Region.hpp"
#include "TestData.hpp"
#include "sphere_operators.hpp"

#include <fstream>

namespace TinMan {

KOKKOS_INLINE_FUNCTION
void preq_hydrostatic(const ExecViewUnmanaged<Real[NP][NP]> phis,
                      const ScratchView<Real[NUM_LEV][NP][NP]> T_v,
                      const ScratchView<Real[NUM_LEV][NP][NP]> p,
                      const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> dp,
                      Real Rgas, ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> phi);

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

struct update_state {
  const Control m_data;
  const Region m_region;

  KOKKOS_INLINE_FUNCTION
  update_state(const Control &data, const Region &region)
      : m_data(data), m_region(region) {}

  // For each thread, requires 3 x NP x NP memory
  // Depends on PHI, PECND
  // Modifies Ephi_grad
  template <typename Grad_View>
  KOKKOS_INLINE_FUNCTION void
  compute_energy_grad(Kokkos::TeamPolicy<>::member_type &team, const int ilev,
                      Grad_View Ephi_grad) const {
    const int ie = team.league_rank();

    ScratchView<Real[NP][NP]> Ephi(team.team_scratch(0));
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      Real v1 = m_region.U(ie, m_data.n0())(ilev, igp, jgp);
      Real v2 = m_region.V(ie, m_data.n0())(ilev, igp, jgp);
      // Kinetic energy + PHI (thermal energy?) + PECND (potential energy?)
      Ephi(igp, jgp) = 0.5 * (v1 * v1 + v2 * v2) +
                       m_region.PHI(ie)(ilev, igp, jgp) +
                       m_region.PECND(ie, ilev)(igp, jgp);
    });
    gradient_sphere(team, Ephi, m_data, m_region.DINV(ie), Ephi_grad);
    // We shouldn't need a block here, as the parallel loops were vector level,
    // not thread level
  }

  // For each thread, requires NP x NP x 6 Scratch Memory
  // Depends on pressure, UN0, VN0, METDET, D, DINV, U, V, FCOR, SPHEREMP, T_v
  // Modifies U, V
  KOKKOS_INLINE_FUNCTION
  void compute_velocity(Kokkos::TeamPolicy<>::member_type &team,
                        ScratchView<Real[NUM_LEV][NP][NP]> pressure,
                        ScratchView<Real[NUM_LEV][NP][NP]> T_v) const {
    const int ie = team.league_rank();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV),
                         [&](const int ilev) {
      ScratchView<Real[NP][NP]> p_ilev =
          subview(pressure, ilev, Kokkos::ALL(), Kokkos::ALL());

      ScratchView<Real[NP][NP]> vort(team.team_scratch(0));
      vorticity_sphere(team, m_region.UN0(ie, ilev), m_region.VN0(ie, ilev),
                       m_data, m_region.METDET(ie), m_region.D(ie), vort);

      ScratchView<Real[2][NP][NP]> Ephi_grad(team.team_scratch(0));
      compute_energy_grad(team, ilev, Ephi_grad);

      ScratchView<Real[2][NP][NP]> grad_p(team.team_scratch(0));
      gradient_sphere(team, p_ilev, m_data, m_region.DINV(ie), grad_p);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                           [&](const int idx) {
        const int igp = idx / NP;
        const int jgp = idx % NP;

        const Real gpterm = T_v(ilev, igp, jgp) / p_ilev(igp, jgp);

        const Real v1 = m_region.U(ie, m_data.n0())(ilev, igp, jgp);
        const Real v2 = m_region.V(ie, m_data.n0())(ilev, igp, jgp);

        const Real fcor_vort_coeff =
            m_region.FCOR(ie)(igp, jgp) + vort(igp, jgp);

        const Real glnps1 =
            PhysicalConstants::Rgas * gpterm * grad_p(0, igp, jgp);
        const Real vtens1 =
            // v_vadv(igp, jgp, 0)
            0.0 + v2 * fcor_vort_coeff - Ephi_grad(0, igp, jgp) - glnps1;

        m_region.U(ie, m_data.np1())(ilev, igp, jgp) =
            m_region.SPHEREMP(ie)(igp, jgp) *
            (m_region.U(ie, m_data.nm1())(ilev, igp, jgp) +
             m_data.dt2() * vtens1);

        const Real glnps2 =
            PhysicalConstants::Rgas * gpterm * grad_p(1, igp, jgp);
        const Real vtens2 =
            // v_vadv(igp, jgp, 1) -
            0.0 - v1 * fcor_vort_coeff - Ephi_grad(1, igp, jgp) - glnps2;

        m_region.V(ie, m_data.np1())(ilev, igp, jgp) =
            m_region.SPHEREMP(ie)(igp, jgp) *
            (m_region.V(ie, m_data.nm1())(ilev, igp, jgp) +
             m_data.dt2() * vtens2);
      });
    });
    team.team_barrier();
  }

  // For each thread, requires 0 Scratch Memory
  // Depends on ETA_DPDN
  // Modifies ETA_DPDN
  KOKKOS_INLINE_FUNCTION
  void compute_eta_dpdn(Kokkos::TeamPolicy<>::member_type &team) const {
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV_P),
                         [&](const int ilev) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                           [&](const int idx) {
        const int igp = idx / NP;
        const int jgp = idx % NP;
        Real eta_dot_dpdn_ie = 0.0;

        m_region.ETA_DPDN(ie)(ilev, igp, jgp) +=
            PhysicalConstants::eta_ave_w * eta_dot_dpdn_ie;
      });
    });
    team.team_barrier();
  }

  // For each thread, requires 0 Scratch Memory
  // Depends on PHIS, DP3D, PHI, T_v
  // Modifies PHI
  KOKKOS_INLINE_FUNCTION
  void
  preq_hydrostatic(const Kokkos::TeamPolicy<>::member_type &team,
                   const ScratchView<Real[NUM_LEV][NP][NP]> pressure,
                   const ScratchView<Real[NUM_LEV][NP][NP]> T_v) const {
    const int ie = team.league_rank();

    ExecViewUnmanaged<Real[NP][NP]> phis = m_region.PHIS(ie);

    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> dp =
        m_region.DP3D(ie, m_data.n0());

    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> phi = m_region.PHI(ie);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NP * NP),
                         KOKKOS_LAMBDA(const int loop_idx) {
      const int jgp = loop_idx / NP;
      const int igp = loop_idx % NP;
      Real phii;

      {
        const Real hk =
            dp(NUM_LEV - 1, igp, jgp) / pressure(NUM_LEV - 1, igp, jgp);
        phii = PhysicalConstants::Rgas * T_v(NUM_LEV - 1, igp, jgp) * hk;
        phi(NUM_LEV - 1, igp, jgp) =
            phis(igp, jgp) +
            PhysicalConstants::Rgas * T_v(NUM_LEV - 1, igp, jgp) * hk * 0.5;
      }

      for (int ilev = NUM_LEV - 2; ilev > 0; --ilev) {
        const Real hk = dp(ilev, igp, jgp) / pressure(ilev, igp, jgp);
        phi(ilev, igp, jgp) = phis(igp, jgp) + phii + PhysicalConstants::Rgas *
                                                          T_v(ilev, igp, jgp) *
                                                          hk * 0.5;

        if (ilev != 1) {
          phii += PhysicalConstants::Rgas * T_v(ilev, igp, jgp) * hk;
        }
      }

      {
        const Real hk = 0.5 * dp(0, igp, jgp) / pressure(0, igp, jgp);
        phi(0, igp, jgp) = phis(igp, jgp) + phii +
                           PhysicalConstants::Rgas * T_v(0, igp, jgp) * hk;
      }
    });
    team.team_barrier();
  }

  // For each thread, requires 3 x NP x NP Scratch Memory
  // Depends on pressure, div_vdp, omega_p
  // Initializes omega_p
  KOKKOS_INLINE_FUNCTION
  void preq_omega_ps(const Kokkos::TeamPolicy<>::member_type &team,
                     const ScratchView<Real[NUM_LEV][NP][NP]> pressure,
                     const ScratchView<Real[NUM_LEV][NP][NP]> div_vdp,
                     const ScratchView<Real[NUM_LEV][NP][NP]> omega_p) const {
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NP * NP),
                         KOKKOS_LAMBDA(const int loop_idx) {
      ScratchView<Real[NP][NP]> suml(team.team_scratch(0));
      const int jgp = loop_idx / NP;
      const int igp = loop_idx % NP;
      ScratchView<Real[2][NP][NP]> grad_p(team.team_scratch(0));
      ScratchView<Real[NP][NP]> p_ilev;

      {
        p_ilev = subview(pressure, 0, Kokkos::ALL(), Kokkos::ALL());
        gradient_sphere(team, p_ilev, m_data, m_region.DINV(ie), grad_p);
        const Real vgrad_p =
            m_region.U(ie, m_data.n0())(0, igp, jgp) * grad_p(0, igp, jgp) +
            m_region.V(ie, m_data.n0())(0, igp, jgp) * grad_p(1, igp, jgp);

        const Real ckk = 0.5 / p_ilev(igp, jgp);
        const Real term = div_vdp(0, igp, jgp);
        omega_p(0, igp, jgp) = vgrad_p / p_ilev(igp, jgp) - ckk * term;
        suml(igp, jgp) = term;
      }

      // Another candidate for parallel scan
      for (int ilev = 1; ilev < NUM_LEV - 1; ++ilev) {
        p_ilev = subview(pressure, ilev, Kokkos::ALL(), Kokkos::ALL());
        gradient_sphere(team, p_ilev, m_data, m_region.DINV(ie), grad_p);
        const Real vgrad_p =
            m_region.U(ie, m_data.n0())(ilev, igp, jgp) * grad_p(0, igp, jgp) +
            m_region.V(ie, m_data.n0())(ilev, igp, jgp) * grad_p(1, igp, jgp);

        const Real ckk = 0.5 / p_ilev(igp, jgp);
        const Real ckl = 2.0 * ckk;
        const Real term = div_vdp(ilev, igp, jgp);
        omega_p(ilev, igp, jgp) =
            vgrad_p / p_ilev(igp, jgp) - ckl * suml(igp, jgp) - ckk * term;

        suml(igp, jgp) += term;
      }

      {
        p_ilev = subview(pressure, NUM_LEV - 1, Kokkos::ALL(), Kokkos::ALL());
        gradient_sphere(team, p_ilev, m_data, m_region.DINV(ie), grad_p);
        const Real vgrad_p =
            m_region.U(ie, m_data.n0())(NUM_LEV - 1, igp, jgp) *
                grad_p(0, igp, jgp) +
            m_region.V(ie, m_data.n0())(NUM_LEV - 1, igp, jgp) *
                grad_p(1, igp, jgp);

        const Real ckk = 0.5 / p_ilev(igp, jgp);
        const Real ckl = 2.0 * ckk;
        const Real term = div_vdp(NUM_LEV - 1, igp, jgp);
        omega_p(NUM_LEV - 1, igp, jgp) =
            vgrad_p / p_ilev(igp, jgp) - ckl * suml(igp, jgp) - ckk * term;
      }
    });
    team.team_barrier();
  }

  // Depends on DP3D
  KOKKOS_INLINE_FUNCTION
  void compute_pressure(Kokkos::TeamPolicy<>::member_type team,
                        ScratchView<Real[NUM_LEV][NP][NP]> pressure) const {
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      pressure(0, igp, jgp) = m_data.hybrid_a(0) * m_data.ps0() +
                              0.5 * m_region.DP3D(ie, m_data.n0())(0, igp, jgp);
    });

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                         [&](const int idx) {
      for (int ilev = 1; ilev < NUM_LEV; ilev++) {
        const int igp = idx / NP;
        const int jgp = idx % NP;
        pressure(ilev, igp, jgp) =
            pressure(ilev - 1, igp, jgp) +
            0.5 * m_region.DP3D(ie, m_data.n0())(ilev - 1, igp, jgp) +
            0.5 * m_region.DP3D(ie, m_data.n0())(ilev, igp, jgp);
      }
    });
  }

  KOKKOS_INLINE_FUNCTION
  void
  compute_T_v_no_tracers_helper(Kokkos::TeamPolicy<>::member_type team,
                                int ilev,
                                ScratchView<Real[NUM_LEV][NP][NP]> T_v) const {
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                         KOKKOS_LAMBDA(const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      T_v(ilev, igp, jgp) = m_region.T(ie, m_data.n0())(ilev, igp, jgp);
    });
  }

  KOKKOS_INLINE_FUNCTION
  void
  compute_T_v_tracers_helper(Kokkos::TeamPolicy<>::member_type team, int ilev,
                             ScratchView<Real[NUM_LEV][NP][NP]> T_v) const {
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                         KOKKOS_LAMBDA(const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      Real Qt = m_region.QDP(ie, m_data.qn0(), 1)(ilev, igp, jgp) /
                m_region.DP3D(ie, m_data.n0())(ilev, igp, jgp);
      T_v(ilev, igp, jgp) =
          m_region.T(ie, m_data.n0())(ilev, igp, jgp) *
          (1.0 +
           (PhysicalConstants::Rwater_vapor / PhysicalConstants::Rgas - 1.0) *
               Qt);
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_T_v(Kokkos::TeamPolicy<>::member_type team,
                   ScratchView<Real[NUM_LEV][NP][NP]> T_v) const {
    if (m_data.qn0() == -1) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV),
                           KOKKOS_LAMBDA(const int ilev) {
        compute_T_v_no_tracers_helper(team, ilev, T_v);
      });
    } else {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV),
                           KOKKOS_LAMBDA(const int ilev) {
        compute_T_v_tracers_helper(team, ilev, T_v);
      });
    }
  }

  // Modifies UN0, VN0, PHI, OMEGA_P, T, and DP3D
  KOKKOS_INLINE_FUNCTION
  void compute_stuff(Kokkos::TeamPolicy<>::member_type team,
                     ScratchView<Real[NUM_LEV][NP][NP]> pressure,
                     ScratchView<Real[NUM_LEV][NP][NP]> T_v) const {
    const int ie = team.league_rank();

    // Initialized in divergence_sphere
    // Used 2 times per index, 1 of which is in preq_omega_ps
    ScratchView<Real[NUM_LEV][NP][NP]> div_vdp(team.team_scratch(0));

    // For each thread, requires 2 x NP x NP Scratch Memory
    // Depends on UN0, VN0, METDET, DINV
    // Initializes div_vdp
    // Modifies UN0, VN0
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV),
                         [&](const int ilev) {

      // Create subviews to explicitly have static dimensions
      ScratchView<Real[2][NP][NP]> vdp_ilev(team.team_scratch(0));

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                           [&](const int idx) {
        const int igp = idx / NP;
        const int jgp = idx % NP;
        Real v1 = m_region.U(ie, m_data.n0())(ilev, igp, jgp);
        Real v2 = m_region.V(ie, m_data.n0())(ilev, igp, jgp);

        vdp_ilev(0, igp, jgp) =
            v1 * m_region.DP3D(ie, m_data.n0())(ilev, igp, jgp);
        vdp_ilev(1, igp, jgp) =
            v2 * m_region.DP3D(ie, m_data.n0())(ilev, igp, jgp);

        m_region.UN0(ie, ilev)(igp, jgp) +=
            PhysicalConstants::eta_ave_w * vdp_ilev(0, igp, jgp);
        m_region.VN0(ie, ilev)(igp, jgp) +=
            PhysicalConstants::eta_ave_w * vdp_ilev(1, igp, jgp);
      });

      ScratchView<Real[NP][NP]> div_vdp_ilev =
        Kokkos::subview(div_vdp, ilev, Kokkos::ALL(), Kokkos::ALL());
      divergence_sphere(team, vdp_ilev, m_data, m_region.METDET(ie),
                        m_region.DINV(ie), div_vdp_ilev);
    });
    // Threads_NL = min(NUM_LEV, Max_Threads)
    // Maximum memory usage so far: (NUM_LEV + 2 x Threads_NL) x NP x NP

    team.team_barrier();

    ExecViewUnmanaged<Real[NP][NP]> phis_ie = m_region.PHIS(ie);
    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> phi_ie = m_region.PHI(ie);

    preq_hydrostatic(team, pressure, T_v);

    // Maximum memory usage so far: (NUM_LEV + 2 x Threads) x NP x NP

    // Initialized in preq_omega_ps
    // Used 2 times per index
    ScratchView<Real[NUM_LEV][NP][NP]> omega_p(team.team_scratch(0));

    // Maximum memory usage so far: (NUM_LEV + max(NUM_LEV, 2 x Threads_NL)) x NP x NP

    preq_omega_ps(team, pressure, div_vdp, omega_p);
    // Threads_NP = min(NP x NP, Max_Threads)
    // Maximum memory usage so far: (NUM_LEV + max(NUM_LEV + 3 x NP x NP x Threads_NP, 2 x Threads_NL)) x NP x NP

    team.team_barrier();

    ScratchView<Real[2][NP][NP]> grad_tmp(team.team_scratch(0));

    // Maximum memory usage so far: (NUM_LEV + max(NUM_LEV + max(3 x NP x NP x Threads_NP, 2 x NP x NP), 2 x Threads_NL)) x NP x NP

    // Updates OMEGA_P, T, and DP3D
    // Depends on T, OMEGA_P, T_v, U, V, SPHEREMP, and omega_p
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV_P),
                         [&](const int ilev) {
      // Create subviews to explicitly have static dimensions
      ExecViewUnmanaged<Real[NP][NP]> T_ie_n0_ilev =
          Kokkos::subview(m_region.T(ie, m_data.n0()), ilev, Kokkos::ALL(), Kokkos::ALL());
      gradient_sphere(team, T_ie_n0_ilev, m_data, m_region.DINV(ie), grad_tmp);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                           [&](const int idx) {
        const int igp = idx / NP;
        const int jgp = idx % NP;

        m_region.OMEGA_P(ie, ilev)(igp, jgp) +=
            PhysicalConstants::eta_ave_w * omega_p(ilev, igp, jgp);

        const Real cur_T_v = T_v(ilev, igp, jgp);

        const Real v1 = m_region.U(ie, m_data.n0())(ilev, igp, jgp);
        const Real v2 = m_region.V(ie, m_data.n0())(ilev, igp, jgp);

        const Real ttens =
            // T_vadv(ilev, igp, jgp)
            0.0 - (v1 * grad_tmp(0, igp, jgp) + v2 * grad_tmp(1, igp, jgp)) +
            // kappa_star(ilev, igp, jgp)
            PhysicalConstants::kappa * cur_T_v * omega_p(ilev, igp, jgp);

        m_region.T(ie, m_data.np1())(ilev, igp, jgp) =
            m_region.SPHEREMP(ie)(igp, jgp) *
            (m_region.T(ie, m_data.nm1())(ilev, igp, jgp) +
             m_data.dt2() * ttens);

        m_region.DP3D(ie, m_data.np1())(ilev, igp, jgp) =
            m_region.SPHEREMP(ie)(igp, jgp) *
            (m_region.DP3D(ie, m_data.nm1())(ilev, igp, jgp) +
             m_data.dt2() * div_vdp(ilev, igp, jgp));
      });
    });
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(Kokkos::TeamPolicy<>::member_type team) const {
    // Used 5 times per index - basically the most important variable
    ScratchView<Real[NUM_LEV][NP][NP]> pressure(team.team_scratch(0));
    compute_pressure(team, pressure);

    // Used 3 times per index
    ScratchView<Real[NUM_LEV][NP][NP]> T_v(team.team_scratch(0));
    compute_T_v(team, T_v);

    compute_stuff(team, pressure, T_v);
    compute_velocity(team, pressure, T_v);
    compute_eta_dpdn(team);
  }

  size_t shmem_size(const int team_size) const {
    const int mem_2d_scalar = ScratchView<Real[NP][NP]>::shmem_size();
    const int mem_2d_vector = ScratchView<Real[2][NP][NP]>::shmem_size();
    const int mem_3d_scalar = ScratchView<Real[NUM_LEV][NP][NP]>::shmem_size();
    const int mem_3d_p_scalar =
        ScratchView<Real[NUM_LEV_P][NP][NP]>::shmem_size();
    const int mem_3d_vector =
        ScratchView<Real[NUM_LEV][2][NP][NP]>::shmem_size();

    constexpr const int num_2d_tmp_scalars = 5;
    constexpr const int num_2d_tmp_vectors = 2;
    constexpr const int num_3d_tmp_scalars = 11;
    constexpr const int num_3d_tmp_vectors = 3;
    constexpr const int num_3d_p_tmp_scalars = 1;

    const int mem_needed = num_2d_tmp_scalars * mem_2d_scalar +
                           num_2d_tmp_vectors * mem_2d_vector +
                           num_3d_tmp_scalars * mem_3d_scalar +
                           num_3d_tmp_vectors * mem_3d_vector +
                           num_3d_p_tmp_scalars * mem_3d_p_scalar;

    return mem_needed;
  }
};

KOKKOS_INLINE_FUNCTION
void preq_hydrostatic(const ExecViewUnmanaged<Real[NP][NP]> phis,
                      const ScratchView<Real[NUM_LEV][NP][NP]> T_v,
                      const ScratchView<Real[NUM_LEV][NP][NP]> p,
                      const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> dp,
                      Real Rgas, ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> phi) {
  Real hkk, hkl;
  Real phii[NUM_LEV][NP][NP];
  for (int jgp = 0; jgp < NP; ++jgp) {
    for (int igp = 0; igp < NP; ++igp) {
      hkk = 0.5 * dp(NUM_LEV - 1, igp, jgp) / p(NUM_LEV - 1, igp, jgp);
      hkl = 2.0 * hkk;
      phii[NUM_LEV - 1][igp][jgp] = Rgas * T_v(NUM_LEV - 1, igp, jgp) * hkl;
      phi(NUM_LEV - 1, igp, jgp) =
          phis(igp, jgp) + Rgas * T_v(NUM_LEV - 1, igp, jgp) * hkk;
    }
    // Another candidate for parallel scan
    for (int ilev = NUM_LEV - 2; ilev > 1; --ilev) {
      for (int igp = 0; igp < NP; ++igp) {
        hkk = 0.5 * dp(ilev, igp, jgp) / p(ilev, igp, jgp);
        hkl = 2.0 * hkk;
        phii[ilev][igp][jgp] =
            phii[ilev + 1][igp][jgp] + Rgas * T_v(ilev, igp, jgp) * hkl;
        phi(ilev, igp, jgp) = phis(igp, jgp) + phii[ilev + 1][igp][jgp] +
                              Rgas * T_v(ilev, igp, jgp) * hkk;
      }
    }
    for (int igp = 0; igp < NP; ++igp) {
      hkk = 0.5 * dp(0, igp, jgp) / p(0, igp, jgp);
      phi(0, igp, jgp) =
          phis(igp, jgp) + phii[1][igp][jgp] + Rgas * T_v(0, igp, jgp) * hkk;
    }
  }
}

void compute_and_apply_rhs(const Control &data, Region &region) {
  update_state f(data, region);
  Kokkos::parallel_for(Kokkos::TeamPolicy<>(data.num_elems(), Kokkos::AUTO), f);
}

KOKKOS_INLINE_FUNCTION
void preq_omega_ps(const ScratchView<Real[NUM_LEV][NP][NP]> pressure,
                   const ScratchView<Real[NUM_LEV][NP][NP]> vgrad_p,
                   const ScratchView<Real[NUM_LEV][NP][NP]> div_vdp,
                   ScratchView<Real[NUM_LEV][NP][NP]> omega_p) {
  Real ckk, ckl, term;
  Real suml[NP][NP];
  for (int jgp = 0; jgp < NP; ++jgp) {
    for (int igp = 0; igp < NP; ++igp) {
      ckk = 0.5 / pressure(0, igp, jgp);
      term = div_vdp(0, igp, jgp);
      omega_p(0, igp, jgp) =
          vgrad_p(0, igp, jgp) / pressure(0, igp, jgp) - ckk * term;
      suml[igp][jgp] = term;
    }
    for (int ilev = 1; ilev < NUM_LEV - 1; ++ilev) {
      for (int igp = 0; igp < NP; ++igp) {
        ckk = 0.5 / pressure(ilev, igp, jgp);
        ckl = 2.0 * ckk;
        term = div_vdp(ilev, igp, jgp);
        omega_p(ilev, igp, jgp) =
            vgrad_p(ilev, igp, jgp) / pressure(ilev, igp, jgp) -
            ckl * suml[igp][jgp] - ckk * term;

        suml[igp][jgp] += term;
      }
    }
    for (int igp = 0; igp < NP; ++igp) {
      ckk = 0.5 / pressure(NUM_LEV - 1, igp, jgp);
      ckl = 2.0 * ckk;
      term = div_vdp(NUM_LEV - 1, igp, jgp);
      omega_p(NUM_LEV - 1, igp, jgp) =
          vgrad_p(NUM_LEV - 1, igp, jgp) / pressure(NUM_LEV - 1, igp, jgp) -
          ckl * suml[igp][jgp] - ckk * term;
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
//           vnorm  += std::pow( scalars_4d(ie,np1,IDX_U,ilev,igp,jgp)
// , 2 );
//           vnorm  += std::pow( scalars_4d(ie,np1,IDX_V,ilev,igp,jgp)
// , 2 );
//           tnorm  += std::pow( scalars_4d(ie,np1,IDX_T,ilev,igp,jgp)
// , 2 );
//           dpnorm += std::pow(
// scalars_4d(ie,np1,IDX_DP3D,ilev,igp,jgp), 2 );
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
