#include "compute_and_apply_rhs.hpp"

#include "Types.hpp"
#include "Region.hpp"
#include "TestData.hpp"
#include "sphere_operators.hpp"

#include <fstream>
#include <iomanip>

#include <assert.h>

namespace TinMan {

struct update_state {
  const Control m_data;
  const Region m_region;

  KOKKOS_INLINE_FUNCTION
  update_state(const Control &data, const Region &region)
      : m_data(data), m_region(region) {}

  // For each thread, requires 3 x NP x NP memory
  // Depends on PHI (after preq_hydrostatic), PECND
  // Modifies Ephi_grad
  KOKKOS_INLINE_FUNCTION void
  compute_energy_grad(TeamPolicy &team, const int ilev,
                      ExecViewUnmanaged<const Real[2][2][NP][NP]> c_dinv,
                      ExecViewUnmanaged<Real[2][NP][NP]> Ephi_grad) const {
    const int ie = team.league_rank();

    // Using a slot larger than we need - Can be up to 4 x 4 x 2 = 32 elements
    Real _tmp_viewptr[NP][NP];
    ExecViewUnmanaged<Real[NP][NP]> Ephi(&_tmp_viewptr[0][0]);
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      Ephi(0, 0) = m_region.U_current(ie)(ilev, igp, jgp);
      Ephi(0, 1) = m_region.V_current(ie)(ilev, igp, jgp);
      // Kinetic energy + PHI (thermal energy?) + PECND (potential energy?)
      Ephi(igp, jgp) =
          0.5 * (Ephi(0, 0) * Ephi(0, 0) + Ephi(0, 1) * Ephi(0, 1)) +
          m_region.PHI_update(ie)(ilev, igp, jgp) +
          m_region.PECND(ie, ilev)(igp, jgp);
    });
    gradient_sphere_update(team, Ephi, m_data, c_dinv, Ephi_grad);
    // We shouldn't need a block here, as the parallel loops were vector level,
    // not thread level
  }

  // For each thread, requires NP x NP x 6 Scratch Memory
  // Depends on pressure, PHI, U_current, V_current, METDET,
  // D, DINV, U, V, FCOR, SPHEREMP, T_v
  // Modifies U, V
  KOKKOS_INLINE_FUNCTION
  void compute_velocity(TeamPolicy &team,
                        ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure,
                        ExecViewUnmanaged<const Real[2][2][NP][NP]> c_dinv,
                        ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_v) const {
    const int ie = team.league_rank();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV),
                         [&](const int ilev) {
      ExecViewUnmanaged<const Real[NP][NP]> p_ilev =
          subview(pressure, ilev, Kokkos::ALL(), Kokkos::ALL());

      ExecViewUnmanaged<Real[2][NP][NP]> grad_buf =
          Kokkos::subview(m_data.vector_buf(team), ilev, Kokkos::ALL(),
                          Kokkos::ALL(), Kokkos::ALL());
      gradient_sphere(team, p_ilev, m_data, c_dinv, grad_buf);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, 2 * NP * NP),
                           [&](const int idx) {
        const int hgp = (idx / NP) / NP;
        const int igp = (idx / NP) % NP;
        const int jgp = idx % NP;

        const Real gpterm = T_v(ilev, igp, jgp) / p_ilev(igp, jgp);
        grad_buf(hgp, igp, jgp) *= PhysicalConstants::Rgas * gpterm;
      });

      // grad_buf -> Ephi_grad + glnpsi
      compute_energy_grad(team, ilev, c_dinv, grad_buf);

      Real _tmp_viewptr[NP][NP];
      ExecViewUnmanaged<Real[NP][NP]> vort(&_tmp_viewptr[0][0]);
      vorticity_sphere(team, m_region.U_current(ie, ilev),
                       m_region.V_current(ie, ilev), m_data,
                       m_region.METDET(ie), m_region.D(ie), vort);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                           [&](const int idx) {
        const int igp = idx / NP;
        const int jgp = idx % NP;

        vort(igp, jgp) += m_region.FCOR(ie)(igp, jgp);

        const Real vtens1 =
            // v_vadv(igp, jgp, 0)
            m_region.V_current(ie)(ilev, igp, jgp) * vort(igp, jgp) -
            grad_buf(0, igp, jgp);

        m_region.U_future(ie)(ilev, igp, jgp) =
            m_region.SPHEREMP(ie)(igp, jgp) *
            (m_region.U_previous(ie)(ilev, igp, jgp) + m_data.dt2() * vtens1);

        const Real vtens2 =
            // v_vadv(igp, jgp, 1) -
            -m_region.U_current(ie)(ilev, igp, jgp) * vort(igp, jgp) -
            grad_buf(1, igp, jgp);

        m_region.V_future(ie)(ilev, igp, jgp) =
            m_region.SPHEREMP(ie)(igp, jgp) *
            (m_region.V_previous(ie)(ilev, igp, jgp) + m_data.dt2() * vtens2);
      });
    });
  }

  // For each thread, requires 0 Scratch Memory
  // Depends on ETA_DPDN
  // Modifies ETA_DPDN
  KOKKOS_INLINE_FUNCTION
  void compute_eta_dpdn(TeamPolicy &team) const {
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV_P),
                         [&](const int ilev) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                           [&](const int idx) {
        const int igp = idx / NP;
        const int jgp = idx % NP;

        // TODO: Compute the actual value for this
        Real eta_dot_dpdn_ie = 0.0;

        m_region.ETA_DPDN_update(ie)(ilev, igp, jgp) =
            m_region.ETA_DPDN(ie)(ilev, igp, jgp) +
            PhysicalConstants::eta_ave_w * eta_dot_dpdn_ie;
      });
    });
  }

  // Depends on PHIS, DP3D, PHI, pressure, T_v
  // Modifies PHI
  KOKKOS_INLINE_FUNCTION
  void preq_hydrostatic(
      const TeamPolicy &team,
      const ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> pressure,
      const ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> T_v) const {
    const int ie = team.league_rank();

    ExecViewUnmanaged<const Real[NP][NP]> phis = m_region.PHIS(ie);

    ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> dp =
        m_region.DP3D_current(ie);

    ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> phi = m_region.PHI(ie);
    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> phi_update =
        m_region.PHI_update(ie);

    // Need to test if false sharing causes Kokkos::single to be faster
    Kokkos::single(Kokkos::PerTeam(team), KOKKOS_LAMBDA() {
      for (int igp = 0; igp < NP; ++igp) {
        for (int jgp = 0; jgp < NP; ++jgp) {

          Real phii;
          {
            const Real hk =
                dp(NUM_LEV - 1, igp, jgp) / pressure(NUM_LEV - 1, igp, jgp);
            phii = PhysicalConstants::Rgas * T_v(NUM_LEV - 1, igp, jgp) * hk;
            phi_update(NUM_LEV - 1, igp, jgp) = phis(igp, jgp) + phii * 0.5;
          }

          for (int ilev = NUM_LEV - 2; ilev > 0; --ilev) {
            const Real hk = dp(ilev, igp, jgp) / pressure(ilev, igp, jgp);
            const Real lev_term =
                PhysicalConstants::Rgas * T_v(ilev, igp, jgp) * hk;
            phi_update(ilev, igp, jgp) = phis(igp, jgp) + phii + lev_term * 0.5;

            phii += lev_term;
          }

          {
            const Real hk = 0.5 * dp(0, igp, jgp) / pressure(0, igp, jgp);
            phi_update(0, igp, jgp) =
                phis(igp, jgp) + phii +
                PhysicalConstants::Rgas * T_v(0, igp, jgp) * hk;
          }
        }
      }
    });
    team.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION
  void preq_omega_ps_init(
      const TeamPolicy &team,
      const ExecViewUnmanaged<const Real[2][2][NP][NP]> c_dinv,
      const ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> div_vdp,
      ExecViewUnmanaged<Real[2][NP][NP]> grad_p,
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure,
      ExecViewUnmanaged<Real[NP][NP]> suml) const {

    int ie = team.league_rank();
    ExecViewUnmanaged<Real[NP][NP]> p_ilev =
        Kokkos::subview(pressure, 0, Kokkos::ALL(), Kokkos::ALL());
    gradient_sphere(team, p_ilev, m_data, c_dinv, grad_p);

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                         KOKKOS_LAMBDA(const int loop_idx) {
      const int jgp = loop_idx / NP;
      const int igp = loop_idx % NP;
      const Real vgrad_p =
          m_region.U_current(ie)(0, igp, jgp) * grad_p(0, igp, jgp) +
          m_region.V_current(ie)(0, igp, jgp) * grad_p(1, igp, jgp);

      const Real ckk = 0.5 / p_ilev(igp, jgp);
      const Real term = div_vdp(0, igp, jgp);
      p_ilev(igp, jgp) = vgrad_p / p_ilev(igp, jgp) - ckk * term;
      suml(igp, jgp) = term;
    });
  }

  KOKKOS_INLINE_FUNCTION
  void preq_omega_ps_loop(
      const TeamPolicy &team,
      const ExecViewUnmanaged<const Real[2][2][NP][NP]> c_dinv,
      const ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> div_vdp,
      ExecViewUnmanaged<Real[2][NP][NP]> grad_p,
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure,
      ExecViewUnmanaged<Real[NP][NP]> suml) const {
    // Another candidate for parallel scan
    int ie = team.league_rank();
    ExecViewUnmanaged<Real[NP][NP]> p_ilev;
    for (int ilev = 1; ilev < NUM_LEV - 1; ++ilev) {
      p_ilev = subview(pressure, ilev, Kokkos::ALL(), Kokkos::ALL());
      gradient_sphere(team, p_ilev, m_data, c_dinv, grad_p);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                           KOKKOS_LAMBDA(const int loop_idx) {
        const int jgp = loop_idx / NP;
        const int igp = loop_idx % NP;
        const Real vgrad_p =
            m_region.U_current(ie)(ilev, igp, jgp) * grad_p(0, igp, jgp) +
            m_region.V_current(ie)(ilev, igp, jgp) * grad_p(1, igp, jgp);

        const Real ckk = 0.5 / p_ilev(igp, jgp);
        const Real ckl = 2.0 * ckk;
        const Real term = div_vdp(ilev, igp, jgp);
        p_ilev(igp, jgp) =
            vgrad_p / p_ilev(igp, jgp) - ckl * suml(igp, jgp) - ckk * term;

        suml(igp, jgp) += term;
      });
    }
  }

  KOKKOS_INLINE_FUNCTION
  void preq_omega_ps_tail(
      const TeamPolicy &team,
      const ExecViewUnmanaged<const Real[2][2][NP][NP]> c_dinv,
      const ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> div_vdp,
      ExecViewUnmanaged<Real[2][NP][NP]> grad_p,
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure,
      ExecViewUnmanaged<Real[NP][NP]> suml) const {
    int ie = team.league_rank();
    ExecViewUnmanaged<Real[NP][NP]> p_ilev =
        subview(pressure, NUM_LEV - 1, Kokkos::ALL(), Kokkos::ALL());
    gradient_sphere(team, p_ilev, m_data, c_dinv, grad_p);
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                         KOKKOS_LAMBDA(const int loop_idx) {
      const int jgp = loop_idx / NP;
      const int igp = loop_idx % NP;
      const Real vgrad_p =
          m_region.U_current(ie)(NUM_LEV - 1, igp, jgp) * grad_p(0, igp, jgp) +
          m_region.V_current(ie)(NUM_LEV - 1, igp, jgp) * grad_p(1, igp, jgp);

      const Real ckk = 0.5 / p_ilev(igp, jgp);
      const Real ckl = 2.0 * ckk;
      const Real term = div_vdp(NUM_LEV - 1, igp, jgp);
      p_ilev(igp, jgp) =
          vgrad_p / p_ilev(igp, jgp) - ckl * suml(igp, jgp) - ckk * term;
    });
  }

  // For each thread, requires 3 x NP x NP Scratch Memory
  // Depends on pressure, div_vdp, omega_p
  // Sets pressure equal to omega_p
  KOKKOS_INLINE_FUNCTION
  void
  preq_omega_ps(const TeamPolicy &team,
                const ExecViewUnmanaged<const Real[2][2][NP][NP]> c_dinv,
                const ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> div_vdp,
                const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure) const {
    const int ie = team.league_rank();
    // NOTE: we can't use a single TeamThreadRange loop, since
    //       gradient_sphere requires a 'consistent' pressure,
    //       meaning that we cannot update the different pressure
    //       points within a level before the gradient is complete!

    Kokkos::single(Kokkos::PerTeam(team), [&]() {
      Real _tmp_viewptr[2][NP][NP];
      ExecViewUnmanaged<Real[NP][NP]> suml(&_tmp_viewptr[0][0][0]);
      ExecViewUnmanaged<Real[2][NP][NP]> grad_p(&_tmp_viewptr[1][0][0]);

      preq_omega_ps_init(team, c_dinv, div_vdp, grad_p, pressure, suml);
      preq_omega_ps_loop(team, c_dinv, div_vdp, grad_p, pressure, suml);
      preq_omega_ps_tail(team, c_dinv, div_vdp, grad_p, pressure, suml);
    });
    team.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION
  void compute_pressure_helper(
      TeamPolicy team,
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure) const {
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                         KOKKOS_LAMBDA(const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      pressure(0, igp, jgp) = m_data.hybrid_a(0) * m_data.ps0() +
                              0.5 * m_region.DP3D_current(ie)(0, igp, jgp);
    });
    for (int ilev = 1; ilev < NUM_LEV; ilev++) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                           KOKKOS_LAMBDA(const int idx) {
        int igp = idx / NP;
        int jgp = idx % NP;
        pressure(ilev, igp, jgp) =
            pressure(ilev - 1, igp, jgp) +
            0.5 * (m_region.DP3D_current(ie)(ilev - 1, igp, jgp) +
                   m_region.DP3D_current(ie)(ilev, igp, jgp));
      });
    }
  }

  // Depends on DP3D
  KOKKOS_INLINE_FUNCTION
  void
  compute_pressure(TeamPolicy team,
                   ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure) const {
    Kokkos::single(Kokkos::PerTeam(team), KOKKOS_LAMBDA() {
      compute_pressure_helper(team, pressure);
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_pressure_omegap_phis(
      const TeamPolicy &team,
      const ExecViewUnmanaged<const Real[2][2][NP][NP]> c_dinv,
      const ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> div_vdp,
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> omega_p,
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure) {
    Kokkos::single(Kokkos::PerTeam(team), KOKKOS_LAMBDA() {
      for (int ilev = 0; ilev < NUM_LEV; ilev++) {
      }
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_temperature_no_tracers_helper(
      TeamPolicy team, int ilev,
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_v) const {
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                         KOKKOS_LAMBDA(const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      T_v(ilev, igp, jgp) = m_region.T_current(ie)(ilev, igp, jgp);
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_temperature_tracers_helper(
      TeamPolicy team, int ilev,
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_v) const {
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                         KOKKOS_LAMBDA(const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;

      Real Qt = m_region.QDP(ie, 0, m_data.qn0())(ilev, igp, jgp) /
                m_region.DP3D_current(ie)(ilev, igp, jgp);
      T_v(ilev, igp, jgp) =
          m_region.T_current(ie)(ilev, igp, jgp) *
          (1.0 +
           (PhysicalConstants::Rwater_vapor / PhysicalConstants::Rgas - 1.0) *
               Qt);
    });
  }

  // Depends on DERIVED_UN0, DERIVED_VN0, METDET, DINV
  // Initializes div_vdp, which is used 2 times afterwards
  // Modifies DERIVED_UN0, DERIVED_VN0
  // Requires NUM_LEV * 5 * NP * NP
  KOKKOS_INLINE_FUNCTION
  void compute_div_vdp(TeamPolicy team, int ilev,
                       const ExecViewUnmanaged<const Real[2][2][NP][NP]> c_dinv,
                       ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> div_vdp) const {
    const int ie = team.league_rank();

    // Create subviews to explicitly have static dimensions
    ExecViewUnmanaged<Real[2][NP][NP]> vdp_ilev =
        Kokkos::subview(m_data.vector_buf(team), ilev, Kokkos::ALL,
                        Kokkos::ALL(), Kokkos::ALL());

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      Real v1 = m_region.U_current(ie)(ilev, igp, jgp);
      Real v2 = m_region.V_current(ie)(ilev, igp, jgp);

      vdp_ilev(0, igp, jgp) = v1 * m_region.DP3D_current(ie)(ilev, igp, jgp);
      vdp_ilev(1, igp, jgp) = v2 * m_region.DP3D_current(ie)(ilev, igp, jgp);

      m_region.DERIVED_UN0_update(ie, ilev)(igp, jgp) =
          m_region.DERIVED_UN0(ie, ilev)(igp, jgp) +
          PhysicalConstants::eta_ave_w * vdp_ilev(0, igp, jgp);
      m_region.DERIVED_VN0_update(ie, ilev)(igp, jgp) =
          m_region.DERIVED_VN0(ie, ilev)(igp, jgp) +
          PhysicalConstants::eta_ave_w * vdp_ilev(1, igp, jgp);
    });

    ExecViewUnmanaged<Real[NP][NP]> div_vdp_ilev =
        Kokkos::subview(div_vdp, ilev, Kokkos::ALL(), Kokkos::ALL());
    divergence_sphere(team, vdp_ilev, m_data, m_region.METDET(ie), c_dinv,
                      div_vdp_ilev);
  }

  // Depends on T_current
  // Might depend on QDP, DP3D_current
  KOKKOS_INLINE_FUNCTION
  void compute_temperature_div_vdp(
      TeamPolicy team, const ExecViewUnmanaged<const Real[2][2][NP][NP]> c_dinv,
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_v,
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> div_vdp) const {
    if (m_data.qn0() == -1) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV),
                           KOKKOS_LAMBDA(const int ilev) {
        compute_temperature_no_tracers_helper(team, ilev, T_v);
      });
    } else {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV),
                           KOKKOS_LAMBDA(const int ilev) {
        compute_temperature_tracers_helper(team, ilev, T_v);
      });
    }
    team.team_barrier();
  }

  // Requires 2 x NUM_LEV x NP x NP team memory
  // Requires 7 x NP x NP thread memory
  // Depends on DERIVED_UN0, DERIVED_VN0, U, V,
  // Modifies DERIVED_UN0, DERIVED_VN0, OMEGA_P, T, and DP3D
  // block_3d_scalars
  KOKKOS_INLINE_FUNCTION
  void compute_stuff(TeamPolicy team,
                     ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure,
                     ExecViewUnmanaged<const Real[2][2][NP][NP]> c_dinv,
                     ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_v) const {
    const int ie = team.league_rank();

    // Used 2 times per index, 1 of which is in preq_omega_ps
    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> div_vdp = m_data.div_vdp(team);
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, NUM_LEV),
        [&](const int ilev) { compute_div_vdp(team, ilev, c_dinv, div_vdp); });

    // TODO: Rename pressure after here to omega_p
    preq_omega_ps(team, c_dinv, div_vdp, pressure);
    // Updates OMEGA_P, T, and DP3D
    // Depends on T (global), OMEGA_P (global), U (global), V (global),
    // SPHEREMP (global), T_v, and omega_p
    // Requires 2 * NUM_LEV * NP * NP scratch memory
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV),
                         [&](const int ilev) {
      // Create subviews to explicitly have static dimensions
      ExecViewUnmanaged<const Real[NP][NP]> T_ie_n0_ilev = Kokkos::subview(
          m_region.T_current(ie), ilev, Kokkos::ALL(), Kokkos::ALL());

      ExecViewUnmanaged<Real[2][NP][NP]> grad_tmp = Kokkos::subview(
          m_data.vector_buf(team), ilev, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);

      gradient_sphere(team, T_ie_n0_ilev, m_data, c_dinv, grad_tmp);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                           [&](const int idx) {
        const int igp = idx / NP;
        const int jgp = idx % NP;

        m_region.OMEGA_P_update(ie, ilev)(igp, jgp) =
            m_region.OMEGA_P(ie, ilev)(igp, jgp) +
            PhysicalConstants::eta_ave_w * pressure(ilev, igp, jgp);

        const Real cur_T_v = T_v(ilev, igp, jgp);

        const Real v1 = m_region.U_current(ie)(ilev, igp, jgp);
        const Real v2 = m_region.V_current(ie)(ilev, igp, jgp);

        const Real ttens =
            // T_vadv(ilev, igp, jgp)
            0.0 - (v1 * grad_tmp(0, igp, jgp) + v2 * grad_tmp(1, igp, jgp)) +
            // kappa_star(ilev, igp, jgp)
            PhysicalConstants::kappa * cur_T_v * pressure(ilev, igp, jgp);

        m_region.T_future(ie)(ilev, igp, jgp) =
            m_region.SPHEREMP(ie)(igp, jgp) *
            (m_region.T_previous(ie)(ilev, igp, jgp) + m_data.dt2() * ttens);

        m_region.DP3D_future(ie)(ilev, igp, jgp) =
            m_region.SPHEREMP(ie)(igp, jgp) *
            (m_region.DP3D_previous(ie)(ilev, igp, jgp) -
             m_data.dt2() * div_vdp(ilev, igp, jgp));
      });
    });
  }

  // call preq_vertadv(fptr%base(ie)%state%T(:,:,:,n0), &
  //                   fptr%base(ie)%state%v(:,:,:,:,n0), &
  //                   eta_dot_dpdn,rdp,T_vadv,v_vadv)
  // ...
  // real (kind=real_kind), intent(in) :: T(np,np,nlev)
  // real (kind=real_kind), intent(in) :: v(np,np,2,nlev)
  // real (kind=real_kind), intent(in) :: eta_dot_dp_deta(np,np,nlevp)
  // real (kind=real_kind), intent(in) :: rpdel(np,np,nlev)

  // real (kind=real_kind), intent(out) :: T_vadv(np,np,nlev)
  // real (kind=real_kind), intent(out) :: v_vadv(np,np,2,nlev)

  // Computes the vertical advection of T and v
  KOKKOS_INLINE_FUNCTION
  void preq_vertadv(
      const TeamPolicy &team,
      const ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> T,
      const ExecViewUnmanaged<const Real[NUM_LEV][2][NP][NP]> v,
      const ExecViewUnmanaged<const Real[NUM_LEV_P][NP][NP]> eta_dp_deta,
      const ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> rpdel,
      ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_vadv,
      ExecViewUnmanaged<Real[NUM_LEV][2][NP][NP]> v_vadv) {
    constexpr const int k_0 = 0;
    for (int j = 0; j < NP; ++j) {
      for (int i = 0; i < NP; ++i) {
        Real facp = 0.5 * rpdel(k_0, j, i) * eta_dp_deta(k_0 + 1, j, i);
        T_vadv(k_0, j, i) = facp * (T(k_0 + 1, j, i) - T(k_0, j, i));
        for (int h = 0; h < 2; ++h) {
          v_vadv(k_0, h, j, i) = facp * (v(k_0 + 1, h, j, i) - v(k_0, h, j, i));
        }
      }
    }
    constexpr const int k_f = NUM_LEV - 1;
    for (int k = k_0 + 1; k < k_f; ++k) {
      for (int j = 0; j < NP; ++j) {
        for (int i = 0; i < NP; ++i) {
          Real facp = 0.5 * rpdel(k, j, i) * eta_dp_deta(k + 1, j, i);
          Real facm = 0.5 * rpdel(k, j, i) * eta_dp_deta(k, j, i);
          T_vadv(k, j, i) = facp * (T(k + 1, j, i) - T(k, j, i)) +
                            facm * (T(k, j, i) - T(k - 1, j, i));
          for (int h = 0; h < 2; ++h) {
            v_vadv(k, h, j, i) = facp * (v(k + 1, h, j, i) - v(k, h, j, i)) +
                                 facm * (v(k, h, j, i) - v(k - 1, h, j, i));
          }
        }
      }
    }
    for (int j = 0; j < NP; ++j) {
      for (int i = 0; i < NP; ++i) {
        Real facm = 0.5 * rpdel(k_f, j, i) * eta_dp_deta(k_f, j, i);
        T_vadv(k_f, j, i) = facm * (T(k_f, j, i) - T(k_f - 1, j, i));
        for (int h = 0; h < 2; ++h) {
          v_vadv(k_f, h, j, i) = facm * (v(k_f, h, j, i) - v(k_f - 1, h, j, i));
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void init_const_cache(const TeamPolicy &team,
                        ExecViewUnmanaged<Real[2][2][NP][NP]> c_dinv) const {
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                         KOKKOS_LAMBDA(int idx) {
      const int hgp = idx / NP;
      const int igp = idx % NP;
      for (int jgp = 0; jgp < 2; ++jgp) {
        for (int kgp = 0; kgp < 2; ++kgp) {
          c_dinv(jgp, kgp, hgp, igp) = m_region.DINV(ie)(jgp, kgp, hgp, igp);
        }
      }
    });
  }

  template <typename Data>
  KOKKOS_INLINE_FUNCTION Real *scratch_to_ptr(TeamPolicy team) const {
    ViewType<Data, ScratchMemSpace, Kokkos::MemoryUnmanaged> view(
        team.team_scratch(0));
    return view.data();
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(TeamPolicy team) const {
    // Used 2 times per index, 1 of which is in preq_omega_ps
    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> div_vdp = m_data.div_vdp(team);

    // Cache dinv, and dvv
    Real *ptr = scratch_to_ptr<Real[2][2][NP][NP]>(team);
    ExecViewUnmanaged<Real[2][2][NP][NP]> c_dinv(ptr);
    ExecViewUnmanaged<const Real[NP][NP]> c_dvv = m_data.dvv();
    init_const_cache(team, c_dinv);

    // Used 3 times per index
    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_v = m_data.T_v(team);
    compute_temperature_div_vdp(team, c_dinv, T_v, div_vdp);

    // Used 5 times per index - basically the most important variable
    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure = m_data.pressure(team);
    compute_pressure(team, pressure);

    preq_hydrostatic(team, pressure, T_v);
    compute_velocity(team, pressure, c_dinv, T_v);
    compute_eta_dpdn(team);
    // Breaks pressure
    compute_stuff(team, pressure, c_dinv, T_v);
  }

  KOKKOS_INLINE_FUNCTION
  size_t shmem_size(const int team_size) const {
    return sizeof(Real[2][2][NP][NP]);
  }
};

void compute_and_apply_rhs(const Control &data, Region &region,
                           int threads_per_team, int vectors_per_thread) {
  update_state f(data, region);
  Kokkos::parallel_for(Kokkos::TeamPolicy<ExecSpace>(data.num_elems(),
                                                     threads_per_team,
                                                     vectors_per_thread),
                       f);
  ExecSpace::fence();
}

void print_results_2norm(const Control &data, const Region &region) {
  Real vnorm(0.), tnorm(0.), dpnorm(0.);
  for (int ie = 0; ie < data.num_elems(); ++ie) {
    auto U = Kokkos::create_mirror_view(region.U_current(ie));
    Kokkos::deep_copy(U, region.U_future(ie));

    auto V = Kokkos::create_mirror_view(region.V_current(ie));
    Kokkos::deep_copy(V, region.V_future(ie));

    auto T = Kokkos::create_mirror_view(region.T_current(ie));
    Kokkos::deep_copy(T, region.T_future(ie));

    auto DP3D = Kokkos::create_mirror_view(region.DP3D_current(ie));
    Kokkos::deep_copy(DP3D, region.DP3D_future(ie));

    vnorm += std::pow(compute_norm(U), 2);
    vnorm += std::pow(compute_norm(V), 2);
    tnorm += std::pow(compute_norm(T), 2);
    dpnorm += std::pow(compute_norm(DP3D), 2);
  }

  std::cout << std::setprecision(15);
  std::cout << "   ---> Norms:\n"
            << "          ||v||_2  = " << std::sqrt(vnorm) << "\n"
            << "          ||T||_2  = " << std::sqrt(tnorm) << "\n"
            << "          ||dp||_2 = " << std::sqrt(dpnorm) << "\n";
}

} // Namespace TinMan
