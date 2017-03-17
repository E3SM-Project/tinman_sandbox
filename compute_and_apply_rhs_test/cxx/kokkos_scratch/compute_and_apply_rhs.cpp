#include "compute_and_apply_rhs.hpp"

#include "Types.hpp"
#include "Region.hpp"
#include "TestData.hpp"
#include "sphere_operators.hpp"

#include "ScratchStack.hpp"

#include "timer.hpp"

#include <fstream>
#include <iomanip>

#include <stdio.h>

namespace TinMan {

struct update_state {
  const Control m_data;
  Region m_region;

  KOKKOS_INLINE_FUNCTION
  update_state(const Control &data, const int &num_elems)
      : m_data(data), m_region(num_elems) {}

  // For each thread, requires 3 x NP x NP memory
  // Depends on PHI (after preq_hydrostatic), PECND
  // Modifies Ephi_grad
  template <typename Grad_View>
  KOKKOS_INLINE_FUNCTION void
  compute_energy_grad(Kokkos::TeamPolicy<>::member_type &team,
                      ScratchStack &scratch_mem, const int ilev,
                      ExecViewUnmanaged<const Real[2][2][NP][NP]> c_dinv,
                      Grad_View Ephi_grad) const {
    const int ie = team.league_rank();

    ExecViewUnmanaged<Real[NP][NP]> Ephi(
        static_cast<Real *>(scratch_mem.allocate_thread(sizeof(Real[NP][NP]))));

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                         [&](const int idx) {
      const int igp = idx / NP;
      const int jgp = idx % NP;
      Real v1 = m_region.U_current(ie)(ilev, igp, jgp);
      Real v2 = m_region.V_current(ie)(ilev, igp, jgp);
      // Kinetic energy + PHI (thermal energy?) + PECND (potential energy?)
      Ephi(igp, jgp) = 0.5 * (v1 * v1 + v2 * v2) +
                       m_region.PHI_update(ie)(ilev, igp, jgp) +
                       m_region.PECND(ie, ilev)(igp, jgp);
    });
    gradient_sphere_update(team, scratch_mem, Ephi, m_data, c_dinv, Ephi_grad);
    // We shouldn't need a block here, as the parallel loops were vector level,
    // not thread level
    scratch_mem.free_thread(sizeof(Real[NP][NP]));
  }

  // For each thread, requires NP x NP x 6 Scratch Memory
  // Depends on pressure, PHI, U_current, V_current, METDET,
  // D, DINV, U, V, FCOR, SPHEREMP, T_v
  // Modifies U, V
  KOKKOS_INLINE_FUNCTION
  void compute_velocity(Kokkos::TeamPolicy<>::member_type &team,
                        ScratchStack &scratch_mem,
                        ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure,
                        ExecViewUnmanaged<const Real[2][2][NP][NP]> c_dinv,
                        ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_v) const {
    const int ie = team.league_rank();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV),
                         [&](const int ilev) {
      ExecViewUnmanaged<const Real[NP][NP]> p_ilev =
          subview(pressure, ilev, Kokkos::ALL(), Kokkos::ALL());

      ExecViewUnmanaged<Real[2][NP][NP]> grad_buf(static_cast<Real *>(
          scratch_mem.allocate_thread(sizeof(Real[2][NP][NP]))));

      gradient_sphere(team, scratch_mem, p_ilev, m_data, c_dinv, grad_buf);
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, 2 * NP * NP),
                           [&](const int idx) {
        const int hgp = (idx / NP) / NP;
        const int igp = (idx / NP) % NP;
        const int jgp = idx % NP;

        const Real gpterm = T_v(ilev, igp, jgp) / p_ilev(igp, jgp);
        grad_buf(hgp, igp, jgp) *= PhysicalConstants::Rgas * gpterm;
      });

      // grad_buf -> Ephi_grad + glnpsi
      compute_energy_grad(team, scratch_mem, ilev, c_dinv, grad_buf);

      ExecViewUnmanaged<Real[NP][NP]> vort(static_cast<Real *>(
          scratch_mem.allocate_thread(sizeof(Real[NP][NP]))));
      vorticity_sphere(team, scratch_mem, m_region.U_current(ie, ilev),
                       m_region.V_current(ie, ilev), m_data,
                       m_region.METDET(ie), m_region.D(ie), vort);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                           [&](const int idx) {
        const int igp = idx / NP;
        const int jgp = idx % NP;

        const Real fcor_vort_coeff =
            m_region.FCOR(ie)(igp, jgp) + vort(igp, jgp);
        const Real v2 = m_region.V_current(ie)(ilev, igp, jgp);

        const Real vtens1 =
            // v_vadv(igp, jgp, 0)
            v2 * fcor_vort_coeff - grad_buf(0, igp, jgp);

        const Real spheremp = m_region.SPHEREMP(ie)(igp, jgp);

        m_region.U_future(ie)(ilev, igp, jgp) =
            spheremp *
            (m_region.U_previous(ie)(ilev, igp, jgp) + m_data.dt2() * vtens1);

        const Real v1 = m_region.U_current(ie)(ilev, igp, jgp);

        const Real vtens2 =
            // v_vadv(igp, jgp, 1) -
            -v1 * fcor_vort_coeff - grad_buf(1, igp, jgp);

        m_region.V_future(ie)(ilev, igp, jgp) =
            spheremp *
            (m_region.V_previous(ie)(ilev, igp, jgp) + m_data.dt2() * vtens2);
      });
    });
    // free grad_buf
    // free vort
    scratch_mem.free_thread(sizeof(Real[2][NP][NP]) + sizeof(Real[NP][NP]));
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

        m_region.ETA_DPDN_update(ie)(ilev, igp, jgp) =
            m_region.ETA_DPDN(ie)(ilev, igp, jgp) +
            PhysicalConstants::eta_ave_w * eta_dot_dpdn_ie;
      });
    });
  }

  // For each thread, requires 0 Scratch Memory
  // Depends on PHIS, DP3D, PHI, T_v
  // Modifies PHI
  KOKKOS_INLINE_FUNCTION
  void
  preq_hydrostatic(const Kokkos::TeamPolicy<>::member_type &team,
                   const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure,
                   const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_v) const {
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

  // For each thread, requires 3 x NP x NP Scratch Memory
  // Depends on pressure, div_vdp, omega_p
  // Sets pressure equal to omega_p
  KOKKOS_INLINE_FUNCTION
  void
  preq_omega_ps(const Kokkos::TeamPolicy<>::member_type &team,
                ScratchStack &scratch_mem,
                const ExecViewUnmanaged<const Real[2][2][NP][NP]> c_dinv,
                const ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> div_vdp,
                const ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure) const {
    const int ie = team.league_rank();
    // Need to test if false sharing causes Kokkos::single to be faster
    ExecViewUnmanaged<Real[NP][NP]> suml(
        static_cast<Real *>(scratch_mem.allocate_team(sizeof(Real[NP][NP]))));
    ExecViewUnmanaged<Real[2][NP][NP]> grad_p(static_cast<Real *>(
        scratch_mem.allocate_team(sizeof(Real[2][NP][NP]))));
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NP * NP),
                         KOKKOS_LAMBDA(const int loop_idx) {
      const int jgp = loop_idx / NP;
      const int igp = loop_idx % NP;
      ExecViewUnmanaged<Real[NP][NP]> p_ilev;

      {
        p_ilev = subview(pressure, 0, Kokkos::ALL(), Kokkos::ALL());
        // const Kokkos::Impl::CudaTeamMember,   ScratchStack *const,
        // TinMan::ExecViewUnmanaged<TinMan::Real [4][4]>,   const
        // TinMan::Control,
        // const TinMan::ExecViewUnmanaged<const TinMan::Real [2][2][4][4]>,
        // const
        // TinMan::ExecViewUnmanaged<TinMan::Real [2][4][4]>
        gradient_sphere(team, scratch_mem, p_ilev, m_data, c_dinv, grad_p);
        const Real vgrad_p =
            m_region.U_current(ie)(0, igp, jgp) * grad_p(0, igp, jgp) +
            m_region.V_current(ie)(0, igp, jgp) * grad_p(1, igp, jgp);

        const Real ckk = 0.5 / p_ilev(igp, jgp);
        const Real term = div_vdp(0, igp, jgp);
        p_ilev(igp, jgp) = vgrad_p / p_ilev(igp, jgp) - ckk * term;
        suml(igp, jgp) = term;
      }

      // Another candidate for parallel scan
      for (int ilev = 1; ilev < NUM_LEV - 1; ++ilev) {
        p_ilev = subview(pressure, ilev, Kokkos::ALL(), Kokkos::ALL());
        gradient_sphere(team, scratch_mem, p_ilev, m_data, c_dinv, grad_p);
        const Real vgrad_p =
            m_region.U_current(ie)(ilev, igp, jgp) * grad_p(0, igp, jgp) +
            m_region.V_current(ie)(ilev, igp, jgp) * grad_p(1, igp, jgp);

        const Real ckk = 0.5 / p_ilev(igp, jgp);
        const Real ckl = 2.0 * ckk;
        const Real term = div_vdp(ilev, igp, jgp);
        p_ilev(igp, jgp) =
            vgrad_p / p_ilev(igp, jgp) - ckl * suml(igp, jgp) - ckk * term;

        suml(igp, jgp) += term;
      }

      {
        p_ilev = subview(pressure, NUM_LEV - 1, Kokkos::ALL(), Kokkos::ALL());
        gradient_sphere(team, scratch_mem, p_ilev, m_data, c_dinv, grad_p);
        const Real vgrad_p =
            m_region.U_current(ie)(NUM_LEV - 1, igp, jgp) *
                grad_p(0, igp, jgp) +
            m_region.V_current(ie)(NUM_LEV - 1, igp, jgp) * grad_p(1, igp, jgp);

        const Real ckk = 0.5 / p_ilev(igp, jgp);
        const Real ckl = 2.0 * ckk;
        const Real term = div_vdp(NUM_LEV - 1, igp, jgp);
        p_ilev(igp, jgp) =
            vgrad_p / p_ilev(igp, jgp) - ckl * suml(igp, jgp) - ckk * term;
      }
    });
    scratch_mem.free_team(sizeof(Real[NP][NP]) + sizeof(Real[2][NP][NP]));
    team.team_barrier();
  }

  KOKKOS_INLINE_FUNCTION
  void compute_pressure_helper(
      Kokkos::TeamPolicy<>::member_type team,
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
  compute_pressure(Kokkos::TeamPolicy<>::member_type team,
                   ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure) const {
    Kokkos::single(Kokkos::PerTeam(team), KOKKOS_LAMBDA() {
      compute_pressure_helper(team, pressure);
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_T_v_no_tracers_helper(
      Kokkos::TeamPolicy<>::member_type team, int ilev,
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
  void compute_T_v_tracers_helper(
      Kokkos::TeamPolicy<>::member_type team, int ilev,
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

  KOKKOS_INLINE_FUNCTION
  void compute_T_v(Kokkos::TeamPolicy<>::member_type team,
                   ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_v) const {
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

  // Requires 2 x NUM_LEV x NP x NP team memory
  // Requires 7 x NP x NP thread memory
  // Depends on DERIVED_UN0, DERIVED_VN0, U, V,
  // Modifies DERIVED_UN0, DERIVED_VN0, OMEGA_P, T, and DP3D
  KOKKOS_INLINE_FUNCTION
  void compute_stuff(Kokkos::TeamPolicy<>::member_type team,
                     ScratchStack &scratch_mem,
                     ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure,
                     ExecViewUnmanaged<const Real[2][2][NP][NP]> c_dinv,
                     ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_v) const {
    const int ie = team.league_rank();

    // Initialized in divergence_sphere
    // Used 2 times per index, 1 of which is in preq_omega_ps
    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> div_vdp(static_cast<Real *>(
        scratch_mem.allocate_team(sizeof(Real[NUM_LEV][NP][NP]))));

    // Depends on DERIVED_UN0, DERIVED_VN0, METDET, DINV
    // Initializes div_vdp
    // Modifies DERIVED_UN0, DERIVED_VN0
    // Requires NUM_LEV * 5 * NP * NP
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV),
                         [&](const int ilev) {

      // Create subviews to explicitly have static dimensions
      ExecViewUnmanaged<Real[2][NP][NP]> vdp_ilev(static_cast<Real *>(
          scratch_mem.allocate_thread(sizeof(Real[2][NP][NP]))));

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
      divergence_sphere(team, scratch_mem, vdp_ilev, m_data,
                        m_region.METDET(ie), c_dinv, div_vdp_ilev);
    });

    scratch_mem.free_thread(sizeof(Real[2][NP][NP]));
    team.team_barrier();

    preq_omega_ps(team, scratch_mem, c_dinv, div_vdp, pressure);
    auto &omega_p = pressure;

    // Updates OMEGA_P, T, and DP3D
    // Depends on T (global), OMEGA_P (global), U (global), V (global),
    // SPHEREMP (global), T_v, and omega_p
    // Requires 2 * NUM_LEV * NP * NP scratch memory
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, NUM_LEV),
                         [&](const int ilev) {
      // Create subviews to explicitly have static dimensions
      ExecViewUnmanaged<const Real[NP][NP]> T_ie_n0_ilev = Kokkos::subview(
          m_region.T_current(ie), ilev, Kokkos::ALL(), Kokkos::ALL());

      ExecViewUnmanaged<Real[2][NP][NP]> grad_tmp(static_cast<Real *>(
          scratch_mem.allocate_thread(sizeof(Real[2][NP][NP]))));

      gradient_sphere(team, scratch_mem, T_ie_n0_ilev, m_data, c_dinv,
                      grad_tmp);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                           [&](const int idx) {
        const int igp = idx / NP;
        const int jgp = idx % NP;

        m_region.OMEGA_P_update(ie, ilev)(igp, jgp) =
            m_region.OMEGA_P(ie, ilev)(igp, jgp) +
            PhysicalConstants::eta_ave_w * omega_p(ilev, igp, jgp);

        const Real cur_T_v = T_v(ilev, igp, jgp);

        const Real v1 = m_region.U_current(ie)(ilev, igp, jgp);
        const Real v2 = m_region.V_current(ie)(ilev, igp, jgp);

        const Real ttens =
            // T_vadv(ilev, igp, jgp)
            0.0 - (v1 * grad_tmp(0, igp, jgp) + v2 * grad_tmp(1, igp, jgp)) +
            // kappa_star(ilev, igp, jgp)
            PhysicalConstants::kappa * cur_T_v * omega_p(ilev, igp, jgp);

        m_region.T_future(ie)(ilev, igp, jgp) =
            m_region.SPHEREMP(ie)(igp, jgp) *
            (m_region.T_previous(ie)(ilev, igp, jgp) + m_data.dt2() * ttens);

        m_region.DP3D_future(ie)(ilev, igp, jgp) =
            m_region.SPHEREMP(ie)(igp, jgp) *
            (m_region.DP3D_previous(ie)(ilev, igp, jgp) -
             m_data.dt2() * div_vdp(ilev, igp, jgp));
      });
    });
    // free grad_tmp
    scratch_mem.free_thread(sizeof(Real[2][NP][NP]));
    // free div_vdp
    scratch_mem.free_team(sizeof(Real[NUM_LEV][NP][NP]));
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
      const Kokkos::TeamPolicy<>::member_type &team,
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
  void init_const_cache(const Kokkos::TeamPolicy<>::member_type &team,
                        ExecViewUnmanaged<Real[2][2][NP][NP]> c_dinv,
                        ExecViewUnmanaged<Real[NP][NP]> c_dvv) const {
    const int ie = team.league_rank();
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, NP * NP),
                         KOKKOS_LAMBDA(int idx) {
      const int hgp = idx / NP;
      const int igp = idx % NP;
      c_dvv(hgp, igp) = m_data.dvv(hgp, igp);
      for (int jgp = 0; jgp < 2; ++jgp) {
        for (int kgp = 0; kgp < 2; ++kgp) {
          c_dinv(jgp, kgp, hgp, igp) = m_region.DINV(ie)(jgp, kgp, hgp, igp);
        }
      }
    });
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(Kokkos::TeamPolicy<>::member_type team) const {
    Real *memory = ExecViewUnmanaged<Real *>(team.team_scratch(0),
                                             shmem_size(team.team_size()) /
                                                 sizeof(Real)).ptr_on_device();
    ScratchStack scratch_mem(&team, memory, shmem_size(team.team_size()));

    // Used 5 times per index - basically the most important variable
    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure(static_cast<Real *>(
        scratch_mem.allocate_team(sizeof(Real[NUM_LEV][NP][NP]))));
    compute_pressure(team, pressure);

    // Cache dinv, and dvv
    ExecViewUnmanaged<Real[2][2][NP][NP]> c_dinv(static_cast<Real *>(
        scratch_mem.allocate_team(sizeof(Real[NUM_LEV][NP][NP]))));
    ExecViewUnmanaged<Real[NP][NP]> c_dvv(
        static_cast<Real *>(scratch_mem.allocate_team(sizeof(Real[NP][NP]))));

    Kokkos::single(Kokkos::PerTeam(team),
                   KOKKOS_LAMBDA() { init_const_cache(team, c_dinv, c_dvv); });

    // Used 3 times per index
    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_v(static_cast<Real *>(
        scratch_mem.allocate_team(sizeof(Real[NUM_LEV][NP][NP]))));
    compute_T_v(team, T_v);

    preq_hydrostatic(team, pressure, T_v);
    compute_velocity(team, scratch_mem, pressure, c_dinv, T_v);
    compute_eta_dpdn(team);
    // Breaks pressure
    compute_stuff(team, scratch_mem, pressure, c_dinv, T_v);
  }

  KOKKOS_INLINE_FUNCTION
  size_t shmem_size(const int team_size) const {
    size_t tmp = 4;
    return tmp;
  }

  void next_compute_apply_rhs() { m_region.next_compute_apply_rhs(); }

  const Region &region() const { return m_region; }
};

void run_simulation(int num_elems, int num_exec, bool dump_results) {
  Control data(num_elems);

  // Print norm of initial states, to check we are using same data in all tests
  update_state f(data, num_elems);
  print_results_2norm(data, f.region());

  std::cout << " --- Performing computations... (" << num_exec
            << " executions of the main loop on " << num_elems
            << " elements)\n";

  // Burn in before timing to reduce cache effect
  Kokkos::parallel_for(
      Kokkos::TeamPolicy<Kokkos::Cuda>(data.num_elems(), Kokkos::AUTO), f);
  ExecSpace::fence();

  Timer::Timer global_timer;
  global_timer.startTimer();
  for (int i = 0; i < num_exec; ++i) {
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<Kokkos::Cuda>(data.num_elems(), Kokkos::AUTO), f);
    ExecSpace::fence();
  }
  global_timer.stopTimer();

  std::cout << "   ---> compute_and_apply_rhs execution total time: "
            << global_timer << "\n";

  //  for(int i = 0; i < num_exec; ++i) {
  //    std::cout << timers[i] << std::endl;
  //  }

  print_results_2norm(data, f.region());

  if (dump_results) {
    f.region().save_state(data);
  }
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

  std::cout << std::setprecision(17);
  std::cout << "   ---> Norms:\n"
            << "          ||v||_2  = " << std::sqrt(vnorm) << "\n"
            << "          ||T||_2  = " << std::sqrt(tnorm) << "\n"
            << "          ||dp||_2 = " << std::sqrt(dpnorm) << "\n";
}

} // Namespace TinMan
