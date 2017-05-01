#include "compute_and_apply_rhs.hpp"

#include "Region.hpp"
#include "TestData.hpp"
#include "Types.hpp"
#include "sphere_operators.hpp"

#include <fstream>
#include <iomanip>

#include <assert.h>

namespace TinMan {

struct update_state {
  const Control m_data;
  const Region m_region;

  struct KernelVariables {
    KOKKOS_INLINE_FUNCTION
    KernelVariables(TeamPolicy &team)
        : team(team), scalar_buf_1(allocate_thread<Real, Real[NP][NP]>(team)),
          scalar_buf_2(allocate_thread<Real, Real[NP][NP]>(team)),
          vector_buf_1(allocate_thread<Real, Real[2][NP][NP]>(team)),
          vector_buf_2(allocate_thread<Real, Real[2][NP][NP]>(team)),
          m_ie(team.league_rank()), m_ilev(-1), m_igp(-1), m_jgp(-1) {}

    template <typename Primitive, typename Data>
    KOKKOS_INLINE_FUNCTION Primitive *allocate_team(TeamPolicy &team) const {
      ViewType<Data, ScratchMemSpace, Kokkos::MemoryUnmanaged> view(
          team.team_scratch(0));
      return view.data();
    }

    template <typename Primitive, typename Data>
    KOKKOS_INLINE_FUNCTION Primitive *allocate_thread(TeamPolicy &team) const {
      ViewType<Data, ScratchMemSpace, Kokkos::MemoryUnmanaged> view(
          team.thread_scratch(0));
      return view.data();
    }

    KOKKOS_INLINE_FUNCTION
    static size_t shmem_size(int team_size) {
      size_t mem_size =
          (2 * sizeof(Real[2][NP][NP]) + 2 * sizeof(Real[NP][NP])) * team_size;
      return mem_size;
    }

    const TeamPolicy &team;
    ExecViewUnmanaged<Real[NP][NP]> scalar_buf_1;
    ExecViewUnmanaged<Real[NP][NP]> scalar_buf_2;
    ExecViewUnmanaged<Real[2][NP][NP]> vector_buf_1;
    ExecViewUnmanaged<Real[2][NP][NP]> vector_buf_2;
    int m_ie;
    int m_ilev, m_igp, m_jgp;
    Real m_temp[2];
  };

  KOKKOS_INLINE_FUNCTION
  update_state(const Control &data, const Region &region)
      : m_data(data), m_region(region) {}

  // Depends on PHI (after preq_hydrostatic), PECND
  // Modifies Ephi_grad
  KOKKOS_INLINE_FUNCTION void
  compute_energy_grad(KernelVariables &k_locals) const {
    // ExecViewUnmanaged<Real[NP][NP]> Ephi = k_locals.scalar_buf_1;
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(k_locals.team, NP * NP),
                         [&](const int idx) {
      k_locals.m_igp = idx / NP;
      k_locals.m_jgp = idx % NP;
      // Kinetic energy + PHI (geopotential energy) + PECND (potential energy?)
      // Ephi(igp, jgp) =
      //     0.5 * (m_region.U_current(k_locals.m_ie, k_locals.m_ilev)(igp, jgp)
      // *
      //                m_region.U_current(k_locals.m_ie, k_locals.m_ilev)(igp,
      // jgp) +
      //            m_region.V_current(k_locals.m_ie, k_locals.m_ilev)(igp, jgp)
      // *
      //                m_region.V_current(k_locals.m_ie, k_locals.m_ilev)(igp,
      // jgp)) +
      //     m_region.PHI_update(k_locals.m_ie)(k_locals.m_ilev, igp, jgp) +
      //     m_region.PECND(k_locals.m_ie, k_locals.m_ilev)(igp, jgp);
      Real &energy = k_locals.m_temp[0];
      energy =
          m_region.U_current(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                             k_locals.m_jgp) *
          m_region.U_current(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                             k_locals.m_jgp) +
          m_region.V_current(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                             k_locals.m_jgp) *
          m_region.V_current(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                             k_locals.m_jgp);
      energy *= 0.5;
      energy +=
        m_region.PHI_update(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                                             k_locals.m_jgp);
      k_locals.scalar_buf_1(k_locals.m_igp, k_locals.m_jgp) = energy + m_region.PECND(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp);
    });
    gradient_sphere_update(k_locals.team, k_locals.scalar_buf_1, m_data,
                           m_region.DINV(k_locals.m_ie), k_locals.vector_buf_1,
                           k_locals.vector_buf_2);
  }

  // Depends on pressure, PHI, U_current, V_current, METDET,
  // D, DINV, U, V, FCOR, SPHEREMP, T_v, ETA_DPDN
  KOKKOS_INLINE_FUNCTION void
  compute_phase_3(KernelVariables &k_locals) const {
    Kokkos::parallel_for(Kokkos::TeamThreadRange(k_locals.team, NUM_LEV),
                         [&](const int &ilev) {
      k_locals.m_ilev = ilev;
      compute_eta_dpdn(k_locals);
      compute_velocity(k_locals);
      compute_update_omega_p(k_locals);
      compute_update_temperature(k_locals);
      compute_dp3d(k_locals);
    });
  }

  // Depends on pressure, PHI, U_current, V_current, METDET,
  // D, DINV, U, V, FCOR, SPHEREMP, T_v
  KOKKOS_INLINE_FUNCTION
  void compute_velocity(KernelVariables &k_locals) const {
    ExecViewUnmanaged<const Real[NP][NP]> p_ilev =
        m_data.pressure(k_locals.m_ie, k_locals.m_ilev);

    gradient_sphere(k_locals.team, p_ilev, m_data, m_region.DINV(k_locals.m_ie),
                    k_locals.vector_buf_1, k_locals.vector_buf_2);
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(k_locals.team, 2 * NP * NP),
                         [&](const int idx) {
      const int hgp = (idx / NP) / NP;
      k_locals.m_igp = (idx / NP) % NP;
      k_locals.m_jgp = idx % NP;

      Real &glnpsi = k_locals.m_temp[0];
      glnpsi = k_locals.vector_buf_2(hgp, k_locals.m_igp, k_locals.m_jgp) *
          PhysicalConstants::Rgas *
          m_data.T_v(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp);
      k_locals.vector_buf_2(hgp, k_locals.m_igp, k_locals.m_jgp) = glnpsi /
          p_ilev(k_locals.m_igp, k_locals.m_jgp);
    });

    // k_locals.vector_buf_2 -> Ephi_grad + glnpsi
    compute_energy_grad(k_locals);

    const ExecViewUnmanaged<Real[NP][NP]> &vort = k_locals.scalar_buf_2;
    vorticity_sphere(k_locals.team,
                     m_region.U_current(k_locals.m_ie, k_locals.m_ilev),
                     m_region.V_current(k_locals.m_ie, k_locals.m_ilev), m_data,
                     m_region.METDET(k_locals.m_ie), m_region.D(k_locals.m_ie),
                     k_locals.vector_buf_1, vort);

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(k_locals.team, NP * NP),
                         [&](const int idx) {
      k_locals.m_igp = idx / NP;
      k_locals.m_jgp = idx % NP;

      Real &vort_fcor = k_locals.m_temp[0];
      vort_fcor = vort(k_locals.m_igp, k_locals.m_jgp) +
          m_region.FCOR(k_locals.m_ie, k_locals.m_igp, k_locals.m_jgp);

      Real &vel_tmp = k_locals.m_temp[1];
      // Compute U
      vel_tmp = -k_locals.vector_buf_2(0, k_locals.m_igp, k_locals.m_jgp);
      vel_tmp += // v_vadv(k_locals.m_igp, k_locals.m_jgp) +
          m_region.V_current(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                             k_locals.m_jgp) *
          vort_fcor;
      vel_tmp *= m_data.dt2();
      vel_tmp += m_region.U_previous(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                                     k_locals.m_jgp);
      m_region.U_future(k_locals.m_ie, k_locals.m_ilev,
                        k_locals.m_igp, k_locals.m_jgp) = vel_tmp *
        m_region.SPHEREMP(k_locals.m_ie, k_locals.m_igp, k_locals.m_jgp);

      // Compute V
      vel_tmp = -k_locals.vector_buf_2(1, k_locals.m_igp, k_locals.m_jgp);
      vel_tmp += // v_vadv(k_locals.m_igp, k_locals.m_jgp) +
          -m_region.U_current(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                              k_locals.m_jgp) *
          vort_fcor;
      vel_tmp *= m_data.dt2();
      vel_tmp += m_region.V_previous(k_locals.m_ie, k_locals.m_ilev,
                                     k_locals.m_igp, k_locals.m_jgp);
      m_region.V_future(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                        k_locals.m_jgp) =
          m_region.SPHEREMP(k_locals.m_ie, k_locals.m_igp, k_locals.m_jgp) *
          vel_tmp;
    });
  }

  // Depends on ETA_DPDN
  KOKKOS_INLINE_FUNCTION
  void compute_eta_dpdn(KernelVariables &k_locals) const {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(k_locals.team, NP * NP),
                         [&](const int idx) {
      k_locals.m_igp = idx / NP;
      k_locals.m_jgp = idx % NP;

      // TODO: Compute the actual value for this
      // Real eta_dot_dpdn_ie = 0.0;

      m_region.ETA_DPDN_update(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                               k_locals.m_jgp) =
          m_region.ETA_DPDN(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                            k_locals.m_jgp);
      // + PhysicalConstants::eta_ave_w * eta_dot_dpdn_ie;
    });
  }

  // Depends on PHIS, DP3D, PHI, pressure, T_v
  // Modifies PHI
  KOKKOS_INLINE_FUNCTION
  void preq_hydrostatic(KernelVariables &k_locals) const {
    const ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> pressure =
        m_data.pressure(k_locals.m_ie);
    const ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> T_v =
        m_data.T_v(k_locals.m_ie);

    ExecViewUnmanaged<const Real[NP][NP]> phis = m_region.PHIS(k_locals.m_ie);

    ExecViewUnmanaged<const Real[NUM_LEV][NP][NP]> dp =
        m_region.DP3D_current(k_locals.m_ie);

    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> phi_update =
        m_region.PHI_update(k_locals.m_ie);

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(k_locals.team, NP * NP),
                         [&](const int loop_idx) {
      k_locals.m_igp = loop_idx / NP;
      k_locals.m_jgp = loop_idx % NP;
      k_locals.scalar_buf_1(k_locals.m_igp, k_locals.m_jgp) = 0.0;
      for (k_locals.m_ilev = NUM_LEV - 1; k_locals.m_ilev >= 0;
           --k_locals.m_ilev) {
        k_locals.vector_buf_2(0, k_locals.m_igp, k_locals.m_jgp) =
            PhysicalConstants::Rgas /
            pressure(k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp);
        k_locals.vector_buf_2(0, k_locals.m_igp, k_locals.m_jgp) *=
            T_v(k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp);
        k_locals.vector_buf_2(0, k_locals.m_igp, k_locals.m_jgp) *=
            dp(k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp);

        phi_update(k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp) =
            phis(k_locals.m_igp, k_locals.m_jgp) +
            k_locals.scalar_buf_1(k_locals.m_igp, k_locals.m_jgp);
        // FMA so no temporary register needed
        phi_update(k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp) +=
            0.5 * k_locals.vector_buf_2(0, k_locals.m_igp, k_locals.m_jgp);

        k_locals.scalar_buf_1(k_locals.m_igp, k_locals.m_jgp) +=
            k_locals.vector_buf_2(0, k_locals.m_igp, k_locals.m_jgp);
      }
    });
  }

  // Depends on pressure, U_current, V_current, div_vdp, omega_p
  KOKKOS_INLINE_FUNCTION
  void preq_omega_ps(KernelVariables &k_locals) const {
    // NOTE: we can't use a single TeamThreadRange loop, since
    //       gradient_sphere requires a 'consistent' pressure,
    //       meaning that we cannot update the different pressure
    //       points within a level before the gradient is complete!
    // Uses k_locals.scalar_buf_1 for intermediate computations registers
    //      k_locals.scalar_buf_2 to store the intermediate integration
    //      k_locals.vector_buf_1 to store the gradient
    //      k_locals.vector_buf_2 for the gradient buffer
    //
    const ExecViewUnmanaged<Real[2][NP][NP]> &grad_p = k_locals.vector_buf_1;
    const ExecViewUnmanaged<Real[NP][NP]> &integral = k_locals.scalar_buf_2;
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(k_locals.team, NP * NP),
                         [&](const int loop_idx) {
      k_locals.m_igp = loop_idx / NP;
      k_locals.m_jgp = loop_idx % NP;
      integral(k_locals.m_igp, k_locals.m_jgp) = 0.0;
    });
    for (k_locals.m_ilev = 0; k_locals.m_ilev < NUM_LEV; ++k_locals.m_ilev) {
      ExecViewUnmanaged<const Real[NP][NP]> p_ilev =
          m_data.pressure(k_locals.m_ie, k_locals.m_ilev);
      gradient_sphere(k_locals.team, p_ilev, m_data,
                      m_region.DINV(k_locals.m_ie), k_locals.vector_buf_2,
                      grad_p);

      Kokkos::parallel_for(Kokkos::ThreadVectorRange(k_locals.team, NP * NP),
                           [&](const int loop_idx) {
        k_locals.m_igp = loop_idx / NP;
        k_locals.m_jgp = loop_idx % NP;
        Real vgrad_p = m_region.U_current(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp) *
                       grad_p(0, k_locals.m_igp, k_locals.m_jgp);
        vgrad_p += m_region.V_current(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp) *
                   grad_p(1, k_locals.m_igp, k_locals.m_jgp);
        vgrad_p -= integral(k_locals.m_igp, k_locals.m_jgp);
        vgrad_p += -0.5 * m_data.div_vdp(k_locals.m_ie, k_locals.m_ilev,
                                         k_locals.m_igp, k_locals.m_jgp);

        m_data.omega_p(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                       k_locals.m_jgp) =
            vgrad_p / p_ilev(k_locals.m_igp, k_locals.m_jgp);

        integral(k_locals.m_igp, k_locals.m_jgp) += m_data.div_vdp(
            k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp);
      });
    }
  }

  // Depends on DP3D
  KOKKOS_INLINE_FUNCTION
  void compute_pressure(KernelVariables &k_locals) const {
    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> pressure =
        m_data.pressure(k_locals.m_ie);
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(k_locals.team, NP * NP),
                         [&](const int idx) {
      k_locals.m_igp = idx / NP;
      k_locals.m_jgp = idx % NP;
      Real &tmp_1 = k_locals.m_temp[0];
      tmp_1 = m_data.hybrid_a(0) * m_data.ps0();

      Real &tmp_2 = k_locals.m_temp[1];
      tmp_2 = 0.5 * m_region.DP3D_current(k_locals.m_ie, 0, k_locals.m_igp, k_locals.m_jgp);
      pressure(0, k_locals.m_igp, k_locals.m_jgp) =
          tmp_1 + tmp_2;
      for (k_locals.m_ilev = 1; k_locals.m_ilev < NUM_LEV; k_locals.m_ilev++) {
        tmp_1 =
            0.5 * (m_region.DP3D_current(k_locals.m_ie, k_locals.m_ilev - 1, k_locals.m_igp, k_locals.m_jgp) +
                   m_region.DP3D_current(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp));
        pressure(k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp) =
            pressure(k_locals.m_ilev - 1, k_locals.m_igp, k_locals.m_jgp) +
            tmp_1;
      }
    });
  }

  // Depends on DP3D, PHIS, DP3D, PHI, T_v
  // Modifies pressure, PHI
  KOKKOS_INLINE_FUNCTION
  void compute_scan_properties(KernelVariables &k_locals) const {
    if (k_locals.team.team_rank() == 0) {
      compute_pressure(k_locals);
      preq_hydrostatic(k_locals);
      preq_omega_ps(k_locals);
    }
  }

  KOKKOS_INLINE_FUNCTION
  void compute_temperature_no_tracers_helper(KernelVariables &k_locals) const {
    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_v = m_data.T_v(k_locals.team);
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(k_locals.team, NP * NP),
                         [&](const int idx) {
      k_locals.m_igp = idx / NP;
      k_locals.m_jgp = idx % NP;
      T_v(k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp) = m_region.T_current(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp);
    });
  }

  KOKKOS_INLINE_FUNCTION
  void compute_temperature_tracers_helper(KernelVariables &k_locals) const {
    ExecViewUnmanaged<Real[NUM_LEV][NP][NP]> T_v = m_data.T_v(k_locals.team);
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(k_locals.team, NP * NP),
                         [&](const int idx) {
      k_locals.m_igp = idx / NP;
      k_locals.m_jgp = idx % NP;

      Real &Qt = k_locals.m_temp[0];
      Qt = m_region.QDP(k_locals.m_ie, 0, m_data.qn0(), k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp) /
           m_region.DP3D_current(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp);
      Qt *= PhysicalConstants::Rwater_vapor / PhysicalConstants::Rgas - 1.0;
      Qt += 1.0;
      T_v(k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp) =
          m_region.T_current(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                                            k_locals.m_jgp) *
          Qt;
    });
  }

  // Depends on DERIVED_UN0, DERIVED_VN0, METDET, DINV
  // Initializes div_vdp, which is used 2 times afterwards
  // Modifies DERIVED_UN0, DERIVED_VN0
  // Requires NUM_LEV * 5 * NP * NP
  KOKKOS_INLINE_FUNCTION
  void compute_div_vdp(KernelVariables &k_locals) const {
    // Create subviews to explicitly have static dimensions
    // ExecViewUnmanaged<Real[2][NP][NP]> vdp_ilev =
    // k_locals.vector_buf_2;

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(k_locals.team, NP * NP),
                         [&](const int idx) {
      k_locals.m_igp = idx / NP;
      k_locals.m_jgp = idx % NP;

      k_locals.vector_buf_2(0, k_locals.m_igp, k_locals.m_jgp) =
          m_region.U_current(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                             k_locals.m_jgp) *
          m_region.DP3D_current(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                                k_locals.m_jgp);

      k_locals.vector_buf_2(1, k_locals.m_igp, k_locals.m_jgp) =
          m_region.V_current(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                             k_locals.m_jgp) *
          m_region.DP3D_current(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                                k_locals.m_jgp);

      k_locals.scalar_buf_1(k_locals.m_igp, k_locals.m_jgp) =
          PhysicalConstants::eta_ave_w *
          k_locals.vector_buf_2(0, k_locals.m_igp, k_locals.m_jgp);

      m_region.DERIVED_UN0_update(k_locals.m_ie, k_locals.m_ilev,
                                  k_locals.m_igp, k_locals.m_jgp) =
          m_region.DERIVED_UN0(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                               k_locals.m_jgp) +
          k_locals.scalar_buf_1(k_locals.m_igp, k_locals.m_jgp);

      Real &tmp = k_locals.m_temp[0];
      // k_locals.scalar_buf_1(0, 0)
      tmp = PhysicalConstants::eta_ave_w *
            k_locals.vector_buf_2(1, k_locals.m_igp, k_locals.m_jgp);
      m_region.DERIVED_VN0_update(k_locals.m_ie, k_locals.m_ilev,
                                  k_locals.m_igp, k_locals.m_jgp) =
          m_region.DERIVED_VN0(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                               k_locals.m_jgp) + tmp;
    });

    ExecViewUnmanaged<Real[NP][NP]> div_vdp_ilev =
        Kokkos::subview(m_data.div_vdp(k_locals.m_ie), k_locals.m_ilev,
                        Kokkos::ALL(), Kokkos::ALL());
    divergence_sphere(k_locals.team, k_locals.vector_buf_2, m_data,
                      m_region.METDET(k_locals.m_ie),
                      m_region.DINV(k_locals.m_ie), k_locals.vector_buf_1,
                      div_vdp_ilev);
  }

  // Depends on T_current, DERIVE_UN0, DERIVED_VN0, METDET, DINV
  // Might depend on QDP, DP3D_current
  KOKKOS_INLINE_FUNCTION
  void compute_temperature_div_vdp(KernelVariables &k_locals) const {
    if (m_data.qn0() == -1) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(k_locals.team, NUM_LEV),
                           [&](const int ilev) {
        k_locals.m_ilev = ilev;
        compute_temperature_no_tracers_helper(k_locals);
        compute_div_vdp(k_locals);
      });
    } else {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(k_locals.team, NUM_LEV),
                           [&](const int ilev) {
        k_locals.m_ilev = ilev;
        compute_temperature_tracers_helper(k_locals);
        compute_div_vdp(k_locals);
      });
    }
  }

  KOKKOS_INLINE_FUNCTION
  void compute_update_omega_p(KernelVariables &k_locals) const {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(k_locals.team, NP * NP),
                         [&](const int idx) {
      k_locals.m_igp = idx / NP;
      k_locals.m_jgp = idx % NP;
      m_region.OMEGA_P_update(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp) =
        m_region.OMEGA_P(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp) +
        PhysicalConstants::eta_ave_w *
        m_data.omega_p(k_locals.m_ie, k_locals.m_ilev,
                       k_locals.m_igp, k_locals.m_jgp);
    });
  }

  // Depends on T (global), OMEGA_P (global), U (global), V (global),
  // SPHEREMP (global), T_v, and omega_p
  // block_3d_scalars
  KOKKOS_INLINE_FUNCTION
  void compute_update_temperature(KernelVariables &k_locals) const {
    ExecViewUnmanaged<const Real[NP][NP]> temperature =
        Kokkos::subview(m_region.T_current(k_locals.m_ie), k_locals.m_ilev,
                        Kokkos::ALL(), Kokkos::ALL());

    const ExecViewUnmanaged<Real[2][NP][NP]> &grad_tmp = k_locals.vector_buf_1;

    gradient_sphere(k_locals.team, temperature, m_data,
                    m_region.DINV(k_locals.m_ie), k_locals.vector_buf_2,
                    grad_tmp);

    Kokkos::parallel_for(Kokkos::ThreadVectorRange(k_locals.team, NP * NP),
                         [&](const int idx) {
      k_locals.m_igp = idx / NP;
      k_locals.m_jgp = idx % NP;

      Real &vgrad_t = k_locals.m_temp[0];
      vgrad_t = m_region.U_current(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                                   k_locals.m_jgp) *
                grad_tmp(0, k_locals.m_igp, k_locals.m_jgp) +
                m_region.V_current(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                                   k_locals.m_jgp) *
                grad_tmp(1, k_locals.m_igp, k_locals.m_jgp);

      // vgrad_t + kappa * T_v * omega_p
      Real &ttens = k_locals.m_temp[0];
      ttens = vgrad_t + PhysicalConstants::kappa *
              m_data.T_v(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                         k_locals.m_jgp) *
              m_data.omega_p(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                             k_locals.m_jgp);

      Real &future_temp = k_locals.m_temp[0];
      future_temp = ttens * m_data.dt2() +
        m_region.T_previous(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp);
        future_temp *= m_region.SPHEREMP(k_locals.m_ie, k_locals.m_igp, k_locals.m_jgp);

      m_region.T_future(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                        k_locals.m_jgp) = future_temp;
    });
  }

  // Depends on DERIVED_UN0, DERIVED_VN0, U, V,
  // Modifies DERIVED_UN0, DERIVED_VN0, OMEGA_P, T, and DP3D
  KOKKOS_INLINE_FUNCTION
  void compute_dp3d(KernelVariables &k_locals) const {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(k_locals.team, NP * NP),
                         [&](const int idx) {
      k_locals.m_igp = idx / NP;
      k_locals.m_jgp = idx % NP;
      Real tmp =
          m_data.div_vdp(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp,
                         k_locals.m_jgp) * -m_data.dt2() +
          m_region.DP3D_previous(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp);
      m_region.DP3D_future(k_locals.m_ie, k_locals.m_ilev, k_locals.m_igp, k_locals.m_jgp) =
          m_region.SPHEREMP(k_locals.m_ie, k_locals.m_igp, k_locals.m_jgp) *
          tmp;
    });
  }

  // Computes the vertical advection of T and v
  // Not currently used
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
  void operator()(TeamPolicy team) const {
    KernelVariables k_locals(team);

    compute_temperature_div_vdp(k_locals);

    compute_scan_properties(k_locals);

    compute_phase_3(k_locals);
  }

  KOKKOS_INLINE_FUNCTION
  size_t shmem_size(const int team_size) const {
    return KernelVariables::shmem_size(team_size);
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
