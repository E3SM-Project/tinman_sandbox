#include "compute_and_apply_rhs.hpp"
#include "test_macros.hpp"
#include "dimensions.hpp"
#include "data_structures.hpp"
#include "sphere_operators.hpp"

#include <cmath>
#include <iostream>
#include <fstream>

namespace Homme {

void compute_and_apply_rhs(TestData &data) {
  // Create local arrays
  real *Ephi = new real[np * np]{};
  real *T_v = new real[nlev * np * np]{};
  real *divdp = new real[nlev * np * np]{};
  real *grad_p = new real[nlev * np * np * 2]{};
  real *eta_dot_dpdn_tmp = new real[nlevp * np * np]{};
  real *kappa_star = new real[nlev * np * np]{};
  real *omega_p_tmp = new real[nlev * np * np]{};
  real *p = new real[nlev * np * np]{};
  real *ttens = new real[nlev * np * np]{};
  real *T_vadv = new real[nlev * np * np]{};
  real *v_vadv = new real[nlev * np * np * 2]{};
  real *vdp = new real[nlev * np * np * 2]{};
  real *vgrad_T = new real[np * np]{};
  real *vgrad_p = new real[nlev * np * np]{};
  real *vort = new real[nlev * np * np]{};
  real *vtemp = new real[np * np * 2]{};
  real *vtens1 = new real[nlev * np * np]{};
  real *vtens2 = new real[nlev * np * np]{};

  // Other accessory variables
  real Qt = 0;
  real glnps1 = 0;
  real glnps2 = 0;
  real gpterm = 0;
  real v1 = 0;
  real v2 = 0;

  real *Qdp_ie = nullptr;
  real *T_n0 = nullptr;
  real *T_nm1 = nullptr;
  real *T_np1 = nullptr;
  real *derived_vn0 = nullptr;
  real *dp3d_n0 = nullptr;
  real *dp3d_nm1 = nullptr;
  real *dp3d_np1 = nullptr;
  real *fcor = nullptr;
  real *omega_p = nullptr;
  real *pecnd = nullptr;
  real *phi = nullptr;
  real *phis = nullptr;
  real *spheremp = nullptr;
  real *v_n0 = nullptr;
  real *v_nm1 = nullptr;
  real *v_np1 = nullptr;
  real *eta_dot_dpdn = nullptr;

  // Input parameters
  const int nets = data.control.nets;
  const int nete = data.control.nete;
  const int n0 = data.control.n0;
  const int np1 = data.control.np1;
  const int nm1 = data.control.nm1;
  const int qn0 = data.control.qn0;
  const real dt2 = data.control.dt2;

  // Explicitly initialize T_vadv and v_vadv to 0 as our code is vertically Lagrangian
  for(int igp = 0; igp < nlev * np * np; ++igp) {
    for(int jgp = 0; jgp < 2; ++jgp) {
      v_vadv[igp * 2 + jgp] = 0.0;
    }
    T_vadv[igp] = 0.0;
  }

  // Loop over elements
  for (int ie = nets; ie < nete; ++ie) {
    dp3d_n0 = SLICE_5D_IJ(data.arrays.elem_state_dp3d, ie, n0, timelevels, nlev,
                          np, np);

    for (int igp = 0; igp < np; ++igp) {
      for (int jgp = 0; jgp < np; ++jgp) {
        AT_3D(p, 0, igp, jgp, np, np) =
            data.hvcoord.hyai[0] * data.hvcoord.ps0 +
            0.5 * AT_3D(dp3d_n0, 0, igp, jgp, np, np);
      }
    }

    for (int ilev = 1; ilev < nlev; ++ilev) {
      for (int igp = 0; igp < np; ++igp) {
        for (int jgp = 0; jgp < np; ++jgp) {
          AT_3D(p, ilev, igp, jgp, np, np) =
              AT_3D(p, (ilev - 1), igp, jgp, np, np) +
              0.5 * (AT_3D(dp3d_n0, (ilev - 1), igp, jgp, np, np) +
                     AT_3D(dp3d_n0, ilev, igp, jgp, np, np));
        }
      }
    }

    derived_vn0 = SLICE_5D(data.arrays.elem_derived_vn0, ie, nlev, np, np, 2);
    v_n0 = SLICE_6D_IJ(data.arrays.elem_state_v, ie, n0, timelevels, nlev, np,
                       np, 2);
    for (int ilev = 0; ilev < nlev; ++ilev) {
      gradient_sphere(SLICE_3D(p, ilev, np, np), data, ie,
                      SLICE_4D(grad_p, ilev, np, np, 2));

      for (int igp = 0; igp < np; ++igp) {
        for (int jgp = 0; jgp < np; ++jgp) {
          v1 = AT_4D(v_n0, ilev, igp, jgp, 0, np, np, 2);
          v2 = AT_4D(v_n0, ilev, igp, jgp, 1, np, np, 2);
          AT_3D(vgrad_p, ilev, igp, jgp, np, np) =
              v1 * AT_4D(grad_p, ilev, igp, jgp, 0, np, np, 2) +
              v2 * AT_4D(grad_p, ilev, igp, jgp, 1, np, np, 2);

          AT_4D(vdp, ilev, igp, jgp, 0, np, np, 2) =
              v1 * AT_3D(dp3d_n0, ilev, igp, jgp, np, np);
          AT_4D(vdp, ilev, igp, jgp, 1, np, np, 2) =
              v2 * AT_3D(dp3d_n0, ilev, igp, jgp, np, np);

          AT_4D(derived_vn0, ilev, igp, jgp, 0, np, np, 2) +=
              data.constants.eta_ave_w *
              AT_4D(vdp, ilev, igp, jgp, 0, np, np, 2);
          AT_4D(derived_vn0, ilev, igp, jgp, 1, np, np, 2) +=
              data.constants.eta_ave_w *
              AT_4D(vdp, ilev, igp, jgp, 1, np, np, 2);
        }
      }

      divergence_sphere(SLICE_4D(vdp, ilev, np, np, 2), data, ie,
                        SLICE_3D(divdp, ilev, np, np));
      vorticity_sphere(SLICE_4D(v_n0, ilev, np, np, 2), data, ie,
                       SLICE_3D(vort, ilev, np, np));
    }

    T_n0 =
        SLICE_5D_IJ(data.arrays.elem_state_T, ie, n0, timelevels, nlev, np, np);
    if (qn0 == -1) {
      for (int ilev = 0; ilev < nlev; ++ilev) {
        for (int igp = 0; igp < np; ++igp) {
          for (int jgp = 0; jgp < np; ++jgp) {
            AT_3D(T_v, ilev, igp, jgp, np, np) =
                AT_3D(T_n0, ilev, igp, jgp, np, np);
            AT_3D(kappa_star, ilev, igp, jgp, np, np) = data.constants.kappa;
          }
        }
      }
    } else {
      Qdp_ie =
          SLICE_6D(data.arrays.elem_state_Qdp, ie, nlev, np, np, qsize_d, 2);
      for (int ilev = 0; ilev < nlev; ++ilev) {
        for (int igp = 0; igp < np; ++igp) {
          for (int jgp = 0; jgp < np; ++jgp) {
            Qt = AT_5D(Qdp_ie, ilev, 1, qn0, igp, jgp, qsize_d, 2, np, np) /
                 AT_3D(dp3d_n0, ilev, igp, jgp, np, np);
            AT_3D(T_v, ilev, igp, jgp, np, np) =
                AT_3D(T_n0, ilev, igp, jgp, np, np) *
                (1.0 +
                 (data.constants.Rwater_vapor / data.constants.Rgas - 1.0) *
                     Qt);
            AT_3D(kappa_star, ilev, igp, jgp, np, np) = data.constants.kappa;
          }
        }
      }
    }

    phis = SLICE_3D(data.arrays.elem_state_phis, ie, np, np);
    phi = SLICE_4D(data.arrays.elem_derived_phi, ie, nlev, np, np);

    preq_hydrostatic(phis, T_v, p, dp3d_n0, data.constants.Rgas, phi);
    preq_omega_ps(p, vgrad_p, divdp, omega_p_tmp);

    omega_p = SLICE_4D(data.arrays.elem_derived_omega_p, ie, nlev, np, np);
    eta_dot_dpdn =
        SLICE_4D(data.arrays.elem_derived_eta_dot_dpdn, ie, nlevp, np, np);
    for (int ilev = 0; ilev < nlev; ++ilev) {
      for (int igp = 0; igp < np; ++igp) {
        for (int jgp = 0; jgp < np; ++jgp) {
          AT_3D(eta_dot_dpdn, ilev, igp, jgp, np, np) +=
              data.constants.eta_ave_w *
              AT_3D(eta_dot_dpdn_tmp, ilev, igp, jgp, np, np);
          AT_3D(omega_p, ilev, igp, jgp, np, np) +=
              data.constants.eta_ave_w *
              AT_3D(omega_p_tmp, ilev, igp, jgp, np, np);
        }
      }
    }
    for (int igp = 0; igp < np; ++igp) {
      for (int jgp = 0; jgp < np; ++jgp) {
        AT_3D(eta_dot_dpdn, nlev, igp, jgp, np, np) +=
            data.constants.eta_ave_w *
            AT_3D(eta_dot_dpdn_tmp, nlev, igp, jgp, np, np);
      }
    }

    pecnd = SLICE_4D(data.arrays.elem_derived_pecnd, ie, nlev, np, np);
    fcor = SLICE_3D(data.arrays.elem_fcor, ie, np, np);

    for (int ilev = 0; ilev < nlev; ++ilev) {
      for (int igp = 0; igp < np; ++igp) {
        for (int jgp = 0; jgp < np; ++jgp) {
          v1 = AT_4D(v_n0, ilev, igp, jgp, 0, np, np, 2);
          v2 = AT_4D(v_n0, ilev, igp, jgp, 1, np, np, 2);

          AT_2D(Ephi, igp, jgp, np) = 0.5 * (v1 * v1 + v2 * v2) +
                                      AT_3D(phi, ilev, igp, jgp, np, np) +
                                      AT_3D(pecnd, ilev, igp, jgp, np, np);
        }
      }

      gradient_sphere(SLICE_3D(T_n0, ilev, np, np), data, ie, vtemp);

      for (int igp = 0; igp < np; ++igp) {
        for (int jgp = 0; jgp < np; ++jgp) {
          v1 = AT_4D(v_n0, ilev, igp, jgp, 0, np, np, 2);
          v2 = AT_4D(v_n0, ilev, igp, jgp, 1, np, np, 2);

          AT_2D(vgrad_T, igp, jgp, np) = v1 * AT_3D(vtemp, igp, jgp, 0, np, 2) +
                                         v2 * AT_3D(vtemp, igp, jgp, 1, np, 2);
        }
      }

      gradient_sphere(Ephi, data, ie, vtemp);

      for (int igp = 0; igp < np; ++igp) {
        for (int jgp = 0; jgp < np; ++jgp) {
          gpterm = AT_3D(T_v, ilev, igp, jgp, np, np) /
                   AT_3D(p, ilev, igp, jgp, np, np);

          glnps1 = data.constants.Rgas * gpterm *
                   AT_4D(grad_p, ilev, igp, jgp, 0, np, np, 2);
          glnps2 = data.constants.Rgas * gpterm *
                   AT_4D(grad_p, ilev, igp, jgp, 1, np, np, 2);

          v1 = AT_4D(v_n0, ilev, igp, jgp, 0, np, np, 2);
          v2 = AT_4D(v_n0, ilev, igp, jgp, 1, np, np, 2);

          AT_3D(vtens1, ilev, igp, jgp, np, np) =
              AT_4D(v_vadv, ilev, igp, jgp, 0, np, np, 2) +
              v2 * (AT_2D(fcor, igp, jgp, np) +
                    AT_3D(vort, ilev, igp, jgp, np, np)) -
              AT_3D(vtemp, igp, jgp, 0, np, 2) - glnps1;
          AT_3D(vtens2, ilev, igp, jgp, np, np) =
              AT_4D(v_vadv, ilev, igp, jgp, 1, np, np, 2) +
              v1 * (AT_2D(fcor, igp, jgp, np) +
                    AT_3D(vort, ilev, igp, jgp, np, np)) -
              AT_3D(vtemp, igp, jgp, 1, np, 2) - glnps2;

          AT_3D(ttens, ilev, igp, jgp, np, np) =
              AT_3D(T_vadv, ilev, igp, jgp, np, np) -
              AT_2D(vgrad_T, igp, jgp, np) +
              AT_3D(kappa_star, ilev, igp, jgp, np, np) *
                  AT_3D(T_v, ilev, igp, jgp, np, np) *
                  AT_3D(omega_p_tmp, ilev, igp, jgp, np, np);
        }
      }
    }
    spheremp = SLICE_3D(data.arrays.elem_spheremp, ie, np, np);

    v_np1 = SLICE_6D_IJ(data.arrays.elem_state_v, ie, np1, timelevels, nlev, np,
                        np, 2);
    T_np1 = SLICE_5D_IJ(data.arrays.elem_state_T, ie, np1, timelevels, nlev, np,
                        np);
    dp3d_np1 = SLICE_5D_IJ(data.arrays.elem_state_dp3d, ie, np1, timelevels,
                           nlev, np, np);

    v_nm1 = SLICE_6D_IJ(data.arrays.elem_state_v, ie, nm1, timelevels, nlev, np,
                        np, 2);
    T_nm1 = SLICE_5D_IJ(data.arrays.elem_state_T, ie, nm1, timelevels, nlev, np,
                        np);
    dp3d_nm1 = SLICE_5D_IJ(data.arrays.elem_state_dp3d, ie, nm1, timelevels,
                           nlev, np, np);

    for (int ilev = 0; ilev < nlev; ++ilev) {
      for (int igp = 0; igp < np; ++igp) {
        for (int jgp = 0; jgp < np; ++jgp) {
          AT_4D(v_np1, ilev, igp, jgp, 0, np, np, 2) =
              AT_2D(spheremp, igp, jgp, np) *
              (AT_4D(v_nm1, ilev, igp, jgp, 0, np, np, 2) +
               dt2 * AT_3D(vtens1, ilev, igp, jgp, np, np));

          AT_4D(v_np1, ilev, igp, jgp, 1, np, np, 2) =
              AT_2D(spheremp, igp, jgp, np) *
              (AT_4D(v_nm1, ilev, igp, jgp, 1, np, np, 2) +
               dt2 * AT_3D(vtens2, ilev, igp, jgp, np, np));

          AT_3D(T_np1, ilev, igp, jgp, np, np) =
              AT_2D(spheremp, igp, jgp, np) *
              (AT_3D(T_nm1, ilev, igp, jgp, np, np) +
               dt2 * AT_3D(ttens, ilev, igp, jgp, np, np));
          AT_3D(dp3d_np1, ilev, igp, jgp, np, np) =
              AT_2D(spheremp, igp, jgp, np) *
              (AT_3D(dp3d_nm1, ilev, igp, jgp, np, np) +
               dt2 * AT_3D(divdp, ilev, igp, jgp, np, np));
        }
      }
    }
  }

  delete[] Ephi;
  delete[] T_v;
  delete[] divdp;
  delete[] grad_p;
  delete[] eta_dot_dpdn_tmp;
  delete[] kappa_star;
  delete[] omega_p_tmp;
  delete[] p;
  delete[] ttens;
  delete[] T_vadv;
  delete[] v_vadv;
  delete[] vdp;
  delete[] vgrad_T;
  delete[] vgrad_p;
  delete[] vort;
  delete[] vtemp;
  delete[] vtens1;
  delete[] vtens2;
}

void preq_hydrostatic(const real *const phis, const real *const T_v,
                      const real *const p, const real *dp, real Rgas,
                      real *const phi) {
  real hkk, hkl;
  real phii[nlev][np][np];

  for (int jgp = 0; jgp < np; ++jgp) {
    for (int igp = 0; igp < np; ++igp) {
      hkk = 0.5 * AT_3D(dp, (nlev - 1), igp, jgp, np, np) /
            AT_3D(p, (nlev - 1), igp, jgp, np, np);
      hkl = 2.0 * hkk;
      phii[nlev - 1][igp][jgp] =
          Rgas * AT_3D(T_v, (nlev - 1), igp, jgp, np, np) * hkl;
      AT_3D(phi, (nlev - 1), igp, jgp, np, np) =
          AT_2D(phis, igp, jgp, np) +
          Rgas * AT_3D(T_v, (nlev - 1), igp, jgp, np, np) * hkk;
    }
    for (int ilev = nlev - 2; ilev > 0; --ilev) {
      for (int igp = 0; igp < np; ++igp) {
        hkk = 0.5 * AT_3D(dp, ilev, igp, jgp, np, np) /
              AT_3D(p, ilev, igp, jgp, np, np);
        hkl = 2.0 * hkk;
        AT_3D(phi, ilev, igp, jgp, np, np) =
            AT_2D(phis, igp, jgp, np) + phii[ilev + 1][igp][jgp] +
            Rgas * AT_3D(T_v, ilev, igp, jgp, np, np) * hkk;
        phii[ilev][igp][jgp] = phii[ilev + 1][igp][jgp] +
                               Rgas * AT_3D(T_v, ilev, igp, jgp, np, np) * hkl;
      }
    }
    for (int igp = 0; igp < np; ++igp) {
      hkk =
          0.5 * AT_3D(dp, 0, igp, jgp, np, np) / AT_3D(p, 0, igp, jgp, np, np);
      AT_3D(phi, 0, igp, jgp, np, np) =
          AT_2D(phis, igp, jgp, np) + phii[1][igp][jgp] +
          Rgas * AT_3D(T_v, 0, igp, jgp, np, np) * hkk;
    }
  }
}

void preq_omega_ps(const real *const p, const real *const vgrad_p,
                   const real *const divdp, real *const omega_p) {
  real ckk, ckl, term;
  real suml[np][np];
  for (int jgp = 0; jgp < np; ++jgp) {
    for (int igp = 0; igp < np; ++igp) {
      ckk = 0.5 / AT_3D(p, 0, igp, jgp, np, np);
      term = AT_3D(divdp, 0, igp, jgp, np, np);
      AT_3D(omega_p, 0, igp, jgp, np, np) =
          AT_3D(vgrad_p, 0, igp, jgp, np, np) / AT_3D(p, 0, igp, jgp, np, np) -
          ckk * term;
      suml[igp][jgp] = term;
    }

    for (int ilev = 1; ilev < nlev - 1; ++ilev) {
      for (int igp = 0; igp < np; ++igp) {
        ckk = 0.5 / AT_3D(p, ilev, igp, jgp, np, np);
        ckl = 2.0 * ckk;
        term = AT_3D(divdp, ilev, igp, jgp, np, np);
        AT_3D(omega_p, ilev, igp, jgp, np, np) =
            AT_3D(vgrad_p, ilev, igp, jgp, np, np) /
                AT_3D(p, ilev, igp, jgp, np, np) -
            ckl * suml[igp][jgp] - ckk * term;

        suml[igp][jgp] += term;
      }
    }

    for (int igp = 0; igp < np; ++igp) {
      ckk = 0.5 / AT_3D(p, (nlev - 1), igp, jgp, np, np);
      ckl = 2.0 * ckk;
      term = AT_3D(divdp, (nlev - 1), igp, jgp, np, np);
      AT_3D(omega_p, (nlev - 1), igp, jgp, np, np) =
          AT_3D(vgrad_p, (nlev - 1), igp, jgp, np, np) /
              AT_3D(p, (nlev - 1), igp, jgp, np, np) -
          ckl * suml[igp][jgp] - ckk * term;
    }
  }
}

void print_results_2norm(const TestData &data) {
  // Input parameters
  const int nets = data.control.nets;
  const int nete = data.control.nete;
  const int np1 = data.control.np1;

  real *v_np1;
  real *T_np1;
  real *dp3d_np1;

  real vnorm(0.), tnorm(0.), dpnorm(0.);
  for (int ie = nets; ie < nete; ++ie) {
    v_np1 = SLICE_6D_IJ(data.arrays.elem_state_v, ie, np1, timelevels, nlev, np,
                        np, 2);
    T_np1 = SLICE_5D_IJ(data.arrays.elem_state_T, ie, np1, timelevels, nlev, np,
                        np);
    dp3d_np1 = SLICE_5D_IJ(data.arrays.elem_state_dp3d, ie, np1, timelevels,
                           nlev, np, np);

    for (int ilev = 0; ilev < nlev; ++ilev) {
      for (int igp = 0; igp < np; ++igp) {
        for (int jgp = 0; jgp < np; ++jgp) {
          vnorm += std::pow(AT_4D(v_np1, ilev, igp, jgp, 0, np, np, 2), 2);
          vnorm += std::pow(AT_4D(v_np1, ilev, igp, jgp, 1, np, np, 2), 2);
          tnorm += std::pow(AT_3D(T_np1, ilev, igp, jgp, np, np), 2);
          dpnorm += std::pow(AT_3D(dp3d_np1, ilev, igp, jgp, np, np), 2);
        }
      }
    }
  }

  std::cout << "   ---> Norms:\n"
            << "          ||v||_2  = " << std::sqrt(vnorm) << "\n"
            << "          ||T||_2  = " << std::sqrt(tnorm) << "\n"
            << "          ||dp||_2 = " << std::sqrt(dpnorm) << "\n";
}

void dump_results_to_file(const TestData &data) {
  // Input parameters
  const int nets = data.control.nets;
  const int nete = data.control.nete;
  const int np1 = data.control.np1;

  real *v_np1;
  real *T_np1;
  real *dp3d_np1;

  std::ofstream vxfile, vyfile, tfile, dpfile;
  vxfile.open("elem_state_vx.txt");
  if (!vxfile.is_open()) {
    std::cout << "Error! Cannot open 'elem_state_vx.txt'.\n";
    std::abort();
  }

  vyfile.open("elem_state_vy.txt");
  if (!vyfile.is_open()) {
    vxfile.close();
    std::cout << "Error! Cannot open 'elem_state_vy.txt'.\n";
    std::abort();
  }

  tfile.open("elem_state_t.txt");
  if (!tfile.is_open()) {
    std::cout << "Error! Cannot open 'elem_state_t.txt'.\n";
    vxfile.close();
    vyfile.close();
    std::abort();
  }

  dpfile.open("elem_state_dp3d.txt");
  if (!dpfile.is_open()) {
    std::cout << "Error! Cannot open 'elem_state_dp3d.txt'.\n";
    vxfile.close();
    vyfile.close();
    tfile.close();
    std::abort();
  }

  vxfile.precision(6);
  vyfile.precision(6);
  tfile.precision(6);
  dpfile.precision(6);

  for (int ie = nets; ie < nete; ++ie) {
    v_np1 = SLICE_6D_IJ(data.arrays.elem_state_v, ie, np1, timelevels, nlev, np,
                        np, 2);
    T_np1 = SLICE_5D_IJ(data.arrays.elem_state_T, ie, np1, timelevels, nlev, np,
                        np);
    dp3d_np1 = SLICE_5D_IJ(data.arrays.elem_state_dp3d, ie, np1, timelevels,
                           nlev, np, np);

    for (int ilev = 0; ilev < nlev; ++ilev) {
      vxfile << "[" << ie << ", " << ilev << "]\n";
      vyfile << "[" << ie << ", " << ilev << "]\n";
      tfile << "[" << ie << ", " << ilev << "]\n";
      dpfile << "[" << ie << ", " << ilev << "]\n";

      for (int igp = 0; igp < np; ++igp) {
        for (int jgp = 0; jgp < np; ++jgp) {
          vxfile << " " << AT_4D(v_np1, ilev, igp, jgp, 0, np, np, 2);
          vyfile << " " << AT_4D(v_np1, ilev, igp, jgp, 1, np, np, 2);
          tfile << " " << AT_3D(T_np1, ilev, igp, jgp, np, np);
          dpfile << " " << AT_3D(dp3d_np1, ilev, igp, jgp, np, np);
        }
        vxfile << "\n";
        vyfile << "\n";
        tfile << "\n";
        dpfile << "\n";
      }
    }
  }

  // Closing files
  vxfile.close();
  vyfile.close();
  tfile.close();
  dpfile.close();
};

} // Namespace Homme
