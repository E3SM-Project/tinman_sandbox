#include "compute_and_apply_rhs.hpp"
#include "test_macros.hpp"
#include "dimensions.hpp"
#include "data_structures.hpp"
#include "sphere_operators.hpp"

#include <cmath>
#include <iostream>
#include <fstream>

namespace Homme
{

void compute_and_apply_rhs (TestData& data)
{
  // Create local arrays
  // Those without restrict shouldn't be accessed without their pointers
  real Ephi[np][np]                      = {};
  real T_v[nlev][np][np]                 = {};
  real divdp[nlev][np][np]               = {};
  real grad_p[nlev][np][np][2]           = {};
  real eta_dot_dpdn_tmp[nlevp][np][np]   = {};
  real omega_p_tmp[nlev][np][np]         = {};
  real p[nlev][np][np]                   = {};
  real vdp[nlev][np][np][2]              = {};
  real vgrad_p[nlev][np][np]             = {};
  real vort[nlev][np][np]                = {};
  real vtemp[np][np][2]                  = {};
  RESTRICT real kappa_star[nlev][np][np] = {};
  RESTRICT real ttens[nlev][np][np]      = {};
  RESTRICT real T_vadv[nlev][np][np]     = {};
  RESTRICT real v_vadv[nlev][np][np][2]  = {};
  RESTRICT real vgrad_T[np][np]          = {};
  RESTRICT real vtens1[nlev][np][np]     = {};
  RESTRICT real vtens2[nlev][np][np]     = {};

  // Get a pointer version so we can use single
  // subroutines interface for both ptrs and arrays
  RESTRICT real* Ephi_ptr             = PTR_FROM_2D(Ephi);
  RESTRICT real* divdp_ptr            = PTR_FROM_3D(divdp);
  RESTRICT real* eta_dot_dpdn_tmp_ptr = PTR_FROM_3D(eta_dot_dpdn_tmp);
  RESTRICT real* grad_p_ptr           = PTR_FROM_4D(grad_p);
  RESTRICT real* p_ptr                = PTR_FROM_3D(p);
  RESTRICT real* vdp_ptr              = PTR_FROM_4D(vdp);
  RESTRICT real* vgrad_p_ptr          = PTR_FROM_3D(vgrad_p);
  RESTRICT real* vort_ptr             = PTR_FROM_3D(vort);
  RESTRICT real* vtemp_ptr            = PTR_FROM_3D(vtemp);
  RESTRICT real* omega_p_tmp_ptr      = PTR_FROM_3D(omega_p_tmp);
  RESTRICT real* T_v_ptr              = PTR_FROM_3D(T_v);

  // Other accessory variables
  real Qt     = 0;
  real glnps1 = 0;
  real glnps2 = 0;
  real gpterm = 0;
  real v1     = 0;
  real v2     = 0;

  RESTRICT real* Qdp_ie            = nullptr;
  RESTRICT real* T_n0              = nullptr;
  RESTRICT real* T_nm1             = nullptr;
  RESTRICT real* T_np1             = nullptr;
  RESTRICT real* derived_vn0       = nullptr;
  RESTRICT real* dp3d_n0           = nullptr;
  RESTRICT real* dp3d_nm1          = nullptr;
  RESTRICT real* dp3d_np1          = nullptr;
  RESTRICT real* fcor              = nullptr;
  RESTRICT real* omega_p           = nullptr;
  RESTRICT real* pecnd             = nullptr;
  RESTRICT real* phi               = nullptr;
  RESTRICT real* phis              = nullptr;
  RESTRICT real* spheremp          = nullptr;
  RESTRICT real* v_n0              = nullptr;
  RESTRICT real* v_nm1             = nullptr;
  RESTRICT real* v_np1             = nullptr;
  RESTRICT real* eta_dot_dpdn      = nullptr;

  // Input parameters
  const int nets = data.control.nets;
  const int nete = data.control.nete;
  const int n0   = data.control.n0;
  const int np1  = data.control.np1;
  const int nm1  = data.control.nm1;
  const int qn0  = data.control.qn0;
  const real dt2 = data.control.dt2;

  // Loop over elements
  for (int ie=nets; ie<nete; ++ie)
  {


    dp3d_n0 = SLICE_5D_IJ(data.arrays.elem_state_dp3d,ie,n0,timelevels,nlev,np,np);

    for (int igp=0; igp<np; ++igp)
    {
      for (int jgp=0; jgp<np; ++jgp)
      {
        AT_3D(p_ptr, np, np, 0, igp, jgp) = data.hvcoord.hyai[0]*data.hvcoord.ps0 + 0.5*AT_3D(dp3d_n0,0,igp,jgp,np,np);
      }
    }

    SIMD
    for (int ilev=1; ilev<nlev; ++ilev)
    {
      for (int igp=0; igp<np; ++igp)
      {
        for (int jgp=0; jgp<np; ++jgp)
        {
          AT_3D(p_ptr, ilev, igp, jgp, np, np) = AT_3D(p_ptr, ilev-1, igp, jgp, np, np)
                            + 0.5*AT_3D(dp3d_n0,(ilev-1),igp,jgp,np,np)
                            + 0.5*AT_3D(dp3d_n0,ilev,igp,jgp,np,np);
        }
      }
    }

    derived_vn0 = SLICE_5D(data.arrays.elem_derived_vn0,ie,nlev,np,np,2);
    v_n0 = SLICE_6D_IJ(data.arrays.elem_state_v,ie,n0,timelevels,nlev,np,np,2);
    for (int ilev=0; ilev<nlev; ++ilev)
    {
      gradient_sphere (SLICE_3D(p_ptr,ilev,np,np), data, ie, SLICE_4D(grad_p_ptr,ilev,np,np,2));

      SIMD
      for (int igp=0; igp<np; ++igp)
      {
        for (int jgp=0; jgp<np; ++jgp)
        {
          AT_3D(vgrad_p_ptr, ilev, igp, jgp, np, np) = AT_4D(v_n0,ilev,igp,jgp,0,np,np,2) * AT_4D(grad_p_ptr, ilev, igp, jgp, 0, np, np, 2) + AT_4D(v_n0,ilev,igp,jgp,1,np,np,2) * AT_4D(grad_p_ptr, ilev, igp, jgp, 1, np, np, 2);

          AT_4D(vdp_ptr, ilev, igp, jgp, 0, np, np, 2) = AT_4D(v_n0,ilev,igp,jgp,0,np,np,2) * AT_3D(dp3d_n0,ilev,igp,jgp,np,np);
          AT_4D(vdp_ptr, ilev, igp, jgp, 1, np, np, 2) = AT_4D(v_n0,ilev,igp,jgp,1,np,np,2) * AT_3D(dp3d_n0,ilev,igp,jgp,np,np);

          AT_4D(derived_vn0,ilev,igp,jgp,0,np,np,2) += Constants::eta_ave_w * AT_4D(vdp_ptr, ilev, igp, jgp, 0, np, np, 2);
          AT_4D(derived_vn0,ilev,igp,jgp,1,np,np,2) += Constants::eta_ave_w * AT_4D(vdp_ptr, ilev, igp, jgp, 1, np, np, 2);
        }
      }

      divergence_sphere(SLICE_4D(vdp_ptr,ilev,np,np,2), data, ie, SLICE_3D (divdp_ptr,ilev,np,np));
      vorticity_sphere(SLICE_4D(v_n0,ilev,np,np,2), data, ie, SLICE_3D (vort_ptr,ilev,np,np));
    }

    T_n0 = SLICE_5D_IJ(data.arrays.elem_state_T,ie,n0,timelevels,nlev,np,np);
    if (qn0==-1)
    {
      SIMD
      for (int ilev=0; ilev<nlev; ++ilev)
      {
        for (int igp=0; igp<np; ++igp)
        {
          for (int jgp=0; jgp<np; ++jgp)
          {
            AT_3D(T_v_ptr, ilev, igp, jgp, np, np) = AT_3D(T_n0,ilev,igp,jgp,np,np);
            kappa_star[ilev][igp][jgp] = Constants::kappa;
          }
        }
      }
    }
    else
    {
      Qdp_ie = SLICE_6D (data.arrays.elem_state_Qdp,ie,nlev,qsize_d,2,np,np);
      SIMD
      for (int ilev=0; ilev<nlev; ++ilev)
      {
        for (int igp=0; igp<np; ++igp)
        {
          for (int jgp=0; jgp<np; ++jgp)
          {
            Qt = AT_5D(Qdp_ie,ilev,1,qn0,igp,jgp,qsize_d,2,np,np) / AT_3D(dp3d_n0,ilev,igp,jgp,np,np);
            AT_3D(T_v_ptr, ilev, igp, jgp, np, np) = AT_3D(T_n0,ilev,igp,jgp,np,np)*(real(1.0)+ (Constants::Rwater_vapor/Constants::Rgas - real(1.0))*Qt);
            kappa_star[ilev][igp][jgp] = Constants::kappa;
          }
        }
      }
    }

    phis = SLICE_3D(data.arrays.elem_state_phis,ie,np,np);
    phi  = SLICE_4D(data.arrays.elem_derived_phi,ie,nlev,np,np);

    preq_hydrostatic (phis,T_v_ptr,p_ptr,dp3d_n0,Constants::Rgas,phi);
    preq_omega_ps (p_ptr,vgrad_p_ptr,divdp_ptr,omega_p_tmp_ptr);

    omega_p      = SLICE_4D(data.arrays.elem_derived_omega_p,ie,nlev,np,np);
    eta_dot_dpdn = SLICE_4D(data.arrays.elem_derived_eta_dot_dpdn,ie,nlevp,np,np);

    SIMD
    for (int ilev=0; ilev<nlev; ++ilev)
    {
      for (int igp=0; igp<np; ++igp)
      {
        for (int jgp=0; jgp<np; ++jgp)
        {
          AT_3D(eta_dot_dpdn,ilev,igp,jgp,np,np) += Constants::eta_ave_w * AT_3D(eta_dot_dpdn_tmp_ptr, ilev, igp, jgp, np, np);
        }
      }
    }

    SIMD
    for (int ilev=0; ilev<nlev; ++ilev)
    {
      for (int igp=0; igp<np; ++igp)
      {
        for (int jgp=0; jgp<np; ++jgp)
        {
          AT_3D(omega_p,ilev,igp,jgp,np,np) += Constants::eta_ave_w * AT_3D(omega_p_tmp_ptr, ilev, igp, jgp, np, np);
        }
      }
    }

    SIMD
    for (int igp=0; igp<np; ++igp)
    {
      for (int jgp=0; jgp<np; ++jgp)
      {
        AT_3D(eta_dot_dpdn,nlev,igp,jgp,np,np) += Constants::eta_ave_w * AT_3D(eta_dot_dpdn_tmp_ptr, nlev, igp, jgp, np, np);
      }
    }

    pecnd = SLICE_4D(data.arrays.elem_derived_pecnd,ie,nlev,np,np);
    fcor  = SLICE_3D(data.arrays.elem_fcor,ie,np,np);
    for (int ilev=0; ilev<nlev; ++ilev)
    {
      SIMD
      for (int igp=0; igp<np; ++igp)
      {
        for (int jgp=0; jgp<np; ++jgp)
        {
          v1 = AT_4D(v_n0,ilev,igp,jgp,0,np,np,2);
          v2 = AT_4D(v_n0,ilev,igp,jgp,1,np,np,2);

          AT_2D(Ephi_ptr, igp, jgp, np) = 0.5 * (v1*v1 + v2*v2) + AT_3D(phi,ilev,igp,jgp,np,np) + AT_3D(pecnd,ilev,igp,jgp,np,np);
        }
      }

      gradient_sphere (SLICE_3D(T_n0,ilev,np,np),data,ie,vtemp_ptr);

      SIMD
      for (int igp=0; igp<np; ++igp)
      {
        for (int jgp=0; jgp<np; ++jgp)
        {
          v1 = AT_4D(v_n0,ilev,igp,jgp,0,np,np,2);
          v2 = AT_4D(v_n0,ilev,igp,jgp,1,np,np,2);

          vgrad_T[igp][jgp] = v1*AT_3D(vtemp_ptr, igp, jgp, 0, np, 2) + v2*AT_3D(vtemp_ptr, igp, jgp, 1, np, 2);
        }
      }

      gradient_sphere (Ephi_ptr, data, ie, vtemp_ptr);

      SIMD
      for (int igp=0; igp<np; ++igp)
      {
        for (int jgp=0; jgp<np; ++jgp)
        {
          gpterm = AT_3D(T_v_ptr, ilev, igp, jgp, np, np) / AT_3D(p_ptr, ilev, igp, jgp, np, np);

          glnps1 = Constants::Rgas*gpterm*AT_4D(grad_p_ptr, ilev, igp, jgp, 0, np, np, 2);
          glnps2 = Constants::Rgas*gpterm*AT_4D(grad_p_ptr, ilev, igp, jgp, 1, np, np, 2);

          v1 = AT_4D(v_n0,ilev,igp,jgp,0,np,np,2);
          v2 = AT_4D(v_n0,ilev,igp,jgp,1,np,np,2);

          vtens1[ilev][igp][jgp] = v_vadv[ilev][igp][jgp][0] + v2 * (AT_2D(fcor,igp,jgp,np) + AT_3D(vort_ptr, ilev, igp, jgp, np, np)) - AT_3D(vtemp_ptr, igp, jgp, 0, np, 2) - glnps1;
          vtens2[ilev][igp][jgp] = v_vadv[ilev][igp][jgp][1] - v1 * (AT_2D(fcor,igp,jgp,np) + AT_3D(vort_ptr, ilev, igp, jgp, np, np)) - AT_3D(vtemp_ptr, igp, jgp, 1, np, 2) - glnps2;

          ttens[ilev][igp][jgp]  = T_vadv[ilev][igp][jgp] - vgrad_T[igp][jgp] + kappa_star[ilev][igp][jgp]*AT_3D(T_v_ptr, ilev, igp, jgp, np, np)*AT_3D(omega_p_tmp_ptr, ilev, igp, jgp, np, np);
        }
      }
    }

    spheremp = SLICE_3D(data.arrays.elem_spheremp,ie,np,np);
    v_np1    = SLICE_6D_IJ(data.arrays.elem_state_v,ie,np1,timelevels,nlev,np,np,2);
    T_np1    = SLICE_5D_IJ(data.arrays.elem_state_T,ie,np1,timelevels,nlev,np,np);
    dp3d_np1 = SLICE_5D_IJ(data.arrays.elem_state_dp3d,ie,np1,timelevels,nlev,np,np);

    v_nm1    = SLICE_6D_IJ(data.arrays.elem_state_v,ie,nm1,timelevels,nlev,np,np,2);
    T_nm1    = SLICE_5D_IJ(data.arrays.elem_state_T,ie,nm1,timelevels,nlev,np,np);
    dp3d_nm1 = SLICE_5D_IJ(data.arrays.elem_state_dp3d,ie,nm1,timelevels,nlev,np,np);

    SIMD
    for (int ilev=0; ilev<nlev; ++ilev)
    {
      for (int igp=0; igp<np; ++igp)
      {
        for (int jgp=0; jgp<np; ++jgp)
        {
          for(int k=0; k<2; ++k)
          {
            AT_4D(v_np1,ilev,igp,jgp,k,np,np,2) = AT_2D(spheremp,igp,jgp,np) * (AT_4D(v_nm1,ilev,igp,jgp,k,np,np,2) + dt2*vtens1[ilev][igp][jgp]);
          }
        }
      }
    }
    SIMD
    for (int ilev=0; ilev<nlev; ++ilev)
    {
      for (int igp=0; igp<np; ++igp)
      {
        for (int jgp=0; jgp<np; ++jgp)
        {
          AT_3D(T_np1,ilev,igp,jgp,np,np)     = AT_2D(spheremp,igp,jgp,np) * (AT_3D(T_nm1,ilev,igp,jgp,np,np) + dt2*ttens[ilev][igp][jgp]);
          AT_3D(dp3d_np1,ilev,igp,jgp,np,np)  = AT_2D(spheremp,igp,jgp,np) * (AT_3D(dp3d_nm1,ilev,igp,jgp,np,np) + dt2*AT_3D(divdp_ptr, ilev, igp, jgp, np, np));
        }
      }
    }
  }
}

void preq_hydrostatic (RESTRICT const real* const phis, RESTRICT const real* const T_v,
                       RESTRICT const real* const p, RESTRICT const real* dp,
                       real Rgas, RESTRICT real* const phi)
{
  real hkk, hkl;
  real phii[nlev][np][np];

  SIMD
  for (int jgp=0; jgp<np; ++jgp)
  {
    for (int igp=0; igp<np; ++igp)
    {
      hkk = 0.5*AT_3D(dp,(nlev-1),igp,jgp,np,np) / AT_3D(p,(nlev-1),igp,jgp,np,np);
      hkl = 2.0*hkk;
      phii[nlev-1][igp][jgp] = Rgas*AT_3D(T_v, (nlev-1),igp,jgp,np,np)*hkl;
      AT_3D(phi,(nlev-1),igp,jgp,np,np) = AT_2D(phis,igp,jgp,np) + Rgas*AT_3D(T_v, (nlev-1),igp,jgp,np,np)*hkk;
    }
    for (int ilev=nlev-2; ilev>0; --ilev)
    {
      for (int igp=0; igp<np; ++igp)
      {
        hkk = 0.5*AT_3D(dp,ilev,igp,jgp,np,np) / AT_3D(p,ilev,igp,jgp,np,np);
        hkl = 2.0*hkk;
        phii[ilev][igp][jgp] = phii[ilev+1][igp][jgp] + Rgas*AT_3D(T_v, ilev,igp,jgp,np,np)*hkl;
        AT_3D(phi,ilev,igp,jgp,np,np) = AT_2D(phis,igp,jgp,np) + phii[ilev+1][igp][jgp] + Rgas*AT_3D(T_v, ilev,igp,jgp,np,np)*hkk;
      }
    }
    for (int igp=0; igp<np; ++igp)
    {
      hkk = 0.5*AT_3D(dp,0,igp,jgp,np,np) / AT_3D(p,0,igp,jgp,np,np);
      AT_3D(phi,0,igp,jgp,np,np) = AT_2D(phis,igp,jgp,np) + phii[1][igp][jgp] + Rgas*AT_3D(T_v,0,igp,jgp,np,np)*hkk;
    }
  }
}

void preq_omega_ps (RESTRICT const real* const p, RESTRICT const real* const vgrad_p,
                    RESTRICT const real* const divdp, RESTRICT real* const omega_p)
{
  real ckk, ckl, term;
  real suml[np][np];
  SIMD
  for (int igp=0; igp<np; ++igp)
  {
    for (int jgp=0; jgp<np; ++jgp)
    {
      ckk = 0.5 / AT_3D(p,0,igp,jgp,np,np);
      term  = AT_3D(divdp,0,igp,jgp,np,np);
      AT_3D (omega_p,0,igp,jgp,np,np) = AT_3D (vgrad_p,0,igp,jgp,np,np)/AT_3D(p,0,igp,jgp,np,np) - ckk*term;
      suml[igp][jgp] = term;
    }

    for (int ilev=1; ilev<nlev-1; ++ilev)
    {
      for (int jgp=0; jgp<np; ++jgp)
      {
        ckk = 0.5 / AT_3D(p,ilev,igp,jgp,np,np);
        ckl = 2.0 * ckk;
        term  = AT_3D(divdp,ilev,igp,jgp,np,np);
        AT_3D (omega_p,ilev,igp,jgp,np,np) = AT_3D (vgrad_p,ilev,igp,jgp,np,np)/AT_3D(p,ilev,igp,jgp,np,np)
                                           - ckl*suml[igp][jgp] - ckk*term;

        suml[igp][jgp] += term;
      }
    }

    for (int jgp=0; jgp<np; ++jgp)
    {
      ckk = 0.5 / AT_3D(p,(nlev-1),igp,jgp,np,np);
      ckl = 2.0 * ckk;
      term  = AT_3D(divdp,(nlev-1),igp,jgp,np,np);
      AT_3D (omega_p,(nlev-1),igp,jgp,np,np) = AT_3D (vgrad_p,(nlev-1),igp,jgp,np,np)/AT_3D(p,(nlev-1),igp,jgp,np,np)
                                             - ckl*suml[igp][jgp] - ckk*term;
    }
  }
}

void print_results_2norm (const TestData& data)
{
  // Input parameters
  const int nets = data.control.nets;
  const int nete = data.control.nete;
  const int np1  = data.control.np1;

  real* v_np1;
  real* T_np1;
  real* dp3d_np1;

  real vnorm(0.), tnorm(0.), dpnorm(0.);
  for (int ie=nets; ie<nete; ++ie)
  {
    v_np1    = SLICE_6D_IJ(data.arrays.elem_state_v,ie,np1,timelevels,nlev,np,np,2);
    T_np1    = SLICE_5D_IJ(data.arrays.elem_state_T,ie,np1,timelevels,nlev,np,np);
    dp3d_np1 = SLICE_5D_IJ(data.arrays.elem_state_dp3d,ie,np1,timelevels,nlev,np,np);

    for (int ilev=0; ilev<nlev; ++ilev)
    {
      for (int igp=0; igp<np; ++igp)
      {
        for (int jgp=0; jgp<np; ++jgp)
        {
          vnorm  += std::pow( AT_4D(v_np1,ilev,igp,jgp,0,np,np,2), 2 );
          vnorm  += std::pow( AT_4D(v_np1,ilev,igp,jgp,1,np,np,2), 2 );
          tnorm  += std::pow( AT_3D(T_np1,ilev,igp,jgp,np,np), 2 );
          dpnorm += std::pow( AT_3D(dp3d_np1,ilev,igp,jgp,np,np), 2 );
        }
      }
    }
  }

  std::cout << "   ---> Norms:\n"
            << "          ||v||_2  = " << std::sqrt (vnorm) << "\n"
            << "          ||T||_2  = " << std::sqrt (tnorm) << "\n"
            << "          ||dp||_2 = " << std::sqrt (dpnorm) << "\n";
}

void dump_results_to_file (const TestData& data)
{
  // Input parameters
  const int nets = data.control.nets;
  const int nete = data.control.nete;
  const int np1  = data.control.np1;

  real* v_np1;
  real* T_np1;
  real* dp3d_np1;

  std::ofstream vxfile, vyfile, tfile, dpfile;
  vxfile.open("elem_state_vx.txt");
  if (!vxfile.is_open())
  {
    std::cout << "Error! Cannot open 'elem_state_vx.txt'.\n";
    std::abort();
  }

  vyfile.open("elem_state_vy.txt");
  if (!vyfile.is_open())
  {
    vxfile.close();
    std::cout << "Error! Cannot open 'elem_state_vy.txt'.\n";
    std::abort();
  }

  tfile.open("elem_state_t.txt");
  if (!tfile.is_open())
  {
    std::cout << "Error! Cannot open 'elem_state_t.txt'.\n";
    vxfile.close();
    vyfile.close();
    std::abort();
  }

  dpfile.open("elem_state_dp3d.txt");
  if (!dpfile.is_open())
  {
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

  for (int ie=nets; ie<nete; ++ie)
  {
    v_np1    = SLICE_6D_IJ(data.arrays.elem_state_v,ie,np1,timelevels,nlev,np,np,2);
    T_np1    = SLICE_5D_IJ(data.arrays.elem_state_T,ie,np1,timelevels,nlev,np,np);
    dp3d_np1 = SLICE_5D_IJ(data.arrays.elem_state_dp3d,ie,np1,timelevels,nlev,np,np);

    for (int ilev=0; ilev<nlev; ++ilev)
    {
      vxfile << "[" << ie << ", " << ilev << "]\n";
      vyfile << "[" << ie << ", " << ilev << "]\n";
      tfile  << "[" << ie << ", " << ilev << "]\n";
      dpfile << "[" << ie << ", " << ilev << "]\n";

      for (int igp=0; igp<np; ++igp)
      {
        for (int jgp=0; jgp<np; ++jgp)
        {
          vxfile << " " << AT_4D(v_np1,ilev,igp,jgp,0,np,np,2);
          vyfile << " " << AT_4D(v_np1,ilev,igp,jgp,1,np,np,2);
          tfile  << " " << AT_3D(T_np1,ilev,igp,jgp,np,np);
          dpfile << " " << AT_3D(dp3d_np1,ilev,igp,jgp,np,np);
        }
        vxfile << "\n";
        vyfile << "\n";
        tfile  << "\n";
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
