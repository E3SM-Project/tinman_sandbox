#include "compute_and_apply_rhs.hpp"
#include "test_macros.hpp"
#include "dimensions.hpp"
#include "data_structures.hpp"
#include "sphere_operators.hpp"

#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

namespace Homme
{

void compute_and_apply_rhs (TestData& data)
{
  // Create local arrays
  real Ephi[np][np]                    = {};
  real T_v[nlev][np][np]               = {};
  real divdp[nlev][np][np]             = {};
  real grad_p[nlev][np][np][2]         = {};
  real eta_dot_dpdn_tmp[nlevp][np][np] = {};
  real kappa_star[nlev][np][np]        = {};
  real omega_p_tmp[nlev][np][np]       = {};
  real p[nlev][np][np]                 = {};
  real ttens[nlev][np][np]             = {};
  real T_vadv[nlev][np][np]            = {};
  real v_vadv[nlev][np][np][2]         = {};
  real vdp[nlev][np][np][2]            = {};
  real vgrad_T[np][np]                 = {};
  real vgrad_p[nlev][np][np]           = {};
  real vort[nlev][np][np]              = {};
  real vtemp[np][np][2]                = {};
  real vtens1[nlev][np][np]            = {};
  real vtens2[nlev][np][np]            = {};

  // Get a pointer version so we can use single
  // subroutines interface for both ptrs and arrays
  real* Ephi_ptr             = PTR_FROM_2D(Ephi);
  real* divdp_ptr            = PTR_FROM_3D(divdp);
  real* eta_dot_dpdn_tmp_ptr = PTR_FROM_3D(eta_dot_dpdn_tmp);
  real* grad_p_ptr           = PTR_FROM_4D(grad_p);
  real* p_ptr                = PTR_FROM_3D(p);
  real* vdp_ptr              = PTR_FROM_4D(vdp);
  real* vgrad_p_ptr          = PTR_FROM_3D(vgrad_p);
  real* vort_ptr             = PTR_FROM_3D(vort);
  real* vtemp_ptr            = PTR_FROM_3D(vtemp);
  real* omega_p_tmp_ptr      = PTR_FROM_3D(omega_p_tmp);
  real* T_v_ptr              = PTR_FROM_3D(T_v);

  // Other accessory variables
  real Qt     = 0;
  real glnps1 = 0;
  real glnps2 = 0;
  real gpterm = 0;
  real v1     = 0;
  real v2     = 0;

  real* Qdp_ie            = nullptr;
  real* T_n0              = nullptr;
  real* T_nm1             = nullptr;
  real* T_np1             = nullptr;
  real* derived_vn0       = nullptr;
  real* dp3d_n0           = nullptr;
  real* dp3d_nm1          = nullptr;
  real* dp3d_np1          = nullptr;
  real* fcor              = nullptr;
  real* omega_p           = nullptr;
  real* pecnd             = nullptr;
  real* phi               = nullptr;
  real* phis              = nullptr;
  real* spheremp          = nullptr;
  real* v_n0              = nullptr;
  real* v_nm1             = nullptr;
  real* v_np1             = nullptr;
  real* eta_dot_dpdn      = nullptr;

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
        p[0][igp][jgp] = data.hvcoord.hyai[0]*data.hvcoord.ps0 + 0.5*AT_3D(dp3d_n0,0,igp,jgp,np,np);
      }
    }

    for (int ilev=1; ilev<nlev; ++ilev)
    {
      for (int igp=0; igp<np; ++igp)
      {
        for (int jgp=0; jgp<np; ++jgp)
        {
          p[ilev][igp][jgp] = p[ilev-1][igp][jgp]
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

      for (int igp=0; igp<np; ++igp)
      {
        for (int jgp=0; jgp<np; ++jgp)
        {
          v1 = AT_4D(v_n0,ilev,igp,jgp,0,np,np,2);
          v2 = AT_4D(v_n0,ilev,igp,jgp,1,np,np,2);
          vgrad_p[ilev][igp][jgp] = v1 * grad_p[ilev][igp][jgp][0] + v2 * grad_p[ilev][igp][jgp][1];

          vdp[ilev][igp][jgp][0] = v1 * AT_3D(dp3d_n0,ilev,igp,jgp,np,np);
          vdp[ilev][igp][jgp][1] = v2 * AT_3D(dp3d_n0,ilev,igp,jgp,np,np);

          AT_4D(derived_vn0,ilev,igp,jgp,0,np,np,2) += data.constants.eta_ave_w * vdp[ilev][igp][jgp][0];
          AT_4D(derived_vn0,ilev,igp,jgp,1,np,np,2) += data.constants.eta_ave_w * vdp[ilev][igp][jgp][1];
        }
      }

      divergence_sphere(SLICE_4D(vdp_ptr,ilev,np,np,2), data, ie, SLICE_3D (divdp_ptr,ilev,np,np));
      vorticity_sphere(SLICE_4D(v_n0,ilev,np,np,2), data, ie, SLICE_3D (vort_ptr,ilev,np,np));
    }

    T_n0 = SLICE_5D_IJ(data.arrays.elem_state_T,ie,n0,timelevels,nlev,np,np);
    if (qn0==-1)
    {
      for (int ilev=0; ilev<nlev; ++ilev)
      {
        for (int igp=0; igp<np; ++igp)
        {
          for (int jgp=0; jgp<np; ++jgp)
          {
            T_v[ilev][igp][jgp] = AT_3D(T_n0,ilev,igp,jgp,np,np);
            kappa_star[ilev][igp][jgp] = data.constants.kappa;
          }
        }
      }
    }
    else
    {
      Qdp_ie = SLICE_6D_IJK (data.arrays.elem_state_Qdp,ie,0,qn0,qsize_d,2,nlev,np,np);
      for (int ilev=0; ilev<nlev; ++ilev)
      {
        for (int igp=0; igp<np; ++igp)
        {
          for (int jgp=0; jgp<np; ++jgp)
          {
            Qt = AT_3D(Qdp_ie,ilev,igp,jgp,np,np) / AT_3D(dp3d_n0,ilev,igp,jgp,np,np);
            T_v[ilev][igp][jgp] = AT_3D(T_n0,ilev,igp,jgp,np,np)*(1.0+ (data.constants.Rwater_vapor/data.constants.Rgas - 1.0)*Qt);
            kappa_star[ilev][igp][jgp] = data.constants.kappa;
          }
        }
      }
    }

    phis = SLICE_3D(data.arrays.elem_state_phis,ie,np,np);
    phi  = SLICE_4D(data.arrays.elem_derived_phi,ie,nlev,np,np);

    preq_hydrostatic (phis,T_v_ptr,p_ptr,dp3d_n0,data.constants.Rgas,phi);
    preq_omega_ps (p_ptr,vgrad_p_ptr,divdp_ptr,omega_p_tmp_ptr);

    omega_p      = SLICE_4D(data.arrays.elem_derived_omega_p,ie,nlev,np,np);
    eta_dot_dpdn = SLICE_4D(data.arrays.elem_derived_eta_dot_dpdn,ie,nlevp,np,np);
    for (int ilev=0; ilev<nlev; ++ilev)
    {
      for (int igp=0; igp<np; ++igp)
      {
        for (int jgp=0; jgp<np; ++jgp)
        {
          AT_3D(eta_dot_dpdn,ilev,igp,jgp,np,np) += data.constants.eta_ave_w * eta_dot_dpdn_tmp[ilev][igp][jgp];
          AT_3D(omega_p,ilev,igp,jgp,np,np) += data.constants.eta_ave_w * omega_p_tmp[ilev][igp][jgp];
        }
      }
    }
    for (int igp=0; igp<np; ++igp)
    {
      for (int jgp=0; jgp<np; ++jgp)
      {
        AT_3D(eta_dot_dpdn,nlev,igp,jgp,np,np) += data.constants.eta_ave_w * eta_dot_dpdn_tmp[nlev][igp][jgp];
      }
    }

    pecnd = SLICE_4D(data.arrays.elem_derived_pecnd,ie,nlev,np,np);
    fcor  = SLICE_3D(data.arrays.elem_fcor,ie,np,np);
    for (int ilev=0; ilev<nlev; ++ilev)
    {
      for (int igp=0; igp<np; ++igp)
      {
        for (int jgp=0; jgp<np; ++jgp)
        {
          v1 = AT_4D(v_n0,ilev,igp,jgp,0,np,np,2);
          v2 = AT_4D(v_n0,ilev,igp,jgp,1,np,np,2);

          Ephi[igp][jgp] = 0.5 * (v1*v1 + v2*v2) + AT_3D(phi,ilev,igp,jgp,np,np) + AT_3D (pecnd,ilev,igp,jgp,np,np);
        }
      }

      gradient_sphere (SLICE_3D(T_n0,ilev,np,np),data,ie,vtemp_ptr);

      for (int igp=0; igp<np; ++igp)
      {
        for (int jgp=0; jgp<np; ++jgp)
        {
          v1 = AT_4D(v_n0,ilev,igp,jgp,0,np,np,2);
          v2 = AT_4D(v_n0,ilev,igp,jgp,1,np,np,2);

          vgrad_T[igp][jgp] = v1*vtemp[igp][jgp][0] + v2*vtemp[igp][jgp][1];
        }
      }

      gradient_sphere (Ephi_ptr, data, ie, vtemp_ptr);

      for (int igp=0; igp<np; ++igp)
      {
        for (int jgp=0; jgp<np; ++jgp)
        {
          gpterm = T_v[ilev][igp][jgp] / p[ilev][igp][jgp];

          glnps1 = data.constants.Rgas*gpterm*grad_p[ilev][igp][jgp][0];
          glnps2 = data.constants.Rgas*gpterm*grad_p[ilev][igp][jgp][1];

          v1 = AT_4D(v_n0,ilev,igp,jgp,0,np,np,2);
          v2 = AT_4D(v_n0,ilev,igp,jgp,1,np,np,2);

          vtens1[ilev][igp][jgp] = v_vadv[ilev][igp][jgp][0] + v2 * (AT_2D(fcor,igp,jgp,np) + vort[ilev][igp][jgp]) - vtemp[igp][jgp][0] - glnps1;
          vtens2[ilev][igp][jgp] = v_vadv[ilev][igp][jgp][1] - v1 * (AT_2D(fcor,igp,jgp,np) + vort[ilev][igp][jgp]) - vtemp[igp][jgp][1] - glnps2;

          ttens[ilev][igp][jgp]  = T_vadv[ilev][igp][jgp] - vgrad_T[igp][jgp] + kappa_star[ilev][igp][jgp]*T_v[ilev][igp][jgp]*omega_p_tmp[ilev][igp][jgp];
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

    for (int ilev=0; ilev<nlev; ++ilev)
    {
      for (int igp=0; igp<np; ++igp)
      {
        for (int jgp=0; jgp<np; ++jgp)
        {
          AT_4D(v_np1,ilev,igp,jgp,0,np,np,2) = AT_2D(spheremp,igp,jgp,np) * (AT_4D(v_nm1,ilev,igp,jgp,0,np,np,2) + dt2*vtens1[ilev][igp][jgp]);
          AT_4D(v_np1,ilev,igp,jgp,1,np,np,2) = AT_2D(spheremp,igp,jgp,np) * (AT_4D(v_nm1,ilev,igp,jgp,1,np,np,2) + dt2*vtens1[ilev][igp][jgp]);
          AT_3D(T_np1,ilev,igp,jgp,np,np)     = AT_2D(spheremp,igp,jgp,np) * (AT_3D(T_nm1,ilev,igp,jgp,np,np) + dt2*ttens[ilev][igp][jgp]);
          AT_3D(dp3d_np1,ilev,igp,jgp,np,np)  = AT_2D(spheremp,igp,jgp,np) * (AT_3D(dp3d_nm1,ilev,igp,jgp,np,np) + dt2*divdp[ilev][igp][jgp]);
        }
      }
    }
  }
}

void preq_hydrostatic (const real* const phis, const real* const T_v,
                       const real* const p, const real* dp,
                       real Rgas, real* const phi)
{
  real hkk, hkl;
  real phii[nlev][np][np];

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

void preq_omega_ps (const real* const p, const real* const vgrad_p,
                    const real* const divdp, real* const omega_p)
{
  real ckk, ckl, term;
  real suml[np][np];
  for (int jgp=0; jgp<np; ++jgp)
  {
    for (int igp=0; igp<np; ++igp)
    {
      ckk = 0.5 / AT_3D(p,0,igp,jgp,np,np);
      term  = AT_3D(divdp,0,igp,jgp,np,np);
      AT_3D (omega_p,0,igp,jgp,np,np) = AT_3D (vgrad_p,0,igp,jgp,np,np)/AT_3D(p,0,igp,jgp,np,np) - ckk*term;
      suml[igp][jgp] = term;
    }

    for (int ilev=1; ilev<nlev-1; ++ilev)
    {
      for (int igp=0; igp<np; ++igp)
      {
        ckk = 0.5 / AT_3D(p,ilev,igp,jgp,np,np);
        ckl = 2.0 * ckk;
        term  = AT_3D(divdp,ilev,igp,jgp,np,np);
        AT_3D (omega_p,ilev,igp,jgp,np,np) = AT_3D (vgrad_p,ilev,igp,jgp,np,np)/AT_3D(p,ilev,igp,jgp,np,np)
                                           - ckl*suml[igp][jgp] - ckk*term;

        suml[igp][jgp] += term;
      }
    }

    for (int igp=0; igp<np; ++igp)
    {
      ckk = 0.5 / AT_3D(p,(nlev-1),igp,jgp,np,np);
      ckl = 2.0 * ckk;
      term  = AT_3D(divdp,(nlev-1),igp,jgp,np,np);
      AT_3D (omega_p,(nlev-1),igp,jgp,np,np) = AT_3D (vgrad_p,(nlev-1),igp,jgp,np,np)/AT_3D(p,(nlev-1),igp,jgp,np,np)
                                             - ckl*suml[igp][jgp] - ckk*term;
    }
  }
}

real compute_norm (const real* const field, int length)
{
  // Note: use Kahan algorithm to maintain accuracy
  real norm = 0;
  real temp;
  real c = 0;
  real y = 0;
  for (int i=0; i<length; ++i)
  {
    y = field[i]*field[i] - c;
    temp = norm + y;
    c = (temp - norm) - y;
    norm = temp;
  }

  return std::sqrt(norm);
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

    vnorm  += std::pow( compute_norm(v_np1,nlev*np*np*2), 2 );
    tnorm  += std::pow( compute_norm(T_np1,nlev*np*np), 2 );
    dpnorm += std::pow( compute_norm(dp3d_np1,nlev*np*np), 2 );
  }

  std::cout << "   ---> Norms:\n"
            << "          ||v||_2  = " << std::setprecision(17) << std::sqrt (vnorm) << "\n"
            << "          ||T||_2  = " << std::setprecision(17) << std::sqrt (tnorm) << "\n"
            << "          ||dp||_2 = " << std::setprecision(17) << std::sqrt (dpnorm) << "\n";
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
