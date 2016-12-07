#include "test_macros.hpp"

#include "dimensions.hpp"
#include "data_structures.hpp"
#include "sphere_operators.hpp"

namespace Homme
{

int compute_and_apply_rhs (TestData& data)
{
  // Create local arrays
  real grad_p[nlev][2][np][np];
  real p[nlev][np][np];
  real vdp[nlev][np][np][2];
  real vgrad_p[nlev][np][np];
  real divdp[nlev][np][np];
  real vort[nlev][np][np];

  // Get a pointer version so we can use single subroutines interface
  real* p_ptr = PTR_FROM_3D(p);
  real* grad_p_ptr = PTR_FROM_4D(grad_p);
  real* vdp_ptr = PTR_FROM_4D(vdp);
  real* vgrad_p_ptr = PTR_FROM_3D(vgrad_p);
  real* divdp_ptr = PTR_FROM_3D(divdp);
  real* vort_ptr = PTR_FROM_3D(vort);

  // Other accessory variables
  real v1,v2;

  // Extract stuff from data (for notation shortness)
  HVCoord& hvcoord = data.hvcoord;

  // Extract arrays from data (for notation shortness)
  real* elem_Dinv         = data.arrays.elem_Dinv;
  real* elem_state_v      = data.arrays.elem_state_v;
  real* elem_state_ps_v   = data.arrays.elem_state_ps_v;
  real* elem_state_dp3d   = data.arrays.elem_state_dp3d;
  real* elem_derived_vn0  = data.arrays.elem_derived_vn0;

  const int nets = 0;
  const int nete = nelems;
  const int n0 = 0;

  for (int ie=nets; ie<nete; ++ie)
  {
    real* dp = SLICE_5D(elem_state_dp3d,ie,timelevels,nlev,np,np);
    dp = SLICE_4D(dp,n0,nlev,np,np);

    for (int ipt=0; ipt<np; ++ipt)
    {
      for (int jpt=0; jpt<np; ++jpt)
      {
        p[0][ipt][jpt] = hvcoord.hyai[0]*hvcoord.ps0 + 0.5*AT_3D(dp,0,ipt,jpt,np,np);
      }
    }

    for (int ilev=1; ilev<nlev; ++ilev)
    {
      for (int ipt=0; ipt<np; ++ipt)
      {
        for (int jpt=0; jpt<np; ++jpt)
        {
          p[ilev][ipt][jpt] = p[ilev-1][ipt][jpt] + 0.5*( AT_3D(dp,ilev-1,ipt,jpt,np,np) + AT_3D(dp,ilev,ipt,jpt,np,np) );
        }
      }
    }

    for (int ilev=0; ilev<nlev; ++ilev)
    {
      real* gradp_ilev = SLICE_4D(grad_p_ptr,ilev,np,np,2);

      gradient_sphere (SLICE_3D(p_ptr,ilev,np,np), data, ie, gradp_ilev);

      real* state_v_n0_ilev = SLICE_6D(elem_state_v,ie,timelevels,nlev,np,np,2);
      state_v_n0_ilev = SLICE_5D(state_v_n0_ilev,n0,nlev,np,np,2);
      state_v_n0_ilev = SLICE_4D(state_v_n0_ilev,ilev,np,np,2);

      real* vdp_ilev = SLICE_4D (vdp_ptr,ilev,np,np,2);
      real* vgrad_p_ilev = SLICE_3D (vgrad_p_ptr,ilev,np,np);

      real* derived_vn0_ilev = SLICE_5D(elem_derived_vn0,ie,nlev,np,np,2);
      derived_vn0_ilev = SLICE_4D(derived_vn0_ilev,ilev,np,np,2);
      for (int ipt=0; ipt<np; ++ipt)
      {
        for (int jpt=0; jpt<np; ++jpt)
        {
          v1 = AT_3D(state_v_n0_ilev,ipt,jpt,0,np,2);
          v2 = AT_3D(state_v_n0_ilev,ipt,jpt,1,np,2);
          AT_2D(vgrad_p_ilev,ipt,jpt,np) = v1 * AT_3D(gradp_ilev,ipt,jpt,0,np,2)
                                         + v2 * AT_3D(gradp_ilev,ipt,jpt,1,np,2);

          AT_3D(vdp_ilev,ipt,jpt,0,np,2) = v1 * AT_3D(dp,ilev,ipt,jpt,np,np);
          AT_3D(vdp_ilev,ipt,jpt,1,np,2) = v2 * AT_3D(dp,ilev,ipt,jpt,np,np);

          AT_3D(derived_vn0_ilev,ipt,jpt,0,np,2) += data.constants.eta_ave_w * AT_3D(vdp_ilev,ipt,jpt,0,np,2);
          AT_3D(derived_vn0_ilev,ipt,jpt,1,np,2) += data.constants.eta_ave_w * AT_3D(vdp_ilev,ipt,jpt,1,np,2);
        }
      }

      divergence_sphere(vdp_ilev, data, ie, SLICE_3D (divdp_ptr,ilev,np,np));
      vorticity_sphere(state_v_n0_ilev, data, ie, SLICE_3D (vort_ptr,ilev,np,np));
    }
  }

/*
     do k=1,nlev
...
     if (qn0 == -1 ) then
        do k=1,nlev
           do j=1,np
              do i=1,np
                 T_v(i,j,k) = elem(ie)%state%T(i,j,k,n0)
                 kappa_star(i,j,k) = kappa
              end do
           end do
        end do
     else
        do k=1,nlev
           do j=1,np
              do i=1,np
                 Qt = elem(ie)%state%Qdp(i,j,k,1,qn0)/dp(i,j,k)
                 T_v(i,j,k) = Virtual_Temperature1d(elem(ie)%state%T(i,j,k,n0),Qt)
                 kappa_star(i,j,k) = kappa
              end do
           end do
        end do
     end if
     call preq_hydrostatic(phi,elem(ie)%state%phis,T_v,p,dp)
     call preq_omega_ps(omega_p,hvcoord,p,vgrad_p,divdp)
     sdot_sum=0
     ! VERTICALLY LAGRANGIAN:   no vertical motion
     eta_dot_dpdn=0
     T_vadv=0
     v_vadv=0
     do k=1,nlev  !  Loop index added (AAM)
        elem(ie)%derived%eta_dot_dpdn(:,:,k) = &
             elem(ie)%derived%eta_dot_dpdn(:,:,k) + eta_ave_w*eta_dot_dpdn(:,:,k)
        elem(ie)%derived%omega_p(:,:,k) = &
             elem(ie)%derived%omega_p(:,:,k) + eta_ave_w*omega_p(:,:,k)
     enddo
     elem(ie)%derived%eta_dot_dpdn(:,:,nlev+1) = &
          elem(ie)%derived%eta_dot_dpdn(:,:,nlev+1) + eta_ave_w*eta_dot_dpdn(:,:,nlev+1)
     vertloop: do k=1,nlev
        do j=1,np
           do i=1,np
              v1     = elem(ie)%state%v(i,j,1,k,n0)
              v2     = elem(ie)%state%v(i,j,2,k,n0)
              E = 0.5D0*( v1*v1 + v2*v2 )
              Ephi(i,j)=E+phi(i,j,k)+elem(ie)%derived%pecnd(i,j,k)
           end do
        end do
        vtemp(:,:,:)   = gradient_sphere(elem(ie)%state%T(:,:,k,n0),deriv,elem(ie)%Dinv)
        do j=1,np
           do i=1,np
              v1     = elem(ie)%state%v(i,j,1,k,n0)
              v2     = elem(ie)%state%v(i,j,2,k,n0)
              vgrad_T(i,j) =  v1*vtemp(i,j,1) + v2*vtemp(i,j,2)
           end do
        end do

        vtemp = gradient_sphere(Ephi(:,:),deriv,elem(ie)%Dinv)
        do j=1,np
           do i=1,np
              gpterm = T_v(i,j,k)/p(i,j,k)
              glnps1 = Rgas*gpterm*grad_p(i,j,1,k)
              glnps2 = Rgas*gpterm*grad_p(i,j,2,k)

              v1     = elem(ie)%state%v(i,j,1,k,n0)
              v2     = elem(ie)%state%v(i,j,2,k,n0)

              vtens1(i,j,k) =   - v_vadv(i,j,1,k)                           &
                   + v2*(elem(ie)%fcor(i,j) + vort(i,j,k))        &
                   - vtemp(i,j,1) - glnps1
              vtens2(i,j,k) =   - v_vadv(i,j,2,k)                            &
                   - v1*(elem(ie)%fcor(i,j) + vort(i,j,k))        &
                   - vtemp(i,j,2) - glnps2
              ttens(i,j,k)  = - T_vadv(i,j,k) - vgrad_T(i,j) + kappa_star(i,j,k)*T_v(i,j,k)*omega_p(i,j,k)
           end do
        end do
     end do vertloop

     do k=1,nlev
        elem(ie)%state%v(:,:,1,k,np1) = elem(ie)%spheremp(:,:)*( elem(ie)%state%v(:,:,1,k,nm1) + dt2*vtens1(:,:,k) )
        elem(ie)%state%v(:,:,2,k,np1) = elem(ie)%spheremp(:,:)*( elem(ie)%state%v(:,:,2,k,nm1) + dt2*vtens2(:,:,k) )
        elem(ie)%state%T(:,:,k,np1) = elem(ie)%spheremp(:,:)*(elem(ie)%state%T(:,:,k,nm1) + dt2*ttens(:,:,k))
        elem(ie)%state%dp3d(:,:,k,np1) = &
             elem(ie)%spheremp(:,:) * (elem(ie)%state%dp3d(:,:,k,nm1) - &
             dt2 * (divdp(:,:,k) + eta_dot_dpdn(:,:,k+1)-eta_dot_dpdn(:,:,k)))
     enddo ! k loop
  end subroutine compute_and_apply_rhs
*/

}

} // Namespace Homme
