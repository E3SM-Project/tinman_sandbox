#include "config1.h"
#include "config2.h"
#include "config3.h"
#include "config4.h"

module routine_mod_ST

implicit none
contains


subroutine compute_and_apply_rhs_st(np1,nm1,n0,qn0,dt2,elem, hvcoord, deriv,nets,nete,eta_ave_w,ST)
!subroutine compute_and_apply_rhs(np1,nm1,n0,qn0,dt2,elem,hvcoord,hybrid,&
!       deriv,nets,nete,compute_diagnostics,eta_ave_w)
  ! ===================================

  use kinds, only : real_kind, np, nlev, ntrac, nelemd, timelevels, numst
  use element_state_mod
  use element_mod
  use derivative_mod_base, only : derivative_t, divergence_sphere, gradient_sphere, vorticity_sphere, &
                                  vorticity_v2
  use hybvcoord_mod, only : hvcoord_t
  use physical_constants, only : cp, cpwater_vapor, Rgas, kappa

implicit none

  type (element_t), intent(inout), target :: elem(:)
!  real (kind=real_kind), intent(inout) :: ST(np,np,nlev,nelemd,numst,timelevels)

!---------------- repeated block
#if STVER1
! I J K IE ST TL
real (kind=real_kind) :: ST(np,np,nlev,nelemd,numst,timelevels)
#endif

#if STVER2
! I J K ST IE TL
real (kind=real_kind) :: ST(np,np,nlev,numst,nelemd,timelevels)
#endif

#if STVER3
! I J K ST TL IE
real (kind=real_kind) :: ST(np,np,nlev,numst,timelevels,nelemd)
#endif

! this is the original layout!
#if STVER4
! I J K TL ST IE
real (kind=real_kind) :: ST(np,np,nlev,timelevels,numst,nelemd)
#endif
!---------------- end of repeated block

  type (derivative_t)  , intent(in) :: deriv
  type (hvcoord_t)     , intent(in) :: hvcoord
  integer, intent(in) :: nets, nete, np1,nm1,n0,qn0
  real*8, intent(in) :: dt2
  real (kind=real_kind), intent(in) :: eta_ave_w 

  integer :: ie

!insert omp here

  do ie  =  nets, nete
    call caar(np1,nm1,n0,qn0,dt2,elem, hvcoord, deriv,ie,eta_ave_w,ST)
  enddo

end subroutine compute_and_apply_rhs_st






subroutine caar(np1,nm1,n0,qn0,dt2,elem, hvcoord, deriv,ie,eta_ave_w,ST)

  use kinds, only : real_kind, np, nlev, ntrac, nelemd, timelevels, numst
  use element_state_mod
  use element_mod
  use derivative_mod_base, only : derivative_t, divergence_sphere, gradient_sphere, vorticity_sphere, &
                                  vorticity_v2
  use hybvcoord_mod, only : hvcoord_t
  use physical_constants, only : cp, cpwater_vapor, Rgas, kappa

implicit none

  type (element_t), intent(inout), target :: elem(:)

!---------------- repeated block
#if STVER1
! I J K IE ST TL
real (kind=real_kind) :: ST(np,np,nlev,nelemd,numst,timelevels)
#endif

#if STVER2
! I J K ST IE TL
real (kind=real_kind) :: ST(np,np,nlev,numst,nelemd,timelevels)
#endif

#if STVER3
! I J K ST TL IE
real (kind=real_kind) :: ST(np,np,nlev,numst,timelevels,nelemd)
#endif

! this is the original layout!
#if STVER4
! I J K TL ST IE
real (kind=real_kind) :: ST(np,np,nlev,timelevels,numst,nelemd)
#endif
!---------------- end of repeated block
  type (derivative_t)  , intent(in) :: deriv
  type (hvcoord_t)     , intent(in) :: hvcoord
  integer, intent(in) :: ie,np1,nm1,n0,qn0
  real*8, intent(in) :: dt2
  real (kind=real_kind), intent(in) :: eta_ave_w 

!local
  real (kind=real_kind), pointer, dimension(:,:,:)   :: phi
  real (kind=real_kind), pointer, dimension(:,:,:)   :: dp
  real (kind=real_kind), dimension(np,np,nlev)   :: omega_p
  real (kind=real_kind), dimension(np,np,nlev)   :: T_v
  real (kind=real_kind), dimension(np,np,nlev)   :: divdp
  real (kind=real_kind), dimension(np,np,nlev+1)   :: eta_dot_dpdn  ! half level vertical velocity on p-grid
  real (kind=real_kind), dimension(np,np)      :: sdot_sum   ! temporary field
  real (kind=real_kind), dimension(np,np,2,nlev)    :: vtemp1, vtemp2     ! generic gradient storage
  real (kind=real_kind), dimension(np,np,2,nlev):: vdp       !                            
  real (kind=real_kind), dimension(np,np,2     ):: v         !                            
  real (kind=real_kind), dimension(np,np,nlev)      :: vgrad_T    ! v.grad(T)
  real (kind=real_kind), dimension(np,np)      :: Ephi       ! kinetic energy + PHI term
  real (kind=real_kind), dimension(np,np,2,nlev) :: grad_p
  real (kind=real_kind), dimension(np,np,2,nlev) :: grad_p_m_pmet  ! gradient(p - p_met)
  real (kind=real_kind), dimension(np,np,nlev)   :: vort       ! vorticity
  real (kind=real_kind), dimension(np,np,nlev)   :: p          ! pressure
  real (kind=real_kind), dimension(np,np,nlev)   :: rdp        ! inverse of delta pressure
  real (kind=real_kind), dimension(np,np,nlev)   :: T_vadv     ! temperature vertical advection
  real (kind=real_kind), dimension(np,np,nlev)   :: vgrad_p    ! v.grad(p)
  real (kind=real_kind), dimension(np,np,nlev+1) :: ph               ! half level pressures on p-grid
  real (kind=real_kind), dimension(np,np,2,nlev) :: v_vadv   ! velocity vertical advection
  real (kind=real_kind), dimension(0:np+1,0:np+1,nlev)          :: corners
  real (kind=real_kind), dimension(2,2,2)                         :: cflux
  real (kind=real_kind) ::  kappa_star(np,np,nlev)
  real (kind=real_kind) ::  vtens1(np,np,nlev)
  real (kind=real_kind) ::  vtens2(np,np,nlev)
  real (kind=real_kind) ::  ttens(np,np,nlev)
  real (kind=real_kind) ::  stashdp3d (np,np,nlev)
  real (kind=real_kind) ::  tempdp3d  (np,np)
  real (kind=real_kind) ::  cp2,cp_ratio,E,de,Qt,v1,v2
  real (kind=real_kind) ::  glnps1,glnps2,gpterm
  integer :: i,j,k,kptr,q
  real (kind=real_kind) ::  u_m_umet, v_m_vmet, t_m_tmet 
  
  integer :: tid, OMP_GET_MAX_THREADS, OMP_GET_THREAD_NUM

!  print *, 'Hello Routine'
!  tid = OMP_GET_MAX_THREADS()     
!  print *, 'Max number TH ', tid
!  tid = OMP_GET_THREAD_NUM()     
!  print *, 'My tid is ', tid

! a dummy loop
!!!$omp parallel do private(k,q)
#if 0
  do k=1,nlev
  do q=2,35
#if STVER1
  ST( :,:,k,ie,6+q,np1) = 1.0d0
  ST( :,:,k,ie,6+q,np1) = ST( :,:,k,ie,6+q,n0)+ST( :,:,k,ie,6+q,nm1)
  ST( :,:,k,ie,6+q,np1) = sin(2.0)

  ST( :,:,k,ie,6+q,np1) = 1.0d0
  ST( :,:,k,ie,6+q,np1) = ST( :,:,k,ie,6+q,n0)+ST( :,:,k,ie,6+q,nm1)
  ST( :,:,k,ie,6+q,np1) = sin(2.0)

  ST( :,:,k,ie,6+q,np1) = 1.0d0
  ST( :,:,k,ie,6+q,np1) = ST( :,:,k,ie,6+q,n0)+ST( :,:,k,ie,6+q,nm1)
  ST( :,:,k,ie,6+q,np1) = sin(2.0)

  ST( :,:,k,ie,6+q,np1) = 1.0d0
  ST( :,:,k,ie,6+q,np1) = ST( :,:,k,ie,6+q,n0)+ST( :,:,k,ie,6+q,nm1)
  ST( :,:,k,ie,6+q,np1) = sin(2.0)
#endif
#if STVER2
  ST( :,:,k,6+q,ie,np1) = 1.0d0
  ST( :,:,k,6+q,ie,np1) = ST( :,:,k,6+q,ie,n0)+ST( :,:,k,6+q,ie,nm1)
  ST( :,:,k,6+q,ie,np1) = sin(2.0)

  ST( :,:,k,6+q,ie,np1) = 1.0d0
  ST( :,:,k,6+q,ie,np1) = ST( :,:,k,6+q,ie,n0)+ST( :,:,k,6+q,ie,nm1)
  ST( :,:,k,6+q,ie,np1) = sin(2.0)

  ST( :,:,k,6+q,ie,np1) = 1.0d0
  ST( :,:,k,6+q,ie,np1) = ST( :,:,k,6+q,ie,n0)+ST( :,:,k,6+q,ie,nm1)
  ST( :,:,k,6+q,ie,np1) = sin(2.0)

  ST( :,:,k,6+q,ie,np1) = 1.0d0
  ST( :,:,k,6+q,ie,np1) = ST( :,:,k,6+q,ie,n0)+ST( :,:,k,6+q,ie,nm1)
  ST( :,:,k,6+q,ie,np1) = sin(2.0)
#endif
  enddo
  enddo
#endif
! end of the dummy


     phi => elem(ie)%derived%phi(:,:,:)

     p(:,:,1)=hvcoord%hyai(1)*hvcoord%ps0 + ST( dXdX1XdpXn0Xie  ) /2

! this can be rewritten
     do k=2,nlev
        p(:,:,k)=p(:,:,k-1) + ST( dXdXkm1XdpXn0Xie )/2 + ST( dXdXkXdpXn0Xie )/2
     enddo

#if HOMP
!$omp parallel do private(k,i,j,v1,v2,Qt,eta_ave_w,E,Ephi)
#endif
     do k=1,nlev

!        tid = OMP_GET_THREAD_NUM()     
!        print *, 'next loop: My tid is ', tid

        grad_p(:,:,:,k) = gradient_sphere(p(:,:,k),deriv,elem(ie)%Dinv)

        rdp(:,:,k) = 1.0D0/ST( dXdXkXdpXn0Xie )

        vtemp1(:,:,:,k)   = gradient_sphere( ST( dXdXkXtXn0Xie ), deriv,elem(ie)%Dinv)
        do j=1,np
           do i=1,np
              v1 = ST( iXjXkXuXn0Xie )
              v2 = ST( iXjXkXvXn0Xie )
              vgrad_p(i,j,k) = (v1*grad_p(i,j,1,k) + v2*grad_p(i,j,2,k))
              vdp(i,j,1,k) = v1*ST( iXjXkXdpXn0Xie )
              vdp(i,j,2,k) = v2*ST( iXjXkXdpXn0Xie )

              Qt = ST( iXjXkXqXqn0Xie )/ ST( iXjXkXdpXn0Xie )
              T_v(i,j,k) = Virtual_Temperature1d( ST( iXjXkXtXn0Xie ),Qt)
              kappa_star(i,j,k) = kappa

              E = 0.5D0*( v1*v1 + v2*v2 )
              Ephi(i,j)=E+phi(i,j,k)+elem(ie)%derived%pecnd(i,j,k)
              vgrad_T(i,j,k) =  v1*vtemp1(i,j,1,k) + v2*vtemp1(i,j,2,k)

           end do
        end do

        vtemp2(:,:,:,k) = gradient_sphere(Ephi(:,:),deriv,elem(ie)%Dinv)

        elem(ie)%derived%vn0(:,:,:,k)=elem(ie)%derived%vn0(:,:,:,k)+eta_ave_w*vdp(:,:,:,k)
        divdp(:,:,k)=divergence_sphere(vdp(:,:,:,k),deriv,elem(ie))
        vort(:,:,k)=vorticity_v2( ST( dXdXkXuXn0Xie ) , ST( dXdXkXvXn0Xie ) ,deriv,elem(ie))

        elem(ie)%derived%omega_p(:,:,k) = &
             elem(ie)%derived%omega_p(:,:,k) + eta_ave_w*omega_p(:,:,k)

     enddo

     call preq_hydrostatic(phi, ST( dXdX1XphisX1Xie ) ,T_v,p, ST( dXdXdXdpXn0Xie ) )
     call preq_omega_ps(omega_p,hvcoord,p,vgrad_p,divdp)

     !sdot_sum=0
     ! VERTICALLY LAGRANGIAN:   no vertical motion
     !eta_dot_dpdn=0
     T_vadv=0
     v_vadv=0

#if HOMP
!$omp parallel do private(k,v1,v2,gpterm,glnps1,glnps2)
#endif
     vertloop: do k=1,nlev
        do j=1,np
           do i=1,np
              gpterm = T_v(i,j,k)/p(i,j,k)
              glnps1 = Rgas*gpterm*grad_p(i,j,1,k)
              glnps2 = Rgas*gpterm*grad_p(i,j,2,k)

              v1 = ST( iXjXkXuXn0Xie )
              v2 = ST( iXjXkXvXn0Xie )
              vtens1(i,j,k) =   - v_vadv(i,j,1,k)                           &
                   + v2*(elem(ie)%fcor(i,j) + vort(i,j,k))        &
                   - vtemp2(i,j,1,k) - glnps1
              vtens2(i,j,k) =   - v_vadv(i,j,2,k)                            &
                   - v1*(elem(ie)%fcor(i,j) + vort(i,j,k))        &
                   - vtemp2(i,j,2,k) - glnps2
              ttens(i,j,k)  = - T_vadv(i,j,k) - vgrad_T(i,j,k) + kappa_star(i,j,k)*T_v(i,j,k)*omega_p(i,j,k)
           end do
        end do

        ST( dXdXkXuXnp1Xie ) = elem(ie)%spheremp(:,:)*( ST( dXdXkXuXnm1Xie ) + dt2*vtens1(:,:,k) )
        ST( dXdXkXuXnp1Xie ) = elem(ie)%spheremp(:,:)*( ST( dXdXkXvXnm1Xie ) + dt2*vtens2(:,:,k) )
        ST( dXdXkXtXnp1Xie ) = elem(ie)%spheremp(:,:)*( ST( dXdXkXtXnm1Xie ) + dt2*ttens(:,:,k)  )
        ST( dXdXkXdpXnp1Xie ) = &
             elem(ie)%spheremp(:,:) * ( ST( dXdXkXdpXnm1Xie ) - &
             dt2 * (divdp(:,:,k) + eta_dot_dpdn(:,:,k+1)-eta_dot_dpdn(:,:,k)))

     end do vertloop


end subroutine caar





  function Virtual_Temperature1d(Tin,rin) result(Tv)  
  use kinds, only : real_kind
  use physical_constants, only : Rwater_vapor, Rgas
    real (kind=real_kind),intent(in) :: Tin
    real (kind=real_kind),intent(in) :: rin
    real (kind=real_kind)            :: Tv
    Tv = Tin*(1_real_kind + (Rwater_vapor/Rgas - 1.0_real_kind)*rin)
  end function Virtual_Temperature1d

  subroutine preq_omega_ps(omega_p,hvcoord,p,vgrad_p,divdp)
    use kinds, only : real_kind, np, nlev
    use hybvcoord_mod, only : hvcoord_t

    implicit none
    real(kind=real_kind), intent(in) :: divdp(np,np,nlev)      ! divergence
    real(kind=real_kind), intent(in) :: vgrad_p(np,np,nlev) ! v.grad(p)
    real(kind=real_kind), intent(in) :: p(np,np,nlev)     ! layer thicknesses (pressure)
    type (hvcoord_t),     intent(in) :: hvcoord
    real(kind=real_kind), intent(out):: omega_p(np,np,nlev)   ! vertical pressure velocity

    integer i,j,k                         ! longitude, level indices
    real(kind=real_kind) term             ! one half of basic term in omega/p summation 
    real(kind=real_kind) Ckk,Ckl          ! diagonal term of energy conversion matrix
    real(kind=real_kind) suml(np,np)      ! partial sum over l = (1, k-1)

#if HOMP
!$omp parallel do private(k,j,i,ckk,term,ckl)
#endif
       do j=1,np   !   Loop inversion (AAM)
          do i=1,np
             ckk = 0.5d0/p(i,j,1)
             term = divdp(i,j,1)
             omega_p(i,j,1) = vgrad_p(i,j,1)/p(i,j,1)
             omega_p(i,j,1) = omega_p(i,j,1) - ckk*term
             suml(i,j) = term
          end do
          do k=2,nlev-1
             do i=1,np
                ckk = 0.5d0/p(i,j,k)
                ckl = 2*ckk
                term = divdp(i,j,k)
                omega_p(i,j,k) = vgrad_p(i,j,k)/p(i,j,k)
                omega_p(i,j,k) = omega_p(i,j,k) - ckl*suml(i,j) - ckk*term
                suml(i,j) = suml(i,j) + term

             end do
          end do
          do i=1,np
             ckk = 0.5d0/p(i,j,nlev)
             ckl = 2*ckk
             term = divdp(i,j,nlev)
             omega_p(i,j,nlev) = vgrad_p(i,j,nlev)/p(i,j,nlev)
             omega_p(i,j,nlev) = omega_p(i,j,nlev) - ckl*suml(i,j) - ckk*term
          end do
       end do
  end subroutine preq_omega_ps


  subroutine preq_hydrostatic(phi,phis,T_v,p,dp)
    use kinds, only : real_kind, np, nlev
    use physical_constants, only : rgas
    implicit none
    real(kind=real_kind), intent(out) :: phi(np,np,nlev)     
    real(kind=real_kind), intent(in) :: phis(np,np)
    real(kind=real_kind), intent(in) :: T_v(np,np,nlev)
    real(kind=real_kind), intent(in) :: p(np,np,nlev)   
    real(kind=real_kind), intent(in) :: dp(np,np,nlev)  
    integer i,j,k                         ! longitude, level indices
    real(kind=real_kind) Hkk,Hkl          ! diagonal term of energy conversion matrix
    real(kind=real_kind), dimension(np,np,nlev) :: phii       ! Geopotential at interfaces
#if HOMP
!$omp parallel do private(k,j,i,hkk,hkl)
#endif
       do j=1,np   !   Loop inversion (AAM)
          do i=1,np
             hkk = dp(i,j,nlev)*0.5d0/p(i,j,nlev)
             hkl = 2*hkk
             phii(i,j,nlev)  = Rgas*T_v(i,j,nlev)*hkl
             phi(i,j,nlev) = phis(i,j) + Rgas*T_v(i,j,nlev)*hkk 
          end do
          do k=nlev-1,2,-1
             do i=1,np
                ! hkk = dp*ckk
                hkk = dp(i,j,k)*0.5d0/p(i,j,k)
                hkl = 2*hkk
                phii(i,j,k) = phii(i,j,k+1) + Rgas*T_v(i,j,k)*hkl
                phi(i,j,k) = phis(i,j) + phii(i,j,k+1) + Rgas*T_v(i,j,k)*hkk
             end do
          end do
          do i=1,np
             ! hkk = dp*ckk
             hkk = 0.5d0*dp(i,j,1)/p(i,j,1)
             phi(i,j,1) = phis(i,j) + phii(i,j,2) + Rgas*T_v(i,j,1)*hkk
          end do
       end do

end subroutine preq_hydrostatic


end module routine_mod_ST
