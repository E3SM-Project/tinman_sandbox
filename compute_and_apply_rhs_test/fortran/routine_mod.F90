module routine_mod

implicit none
contains


subroutine compute_and_apply_rhs(np1,nm1,n0,qn0,dt2,elem, hvcoord, deriv,nets,nete,eta_ave_w)
!subroutine compute_and_apply_rhs(np1,nm1,n0,qn0,dt2,elem,hvcoord,hybrid,&
!       deriv,nets,nete,compute_diagnostics,eta_ave_w)
  ! ===================================

  use kinds, only : real_kind, np, nlev, ntrac, nelemd, timelevels
  use element_mod, only : element_t
  use element_state_mod, only : elem_state_t
                          !indu, indv, indT, inddp,  indps, indphis
  use derivative_mod_base, only : derivative_t, divergence_sphere, gradient_sphere, vorticity_sphere, &
                                  vorticity_v2
  use hybvcoord_mod, only : hvcoord_t

  use physical_constants, only : cp, cpwater_vapor, Rgas, kappa

implicit none

  type (element_t), intent(inout), target :: elem(:)
!  real (kind=real_kind), intent(in) :: ST(np,np,nlev,nelemd,numst,timelevels)
  type (derivative_t)  , intent(in) :: deriv
  type (hvcoord_t)     , intent(in) :: hvcoord
  integer, intent(in) :: nets, nete, np1,nm1,n0,qn0
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
  real (kind=real_kind), dimension(np,np,2)    :: vtemp     ! generic gradient storage
  real (kind=real_kind), dimension(np,np,2,nlev):: vdp      !
  real (kind=real_kind), dimension(np,np,2     ):: v        !
  real (kind=real_kind), dimension(np,np)      :: vgrad_T   ! v.grad(T)
  real (kind=real_kind), dimension(np,np)      :: Ephi      ! kinetic energy + PHI term
  real (kind=real_kind), dimension(np,np,2,nlev) :: grad_p
  real (kind=real_kind), dimension(np,np,2,nlev) :: grad_p_m_pmet  ! gradient(p - p_met)
  real (kind=real_kind), dimension(np,np,nlev)   :: vort       ! vorticity
  real (kind=real_kind), dimension(np,np,nlev)   :: p          ! pressure
  real (kind=real_kind), dimension(np,np,nlev)   :: rdp        ! inverse of delta pressure
  real (kind=real_kind), dimension(np,np,nlev)   :: T_vadv     ! temperature vertical advection
  real (kind=real_kind), dimension(np,np,nlev)   :: vgrad_p    ! v.grad(p)
  real (kind=real_kind), dimension(np,np,nlev+1) :: ph         ! half level pressures on p-grid
  real (kind=real_kind), dimension(np,np,2,nlev) :: v_vadv     ! velocity vertical advection
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
  integer :: i,j,k,kptr,ie
  real (kind=real_kind) ::  u_m_umet, v_m_vmet, t_m_tmet


!  print *, 'Hello Original Routine'
  do ie=nets,nete
     phi => elem(ie)%derived%phi(:,:,:)
     dp  => elem(ie)%state%dp3d(:,:,:,n0)
     p(:,:,1)=hvcoord%hyai(1)*hvcoord%ps0 + dp(:,:,1)/2
     do k=2,nlev
        p(:,:,k)=p(:,:,k-1) + dp(:,:,k-1)/2 + dp(:,:,k)/2
     enddo
#if HOMP
!$omp parallel do private(k,i,j,v1,v2,vtemp)
#endif
     do k=1,nlev
        grad_p(:,:,:,k) = gradient_sphere(p(:,:,k),deriv,elem(ie)%Dinv)
        rdp(:,:,k) = 1.0D0/dp(:,:,k)
        do j=1,np
           do i=1,np
              v1 = elem(ie)%state%v(i,j,1,k,n0)
              v2 = elem(ie)%state%v(i,j,2,k,n0)
              vgrad_p(i,j,k) = (v1*grad_p(i,j,1,k) + v2*grad_p(i,j,2,k))
              vdp(i,j,1,k) = v1*dp(i,j,k)
              vdp(i,j,2,k) = v2*dp(i,j,k)
           end do
        end do
        elem(ie)%derived%vn0(:,:,:,k)=elem(ie)%derived%vn0(:,:,:,k)+eta_ave_w*vdp(:,:,:,k)
        divdp(:,:,k)=divergence_sphere(vdp(:,:,:,k),deriv,elem(ie))
        vort(:,:,k)=vorticity_v2(elem(ie)%state%v(:,:,1,k,n0),elem(ie)%state%v(:,:,2,k,n0),deriv,elem(ie))
     enddo
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
!this loop, moisture
#if HOMP
!$omp parallel do private(k,i,j,Qt)
#endif
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
#if HOMP
     !$omp parallel do private(k)
#endif
     do k=1,nlev  !  Loop index added (AAM)
        elem(ie)%derived%eta_dot_dpdn(:,:,k) = &
             elem(ie)%derived%eta_dot_dpdn(:,:,k) + eta_ave_w*eta_dot_dpdn(:,:,k)
        elem(ie)%derived%omega_p(:,:,k) = &
             elem(ie)%derived%omega_p(:,:,k) + eta_ave_w*omega_p(:,:,k)
     enddo
     elem(ie)%derived%eta_dot_dpdn(:,:,nlev+1) = &
          elem(ie)%derived%eta_dot_dpdn(:,:,nlev+1) + eta_ave_w*eta_dot_dpdn(:,:,nlev+1)
#if HOMP
!$omp parallel do private(k,i,j,v1,v2,E,Ephi,vtemp,vgrad_T,gpterm,glnps1,glnps2)
#endif
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

#if HOMP
!$omp parallel do private(k)
#endif
     do k=1,nlev
        elem(ie)%state%v(:,:,1,k,np1) = elem(ie)%spheremp(:,:)*( elem(ie)%state%v(:,:,1,k,nm1) + dt2*vtens1(:,:,k) )
        elem(ie)%state%v(:,:,2,k,np1) = elem(ie)%spheremp(:,:)*( elem(ie)%state%v(:,:,2,k,nm1) + dt2*vtens2(:,:,k) )
        elem(ie)%state%T(:,:,k,np1) = elem(ie)%spheremp(:,:)*(elem(ie)%state%T(:,:,k,nm1) + dt2*ttens(:,:,k))
        elem(ie)%state%dp3d(:,:,k,np1) = &
             elem(ie)%spheremp(:,:) * (elem(ie)%state%dp3d(:,:,k,nm1) - &
             dt2 * (divdp(:,:,k) + eta_dot_dpdn(:,:,k+1)-eta_dot_dpdn(:,:,k)))

     enddo ! k loop
  end do

end subroutine compute_and_apply_rhs


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


end module routine_mod
