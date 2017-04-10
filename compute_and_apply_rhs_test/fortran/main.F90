#include "config1.h"
#include "config2.h"
#include "config3.h"
#include "config4.h"


program main
  use kinds, only: nelemd

implicit none

  integer :: num_args
  character(len=20)     :: arg

  num_args = command_argument_count()

  if (num_args==1) then
    call get_command_argument(1,arg)
    read (arg, *)  nelemd
  endif

  call main_body

end program main

subroutine main_body

  use kinds
  use element_state_mod
  use element_mod
  use derivative_mod_base
  use hybvcoord_mod
  use test_mod
  use utils_mod

#if ORIG
  use routine_mod         , only : compute_and_apply_rhs
#else
  use routine_mod_ST      , only : compute_and_apply_rhs_st
#endif

implicit none

  type (element_t), allocatable :: elem(:)
  type (derivative_t)           :: deriv

! init params

  real (kind=real_kind) :: Dvv_init(np*np)
  type (hvcoord_t)      :: hvcoord
  integer               :: nets, nete
  real (kind=real_kind) :: dt2, start, finish
  real (kind=real_kind) :: eta_ave_w
  real (kind=real_kind) :: ii, jj, kk, iee

! local
  integer :: i,j,k,ie,tl,ind

  real (kind=real_kind) :: Tt(np,np,nlev)
  real (kind=real_kind) :: v_norm(nelemd), t_norm(nelemd), dp_norm(nelemd)

#if   STVER1
  real (kind=real_kind) :: ST(np,np,nlev,nelemd,numst,timelevels)
#elif STVER2
  real (kind=real_kind) :: ST(np,np,nlev,numst,nelemd,timelevels)
#elif STVER3
  real (kind=real_kind) :: ST(np,np,nlev,numst,timelevels,nelemd)
#elif STVER4
  real (kind=real_kind) :: ST(np,np,nlev,timelevels,numst,nelemd)
#endif

print *, "Main: nelemd = ", nelemd

  dt2 = 1.0
  eta_ave_w = 1.0
  nets = 1
  nete = nelemd

!----------------- INITIALIZATION --------------------

  ! Derivative structure

  ! This is a douple-prec array init-ed with single prec values.
  ! Since Ttest values were computed based on this init, switching to double
  ! prec here would break tests (unless Ttest is recomputed).
  Dvv_init(1:16) = (/ -3.0,  -0.80901699437494745,   0.30901699437494745, &
  -0.50000000000000000 ,4.0450849718747373, 0.0, -1.1180339887498949, &
   1.5450849718747370, -1.5450849718747370, 1.1180339887498949, &
   0.0, -4.0450849718747373, 0.50, -0.30901699437494745, 0.80901699437494745, 3.0 /)

  do j =1 , np
   do i = 1, np
     deriv%Dvv(i,j) = Dvv_init((j-1)*np+i)
   enddo
  enddo

  ! Allocate elements
  allocate(elem(nelemd))

  ! Fill element attributes
  ! Note: SIN and COS expect REAL as input, so we create REAL's ii,jj,kk,iee from INTEGER's i,j,k,ie
  do ie = nets,nete
   iee = ie
   do j = 1,np
    jj = j
    do i = 1,np
     ii = i

     ! Initializing 2D fields
     elem(ie)%fcor(i,j)       = SIN(ii+jj)
     elem(ie)%metdet(i,j)     = ii*jj
     elem(ie)%rmetdet(i,j)    = 1.0d0/elem(ie)%metdet(i,j)
     elem(ie)%spheremp(i,j)   = 2*ii

     elem(ie)%D(i,j,1,1) = 1.0
     elem(ie)%D(i,j,1,2) = 0.0
     elem(ie)%D(i,j,2,1) = 0.0
     elem(ie)%D(i,j,2,2) = 2.0

     elem(ie)%Dinv(i,j,1,1) = 1.0
     elem(ie)%Dinv(i,j,1,2) = 0.0
     elem(ie)%Dinv(i,j,2,1) = 0.0
     elem(ie)%Dinv(i,j,2,2) = 0.5

     do k = 1,nlev
      kk = k

      ! Initializing 3D fields
      elem(ie)%derived%phi(i,j,k) = COS(ii+3*jj)+kk
      elem(ie)%derived%vn0(i,j,1:2,k) = 1.0
      elem(ie)%derived%pecnd(i,j,k) = 1.0
      elem(ie)%derived%omega_p(i,j,k) = jj*jj

      ! Initializing 4D fields
#if ORIG
      elem(ie)%state%dp3d(i,j,k,1:timelevels) = 10*kk+iee+ii+jj + (/1,2,3/)
      elem(ie)%state%v(i,j,1,k,1:timelevels) = 1.0+kk/2+ii+jj+iee/5 + (/1,2,3/)*2.0
      elem(ie)%state%v(i,j,2,k,1:timelevels) = 1.0+kk/2+ii+jj+iee/5 + (/1,2,3/)*3.0
      elem(ie)%state%T(i,j,k,1:timelevels) = 1000-kk-ii-jj+iee/10 + (/1,2,3/)
      elem(ie)%state%Qdp(i,j,k,1,qn0) = 1.0+SIN(ii*jj*kk)
      elem(ie)%state%phis(i,j) = 0.0d0
#else
      ST( iXjXkXdpXdXie )   = 10*kk+iee+ii+jj + (/1,2,3/)
      ST( iXjXkXuXdXie )    = 1.0+kk/2+ii+jj+iee/5 + (/1,2,3/)*2.0
      ST( iXjXkXvXdXie )    = 1.0+kk/2+ii+jj+iee/5 + (/1,2,3/)*3.0
      ST( iXjXkXtXdXie )    = 1000-kk-ii-jj+iee/10 + (/1,2,3/)
      ST( iXjXkXqXdXie )    = 1.0+SIN(ii*jj*kk)
      ST( iXjX1XphisX1Xie ) = 0.0d0
#endif
     enddo
    enddo
   enddo
  enddo

! Init hybrid
! Actually, most of hvcoord does not matter since only Lagrangian code is present.
! What about ps0?
  hvcoord%ps0 = 10.0
  do k = 1, nlev + 1
    hvcoord%hyai(k) = nlev + 2 - k
  enddo
!----------------- END OF INITIALIZATION -----------------------

!---------------- INITIAL NORMS OF (u,v),t and dp states at np1 -----------------
! This is to make sure fortran and cxx are initialized in the same way

  do ie=1,nelemd
#if STVER1
    v_norm(ie)  = compute_norm(ST(:,:,:,ie,1:2,np1),np*np*nlev*2)
    t_norm(ie)  = compute_norm(ST(:,:,:,ie,3  ,np1),np*np*nlev)
    dp_norm(ie) = compute_norm(ST(:,:,:,ie,4  ,np1),np*np*nlev)
#elif STVER2
    v_norm(ie)  = compute_norm(ST(:,:,:,1:2,ie,np1),np*np*nlev*2)
    t_norm(ie)  = compute_norm(ST(:,:,:,3  ,ie,np1),np*np*nlev)
    dp_norm(ie) = compute_norm(ST(:,:,:,4  ,ie,np1),np*np*nlev)
#elif STVER3
    v_norm(ie)  = compute_norm(ST(:,:,:,1:2,np1,ie),np*np*nlev*2)
    t_norm(ie)  = compute_norm(ST(:,:,:,3  ,np1,ie),np*np*nlev)
    dp_norm(ie) = compute_norm(ST(:,:,:,4  ,np1,ie),np*np*nlev)
#elif STVER4
    v_norm(ie)  = compute_norm(ST(:,:,:,np1,1:2,ie),np*np*nlev*2)
    t_norm(ie)  = compute_norm(ST(:,:,:,np1,3  ,ie),np*np*nlev)
    dp_norm(ie) = compute_norm(ST(:,:,:,np1,4  ,ie),np*np*nlev)
#else
    v_norm(ie)  = compute_norm(elem(ie)%state%v(:,:,:,:,np1),np*np*nlev*2)
    t_norm(ie)  = compute_norm(elem(ie)%state%T(:,:,:,np1),np*np*nlev)
    dp_norm(ie) = compute_norm(elem(ie)%state%dp3d(:,:,:,np1),np*np*nlev)
#endif
  enddo

  print *, "||v||_2  = ", compute_norm(v_norm,nelemd)
  print *, "||T||_2  = ", compute_norm(t_norm ,nelemd)
  print *, "||dp||_2 = ", compute_norm(dp_norm,nelemd)

  print *, 'Main, np=', np

!np1 fields will be changed

  call cpu_time(start)
  do ind = 1, loopmax
#if ORIG
    call compute_and_apply_rhs(np1,nm1,n0,qn0, dt2,elem, hvcoord, deriv, nets,nete, eta_ave_w)
#else
    call compute_and_apply_rhs_st(np1,nm1,n0,qn0,dt2,elem, hvcoord, deriv,nets,nete,eta_ave_w,ST)
#endif
    ! I'm not sure if the compiler is smart enough to see that we are doing THE
    ! SAME calculations. To avoid caching, we update time levels
    !call update_time_levels
  enddo
  call cpu_time(finish)

  print '("Time = ",f10.4," seconds.")',finish-start
  print *, 'Raw time = ', finish-start

! ---------------- DO NOT MODIFY ------------------------
  ie = 1
  ind = 1
  do k = 1,nlev;
   do j = 1,np;
    do i = 1,np
     Tt(i,j,k) = Ttest(ind)
     ind = ind+1
    enddo
   enddo
  enddo

#if STVER1
  print *, 'STVER1 diff', maxval(abs(Tt - ST( dXdXdXtXnp1Xie )))
#elif STVER2
  print *, 'STVER2 diff', maxval(abs(Tt - ST( dXdXdXtXnp1Xie )))
#elif STVER3
  print *, 'STVER3 diff', maxval(abs(Tt - ST( dXdXdXtXnp1Xie )))
#elif STVER4
  print *, 'STVER4 diff', maxval(abs(Tt - ST( dXdXdXtXnp1Xie )))
#else
  print *, 'ORIGINAL diff', maxval(abs(Tt - elem(ie)%state%T(:,:,:,np1)))
#endif
!---------------------------------------------------------

  ! Computing states norms (for comparison with C++)
  do ie=1,nelemd
#if STVER1
    v_norm(ie)  = compute_norm(ST(:,:,:,ie,1:2,np1),np*np*nlev*2)
    t_norm(ie)  = compute_norm(ST(:,:,:,ie,3  ,np1),np*np*nlev)
    dp_norm(ie) = compute_norm(ST(:,:,:,ie,4  ,np1),np*np*nlev)
#elif STVER2
    v_norm(ie)  = compute_norm(ST(:,:,:,1:2,ie,np1),np*np*nlev*2)
    t_norm(ie)  = compute_norm(ST(:,:,:,3  ,ie,np1),np*np*nlev)
    dp_norm(ie) = compute_norm(ST(:,:,:,4  ,ie,np1),np*np*nlev)
#elif STVER3
    v_norm(ie)  = compute_norm(ST(:,:,:,1:2,np1,ie),np*np*nlev*2)
    t_norm(ie)  = compute_norm(ST(:,:,:,3  ,np1,ie),np*np*nlev)
    dp_norm(ie) = compute_norm(ST(:,:,:,4  ,np1,ie),np*np*nlev)
#elif STVER4
    v_norm(ie)  = compute_norm(ST(:,:,:,np1,1:2,ie),np*np*nlev*2)
    t_norm(ie)  = compute_norm(ST(:,:,:,np1,3  ,ie),np*np*nlev)
    dp_norm(ie) = compute_norm(ST(:,:,:,np1,4  ,ie),np*np*nlev)
#else
    v_norm(ie)  = compute_norm(elem(ie)%state%v(:,:,:,:,np1),np*np*nlev*2)
    t_norm(ie)  = compute_norm(elem(ie)%state%T(:,:,:,np1),np*np*nlev)
    dp_norm(ie) = compute_norm(elem(ie)%state%dp3d(:,:,:,np1),np*np*nlev)
#endif
  enddo

  print *, "||v||_2  = ", compute_norm(v_norm,nelemd)
  print *, "||T||_2  = ", compute_norm(t_norm ,nelemd)
  print *, "||dp||_2 = ", compute_norm(dp_norm,nelemd)

end subroutine main_body
