#include "config1.h"
#include "config2.h"
#include "config3.h"
#include "config4.h"


program main
  use kinds, only: nelemd, nlev

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
  use utils_mod
  use physical_constants
implicit none

  type (element_t), allocatable :: elem(:)
  type (derivative_t)           :: deriv

! init params

  real (kind=real_kind) :: Dvv_init(np*np)
  type (hvcoord_t)      :: hvcoord
  integer               :: nets, nete, start
  real (kind=real_kind) :: dt2, finish
  real (kind=real_kind) :: eta_ave_w
  real (kind=real_kind) :: ii, jj, kk, iee

! local
!  real (kind=real_kind), pointer, dimension(:,:,:)   :: phi
  real (kind=real_kind),  dimension(np,np,nlev)   :: phi

  real (kind=real_kind), pointer, dimension(:,:,:)   :: dp
  real (kind=real_kind), dimension(np,np,nlev)   :: omega_p
  real (kind=real_kind), dimension(np,np,nlev)   :: T_v
  real (kind=real_kind), dimension(np,np,nlev)   :: divdp
  real (kind=real_kind), dimension(np,np,nlev+1)   :: eta_dot_dpdn  ! half levelvertical velocity on p-grid
  real (kind=real_kind), dimension(np,np)      :: sdot_sum   ! temporary field
  real (kind=real_kind), dimension(np,np,2,nlev)    :: vtemp1, vtemp2     !generic gradient storage
  real (kind=real_kind), dimension(np,np,2,nlev):: vdp       !                            
  real (kind=real_kind), dimension(np,np,2     ):: v         !                            
  real (kind=real_kind), dimension(np,np,nlev)      :: vgrad_T    ! v.grad(T)
  real (kind=real_kind), dimension(np,np)      :: Ephi       ! kinetic energy +PHI term
  real (kind=real_kind), dimension(np,np,2,nlev) :: grad_p
  real (kind=real_kind), dimension(np,np,2,nlev) :: grad_p_m_pmet  ! gradient(p- p_met)
  real (kind=real_kind), dimension(np,np,nlev)   :: vort       ! vorticity
  real (kind=real_kind), dimension(np,np,nlev)   :: p          ! pressure
  real (kind=real_kind), dimension(np,np,nlev)   :: rdp        ! inverse ofdelta pressure
  real (kind=real_kind), dimension(np,np,nlev)   :: T_vadv     ! temperaturevertical advection
  real (kind=real_kind), dimension(np,np,nlev)   :: vgrad_p    ! v.grad(p)
  real (kind=real_kind), dimension(np,np,nlev+1) :: ph         ! halflevel pressures on p-grid
  real (kind=real_kind), dimension(np,np,2,nlev) :: v_vadv   ! velocity verticaladvection
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
!  integer :: i,j,k,kptr,q

  integer :: i,j,k,ie,tl,ind
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

  print *, 'Main, np=', np

!loop 1, like in caar
  call tick(start)
  do ind = 1, loopmax
    do ie = nets, nete

#if ORIG

#else

    phi = elem(ie)%derived%phi(:,:,:)
    p(:,:,1)=hvcoord%hyai(1)*hvcoord%ps0 + ST( dXdX1XdpXn0Xie  ) /2
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
           ! T_v(i,j,k) = Virtual_Temperature1d( ST( iXjXkXtXn0Xie ),Qt)
           kappa_star(i,j,k) = kappa

           E = 0.5D0*( v1*v1 + v2*v2 )
           Ephi(i,j)=E+phi(i,j,k)+elem(ie)%derived%pecnd(i,j,k)
           vgrad_T(i,j,k) =  v1*vtemp1(i,j,1,k) + v2*vtemp1(i,j,2,k)
        end do
     end do

     vtemp2(:,:,:,k) = gradient_sphere(Ephi(:,:),deriv,elem(ie)%Dinv)

     elem(ie)%derived%vn0(:,:,:,k)=elem(ie)%derived%vn0(:,:,:,k)+eta_ave_w*vdp(:,:,:,k)
     divdp(:,:,k)=divergence_sphere(vdp(:,:,:,k),deriv,elem(ie))
     vort(:,:,k)=vorticity_v2( ST( dXdXkXuXn0Xie ) , ST( dXdXkXvXn0Xie ),deriv,elem(ie))

     elem(ie)%derived%omega_p(:,:,k) = &
          elem(ie)%derived%omega_p(:,:,k) + eta_ave_w*omega_p(:,:,k)

  enddo
#endif
  enddo !ie
  enddo !loopmax
  finish = tock(start)

  print '("caar timer, time = ",f10.4," seconds.")',finish
  print *, 'caar timer, raw time = ', finish

!loop 2, tracers...

end subroutine main_body
