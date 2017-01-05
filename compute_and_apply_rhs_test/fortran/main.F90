function init_map(x,n) result(y)

  use kinds

  implicit none

  real, intent(in) :: x
  integer :: n
  real    :: y

  y = sin(n*x)
  y = y*y
  n = n+1

end function init_map

program main

use kinds
use dimensions_mod, only: np, nlev, qsize_d, nelemd
use element_mod
use routine_mod
use derivative_mod_base
use hybvcoord_mod
use results_mod

implicit none

type (element_t), allocatable  :: elem(:)
type (derivative_t) :: deriv

! init params

real (kind=real_kind) :: Dvv_init(np*np)
type (hvcoord_t)   :: hvcoord
integer :: nets, nete, nelem, np1,nm1,n0,qn0
real*8 :: dt2
real (kind=real_kind) :: eta_ave_w
real, parameter :: x = 0.123456789
real :: detD
real :: init_map

! local
integer :: i,j,k,ie,iq,it,n

!----------------- INIT

  np1 = 2
  nm1 = 3
  n0 = 1
  qn0 = 1
  dt2 = 1.0
  eta_ave_w = 1.0
  nets = 1
  nete = 3
  nelem = nelemd

  Dvv_init(1:16) = (/ -3.0,  -0.80901699437494745,   0.30901699437494745, &
 -0.50000000000000000 ,4.0450849718747373, 0.0, -1.1180339887498949, &
   1.5450849718747370, -1.5450849718747370, 1.1180339887498949, &
   0.0, -4.0450849718747373, 0.50, -0.30901699437494745, 0.80901699437494745, 3.0 /)

  do j =1 , np
   do i = 1, np
     deriv%Dvv(i,j) = Dvv_init((j-1)*np+i)
   enddo
  enddo

  allocate(elem(nelem))

  n = 1
  do ie = 1,nelem
    do i = 1, np
      do j =1 , np
!not nlev arrays

        elem(ie)%D(i,j,1,1) = init_map(x,n)
        elem(ie)%D(i,j,1,2) = init_map(x,n)
        elem(ie)%D(i,j,2,1) = init_map(x,n)
        elem(ie)%D(i,j,2,2) = init_map(x,n)

        detD = elem(ie)%D(i,j,1,1)*elem(ie)%D(i,j,2,2) -&
               elem(ie)%D(i,j,1,2)*elem(ie)%D(i,j,2,1)

        elem(ie)%Dinv(i,j,1,1) =  elem(ie)%D(i,j,2,2) / detD
        elem(ie)%Dinv(i,j,1,2) = -elem(ie)%D(i,j,1,2) / detD
        elem(ie)%Dinv(i,j,2,1) = -elem(ie)%D(i,j,2,1) / detD
        elem(ie)%Dinv(i,j,2,2) =  elem(ie)%D(i,j,1,1) / detD

        elem(ie)%fcor(i,j)     = init_map(x,n)
        elem(ie)%spheremp(i,j) = init_map(x,n)
        elem(ie)%metdet(i,j)   = init_map(x,n)
        elem(ie)%rmetdet(i,j)  = 1.0 / elem(ie)%metdet(i,j)

        elem(ie)%state%phis(i,j) = init_map(x,n)

        do k = 1,nlev
          elem(ie)%derived%omega_p(i,j,k) = init_map(x,n)
          elem(ie)%derived%pecnd(i,j,k)   = init_map(x,n)
          elem(ie)%derived%vn0(i,j,1,k)   = init_map(x,n)
          elem(ie)%derived%vn0(i,j,2,k)   = init_map(x,n)

          elem(ie)%derived%phi(i,j,k)     = init_map(x,n)

!only vapor
          do iq=1,qsize_d
            elem(ie)%state%Qdp(i,j,k,iq,1) = init_map(x,n)
            elem(ie)%state%Qdp(i,j,k,iq,2) = init_map(x,n)
          enddo

          do it=1,timelevels
            elem(ie)%state%dp3d(i,j,k,it) = init_map(x,n)

            elem(ie)%state%T(i,j,k,it)    = init_map(x,n)
            elem(ie)%state%v(i,j,1,k,it)  = init_map(x,n)
            elem(ie)%state%v(i,j,2,k,it)  = init_map(x,n)
          enddo

        enddo !k=1,nlev

        do k=1,nlev+1
          elem(ie)%derived%eta_dot_dpdn(i,j,k) = init_map(x,n)
        enddo
      enddo !j=1,np
    enddo !i=1,np
  enddo !ie=1,nelem


! init hybrid!!!!!!!!!!!!!!!!!!!!!!


!----------------- END OF INIT







print *, 'hey', np

call compute_and_apply_rhs(np1,nm1,n0,qn0, dt2,elem, hvcoord, deriv, nets,nete,eta_ave_w)

!do ie = 1,nelem
!print *, 'From element', ie, 'variable T =', elem(ie)%state%T(:,:,:,:)
!enddo

call compute_results_2norm (elem, np1)

end program main
