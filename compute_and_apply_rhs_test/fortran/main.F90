
program main

use kinds
use element_mod
use routine_mod
use derivative_mod_base
use hybvcoord_mod

implicit none

type (element_t), allocatable  :: elem(:)
type (derivative_t) :: deriv

! init params

real (kind=real_kind) :: Dvv_init(np*np)
type (hvcoord_t)   :: hvcoord
integer :: nets, nete, nelem, np1,nm1,n0,qn0
real*8 :: dt2
real (kind=real_kind) :: eta_ave_w 

! local
integer :: i,j,k,ie

!----------------- INIT

  np1 = 2
  nm1 = 3
  n0 = 1
  qn0 = 1
  dt2 = 1.0
  eta_ave_w = 1.0
  nets = 1
  nete = 3
  nelem = nete-nets+1

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

  do ie = nets,nete
  do k = 1,nlev
   do j =1 , np
    do i = 1, np
!not nlev arrays
      elem(ie)%fcor(i,j) = 1.0
      elem(ie)%metdet(i,j) = 1.0
      elem(ie)%rmetdet(i,j) = 1.0
      elem(ie)%spheremp(i,j) = 1.0
!      elem(ie)%rspheremp(i,j) = 1.0 
      
      elem(ie)%derived%phi(i,j,k) = 1.0
      elem(ie)%derived%vn0(i,j,1:2,k) = 1.0
      elem(ie)%derived%pecnd(i,j,k) = 1.0
      elem(ie)%derived%omega_p(i,j,k) = 1.0

      elem(ie)%state%dp3d(i,j,k,1:timelevels) = 1.0
      elem(ie)%state%v(i,j,1:2,k,1:timelevels) = 1.0
      elem(ie)%state%T(i,j,k,1:timelevels) = 1.0

!only vapor
      elem(ie)%state%Qdp(i,j,k,1,qn0) = 1.0

      elem(ie)%Dinv(i,j,1,1) = 1.0
      elem(ie)%Dinv(i,j,1,2) = 0.0
      elem(ie)%Dinv(i,j,2,1) = 0.0
      elem(ie)%Dinv(i,j,2,2) = 1.0

      elem(ie)%D(i,j,1,1) = 1.0
      elem(ie)%D(i,j,1,2) = 0.0
      elem(ie)%D(i,j,2,1) = 0.0
      elem(ie)%D(i,j,2,2) = 1.0
    enddo
   enddo 
  enddo
  enddo

! init hybrid!!!!!!!!!!!!!!!!!!!!!!


!----------------- END OF INIT







print *, 'hey', np

call compute_and_apply_rhs(np1,nm1,n0,qn0, dt2,elem, hvcoord, deriv, nets,nete,eta_ave_w)

do ie = 1,nelem
print *, 'From element', ie, 'variable T =', elem(ie)%state%T(:,:,:,:)
enddo

end program main

