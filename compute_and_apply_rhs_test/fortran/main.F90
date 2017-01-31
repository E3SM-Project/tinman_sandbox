#include "config1.h"
#include "config2.h"

program main

use kinds
use element_state_mod
use element_mod
use routine_mod_st, only : compute_and_apply_rhs_st
use derivative_mod_base
use hybvcoord_mod
use test_mod

implicit none

type (element_t), allocatable  :: elem(:)

#if STVER1
! I J K IE ST TL
real (kind=real_kind) :: ST(np,np,nlev,nelemd,numst,timelevels)
#endif

#if STVER2
! I J K ST IE TL
real (kind=real_kind) :: ST(np,np,nlev,numst,nelemd,timelevels)
#endif

type (derivative_t) :: deriv

! init params

real (kind=real_kind) :: Dvv_init(np*np)
type (hvcoord_t)   :: hvcoord
integer :: nets, nete
real*8 :: dt2, start, finish
real (kind=real_kind) :: eta_ave_w 
real (kind=real_kind) :: ii, jj, kk, iee

! local
integer :: i,j,k,ie,tl,ind
integer, parameter :: loopmax = 10000

real (kind=real_kind) :: Tt(np,np,nlev)

print *, 'WOORLD'
!call flush(6)
!stop

  dt2 = 1.0
  eta_ave_w = 1.0
  nets = 1
  nete = nelemd
!----------------- INIT

  Dvv_init(1:16) = (/ -3.0,  -0.80901699437494745,   0.30901699437494745, &     
 -0.50000000000000000 ,4.0450849718747373, 0.0, -1.1180339887498949, &     
   1.5450849718747370, -1.5450849718747370, 1.1180339887498949, &     
   0.0, -4.0450849718747373, 0.50, -0.30901699437494745, 0.80901699437494745, 3.0 /)

  do j =1 , np
   do i = 1, np
     deriv%Dvv(i,j) = Dvv_init((j-1)*np+i)
   enddo
  enddo 
print *, nelemd


  allocate(elem(nelemd))
  ST = 0.0d0

  do ie = nets,nete
  do k = 1,nlev
   do j = 1,np
    do i = 1,np
      ii = i
      jj = j
      kk = k
      iee = ie
!not nlev arrays
      elem(ie)%fcor(i,j) = SIN(ii+jj) 
      elem(ie)%metdet(i,j) = ii*jj
      elem(ie)%rmetdet(i,j) = 1.0d0/elem(ie)%metdet(i,j)
      elem(ie)%spheremp(i,j) = 2*ii
      elem(ie)%rspheremp(i,j) = 1.0d0/elem(ie)%spheremp(i,j)
      
      elem(ie)%derived%phi(i,j,k) = COS(ii+3*jj)+kk
      elem(ie)%derived%vn0(i,j,1:2,k) = 1.0
      elem(ie)%derived%pecnd(i,j,k) = 1.0
      elem(ie)%derived%omega_p(i,j,k) = jj*jj

!      elem(ie)%state%dp3d(i,j,k,1:timelevels) = 10*kk+iee+ii+jj + (/1,2,3/)
!      elem(ie)%state%v(i,j,1,k,1:timelevels) = 1.0+kk/2+ii+jj+iee/5 + (/1,2,3/)*2.0
!      elem(ie)%state%v(i,j,2,k,1:timelevels) = 1.0+kk/2+ii+jj+iee/5 + (/1,2,3/)*3.0
!      elem(ie)%state%T(i,j,k,1:timelevels) = 1000-kk-ii-jj+iee/10 + (/1,2,3/)
!only vapor
!      elem(ie)%state%Qdp(i,j,k,1,qn0) = 1.0+SIN(ii*jj*kk)

!        ST( iXjXkXdpX1Xie ) = 10*kk+iee+ii+jj + 1
!        ST( iXjXkXdpX2Xie ) = 10*kk+iee+ii+jj + 2
!        ST( iXjXkXdpX3Xie ) = 10*kk+iee+ii+jj + 3

      ST( iXjXkXdpXdXie ) = 10*kk+iee+ii+jj + (/1,2,3/)
      ST( iXjXkXuXdXie ) = 1.0+kk/2+ii+jj+iee/5 + (/1,2,3/)*2.0
      ST( iXjXkXvXdXie ) = 1.0+kk/2+ii+jj+iee/5 + (/1,2,3/)*3.0
      ST( iXjXkXtXdXie ) = 1000-kk-ii-jj+iee/10 + (/1,2,3/)
      ST( iXjXkXqXdXie )= 1.0+SIN(ii*jj*kk)

      elem(ie)%Dinv(i,j,1,1) = 1.0
      elem(ie)%Dinv(i,j,1,2) = 0.0
      elem(ie)%Dinv(i,j,2,1) = 0.0
      elem(ie)%Dinv(i,j,2,2) = 0.5

      elem(ie)%D(i,j,1,1) = 1.0
      elem(ie)%D(i,j,1,2) = 0.0
      elem(ie)%D(i,j,2,1) = 0.0
      elem(ie)%D(i,j,2,2) = 2.0
    enddo
   enddo 
  enddo
  enddo

! Init hybrid
! Actually, most of hvcoord does not matter since only Lagrangian code is present.
! What about ps0?
  hvcoord%ps0 = 10.0
  hvcoord%nprlev = 1 ! does this matter?

  do k = 1, nlev + 1
    hvcoord%hyai(k) = nlev + 2 - k
    hvcoord%hybi(k) = k - 1
  enddo
  hvcoord%hyam(1:nlev) = (hvcoord%hyai(1:nlev) + hvcoord%hyai(2:nlev+1))/2.0
  hvcoord%hybm(1:nlev) = (hvcoord%hybi(1:nlev) + hvcoord%hybi(2:nlev+1))/2.0
  hvcoord%hybd(1:nlev) = (hvcoord%hybi(2:nlev+1) - hvcoord%hybi(1:nlev))/2.0

!----------------- END OF INIT


print *, 'Main ST routine', np

! ---------------- DO NOT MODIFY ------------------------
ie = 1
!do k = 1,nlev; do j = 1,np; do i = 1,np
!print *, elem(ie)%state%T(i,j,k,np1)
!enddo; enddo; enddo

ind = 1
do k = 1,nlev; do j = 1,np; do i = 1,np
Tt(i,j,k) = Ttest(ind)
ind = ind+1
enddo; enddo; enddo


call cpu_time(start)
do ind = 1, 1!loopmax
call compute_and_apply_rhs_st(np1,nm1,n0,qn0, dt2,elem, hvcoord, deriv, nets,nete, eta_ave_w, ST)
enddo
call cpu_time(finish)
print '("Time = ",f10.4," seconds.")',finish-start
print *, 'Raw time = ', finish-start

#if STVER1
print *, 'STVER1 diff', maxval(abs(Tt - ST( dXdXdXtXnp1Xie )))
#endif
#if STVER2
print *, 'STVER2 diff', maxval(abs(Tt - ST( dXdXdXtXnp1Xie )))
#endif


end program main

