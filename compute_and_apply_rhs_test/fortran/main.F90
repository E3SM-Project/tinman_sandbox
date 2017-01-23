
program main

use kinds
use element_mod
use routine_mod, only : compute_and_apply_rhs
use routine_mod_st, only : compute_and_apply_rhs_st
use derivative_mod_base
use hybvcoord_mod

implicit none

type (element_t), allocatable  :: elem(:)
real (kind=real_kind) :: ST(np,np,nlev,nelemd,numst,timelevels)
type (derivative_t) :: deriv

! init params

real (kind=real_kind) :: Dvv_init(np*np)
type (hvcoord_t)   :: hvcoord
integer :: nets, nete, nelem, np1,nm1,n0,qn0
real*8 :: dt2
real (kind=real_kind) :: eta_ave_w 
real (kind=real_kind) :: ii, jj, kk, iee

! local
integer :: i,j,k,ie,tl, ind

real (kind=real_kind) :: Ttest(np*np*nlev), Tt(np,np,nlev)

!----------------- INIT

  np1 = 2
  nm1 = 3
  n0 = 1
  qn0 = 1
  dt2 = 1.0
  eta_ave_w = 1.0
  nets = 1
  nete = nelemd


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

      elem(ie)%state%dp3d(i,j,k,1:timelevels) = 10*kk+iee+ii+jj + (/1,2,3/)
      elem(ie)%state%v(i,j,1,k,1:timelevels) = 1.0+kk/2+ii+jj+iee/5 + (/1,2,3/)*2.0
      elem(ie)%state%v(i,j,2,k,1:timelevels) = 1.0+kk/2+ii+jj+iee/5 + (/1,2,3/)*3.0
      elem(ie)%state%T(i,j,k,1:timelevels) = 1000-kk-ii-jj+iee/10 + (/1,2,3/)

!only vapor
      elem(ie)%state%Qdp(i,j,k,1,qn0) = 1.0+SIN(ii*jj*kk)

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

!----------------- for testing, T from element 3

Ttest = (/ &
   2000.6094651704889D0,        3997.1931528528748D0,        5989.8398982196022D0,        7978.3856103862945D0, &
       1998.6027685251026D0,        3993.1731054596098D0,        5983.8137529944115D0,        7970.3388653861866D0, &
       1996.6129767159553D0,        3989.1917322978047D0,        5977.8510410234785D0,        7962.3842417695851D0, &
       1994.6043241102957D0,        3985.1684208768665D0,        5971.8192687809105D0,        7954.3289175955097D0, &
       1998.6118224867689D0,        3993.1923384174470D0,        5983.8479943602579D0,        7970.3849563692938D0, &
       1996.6037569001460D0,        3989.1694128231252D0,        5977.8170094756688D0,        7962.3295593554531D0, &
       1994.6157077757948D0,        3985.1908740184062D0,        5971.8604368625674D0,        7954.3828923900028D0, &
       1992.6055695597358D0,       3981.1636178783515D0,        5965.8232560683764D0,        7946.3197588135590D0, &
       1996.6137309424748D0,        3989.1917428412585D0,        5977.8541842367322D0,        7962.3838101180809D0, &
       1994.6045282373668D0,        3985.1659859260858D0,        5971.8194872856820D0,       7954.3228311700886D0, &
       1992.6177900603641D0,        3981.1901527934219D0,        5965.8670182476853D0,        7946.3817285282630D0, &
       1990.6064812921757D0,        3977.1601611124538D0,        5959.8261093485226D0,        7938.3124587735329D0 /)

!------------- end of definign T for test

!------------- copying elem%state into ST array
!state vars in STATE: u,v,T,dp3d,ps_v = 5 vars + 35 tracers
do ie = 1, nelemd; do k=1, nlev; do j=1,np; do i=1,np
  ! u, v
  ST(i,j,k,ie,indu,1:timelevels) = elem(ie)%state%v(i,j,1,k,1:timelevels)
  ST(i,j,k,ie,indv,1:timelevels) = elem(ie)%state%v(i,j,2,k,1:timelevels)
  ! T
  ST(i,j,k,ie,indT,1:timelevels) = elem(ie)%state%T(i,j,k,1:timelevels)
  ! dp
  ST(i,j,k,ie,inddp,1:timelevels) = elem(ie)%state%dp3d(i,j,k,1:timelevels)
  ! vapor
  ST(i,j,k,ie,indvapor,1:timelevels) = elem(ie)%state%Qdp(i,j,k,1,1:timelevels)
enddo; enddo; enddo; enddo
! now, ps and phis are a different story, not a leveled var
do ie = 1, nelemd; do k=1, nlev; do j=1,np; do i=1,np
  ! ps
  ST(i,j,k,ie,indps,1:timelevels) = elem(ie)%state%ps_v(i,j,1:timelevels)
  ! phis
  ST(i,j,k,ie,indphis,1:timelevels) = elem(ie)%state%phis(i,j)
enddo; enddo; enddo; enddo

print *, 'hey', np

!np1 fields will be changed

!before:
!print *, 'T ORIGINAL BEFORE ', elem(3)%state%T(:,:,:,np1)
call compute_and_apply_rhs(np1,nm1,n0,qn0, dt2,elem, hvcoord, deriv, nets,nete, eta_ave_w, ST)
!print *, 'T ORIGINAL AFTER ', elem(3)%state%T(:,:,:,np1)

! ---------------- DO NOT MODIFY ------------------------
ie = 3
!do k = 1,nlev; do j = 1,np; do i = 1,np
!print *, elem(ie)%state%T(i,j,k,np1)
!enddo; enddo; enddo

ind = 1
do k = 1,nlev; do j = 1,np; do i = 1,np
Tt(i,j,k) = Ttest(ind)
ind = ind+1
enddo; enddo; enddo

print *, 'ORIGINAL diff', maxval(abs(Tt - elem(ie)%state%T(:,:,:,np1)))

call compute_and_apply_rhs_st(np1,nm1,n0,qn0, dt2,elem, hvcoord, deriv, nets,nete, eta_ave_w, ST)
print *, 'ST diff', maxval(abs(Tt - ST(:,:,:,ie,3,np1)))



end program main

