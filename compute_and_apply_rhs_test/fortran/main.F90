
program main

use kinds
use element_mod
use routine_mod
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

      elem(ie)%state%dp3d(i,j,k,1:timelevels) = 10*kk+iee+ii+jj
      elem(ie)%state%v(i,j,1:2,k,1:timelevels) = 1.0+kk/2+ii+jj+iee/5
      elem(ie)%state%T(i,j,k,1:timelevels) = 1000-kk-ii-jj+iee/10

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
Ttest = (/ 1994.6060414921399D0,     &
   3985.1944059666303D0,   5971.8288730432123D0, 7954.3873066989408D0, 1992.6023599782336D0,&
   3981.1805774651789D0,   5965.8119284451077D0, 7946.3528550804112D0, 1990.6093477237437D0,&
   3977.1927219435743D0,   5959.8395065266241D0, 7938.3853758170817D0, 1988.6037427888766D0,&
   3973.1755220801824D0,    5953.8169318880446D0,    7930.3424947341891D0,    1992.6078804174476D0,&
   3981.1935478349997D0,     5965.8354987738894D0,    7946.3864826888175D0,    1990.6031963942496D0,&
   3977.1774388985023D0,    5959.8147927500886D0,    7938.3449547062282D0,    1988.6115818353771D0,&
   3973.1918352429375D0,     5953.8474407528065D0,    7930.3839702448904D0,    1986.6048270374827D0,&
   3969.1714221867760D0,    5947.8205235312034D0,    7922.3345892940542D0,    1990.6094623943170D0,&
   3977.1928946132675D0,    5959.8408373068978D0,    7938.3852721680478D0,    1988.6038726151007D0,&
   3973.1744292809399D0,    5953.8170518575153D0,    7930.3389642544762D0,    1986.6133762856102D0,&
   3969.1910878050408D0,    5947.8532660252749D0,    7922.3827651758893D0,    1984.6056557594145D0,&
   3965.1683325125096D0,    5941.8231758430275D0,    7914.3280317362323D0 /)

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
call compute_and_apply_rhs(np1,nm1,n0,qn0, dt2,elem, hvcoord, deriv, nets,nete, eta_ave_w, ST)




! ---------------- DO NOT MODIFY ------------------------
ie = 3
!do k = 1,nlev; do j = 1,np; do i = 1,np
!print *, elem(ie)%state%T(i,j,k,np1)
!enddo; enddo; enddo

ind = 1
do k = 1,nlev; do j = 1,np; do i = 1,np
Tt(i,j,k) = Ttest(ind)
!print *, Tt(i,j,k), Ttest(ind)
ind = ind+1
enddo; enddo; enddo

print *, 'diff', maxval(abs(Tt - elem(ie)%state%T(:,:,:,np1)))

!do k = 1,nlev; do j = 1,np; do i = 1,np
!print *,i,j,k, Tt(i,j,k), elem(ie)%state%T(i,j,k,np1), '\n'
!enddo; enddo; enddo


!print *, 'difference = ', RESHAPE(elem(ie)%state%T,(/1,np*np*timelevels*nlev/))
!Ttest(1:np*np*nlev*timelevels) - 


end program main

