module results_mod

implicit none

contains
subroutine compute_results_2norm (elem,np1)

  use dimensions_mod, only: np, nlev
  use element_mod, only: element_t

  type(element_t) :: elem(:)

  real    :: vnorm, tnorm, dpnorm
  integer :: ie, ilev, i, j, nelems
  integer, intent(in) :: np1

  nelems = size(elem)

  vnorm = 0;
  tnorm = 0;
  dpnorm = 0;
  do ie=1,nelems
    do ilev=1,nlev
      do i=1,np
        do j=1,np
          vnorm  = vnorm  + elem(ie)%state%v(i,j,1,ilev,np1)**2
          vnorm  = vnorm  + elem(ie)%state%v(i,j,2,ilev,np1)**2
          tnorm  = tnorm  + elem(ie)%state%T(i,j,ilev,np1)**2
          dpnorm = dpnorm + elem(ie)%state%dp3d(i,j,ilev,np1)**2
        enddo
      enddo
    enddo
  enddo

  print *, "||v||_2 = ", SQRT(vnorm)
  print *, "||T||_2 = ", SQRT(tnorm)
  print *, "||dp||_2 = ", SQRT(dpnorm)
end subroutine

end module results_mod
