module derivative_mod_base

  use kinds
  use element_state_mod
  use element_mod
  use physical_constants, only : rrearth

implicit none
private

  type, public :: derivative_t
     real (kind=real_kind) :: Dvv(np,np)
  end type derivative_t

  public  :: gradient_sphere
  public  :: vorticity_sphere
  public  :: vorticity_v2
  public  :: divergence_sphere

contains

!----------------------------------------------------------------

!DIR$ ATTRIBUTES FORCEINLINE :: gradient_sphere
  function gradient_sphere(s,deriv,Dinv) result(ds)
!
!   input s:  scalar
!   output  ds: spherical gradient of s, lat-lon coordinates
!

    type (derivative_t), intent(in) :: deriv
    real(kind=real_kind), intent(in), dimension(np,np,2,2) :: Dinv
    real(kind=real_kind), intent(in) :: s(np,np)

    real(kind=real_kind) :: ds(np,np,2)

    integer i
    integer j
    integer l

    real(kind=real_kind) ::  dsdx00, dsdy00
    real(kind=real_kind) ::  v1(np,np),v2(np,np)

    do j=1,np
       do l=1,np
          dsdx00=0.0d0
          dsdy00=0.0d0
!DIR$ UNROLL(NP)
          do i=1,np
             dsdx00 = dsdx00 + deriv%Dvv(i,l  )*s(i,j  )
             dsdy00 = dsdy00 + deriv%Dvv(i,l  )*s(j  ,i)
          end do
          v1(l  ,j  ) = dsdx00*rrearth
          v2(j  ,l  ) = dsdy00*rrearth
       end do
    end do
    ! convert covarient to latlon
    do j=1,np
       do i=1,np
          ds(i,j,1)=Dinv(i,j,1,1)*v1(i,j) + Dinv(i,j,2,1)*v2(i,j)
          ds(i,j,2)=Dinv(i,j,1,2)*v1(i,j) + Dinv(i,j,2,2)*v2(i,j)
       enddo
    enddo

    end function gradient_sphere




!DIR$ ATTRIBUTES FORCEINLINE :: vorticity_sphere
  function vorticity_sphere(v,deriv,elem) result(vort)
!
!   input:  v = velocity in lat-lon coordinates
!   ouput:  spherical vorticity of v
!

    type (derivative_t), intent(in) :: deriv
    type (element_t), intent(in) :: elem
    real(kind=real_kind), intent(in) :: v(np,np,2)

    real(kind=real_kind) :: vort(np,np)

    integer i
    integer j
    integer l

    real(kind=real_kind) ::  dvdx00,dudy00
    real(kind=real_kind) ::  vco(np,np,2)
    real(kind=real_kind) ::  vtemp(np,np)

    ! convert to covariant form
    do j=1,np
       do i=1,np
          vco(i,j,1)=(elem%D(i,j,1,1)*v(i,j,1) + elem%D(i,j,2,1)*v(i,j,2))
          vco(i,j,2)=(elem%D(i,j,1,2)*v(i,j,1) + elem%D(i,j,2,2)*v(i,j,2))
       enddo
    enddo

    do j=1,np
       do l=1,np

          dudy00=0.0d0
          dvdx00=0.0d0

!DIR$ UNROLL(NP)
          do i=1,np
             dvdx00 = dvdx00 + deriv%Dvv(i,l  )*vco(i,j  ,2)
             dudy00 = dudy00 + deriv%Dvv(i,l  )*vco(j  ,i,1)
          enddo

          vort(l  ,j  ) = dvdx00
          vtemp(j  ,l  ) = dudy00
       enddo
    enddo

    do j=1,np
       do i=1,np
          vort(i,j)=(vort(i,j)-vtemp(i,j))*(elem%rmetdet(i,j)*rrearth)
       end do
    end do

  end function vorticity_sphere


! separate velocity input
!DIR$ ATTRIBUTES FORCEINLINE :: vorticity_v2
  function vorticity_v2(u,v,deriv,elem) result(vort)
!
!   input:  v = velocity in lat-lon coordinates
!   ouput:  spherical vorticity of v
!
    type (derivative_t), intent(in) :: deriv
    type (element_t), intent(in) :: elem
    real(kind=real_kind), intent(in) :: u(np,np),v(np,np)

    real(kind=real_kind) :: vort(np,np)

    integer i
    integer j
    integer l
    
    real(kind=real_kind) ::  dvdx00,dudy00
    real(kind=real_kind) ::  vco(np,np,2)
    real(kind=real_kind) ::  vtemp(np,np)

    ! convert to covariant form
    do j=1,np
       do i=1,np
          vco(i,j,1)=(elem%D(i,j,1,1)*v(i,j) + elem%D(i,j,2,1)*u(i,j))
          vco(i,j,2)=(elem%D(i,j,1,2)*v(i,j) + elem%D(i,j,2,2)*u(i,j))
       enddo
    enddo

    do j=1,np
       do l=1,np

          dudy00=0.0d0
          dvdx00=0.0d0

!DIR$ UNROLL(NP)
          do i=1,np
             dvdx00 = dvdx00 + deriv%Dvv(i,l  )*vco(i,j  ,2)
             dudy00 = dudy00 + deriv%Dvv(i,l  )*vco(j  ,i,1)
          enddo
 
          vort(l  ,j  ) = dvdx00
          vtemp(j  ,l  ) = dudy00
       enddo
    enddo

    do j=1,np
       do i=1,np
          vort(i,j)=(vort(i,j)-vtemp(i,j))*(elem%rmetdet(i,j)*rrearth)
       end do
    end do

  end function vorticity_v2



!DIR$ ATTRIBUTES FORCEINLINE :: divergence_sphere
  function divergence_sphere(v,deriv,elem) result(div)
!
!   input:  v = velocity in lat-lon coordinates
!   ouput:  div(v)  spherical divergence of v
!


    real(kind=real_kind), intent(in) :: v(np,np,2)  ! in lat-lon coordinates
    type (derivative_t), intent(in) :: deriv
    type (element_t), intent(in) :: elem
    real(kind=real_kind) :: div(np,np)

    ! Local

    integer i
    integer j
    integer l

    real(kind=real_kind) ::  dudx00
    real(kind=real_kind) ::  dvdy00
    real(kind=real_kind) ::  gv(np,np,2),vvtemp(np,np)

    ! convert to contra variant form and multiply by g
    do j=1,np
       do i=1,np
          gv(i,j,1)=elem%metdet(i,j)*(elem%Dinv(i,j,1,1)*v(i,j,1) + elem%Dinv(i,j,1,2)*v(i,j,2))
          gv(i,j,2)=elem%metdet(i,j)*(elem%Dinv(i,j,2,1)*v(i,j,1) + elem%Dinv(i,j,2,2)*v(i,j,2))
       enddo
    enddo

    ! compute d/dx and d/dy
    do j=1,np
       do l=1,np
          dudx00=0.0d0
          dvdy00=0.0d0
!DIR$ UNROLL(NP)
          do i=1,np
             dudx00 = dudx00 + deriv%Dvv(i,l  )*gv(i,j  ,1)
             dvdy00 = dvdy00 + deriv%Dvv(i,l  )*gv(j  ,i,2)
          end do
          div(l  ,j  ) = dudx00
          vvtemp(j  ,l  ) = dvdy00
       end do
    end do

!dir$ simd
    div(:,:)=(div(:,:)+vvtemp(:,:))*(elem%rmetdet(:,:)*rrearth)

  end function divergence_sphere


end module derivative_mod_base
