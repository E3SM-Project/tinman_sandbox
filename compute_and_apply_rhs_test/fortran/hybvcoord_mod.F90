
module hybvcoord_mod
use kinds,              only: r8 => real_kind, iulog
use kinds,              only: plev => nlev
use physical_constants, only: p0

implicit none
private

!----------------------------------------------------------------------- 
! hvcoord_t: Hybrid level definitions: p = a*p0 + b*ps
!            interfaces   p(k) = hyai(k)*ps0 + hybi(k)*ps
!            midpoints    p(k) = hyam(k)*ps0 + hybm(k)*ps
!-----------------------------------------------------------------------
type, public :: hvcoord_t
  real(r8) ps0          ! base state surface-pressure for level definitions
  real(r8) hyai(plev+1)  ! ps0 component of hybrid coordinate - interfaces
  real(r8) hyam(plev)   ! ps0 component of hybrid coordinate - midpoints
  real(r8) hybi(plev+1)  ! ps  component of hybrid coordinate - interfaces
  real(r8) hybm(plev)   ! ps  component of hybrid coordinate - midpoints
  real(r8) hybd(plev)   ! difference in b (hybi) across layers
  real(r8) prsfac       ! log pressure extrapolation factor (time, space independent)
  real(r8) etam(plev)   ! eta-levels at midpoints
  real(r8) etai(plev+1)  ! eta-levels at interfaces
  integer  nprlev       ! number of pure pressure levels at top  
  integer  pad
end type

contains

end module hybvcoord_mod
