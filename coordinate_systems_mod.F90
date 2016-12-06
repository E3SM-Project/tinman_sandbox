

module coordinate_systems_mod


  use kinds, only : real_kind, longdouble_kind
  implicit none
  private

  real(kind=real_kind), public, parameter :: DIST_THRESHOLD= 1.0D-9
  real(kind=real_kind), parameter :: one=1.0D0, two=2.0D0

  type, public :: cartesian2D_t
     real(real_kind) :: x             ! x coordinate
     real(real_kind) :: y             ! y coordinate
  end type cartesian2D_t

  type, public :: cartesian3D_t
     real(real_kind) :: x             ! x coordinate
     real(real_kind) :: y             ! y coordinate
     real(real_kind) :: z             ! z coordinate
  end type cartesian3D_t

  type, public :: spherical_polar_t
     real(real_kind) :: r             ! radius
     real(real_kind) :: lon           ! longitude
     real(real_kind) :: lat           ! latitude
  end type spherical_polar_t




  ! ==========================================
  ! Public Interfaces
  ! ==========================================


contains


end module coordinate_systems_mod
