module kinds

implicit none
public
  integer (kind=4), public, parameter::  &
  int_kind     = 4,                      &
  long_kind    = 8,                      &
  log_kind     = 4,                      &
  real_kind    = 8,                      &
  iulog        = 6,                      & ! stderr file handle
  longdouble_kind    = 8

  integer, public, parameter :: timelevels = 3
  integer, public, parameter :: np=4
  integer, public, parameter :: nlev=72
  integer, public, parameter :: nlevp=nlev+1
  integer, public, parameter :: qsize_d=1
  integer, public, parameter :: ntrac = 1
  integer, public            :: nelemd = 3
  integer, public, parameter :: npsq=np*np
  integer, public            :: np1 = 2
  integer, public            :: nm1 = 3
  integer, public            :: n0 = 1
  integer, public, parameter :: qn0 = 1
  integer, public, parameter :: numst = 41

  integer, public, parameter :: loopmax =  1  ! Do not go above ~10 if you update
                                              ! time levels, or will get NaN's

contains

subroutine tick(t)
    integer, intent(OUT) :: t

    call system_clock(t)
end subroutine tick

! returns time in seconds from now to time described by t
real function tock(t)
    integer, intent(in) :: t
    integer :: now, clock_rate

    call system_clock(now,clock_rate)

    tock = real(now - t)/real(clock_rate)
end function tock

end module kinds

