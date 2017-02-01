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
  integer, public, parameter :: nelemd = 3
  integer, public, parameter :: npsq=np*np
  integer, public, parameter :: np1 = 2
  integer, public, parameter :: nm1 = 3
  integer, public, parameter :: n0 = 1
  integer, public, parameter :: qn0 = 1
  integer, public, parameter :: numst = 41

end module kinds

