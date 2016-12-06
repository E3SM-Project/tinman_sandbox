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

end module kinds

