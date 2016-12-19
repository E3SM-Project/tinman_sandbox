#include <config.h>

PROGRAM main

  USE loop_bounds
  USE loop_test

  IMPLICIT NONE

  Real(kind=8), Dimension(:, :, :), ALLOCATABLE :: x, y

  Integer :: i, j, k

  Integer :: clock_rate, clock_start, clock_end

  Integer :: nargs
  CHARACTER(len=32) :: arg

  nargs = iargc()

  if (nargs >= 1) then
    CALL GETARG(1, arg)
    READ(arg,*) I1
  else
    I1 = I1_MACRO
  endif

  ALLOCATE( X(I3,I2,I1) )
  ALLOCATE( Y(I3,I2,I1) )

  WRITE(*,*) I1, I2, I3

  CALL SYSTEM_CLOCK(clock_start, clock_rate)
  !$omp parallel do
  DO i=1,I1
    DO j=1,I2
      DO k=1,I3
        x(k,j,i) = i*j*k
        y(k,j,i) = i*i*j*j*k*k
      END DO
    END DO
  END DO
  !$omp end parallel do
  CALL SYSTEM_CLOCK(clock_end)
  print *, "Init: ", Real(clock_end-clock_start) / Real(clock_rate)

  CALL SYSTEM_CLOCK(clock_start)
  DO i=1,100
    CALL saxpby( 3.0_8, 5.0_8, x, y )
  END DO
  CALL SYSTEM_CLOCK(clock_end)
  print *, "SAXPBY: ", Real(clock_end-clock_start) / Real(clock_rate)

  DEALLOCATE( X )
  DEALLOCATE( Y )

END Program
