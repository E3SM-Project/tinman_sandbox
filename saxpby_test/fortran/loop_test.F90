
MODULE loop_test

  IMPLICIT NONE

  CONTAINS

  SUBROUTINE saxpby(a, b, x, y)
    USE loop_bounds
    IMPLICIT NONE

    Integer :: i, j, k

    Real(kind=8), Intent(in) :: a, b

    Real(kind=8), Dimension(I3, I2, I1), Intent(inout) :: x, y

    !$omp parallel
    !$omp workshare
    x(:,:,:) = a*x(:,:,:) + b*y(:,:,:)
    !$omp end workshare
    !$omp end parallel

  END SUBROUTINE saxpby

END MODULE loop_test
