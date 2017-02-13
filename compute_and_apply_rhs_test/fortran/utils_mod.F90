module utils_mod

  use kinds, only : real_kind, np, nlev, nelemd, n0, np1, nm1

  implicit none

contains

  function compute_norm (field, total_length) result(norm)
    ! Note: use Kahan summation to maintain accuracy
    integer              , intent(in)  :: total_length
    real (kind=real_kind), intent(in)  :: field(total_length)
    real (kind=real_kind)              :: norm, temp, c, y

    integer :: i

    norm = 0
    c = 0
    y = 0

    do i=1,total_length
      y = field(i)**2 - c
      temp = norm + y
      c = (temp - norm) - y
      norm = temp
    enddo

    norm = sqrt(norm)

  end function compute_norm

  subroutine update_time_levels
   integer :: tmp

    tmp = nm1
    nm1 = n0
    n0  = np1
    np1 = nm1
  end subroutine update_time_levels

end module utils_mod
