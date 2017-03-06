module element_state_mod

  use kinds

  implicit none

  private


#if ORIG
  type, public :: elem_state_t

    ! prognostic variables for preqx solver

    ! prognostics must match those in prim_restart_mod.F90
    ! vertically-lagrangian code advects dp3d instead of ps_v
    ! tracers Q, Qdp always use 2 level time scheme

    real (kind=real_kind) :: v   (np,np,2,nlev,timelevels)            ! velocity                           1
    real (kind=real_kind) :: T   (np,np,nlev,timelevels)              ! temperature                        2
    real (kind=real_kind) :: dp3d(np,np,nlev,timelevels)              ! delta p on levels                  8
    real (kind=real_kind) :: ps_v(np,np,timelevels)                   ! surface pressure                   4
    real (kind=real_kind) :: phis(np,np)                              ! surface geopotential (prescribed)  5
    real (kind=real_kind) :: Q   (np,np,nlev,qsize_d)                 ! Tracer concentration               6
    real (kind=real_kind) :: Qdp (np,np,nlev,qsize_d,timelevels)      ! Tracer mass                        7

  end type elem_state_t
#endif


  contains
end module element_state_mod