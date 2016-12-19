
module element_mod

  use coordinate_systems_mod, only: spherical_polar_t, cartesian2D_t, cartesian3D_t
  use kinds, only :   int_kind, long_kind, real_kind 

  implicit none
  private
  integer, public, parameter :: timelevels = 3


  integer, public, parameter :: np=4
  integer, public, parameter :: nlev=3
  integer, public, parameter :: nlevp=nlev+1
  integer, public, parameter :: qsize_d=1
  integer, public, parameter :: ntrac = 1
  integer, public, parameter :: nelemd = 1


!  real (kind=real_kind), allocatable, target, public :: state_Qdp                (:,:,:,:,:,:)    ! (np,np,nlev,qsize_d,2,nelemd)   
!  real (kind=real_kind), allocatable, target, public :: derived_vn0              (:,:,:,:,:)      ! (np,np,2,nlev,nelemd)                   velocity for SE tracer advection
!  real (kind=real_kind), allocatable, target, public :: derived_divdp            (:,:,:,:)        ! (np,np,nlev,nelemd)                     divergence of dp
!  real (kind=real_kind), allocatable, target, public :: derived_divdp_proj       (:,:,:,:)        ! (np,np,nlev,nelemd)                     DSSed divdp


! =========== PRIMITIVE-EQUATION DATA-STRUCTURES =====================

  type, public :: elem_state_t

    ! prognostic variables for preqx solver

    ! prognostics must match those in prim_restart_mod.F90
    ! vertically-lagrangian code advects dp3d instead of ps_v
    ! tracers Q, Qdp always use 2 level time scheme

    real (kind=real_kind) :: v   (np,np,2,nlev,timelevels)            ! velocity                           1
    real (kind=real_kind) :: T   (np,np,nlev,timelevels)              ! temperature                        2
    real (kind=real_kind) :: dp3d(np,np,nlev,timelevels)              ! delta p on levels                  8
    real (kind=real_kind) :: phis(np,np)                              ! surface geopotential (prescribed)  5
    real (kind=real_kind) :: Qdp (np,np,nlev,qsize_d,2)               ! Tracer mass                        7

  end type elem_state_t

  !___________________________________________________________________
  type, public :: derived_state_t

    ! diagnostic variables for preqx solver

    ! storage for subcycling tracers/dynamics

    real (kind=real_kind) :: vn0  (np,np,2,nlev)                      ! velocity for SE tracer advection

    ! diagnostics for explicit timestep
    real (kind=real_kind) :: phi(np,np,nlev)                          ! geopotential
    real (kind=real_kind) :: omega_p(np,np,nlev)                      ! vertical tendency (derived)
    real (kind=real_kind) :: eta_dot_dpdn(np,np,nlevp)                ! mean vertical flux from dynamics

    ! tracer advection fields used for consistency and limiters
    real (kind=real_kind) :: dp(np,np,nlev)                           ! for dp_tracers at physics timestep

    real (kind=real_kind) :: pecnd(np,np,nlev)                        ! pressure perturbation from condensate

  end type derived_state_t
  
  !___________________________________________________________________
  type, public :: element_t

     type (elem_state_t)      :: state

     type (derived_state_t)   :: derived

     ! Metric terms
     real (kind=real_kind)    :: metdet(np,np)                        ! g = SQRT(det(g_ij)) on velocity and pressure grid
     real (kind=real_kind)    :: rmetdet(np,np)                       ! 1/metdet on velocity pressure grid
     real (kind=real_kind)    :: D(np,np,2,2)                         ! Map covariant field on cube to vector field on the sphere
     real (kind=real_kind)    :: Dinv(np,np,2,2)                      ! Map vector field on the sphere to covariant v on cube

     real (kind=real_kind)    :: spheremp(np,np)                      ! mass matrix on v and p grid
     real (kind=real_kind)    :: fcor(np,np)                          ! Coreolis term

  end type element_t

 
contains





end module element_mod
