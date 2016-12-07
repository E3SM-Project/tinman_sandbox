
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

  integer(kind=int_kind),public,parameter::StateComponents=8! num prognistics variables (for prim_restart_mod.F90)

  !___________________________________________________________________
  type, public :: derived_state_t

    ! diagnostic variables for preqx solver

    ! storage for subcycling tracers/dynamics

    real (kind=real_kind) :: vn0  (np,np,2,nlev)                      ! velocity for SE tracer advection
    real (kind=real_kind) :: vstar(np,np,2,nlev)                      ! velocity on Lagrangian surfaces
    real (kind=real_kind) :: dpdiss_biharmonic(np,np,nlev)            ! mean dp dissipation tendency, if nu_p>0
    real (kind=real_kind) :: dpdiss_ave(np,np,nlev)                   ! mean dp used to compute psdiss_tens

    ! diagnostics for explicit timestep
    real (kind=real_kind) :: phi(np,np,nlev)                          ! geopotential
    real (kind=real_kind) :: omega_p(np,np,nlev)                      ! vertical tendency (derived)
    real (kind=real_kind) :: eta_dot_dpdn(np,np,nlevp)                ! mean vertical flux from dynamics

    ! tracer advection fields used for consistency and limiters
    real (kind=real_kind) :: dp(np,np,nlev)                           ! for dp_tracers at physics timestep
    real (kind=real_kind) :: divdp(np,np,nlev)                        ! divergence of dp
    real (kind=real_kind) :: divdp_proj(np,np,nlev)                   ! DSSed divdp

    real (kind=real_kind) :: pecnd(np,np,nlev)                        ! pressure perturbation from condensate

  end type derived_state_t
  
  !___________________________________________________________________
  type, public :: element_t
!     integer(kind=int_kind) :: LocalId
!     integer(kind=int_kind) :: GlobalId

     ! Coordinate values of element points
!     type (spherical_polar_t) :: spherep(np,np)                       ! Spherical coords of GLL points

     ! Equ-angular gnomonic projection coordinates
!     type (cartesian2D_t)     :: cartp(np,np)                         ! gnomonic coords of GLL points
!     type (cartesian2D_t)     :: corners(4)                           ! gnomonic coords of element corners

     ! 3D cartesian coordinates
     type (cartesian3D_t)     :: corners3D(4)

     ! Element diagnostics
!     real (kind=real_kind)    :: area                                 ! Area of element
!     real (kind=real_kind)    :: normDinv                             ! some type of norm of Dinv used for CFL
!     real (kind=real_kind)    :: dx_short                             ! short length scale in km
!     real (kind=real_kind)    :: dx_long                              ! long length scale in km

!     real (kind=real_kind)    :: variable_hyperviscosity(np,np)       ! hyperviscosity based on above
!     real (kind=real_kind)    :: hv_courant                           ! hyperviscosity courant number
!     real (kind=real_kind)    :: tensorVisc(np,np,2,2)                !og, matrix V for tensor viscosity

     type (elem_state_t)      :: state

     type (derived_state_t)   :: derived
!#if defined _PRIM 
!     type (elem_accum_t)       :: accum
!#endif
     ! Metric terms
     real (kind=real_kind)    :: met(np,np,2,2)                       ! metric tensor on velocity and pressure grid
     real (kind=real_kind)    :: metinv(np,np,2,2)                    ! metric tensor on velocity and pressure grid
     real (kind=real_kind)    :: metdet(np,np)                        ! g = SQRT(det(g_ij)) on velocity and pressure grid
     real (kind=real_kind)    :: rmetdet(np,np)                       ! 1/metdet on velocity pressure grid
     real (kind=real_kind)    :: D(np,np,2,2)                         ! Map covariant field on cube to vector field on the sphere
     real (kind=real_kind)    :: Dinv(np,np,2,2)                      ! Map vector field on the sphere to covariant v on cube



     ! Convert vector fields from spherical to rectangular components
     ! The transpose of this operation is its pseudoinverse.
!     real (kind=real_kind)    :: vec_sphere2cart(np,np,3,2)

     ! Mass matrix terms for an element on a cube face
!     real (kind=real_kind)    :: mp(np,np)                            ! mass matrix on v and p grid
!     real (kind=real_kind)    :: rmp(np,np)                           ! inverse mass matrix on v and p grid

     ! Mass matrix terms for an element on the sphere
     ! This mass matrix is used when solving the equations in weak form
     ! with the natural (surface area of the sphere) inner product
     real (kind=real_kind)    :: spheremp(np,np)                      ! mass matrix on v and p grid
     real (kind=real_kind)    :: rspheremp(np,np)                     ! inverse mass matrix on v and p grid

!     integer(kind=long_kind)  :: gdofP(np,np)                         ! global degree of freedom (P-grid)

     real (kind=real_kind)    :: fcor(np,np)                          ! Coreolis term

!     type (index_t) :: idxP
!     type (index_t),pointer :: idxV
!     integer :: FaceNum

  end type element_t

 
contains





end module element_mod
