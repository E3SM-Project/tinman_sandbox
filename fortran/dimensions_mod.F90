module dimensions_mod
  implicit none
  private

  integer, parameter         :: qsize_d=1

  integer, parameter, public :: np = 4

  integer         :: ntrac = 0
  integer         :: qsize = 0


  integer, parameter, public :: nlev=3
  integer, parameter, public :: nlevp=nlev+1
  public :: qsize,qsize_d,ntrac_d,ntrac

  integer, public :: ne
  integer, public :: nelem       ! total number of elements
  integer, public :: nelemd      ! number of elements per MPI task
  integer, public :: nelemdmax   ! max number of elements on any MPI task
  integer, public :: nPhysProc                          ! This is the number of physics processors/ per dynamics processor
  integer, public :: nnodes,npart,nmpi_per_node
  integer, public :: GlobalUniqueCols

  public :: set_mesh_dimensions

contains


end module dimensions_mod

