#include "config.h"

module dimensions_mod
  implicit none
  private

  integer, parameter, public :: np = NP

  integer, parameter, public :: qsize_d = QSIZE_D


  integer, parameter, public :: nlev=PLEV
  integer, parameter, public :: nlevp=nlev+1

  integer, parameter, public :: nelemd=10      ! number of elements per MPI task

  integer, public :: ne
  integer, public :: nelem       ! total number of elements
  integer, public :: nelemdmax   ! max number of elements on any MPI task
  integer, public, parameter :: ntrac = 1
  integer, public :: nPhysProc                          ! This is the number of physics processors/ per dynamics processor
  integer, public :: nnodes,npart,nmpi_per_node
  integer, public :: GlobalUniqueCols

contains


end module dimensions_mod

