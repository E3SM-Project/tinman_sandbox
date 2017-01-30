CMAKE_MINIMUM_REQUIRED(VERSION 2.8.5)

SET (CMAKE_BUILD_TYPE "RELEASE" CACHE STRING "Select build type.")

PROJECT (TinMan Fortran C CXX)

SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

IF("${KOKKOS_PATH}" STREQUAL "")
  MESSAGE(ABORT "No Kokkos installation specified, define KOKKOS_PATH")
ENDIF()

OPTION(CUDA_BUILD "Whether CUDA is being targetted or not")

IF(${CUDA_BUILD})
  MESSAGE("Building for CUDA")
ENDIF()

INCLUDE (SetCompilerFlags)

ADD_SUBDIRECTORY (compute_and_apply_rhs_test)
ADD_SUBDIRECTORY (saxpby_test)
