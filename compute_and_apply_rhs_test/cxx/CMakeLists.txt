ADD_SUBDIRECTORY(basic)
ADD_SUBDIRECTORY(pointers_only)

OPTION (KOKKOS_CMAKE_BUILD, "Whether Kokkos was build with CMake. This is needed to get the right name for the libraries.\n")

ADD_SUBDIRECTORY(kokkos_basic)
ADD_SUBDIRECTORY(kokkos_scratch)
ADD_SUBDIRECTORY(tiled_vectorized_ppscan)
