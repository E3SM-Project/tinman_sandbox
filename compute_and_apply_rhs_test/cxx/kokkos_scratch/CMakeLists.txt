INCLUDE_DIRECTORIES (${KOKKOS_PATH}/include)

SET(TEST_SRCS
  main.cpp
  compute_and_apply_rhs.cpp
  timer.cpp
  Region.cpp
  TestData.cpp
)

ADD_EXECUTABLE(kokkos-scratch ${TEST_SRCS})

IF(${CUDA_BUILD})
  TARGET_COMPILE_OPTIONS(kokkos-scratch PUBLIC -expt-extended-lambda)
ENDIF()

IF (KOKKOS_CMAKE_BUILD)
  SET(Kokkos_LIBRARIES "kokkoscore")
ELSE()
  SET(Kokkos_LIBRARIES "kokkos")
ENDIF()

TARGET_LINK_LIBRARIES(kokkos-scratch ${Kokkos_LIBRARIES} -L${KOKKOS_PATH}/lib)

IF(${USE_HWLOC})
  TARGET_LINK_LIBRARIES(kokkos-scratch hwloc numa)
ENDIF()

SET_TARGET_PROPERTIES(kokkos-scratch PROPERTIES LINKER_LANGUAGE CXX)
