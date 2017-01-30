INCLUDE_DIRECTORIES (${KOKKOS_PATH}/include)

SET(TEST_SRCS
  main.cpp
  compute_and_apply_rhs.cpp
  sphere_operators.cpp
  timer.cpp
  Region.cpp
  TestData.cpp
)

ADD_EXECUTABLE(kokkos-basic ${TEST_SRCS})

IF(${CUDA_BUILD})
  TARGET_COMPILE_OPTIONS(kokkos-basic PUBLIC -expt-extended-lambda)
ENDIF()

SET(Kokkos_LIBRARIES "kokkos")
TARGET_LINK_LIBRARIES(kokkos-basic ${Kokkos_LIBRARIES} -L${KOKKOS_PATH}/lib)

SET_TARGET_PROPERTIES(kokkos-basic PROPERTIES LINKER_LANGUAGE CXX)
