MESSAGE (STATUS "Building f90_vs_cxx_test (kokkos version)")

INCLUDE_DIRECTORIES (~/prefix/include)

SET(TEST_SRCS
  main.cpp
  compute_and_apply_rhs.cpp
  sphere_operators.cpp
  TestData.cpp
  Region.cpp
)

ADD_EXECUTABLE(compute_and_apply_rhs_kokkos ${TEST_SRCS})

TARGET_COMPILE_OPTIONS(compute_and_apply_rhs_kokkos PUBLIC -expt-extended-lambda)

SET(Kokkos_LIBRARIES "kokkos")
TARGET_LINK_LIBRARIES(compute_and_apply_rhs_kokkos ${Kokkos_LIBRARIES} -L/home/mdeakin/prefix/lib)

SET_TARGET_PROPERTIES(compute_and_apply_rhs_kokkos PROPERTIES LINKER_LANGUAGE CXX)
