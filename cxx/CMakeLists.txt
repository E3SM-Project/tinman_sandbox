MESSAGE (STATUS "Building f90_vs_cxx_test (cxx version)")

SET (TEST_SRCS
  main.cpp
  compute_and_apply_rhs.cpp
  data_structures.cpp
  sphere_operators.cpp
)

INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR})

ADD_EXECUTABLE (f90_vs_cxx_test_cxx ${TEST_SRCS})

SET_TARGET_PROPERTIES (f90_vs_cxx_test_cxx PROPERTIES LINKER_LANGUAGE CXX)
