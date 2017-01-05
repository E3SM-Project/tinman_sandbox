MESSAGE (STATUS "Building f90_vs_cxx_test (cxx_basic version)")

SET (TEST_SRCS
  main.cpp
  compute_and_apply_rhs.cpp
  data_structures.cpp
  sphere_operators.cpp
)

CONFIGURE_FILE( ${CMAKE_SOURCE_DIR}/compute_and_apply_rhs_test/config.h.in config.h)

INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR})

ADD_EXECUTABLE (compute_and_apply_rhs_cxx ${TEST_SRCS})

SET_TARGET_PROPERTIES (compute_and_apply_rhs_cxx PROPERTIES LINKER_LANGUAGE CXX)
