MESSAGE (STATUS "Building f90_vs_cxx_test (f90 version)")

SET (EXEC_MODULE_DIR "${CMAKE_CURRENT_BINARY_DIR}/modules")

SET (SOURCES
  coordinate_systems_mod.F90
  derivative_mod_base.F90
  dimensions_mod.F90
  element_mod.F90
  element_state_mod.F90
  hybvcoord_mod.F90
  kinds.F90
  main.F90
  physical_constants.F90
  results_mod.F90
  routine_mod.F90
  routine_mod_ST.F90
  test_mod.F90
)

GET_FILENAME_COMPONENT(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
CONFIGURE_FILE(${PARENT_DIR}/config.h.in config.h)

EXECUTE_PROCESS(COMMAND sed -i -e "s;^/;!/;g" config.h
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR})

ADD_EXECUTABLE (compute_and_apply_rhs_cxx_f90 ${SOURCES})

SET_TARGET_PROPERTIES(compute_and_apply_rhs_cxx_f90 PROPERTIES Fortran_MODULE_DIRECTORY ${EXEC_MODULE_DIR})

SET_TARGET_PROPERTIES (compute_and_apply_rhs_cxx_f90 PROPERTIES LINKER_LANGUAGE Fortran)

SET (CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -DORIG=1 -DSTVER1")