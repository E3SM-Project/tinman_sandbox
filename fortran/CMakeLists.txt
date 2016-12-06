MESSAGE (STATUS "Building f90_vs_cxx_test (f90 version)")

SET (EXEC_MODULE_DIR "${CMAKE_CURRENT_BINARY_DIR}/modules")

SET (SOURCES
  coordinate_systems_mod.F90
  derivative_mod_base.F90
  dimensions_mod.F90
  element_mod.F90
  hybvcoord_mod.F90
  kinds.F90
  main.F90
  physical_constants.F90
  routine_extracted.F90
  routine_mod.F90
)

ADD_EXECUTABLE (f90_vs_cxx_test_f90 ${SOURCES})

SET_TARGET_PROPERTIES(f90_vs_cxx_test_f90 PROPERTIES Fortran_MODULE_DIRECTORY ${EXEC_MODULE_DIR})

SET_TARGET_PROPERTIES (f90_vs_cxx_test_f90 PROPERTIES LINKER_LANGUAGE Fortran)
