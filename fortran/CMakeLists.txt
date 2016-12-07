MESSAGE (STATUS "Building f90_vs_cxx_test (f90 version)")

SET (EXEC_MODULE_DIR "${CMAKE_CURRENT_BINARY_DIR}/modules")

SET (SOURCES
  coordinate_systems_mod.F90
  derivative_mod_base.F90
  element_mod.F90
  hybvcoord_mod.F90
  kinds.F90
  main.F90
  physical_constants.F90
  routine_mod.F90
)

CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/config.h.in config.h)

MESSAGE (STATUS "ABout to do sed")
# Next reformat the file as Fortran by appending comment lines with an exclamation mark
EXECUTE_PROCESS(COMMAND sed -i -e "s;^/;!/;g" config.h
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

INCLUDE_DIRECTORIES (${CMAKE_CURRENT_BINARY_DIR})

ADD_EXECUTABLE (f90_vs_cxx_test_f90 ${SOURCES})

SET_TARGET_PROPERTIES(f90_vs_cxx_test_f90 PROPERTIES Fortran_MODULE_DIRECTORY ${EXEC_MODULE_DIR})

SET_TARGET_PROPERTIES (f90_vs_cxx_test_f90 PROPERTIES LINKER_LANGUAGE Fortran)
