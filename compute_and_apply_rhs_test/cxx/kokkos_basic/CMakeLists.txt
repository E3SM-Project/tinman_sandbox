MESSAGE (STATUS "Building f90_vs_cxx_test (cxx_kokkos version)")

STRING (TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
IF (${CMAKE_BUILD_TYPE_UPPER} MATCHES "RELEASE")
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
ELSE()
  SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
ENDIF()

CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/Makefile.custom.in Makefile.custom)

# semi-hack, to forward 'all' to our makefile
ADD_CUSTOM_TARGET(kokkos COMMAND $(MAKE) -f Makefile.custom
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
SET_TARGET_PROPERTIES(kokkos PROPERTIES EXCLUDE_FROM_ALL FALSE)

# Hack, to forward 'clean' to our makefile
ADD_CUSTOM_TARGET(really-clean
                  COMMAND "$(MAKE)" clean
                  COMMAND "$(MAKE)" -f Makefile.custom clean WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
