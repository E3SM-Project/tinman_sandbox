CMAKE_MINIMUM_REQUIRED(VERSION 2.8.5)

PROJECT (HommexxForge Fortran C CXX)

SET (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

INCLUDE (SetCompilerFlags)

SET (NUM_ELEMENTS 100 CACHE INT "Number of cells in each element")
SET (NUM_CELLS      4 CACHE INT "Number of cells in each element")
SET (NUM_POINTS     4 CACHE INT "Number of gauss points in the element")
SET (NUM_PLEV       4 CACHE INT "Number of vertical levels")
SET (Q_SIZE_D       4 CACHE INT "Have no idea what this number is...")

ADD_SUBDIRECTORY (fortran)
ADD_SUBDIRECTORY (cxx)
