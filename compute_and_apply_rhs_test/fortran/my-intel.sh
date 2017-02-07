#!/bin/bash   

rm orig
rm st-ver1
rm st-ver2

#original
ifort -vec-report=6 -openmp -DORIG=1 kinds.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod.F90 main-original.F90 -o orig

ifort -vec-report=6 -openmp -DORIG=0 -DSTVER1=1 -DSTVER2=0  kinds.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod_ST.F90 main.F90 -o st-ver1

#ifort -openmp -DORIG=0 -DSTVER1=0 -DSTVER2=1 config1.h config2.h kinds.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod_ST.F90 main.F90 -o st-ver2


# gfortran kinds.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 routine_mod.F90 routine_mod_ST.F90 main-original.F90
