#!/bin/bash   

rm orig
rm s1 s2 s3 s4
rm s1omp s2omp s3omp s4omp 
rm fs1omp fs2omp fs3omp fs4omp

#DHOMP -s for omp
#original --------------------------------------------------------
gfortran -DHOMP=0 -DORIG=1 kinds.F90 utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod.F90 main.F90 -o orig

#original --------------------------------------------------------
gfortran -fopenmp -DHOMP=1 -DORIG=1 kinds.F90 utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod.F90 main.F90 -o origomp


# all versions not fused, no omp ---------------------------------
gfortran -DHOMP=0 -DORIG=0 -DSTVER1=1 -DSTVER2=0 -DSTVER3=0 -DSTVER4=0 config1.h config2.h config3.h config4.h kinds.F90  utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod_ST.F90 main.F90 -o s1

gfortran -DHOMP=0 -DORIG=0 -DSTVER1=0 -DSTVER2=1 -DSTVER3=0 -DSTVER4=0 config1.h config2.h config3.h config4.h kinds.F90  utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod_ST.F90 main.F90 -o s2

gfortran -DHOMP=0 -DORIG=0 -DSTVER1=0 -DSTVER2=0 -DSTVER3=1 -DSTVER4=0 config1.h config2.h config3.h config4.h kinds.F90  utils_mod.F90  coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod_ST.F90 main.F90 -o s3

gfortran -DHOMP=0 -DORIG=0 -DSTVER1=0 -DSTVER2=0 -DSTVER3=0 -DSTVER4=1 config1.h config2.h config3.h config4.h kinds.F90  utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod_ST.F90 main.F90 -o s4

# all versions not fused, omp ---------------------------------
gfortran -fopenmp -DHOMP=1 -DORIG=0 -DSTVER1=1 -DSTVER2=0 -DSTVER3=0 -DSTVER4=0 config1.h config2.h config3.h config4.h kinds.F90  utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod_ST.F90 main.F90 -o s1omp

gfortran -fopenmp -DHOMP=1 -DORIG=0 -DSTVER1=0 -DSTVER2=1 -DSTVER3=0 -DSTVER4=0 config1.h config2.h config3.h config4.h kinds.F90 utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod_ST.F90 main.F90 -o s2omp

gfortran -fopenmp -DHOMP=1 -DORIG=0 -DSTVER1=0 -DSTVER2=0 -DSTVER3=1 -DSTVER4=0 config1.h config2.h config3.h config4.h kinds.F90  utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod_ST.F90 main.F90 -o s3omp

gfortran -fopenmp -DHOMP=1 -DORIG=0 -DSTVER1=0 -DSTVER2=0 -DSTVER3=0 -DSTVER4=1 config1.h config2.h config3.h config4.h kinds.F90  utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod_ST.F90 main.F90 -o s4omp


# all versions fused, omp ---------------------------------
gfortran -fopenmp -DHOMP=1 -DORIG=0 -DSTVER1=1 -DSTVER2=0 -DSTVER3=0 -DSTVER4=0 config1.h config2.h config3.h config4.h kinds.F90  utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_st_fused.F90 main.F90 -o fs1omp

gfortran -fopenmp -DHOMP=1 -DORIG=0 -DSTVER1=0 -DSTVER2=1 -DSTVER3=0 -DSTVER4=0 config1.h config2.h config3.h config4.h kinds.F90  utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_st_fused.F90 main.F90 -o fs2omp

gfortran -fopenmp -DHOMP=1 -DORIG=0 -DSTVER1=0 -DSTVER2=0 -DSTVER3=1 -DSTVER4=0 config1.h config2.h config3.h config4.h kinds.F90 utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_st_fused.F90 main.F90 -o fs3omp

gfortran -fopenmp -DHOMP=1 -DORIG=0 -DSTVER1=0 -DSTVER2=0 -DSTVER3=0 -DSTVER4=1 config1.h config2.h config3.h config4.h kinds.F90 utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_st_fused.F90 main.F90 -o fs4omp



# gfortran kinds.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 routine_mod.F90 routine_mod_ST.F90 main-original.F90
