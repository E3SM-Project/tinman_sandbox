#!/bin/bash   

rm orig
rm s1 s2 s3 s4
rm s1omp s2omp s3omp s4omp 
rm fs1omp fs2omp fs3omp fs4omp

export fl="-fp-model fast -ftz -O3"

#DHOMP -s for omp
#original --------------------------------------------------------
ifort $fl -DHOMP=0 -DORIG=1 kinds.F90 utils_mod.F90  coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod.F90 main.F90 -o orig

#original --------------------------------------------------------
ifort $fl -fopenmp -DHOMP=1 -DORIG=1 kinds.F90 utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod.F90 main.F90 -o origomp


# all versions not fused, no omp ---------------------------------
#ifort $fl -DHOMP=0 -DORIG=0 -DSTVER1=1 -DSTVER2=0 -DSTVER3=0 -DSTVER4=0 kinds.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod_ST.F90 main.F90 -o s1

#ifort $fl -DHOMP=0 -DORIG=0 -DSTVER1=0 -DSTVER2=1 -DSTVER3=0 -DSTVER4=0 kinds.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod_ST.F90 main.F90 -o s2

#ifort $fl -DHOMP=0 -DORIG=0 -DSTVER1=0 -DSTVER2=0 -DSTVER3=1 -DSTVER4=0 kinds.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod_ST.F90 main.F90 -o s3

#ifort $fl -DHOMP=0 -DORIG=0 -DSTVER1=0 -DSTVER2=0 -DSTVER3=0 -DSTVER4=1 kinds.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod_ST.F90 main.F90 -o s4

# all versions not fused, omp ---------------------------------
ifort  $fl -fopenmp -DHOMP=1 -DORIG=0 -DSTVER1=1 -DSTVER2=0 -DSTVER3=0 -DSTVER4=0 kinds.F90 utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod_ST.F90 main.F90 -o s1omp

ifort $fl -fopenmp -DHOMP=1 -DORIG=0 -DSTVER1=0 -DSTVER2=1 -DSTVER3=0 -DSTVER4=0 kinds.F90 utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod_ST.F90 main.F90 -o s2omp

ifort $fl -fopenmp -DHOMP=1 -DORIG=0 -DSTVER1=0 -DSTVER2=0 -DSTVER3=1 -DSTVER4=0 kinds.F90 utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod_ST.F90 main.F90 -o s3omp

ifort $fl -fopenmp -DHOMP=1 -DORIG=0 -DSTVER1=0 -DSTVER2=0 -DSTVER3=0 -DSTVER4=1 kinds.F90 utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_mod_ST.F90 main.F90 -o s4omp


# all versions fused, omp ---------------------------------
ifort $fl -fopenmp -DHOMP=1 -DORIG=0 -DSTVER1=1 -DSTVER2=0 -DSTVER3=0 -DSTVER4=0 kinds.F90 utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_st_fused.F90 main.F90 -o fs1omp

ifort $fl -fopenmp -DHOMP=1 -DORIG=0 -DSTVER1=0 -DSTVER2=1 -DSTVER3=0 -DSTVER4=0 kinds.F90 utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_st_fused.F90 main.F90 -o fs2omp

ifort $fl -fopenmp -DHOMP=1 -DORIG=0 -DSTVER1=0 -DSTVER2=0 -DSTVER3=1 -DSTVER4=0  kinds.F90 utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_st_fused.F90 main.F90 -o fs3omp

ifort $fl -fopenmp -DHOMP=1 -DORIG=0 -DSTVER1=0 -DSTVER2=0 -DSTVER3=0 -DSTVER4=1  kinds.F90 utils_mod.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 test_mod.F90 routine_st_fused.F90 main.F90 -o fs4omp



# gfortran kinds.F90 coordinate_systems_mod.F90 element_state_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 routine_mod.F90 routine_mod_ST.F90 main-original.F90
