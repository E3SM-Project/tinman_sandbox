#!/bin/bash   

rm a.out

gfortran kinds.F90 coordinate_systems_mod.F90 element_mod.F90 physical_constants.F90 derivative_mod_base.F90 hybvcoord_mod.F90 routine_mod.F90 routine_mod_ST.F90 main.F90
