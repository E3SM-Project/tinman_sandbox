### Template configure script

rm -f CMakeCache.txt
rm -rf CMakeFiles

cmake \
  -D CMAKE_BUILD_TYPE:STRING=RELEASE \
  -D CMAKE_C_COMPILER:STRING=mpicc \
  -D CMAKE_CXX_COMPILER:STRING=mpicxx \
  -D CMAKE_Fortran_COMPILER:String=mpif90 \
  ../hommexx-forge-src
