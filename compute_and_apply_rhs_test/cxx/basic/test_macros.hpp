#ifndef TEST_MACROS_HPP
#define TEST_MACROS_HPP

// ==================== ARRAY ACCESSING ================= //

// Access 2D array given last dimension and indices
#define AT_2D(pointer, i, j, dim2) \
   pointer[ dim2*i + j ]

// Access 3D array given last 2 dimensions and indices
#define AT_3D(pointer, i, j, k, dim2, dim3) \
   pointer[ dim2*dim3*i + dim3*j + k ]

// Access 4D array given last 3 dimensions and indices
#define AT_4D(pointer, i, j, k, l, dim2, dim3, dim4) \
   pointer[ dim2*dim3*dim4*i + dim3*dim4*j + dim4*k + l ]

// Access 5D array given last 4 dimensions and indices
#define AT_5D(pointer, i, j, k, l, m, dim2, dim3, dim4, dim5) \
   pointer[ dim2*dim3*dim4*dim5*i + dim3*dim4*dim5*j + dim4*dim5*k + dim5*l + m ]

// Access 6D array given last 5 dimensions and indices
#define AT_6D(pointer, i, j, k, l, m, n, dim2, dim3, dim4, dim5, dim6) \
   pointer[ dim2*dim3*dim4*dim5*dim6*i + dim3*dim4*dim5*dim6*j + dim4*dim5*dim6*k + dim5*dim6*l + m*dim6 + n ]

// =================== ARRAY SLICING ================== //

// Take slice of 3d array with given first index and last two dimensions
#define SLICE_3D(pointer, i, dim2, dim3) \
   &pointer[ i*dim2*dim3 ]

// Take slice of 4d array with given first index and last three dimensions
#define SLICE_4D(pointer, i, dim2, dim3, dim4) \
   &pointer[ i*dim2*dim3*dim4 ]

// Take slice of 5d array with given first index and last four dimensions
#define SLICE_5D(pointer, i, dim2, dim3, dim4, dim5) \
   &pointer[ i*dim2*dim3*dim4*dim5 ]

// Take slice of 5d array with given first two indices and last four dimensions
#define SLICE_5D_IJ(pointer, i, j, dim2, dim3, dim4, dim5) \
   &pointer[ i*dim2*dim3*dim4*dim5 + j*dim3*dim4*dim5 ]

// Take slice of 6d array with given first index and last five dimensions
#define SLICE_6D(pointer, i, dim2, dim3, dim4, dim5, dim6) \
   &pointer[ i*dim2*dim3*dim4*dim5*dim6 ]

// Take slice of 6d array with given first two indices and last five dimensions
#define SLICE_6D_IJ(pointer, i, j, dim2, dim3, dim4, dim5, dim6) \
   &pointer[ i*dim2*dim3*dim4*dim5*dim6 + j*dim3*dim4*dim5*dim6 ]

// Take slice of 6d array with given first three indices and last five dimensions
#define SLICE_6D_IJK(pointer, i, j, k, dim2, dim3, dim4, dim5, dim6) \
   &pointer[ i*dim2*dim3*dim4*dim5*dim6 + j*dim3*dim4*dim5*dim6 + k*dim4*dim5*dim6 ]

// ================= ARRAY TO POINTER ================ //

// Convert a 2D array into a 1D pointer
#define PTR_FROM_2D(array) (&array[0][0])

// Convert a 3D array into a 1D pointer
#define PTR_FROM_3D(array) (&array[0][0][0])

// Convert a 4D array into a 1D pointer
#define PTR_FROM_4D(array) (&array[0][0][0][0])

// Convert a 5D array into a 1D pointer
#define PTR_FROM_5D(array) (&array[0][0][0][0][0])

#endif // TEST_MACROS_HPP
