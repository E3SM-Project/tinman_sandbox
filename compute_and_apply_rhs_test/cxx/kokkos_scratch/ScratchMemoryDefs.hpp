#ifndef TINMAN_SCRATCH_MEMORY_DEFS_HPP
#define TINMAN_SCRATCH_MEMORY_DEFS_HPP

#include "Types.hpp"
#include "ScratchManager.hpp"

namespace TinMan
{

// Team views
constexpr int ID_3D_SCALAR   = 0 ;
constexpr int ID_3D_VECTOR   = 1 ;
constexpr int ID_3D_P_SCALAR = 2;

// Thread views
constexpr int ID_2D_SCALAR = 0;
constexpr int ID_2D_VECTOR = 1;

namespace ScratchMemoryDefs
{

// Compute the amount of scratch memory needed
constexpr size_t size_2d_scalar   = NP*NP;
constexpr size_t size_2d_vector   = 2*NP*NP;
constexpr size_t size_3d_scalar   = NUM_LEV*NP*NP;
constexpr size_t size_3d_vector   = NUM_LEV*2*NP*NP;
constexpr size_t size_3d_p_scalar = NUM_LEV_P*NP*NP;

constexpr size_t num_2d_scalars   = 2;
constexpr size_t num_2d_vectors   = 1;
constexpr size_t num_3d_scalars   = 8;
constexpr size_t num_3d_vectors   = 2;
constexpr size_t num_3d_p_scalars = 1;

constexpr int team_mem_needed = ( num_3d_scalars   * size_3d_scalar
                                + num_3d_vectors   * size_3d_vector
                                + num_3d_p_scalars * size_3d_p_scalar) * sizeof(Real);

constexpr int thread_mem_needed = ( num_2d_scalars   * size_2d_scalar
                                  + num_2d_vectors   * size_2d_vector ) * sizeof(Real);

typedef CountAndSizePack<num_3d_scalars  , size_3d_scalar,
                         num_3d_vectors  , size_3d_vector,
                         num_3d_p_scalars, size_3d_p_scalar
                      > TeamPack;

typedef CountAndSizePack<num_2d_scalars, size_2d_scalar,
                         num_2d_vectors, size_2d_vector
                        > ThreadPack;

using CAARS_ScratchManager = ScratchManager<TeamPack, ThreadPack>;

} // namespace ScratchMemoryDefs

} // namespace TinMan

#endif // TINMAN_SCRATCH_MEMORY_DEFS_HPP
