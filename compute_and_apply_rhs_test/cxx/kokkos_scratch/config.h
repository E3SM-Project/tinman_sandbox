#ifndef TINMAN_CONFIG_H
#define TINMAN_CONFIG_H

#include <Kokkos_Core.hpp>

namespace TinMan {

#ifdef CUDA_BUILD

// Until whenever CUDA supports constexpr properly
#define NP 4
#define NUM_LEV 72
#define NUM_LEV_P (NUM_LEV + 1)
#define QSIZE_D 1
#define NUM_TIME_LEVELS 3

#else

// For CPU, constexpr works, and it's better
static constexpr const int NP = 4;
static constexpr const int NUM_LEV = 72;
static constexpr const int NUM_LEV_P = NUM_LEV + 1;
static constexpr const int QSIZE_D = 1;
static constexpr const int NUM_TIME_LEVELS = 3;

#endif

} // namespace TinMan

#endif // TINMAN_CONFIG_H
