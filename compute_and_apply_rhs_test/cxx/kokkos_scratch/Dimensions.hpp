#ifndef TINMAN_DIMENSIONS_HPP
#define TINMAN_DIMENSIONS_HPP

#include <Kokkos_Core.hpp>

namespace TinMan {

// Until whenever CUDA supports constexpr properly
#ifdef CUDA_BUILD

#define NP                4
#define NUM_LEV           4
#define NUM_LEV_P         (NUM_LEV + 1)
#define QSIZE_D           1
#define NUM_TIME_LEVELS   3
#define Q_NUM_TIME_LEVELS 2

#else

static constexpr int NP                = 4;
static constexpr int NUM_LEV           = 4;
static constexpr int NUM_LEV_P         = NUM_LEV + 1;
static constexpr int QSIZE_D           = 1;
static constexpr int NUM_TIME_LEVELS   = 3;
static constexpr int Q_NUM_TIME_LEVELS = 2;

#endif

} // namespace TinMan

#endif // TINMAN_DIMENSIONS_HPP
