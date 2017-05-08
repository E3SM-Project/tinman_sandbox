#ifndef DIMENSIONS_HPP
#define DIMENSIONS_HPP

#include <config.h>

namespace Homme {

extern int nelems;
constexpr const int np         = NP;
constexpr const int qsize_d    = QSIZE_D;
constexpr const int nlev       = PLEV;
constexpr const int nlevp      = nlev + 1;
constexpr const int timelevels = NUM_TIME_LEVELS;
constexpr const int q_num_time_levels = 2;

} // namespace Homme

#endif // DIMENSIONS_HPP
