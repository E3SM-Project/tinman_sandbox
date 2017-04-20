#ifndef COMPUTE_AND_APPLY_RHS_HPP
#define COMPUTE_AND_APPLY_RHS_HPP

#include "Types.hpp"

namespace TinMan {

// Forward declarations
class Control;
class Region;

void compute_and_apply_rhs(const Control &data, Region &region,
                           int threads_per_team, int vectors_per_thread);

void print_results_2norm(const Control &data, const Region &region);

// void dump_results_to_file (const Control& data, const Region& region);

template <typename RealViewType> Real compute_norm(const RealViewType view) {
  size_t length = view.size();
  typename RealViewType::pointer_type data = view.data();

  // Note: use Kahan algorithm to maintain accuracy
  Real norm = 0;
  Real c = 0;
  Real temp, y;
  for (size_t i = 0; i < length; ++i) {
    y = data[i] * data[i] - c;
    temp = norm + y;
    c = (temp - norm) - y;
    norm = temp;
  }

  return std::sqrt(norm);
}

} // Namespace TinMan

#endif // COMPUTE_AND_APPLY_RHS_HPP
