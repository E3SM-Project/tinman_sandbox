#ifndef COMPUTE_AND_APPLY_RHS_HPP
#define COMPUTE_AND_APPLY_RHS_HPP

#include "config.h"

#include "Types.hpp"

namespace TinMan
{

// Forward declarations
class TestData;
class Region;

void compute_and_apply_rhs (const TestData& data, Region& region);

void print_results_2norm (const TestData& data, const Region& region);

void dump_results_to_file (const TestData& data, const Region& region);

template<typename ViewType>
Real compute_norm (const ViewType view)
{
  size_t length = view.size();
  typename ViewType::pointer_type data = view.data();

  // Note: use Kahan algorithm to maintain accuracy
  Real norm = 0;
  Real c = 0;
  Real temp, y;
  for (size_t i=0; i<length; ++i)
  {
    y = data[i]*data[i] - c;
    temp = norm + y;
    c = (temp - norm) - y;
    norm = temp;
  }

  return std::sqrt(norm);
}

} // Namespace TinMan

#endif  // COMPUTE_AND_APPLY_RHS_HPP
