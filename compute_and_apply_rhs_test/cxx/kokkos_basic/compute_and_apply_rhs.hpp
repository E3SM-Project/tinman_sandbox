#ifndef COMPUTE_AND_APPLY_RHS_HPP
#define COMPUTE_AND_APPLY_RHS_HPP

#include <config.h>

#include <Types.hpp>

namespace TinMan
{

// Forward declarations
class TestData;
class Region;

void compute_and_apply_rhs (const TestData& data, Region& region);

void print_results_2norm (const TestData& data, const Region& region);

void dump_results_to_file (const TestData& data, const Region& region);

} // Namespace TinMan

#endif  // COMPUTE_AND_APPLY_RHS_HPP
