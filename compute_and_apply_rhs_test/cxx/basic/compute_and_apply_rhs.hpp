#ifndef COMPUTE_AND_APPLY_RHS_HPP
#define COMPUTE_AND_APPLY_RHS_HPP

#include "data_structures.hpp"

namespace Homme
{

void compute_and_apply_rhs (TestData& data);

void preq_hydrostatic (const real* const phis, const real* const T_v,
                       const real* const p, const real* dp,
                       real Rgas, real* const phi);

void preq_omega_ps (const real* const p, const real* const vgrad_p,
                   const real* const divdp, real* const omega_p);

real compute_norm (const real* const field, int length);

void print_results_2norm (const TestData& data);

void dump_results_to_file (const TestData& data);

} // Namespace Homme

#endif  // COMPUTE_AND_APPLY_RHS_HPP
