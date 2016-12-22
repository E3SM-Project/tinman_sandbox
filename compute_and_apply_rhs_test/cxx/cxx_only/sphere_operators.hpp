#ifndef SPHERE_OPERATORS_HPP
#define SPHERE_OPERATORS_HPP

#include "data_structures.hpp"

namespace Homme
{

void gradient_sphere (const real* const s, const TestData& data,
                      int ielem, real* const ds);

void divergence_sphere (const real* const v, const TestData& data,
                        int ielem, real* const div);

void vorticity_sphere (const real* const v, const TestData& data,
                       int ielem, real* const vort);

} // Namespace Homme

#endif // SPHERE_OPERATORS_HPP
