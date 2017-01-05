#ifndef TINMAN_TYPES_HPP
#define TINMAN_TYPES_HPP

#include <Kokkos_Core.hpp>

namespace TinMan {

// Usual typedef for real scalar type
typedef double Real;

// Short name for views with layout right
template<typename DataType, typename MemoryManagement>
using ViewType = Kokkos::View<DataType,Kokkos::LayoutRight,MemoryManagement>;

// Further specializations for managed/unmanaged memory
template<typename DataType>
using ViewManaged = ViewType<DataType,Kokkos::MemoryManaged>;
template<typename DataType>
using ViewUnmanaged = ViewType<DataType,Kokkos::MemoryUnmanaged>;

} // TinMan

#endif // TINMAN_KOKKOS_TYPES_HPP
