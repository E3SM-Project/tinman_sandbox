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

// To view the fully expanded name of a complicated template type T,
// just try to access some non-existent field of MyDebug<T>. E.g.:
// MyDebug<T>::type i;
template<typename T>
struct MyDebug {};

} // TinMan

#endif // TINMAN_KOKKOS_TYPES_HPP
