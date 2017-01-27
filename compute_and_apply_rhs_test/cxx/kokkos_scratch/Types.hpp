#ifndef TINMAN_TYPES_HPP
#define TINMAN_TYPES_HPP

#include <Kokkos_Core.hpp>

namespace TinMan {

// Usual typedef for real scalar type
typedef double Real;

// The memory spaces
using ExecSpace    = Kokkos::DefaultExecutionSpace::memory_space;
using ScratchSpace = Kokkos::DefaultExecutionSpace::scratch_memory_space;

// Short name for views with layout right
template<typename DataType, typename MemorySpace, typename MemoryManagement>
using ViewType = Kokkos::View<DataType,Kokkos::LayoutRight,MemorySpace,MemoryManagement>;

// Further specializations for execution space and managed/unmanaged memory
template<typename DataType>
using ExecViewManaged = ViewType<DataType,ExecSpace,Kokkos::MemoryManaged>;
template<typename DataType>
using ExecViewUnmanaged = ViewType<DataType,ExecSpace,Kokkos::MemoryUnmanaged>;

// The scratch view type (always unmanaged)
template<typename DataType>
using ScratchView = ViewType<DataType,ScratchSpace,Kokkos::MemoryUnmanaged>;

// To view the fully expanded name of a complicated template type T,
// just try to access some non-existent field of MyDebug<T>. E.g.:
// MyDebug<T>::type i;
template<typename T>
struct MyDebug {};

} // TinMan

#endif // TINMAN_KOKKOS_TYPES_HPP
