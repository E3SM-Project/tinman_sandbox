#ifndef TINMAN_TYPES_HPP
#define TINMAN_TYPES_HPP

#include <Kokkos_Core.hpp>
#include <config.h>

namespace TinMan {

// Usual typedef for real scalar type
typedef double Real;

// Selecting the execution space. If no specific request, use Kokkos default exec space
#ifdef TINMAN_CUDA_SPACE
using ExecSpace = Kokkos::Cuda;
#elif defined(TINMAN_OPENMP_SPACE)
using ExecSpace = Kokkos::OpenMP;
#elif defined(TINMAN_THREADS_SPACE)
using ExecSpace = Kokkos::Threads;
#elif defined(TINMAN_SERIAL_SPACE)
using ExecSpace = Kokkos::Serial;
#elif defined(TINMAN_DEFAULT_SPACE)
using ExecSpace = Kokkos::DefaultExecutionSpace::execution_space;
#else
#error "No valid execution space choice"
#endif

// The memory spaces
using ExecMemSpace    = ExecSpace::memory_space;
using ScratchMemSpace = ExecSpace::scratch_memory_space;

// Short name for views with layout right
template<typename DataType, typename MemorySpace, typename MemoryManagement>
using ViewType = Kokkos::View<DataType,Kokkos::LayoutRight,MemorySpace,MemoryManagement>;

// Further specializations for execution space and managed/unmanaged memory
template<typename DataType>
using ExecViewManaged = ViewType<DataType,ExecMemSpace,Kokkos::MemoryManaged>;
template<typename DataType>
using ExecViewUnmanaged = ViewType<DataType,ExecMemSpace,Kokkos::MemoryUnmanaged>;

// To view the fully expanded name of a complicated template type T,
// just try to access some non-existent field of MyDebug<T>. E.g.:
// MyDebug<T>::type i;
template<typename T>
struct MyDebug {};

} // TinMan

#endif // TINMAN_KOKKOS_TYPES_HPP
