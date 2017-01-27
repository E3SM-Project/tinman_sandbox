#ifndef TINMAN_HELPERS_HPP
#define TINMAN_HELPERS_HPP

#include "Types.hpp"

namespace TinMan
{

template<typename T>
struct is_single_integral_index
{
  static constexpr bool value = std::is_integral<T>::value;
};

// Empty base type
template<typename T, typename Layout, int RankIn, int CurrentI, typename ... Args>
struct SubDataType {};

// If input layout is strided, in general we can't extract
template<typename T, int RankIn, int CurrentI, typename ... Args>
struct SubDataType<T,Kokkos::LayoutStride,RankIn,CurrentI,Args...>
{
  typedef T data_type;
};

template<typename T, typename ...Args, >
struct SubarrayData<T*,Args...>
{
  typedef std::conditional<is_single_integral_index<iType>::value,T data_type;
};

template<typename InputDataType>
struct SubviewDataType
{

};

} // namespace TinMan

#endif
