#ifndef COMMON_HPP
#define COMMON_HPP

#include <iostream>
#include <chrono>
#include <string>

extern int I1;
constexpr int I2=128;
constexpr int I3=256;

struct Timer {

  Timer( std::string const& name )
    : m_name(name)
  {}

  Timer( Timer const& ) = delete;
  Timer & operator=( Timer const& ) = delete;

  Timer( Timer && ) = default;
  Timer & operator=( Timer && ) = default;


  ~Timer()
  {
    const double sec = seconds();
    std::cout << m_name << ": " << sec << " s" << std::endl;
  }

  using clock_type = std::chrono::high_resolution_clock;

  double seconds() const {
    return 1.0e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>( clock_type::now() - m_start ).count();
  }

  clock_type::time_point m_start{ clock_type::now() };
  std::string m_name;
};

inline
constexpr int IDX(int i, int j, int k) {
  return k + I3*j + I2*I3*i;
}

void saxpby( const double a
           , const double b
           , double * __restrict__ x
           , const double * __restrict__ y
           );

#endif // COMMON_HPP
