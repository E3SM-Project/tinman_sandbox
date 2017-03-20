
#ifndef _SCRATCHSTACK_HPP_
#define _SCRATCHSTACK_HPP_

#include <Kokkos_Core.hpp>
#include <assert.h>

#include "Types.hpp"

namespace TinMan {

// A LIFO memory manager
// The user must ensure that the memory is no longer in use when freeing!
// This object can only be used inside of a Kokkos TeamThreadRange functor
class ScratchStack {
public:
#ifdef NDEBUG
  KOKKOS_INLINE_FUNCTION
  ScratchStack(const Kokkos::TeamPolicy<ExecSpace>::member_type *team,
               void *memory, int max_mem)
      : m_team(team), m_mem_ptr(memory), m_thread_region(false) {}
#else
  KOKKOS_INLINE_FUNCTION
  ScratchStack(const Kokkos::TeamPolicy<ExecSpace>::member_type *team,
               void *memory, int max_mem)
      : m_team(team), m_mem_ptr(memory), m_thread_region(false),
        m_mem_max(max_mem), m_mem_used(0), m_thread_mem_used(0),
        m_mem_max_used(0), m_thread_max_used(0) {}
#endif // NDEBUG

  KOKKOS_INLINE_FUNCTION
  ~ScratchStack() {
    std::cout << "Scratch Stack: max memory used: " << m_mem_max_used
              << "; max thread memory used: " << m_thread_max_used << std::endl;
  }

  KOKKOS_INLINE_FUNCTION void *allocate(int mem_size) {
    if (m_thread_region) {
      return allocate_thread(mem_size);
    } else {
      return allocate_team(mem_size);
    }
  }

  KOKKOS_INLINE_FUNCTION void *allocate_transient(int mem_size) const {
    if (m_thread_region) {
      return allocate_thread_transient(mem_size);
    } else {
      return allocate_team_transient(mem_size);
    }
  }

  KOKKOS_INLINE_FUNCTION void free(int mem_size) {
    if (m_thread_region) {
      free_team(mem_size);
    } else {
      free_thread(mem_size);
    }
  }

  void enter_thread_region() {
    // TODO: Remove team_barriers!!!
    // This significantly slows down OpenMP
    // We need to measure how much thread memory we're using,
    // then allocate independent blocks for them
    m_team->team_barrier();
    assert(m_thread_region == false);
    Kokkos::single(Kokkos::PerTeam(*m_team), [&]() { m_thread_region = true; });
  }

  void exit_thread_region() {
    m_team->team_barrier();
    assert(m_thread_region == true);
    Kokkos::single(Kokkos::PerTeam(*m_team),
                   [&]() { m_thread_region = false; });
  }

private:
  const Kokkos::TeamPolicy<ExecSpace>::member_type *m_team;
  void *m_mem_ptr;
  bool m_thread_region;

  KOKKOS_INLINE_FUNCTION void *allocate_team(int mem_size) {
    m_team->team_barrier();
    void *mem = m_mem_ptr;
    update_mem_usage(mem_size);
    return mem;
  }

  KOKKOS_INLINE_FUNCTION void *allocate_team_transient(int mem_size) const {
    m_team->team_barrier();
    check_mem_usage(mem_size);
    return m_mem_ptr;
  }

  KOKKOS_INLINE_FUNCTION void *allocate_thread(int mem_size) {
    m_team->team_barrier();
    void *mem = static_cast<void *>(static_cast<char *>(m_mem_ptr) +
                                    m_team->team_rank() * mem_size);
    int total_size = m_team->team_size() * mem_size;
    update_thread_usage(total_size);
    update_mem_usage(total_size);
    return mem;
  }

  KOKKOS_INLINE_FUNCTION void *allocate_thread_transient(int mem_size) const {
    m_team->team_barrier();
    void *mem = static_cast<void *>(static_cast<char *>(m_mem_ptr) +
                                    m_team->team_rank() * mem_size);
    const int total_size = m_team->team_size() * mem_size;
    check_mem_usage(total_size);
    check_thread_usage(total_size);
    return mem;
  }

  KOKKOS_INLINE_FUNCTION void free_team(int mem_size) {
    m_team->team_barrier();
    update_mem_usage(-mem_size);
  }

  KOKKOS_INLINE_FUNCTION void free_thread(int mem_size) {
    m_team->team_barrier();
    int total_mem = mem_size * m_team->team_size();
    update_thread_usage(-total_mem);
    update_mem_usage(-total_mem);
  }

#ifdef NDEBUG
  KOKKOS_INLINE_FUNCTION void update_mem_usage(int mem_size) {
    Kokkos::single(Kokkos::PerTeam(*m_team), [&]() {
      m_mem_ptr =
          static_cast<void *>(static_cast<char *>(m_mem_ptr) + mem_size);
    });
  }

  KOKKOS_INLINE_FUNCTION void check_mem_usage(int transient) const {}

  KOKKOS_INLINE_FUNCTION void update_thread_usage(int mem_size) {}

  KOKKOS_INLINE_FUNCTION void check_thread_usage(int transient) const {}
#else
  // Used for error checking - ensures the memory doesn't exceed a set amount
  // Use integers so we can subtract from them and don't have to worry about the
  // compiler attempting to cast them
  const int m_mem_max;
  int m_mem_used;
  int m_thread_mem_used;
  // Used for diagnostics - captures the maximum amount of memory used
  mutable int m_mem_max_used;
  mutable int m_thread_max_used;

  KOKKOS_INLINE_FUNCTION void update_mem_usage(int mem_size) {
    Kokkos::single(Kokkos::PerTeam(*m_team), [&]() { m_mem_used += mem_size; });
    check_mem_usage();
  }

  KOKKOS_INLINE_FUNCTION void check_mem_usage(int transient = 0) const {
    if ((m_mem_used + transient) < 0) {
      std::cout << "m_mem_used: " << m_mem_used << std::endl;
    }
    assert((m_mem_used + transient) >= 0);
    if ((m_mem_used + transient) >= m_mem_max) {
      std::cout << "m_mem_used / m_mem_max: " << (m_mem_used + transient)
                << " / " << m_mem_max << std::endl;
    }
    if (m_mem_used > m_mem_max_used) {
      Kokkos::single(Kokkos::PerTeam(*m_team),
                     [&]() { m_mem_max_used = m_mem_used; });
    }
    assert((m_mem_used + transient) < m_mem_max);
  }

  KOKKOS_INLINE_FUNCTION void update_thread_usage(int mem_size) {
    Kokkos::single(Kokkos::PerTeam(*m_team), [&]() {
      m_thread_mem_used += mem_size;
      if (m_thread_mem_used > m_thread_max_used) {
        m_thread_max_used = m_thread_mem_used;
      }
    });
    check_thread_usage();
  }

  KOKKOS_INLINE_FUNCTION void check_thread_usage(int transient = 0) const {
    if (m_thread_mem_used + transient > m_thread_max_used) {
      Kokkos::single(Kokkos::PerTeam(*m_team),
                     [&]() { m_thread_max_used = m_thread_mem_used; });
    }
  }
#endif // NDEBUG
};

} // TinMan

#endif
