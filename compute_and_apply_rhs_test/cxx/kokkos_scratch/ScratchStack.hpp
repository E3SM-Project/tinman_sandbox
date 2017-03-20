
#ifndef _SCRATCHSTACK_HPP_
#define _SCRATCHSTACK_HPP_

#include <Kokkos_Core.hpp>
#include <assert.h>

// A LIFO memory manager
// The user must ensure that the memory is no longer in use when freeing!
// This object can only be used inside of a Kokkos TeamThreadRange functor
class ScratchStack {
public:
#ifdef NDEBUG
  KOKKOS_INLINE_FUNCTION
  ScratchStack(const Kokkos::TeamPolicy<>::member_type *team, void *memory,
               int max_mem)
      : m_team(team), m_mem_ptr(memory) {}
#else
  KOKKOS_INLINE_FUNCTION
  ScratchStack(const Kokkos::TeamPolicy<>::member_type *team, void *memory,
               int max_mem)
      : m_team(team), m_mem_ptr(memory), m_mem_max(max_mem), m_mem_max_used(0),
        m_mem_used(0) {}
#endif // NDEBUG

  KOKKOS_INLINE_FUNCTION void *allocate_team(int mem_size) {
    void *mem = m_mem_ptr;
    update_mem_usage(mem_size);
    return mem;
  }

  KOKKOS_INLINE_FUNCTION void *allocate_team_transient(int mem_size) const {
    return m_mem_ptr;
  }

  KOKKOS_INLINE_FUNCTION void *allocate_thread(int mem_size) {
    void *mem = static_cast<void *>(static_cast<char *>(m_mem_ptr) +
                                    m_team->team_rank() * mem_size);
    int total_size = m_team->team_size() * mem_size;
    update_mem_usage(total_size);
    return mem;
  }

  KOKKOS_INLINE_FUNCTION void *allocate_thread_transient(int mem_size) const {
    void *mem = static_cast<void *>(static_cast<char *>(m_mem_ptr) +
                                    m_team->team_rank() * mem_size);
    return mem;
  }

  KOKKOS_INLINE_FUNCTION void free_team(int mem_size) {
    update_mem_usage(-mem_size);
  }

  KOKKOS_INLINE_FUNCTION void free_thread(int mem_size) {
    int total_mem = mem_size * m_team->team_size();
    update_mem_usage(-total_mem);
  }

private:
  const Kokkos::TeamPolicy<>::member_type *m_team;
  void *m_mem_ptr;

#ifdef NDEBUG
  KOKKOS_INLINE_FUNCTION void update_mem_usage(int mem_size) {
    Kokkos::single(Kokkos::PerTeam(*m_team), [&]() {
      m_mem_ptr =
          static_cast<void *>(static_cast<char *>(m_mem_ptr) + mem_size);
    });
  }
#else
  // Used for error checking - ensures the memory doesn't exceed a set amount
  // Use integers so we can subtract from them and don't have to worry about the
  // compiler attempting to cast them
  const int m_mem_max;
  int m_mem_used;
  // Used for diagnostics - captures the maximum amount of memory used
  int m_mem_max_used;

  KOKKOS_INLINE_FUNCTION void update_mem_usage(int mem_size) {
    Kokkos::single(Kokkos::PerTeam(*m_team), [&]() {
      m_mem_used += mem_size;
      if (m_mem_used < m_mem_max_used) {
        m_mem_max_used = m_mem_used;
      }
    });
    assert(m_mem_used >= 0);
    assert(m_mem_used < m_mem_max);
  }
#endif // NDEBUG
};

#endif
