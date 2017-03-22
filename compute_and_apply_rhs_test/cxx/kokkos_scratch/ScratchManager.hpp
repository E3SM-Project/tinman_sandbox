#ifndef TINMAN_SCRATCH_MANAGER_HPP
#define TINMAN_SCRATCH_MANAGER_HPP

#include "Types.hpp"

namespace TinMan {

// A simple holder for a pair <Count,Size>
template <size_t Count, size_t Size> struct CountAndSize {
  static constexpr size_t count = Count;
  static constexpr size_t size = Size;
};

// A pack of pairs <Count,Size> flattened
template <size_t Count, size_t Size, size_t... Tail> struct CountAndSizePack {
  using head = CountAndSize<Count, Size>;
  using tail = CountAndSizePack<Tail...>;

  static constexpr size_t num_blocks = 1 + tail::num_blocks;
  static constexpr size_t total_count = head::count + tail::total_count;
  static constexpr size_t total_size =
      head::size * head::count + tail::total_size;
};

template <size_t Count, size_t Size> struct CountAndSizePack<Count, Size> {
  using head = CountAndSize<Count, Size>;

  static constexpr size_t num_blocks = 1;
  static constexpr size_t total_count = head::count;
  static constexpr size_t total_size = head::size * head::count;
};

// Given a CountAndSizePack, we compute the offset of the COUNTER_ID-th view in
// the BLOCK_ID-th CountAndSize pair
template <typename CountAndSizePack, size_t BLOCK_ID, size_t COUNTER_ID>
struct ScratchOffset {
  using pack = CountAndSizePack;
  using head = typename pack::head;
  using tail = typename pack::tail;

  static constexpr size_t value =
      head::count * head::size +
      ScratchOffset<tail, BLOCK_ID - 1, COUNTER_ID>::value;
};

template <typename CountAndSizePack, size_t BLOCK_ID>
struct ScratchOffset<CountAndSizePack, BLOCK_ID, 0> {
  using pack = CountAndSizePack;
  using head = typename pack::head;
  using tail = typename pack::tail;

  static_assert(BLOCK_ID < pack::num_blocks,
                "Error! The BLOCK_ID parameter is out of bounds.\n");

  static constexpr size_t value =
      head::count * head::size + ScratchOffset<tail, BLOCK_ID - 1, 0>::value;
};

template <typename CountAndSizePack, size_t COUNTER_ID>
struct ScratchOffset<CountAndSizePack, 0, COUNTER_ID> {
  using pack = CountAndSizePack;
  using head = typename pack::head;

  static_assert(0 < pack::num_blocks,
                "Error! The pack does not store any block.\n");
  static_assert(COUNTER_ID < head::count,
                "Error! The COUNTER_ID parameter is out of bounds.\n");

  static constexpr size_t value = COUNTER_ID * head::size;
};

template <typename CountAndSizePack>
struct ScratchOffset<CountAndSizePack, 0, 0> {
  using pack = CountAndSizePack;
  using head = typename pack::head;

  static_assert(0 < pack::num_blocks,
                "Error! The pack does not store any block.\n");
  static_assert(0 < head::count,
                "Error! The COUNTER_ID parameter is out of bounds.\n");

  static constexpr size_t value = 0;
};

template <typename TeamSizesPack, typename ThreadSizesPack>
class ScratchManager {
public:
  using team_sizes_pack = TeamSizesPack;
  using thread_sizes_pack = ThreadSizesPack;

  static constexpr size_t sum_team_sizes = team_sizes_pack::total_size;
  static constexpr size_t sum_thread_sizes = thread_sizes_pack::total_size;

  KOKKOS_INLINE_FUNCTION
  ScratchManager(TeamPolicy &team)
    : m_team_scratch(ScratchView(team.team_scratch(0), sum_team_sizes).ptr_on_device()),
      m_thread_scratch(ScratchView(team.thread_scratch(0), sum_thread_sizes)
                             .ptr_on_device()) {}

  // Get a block of memory for a given block ID
  template <size_t BLOCK_ID, size_t COUNTER_ID>
  KOKKOS_INLINE_FUNCTION Real *get_team_scratch() const {
    return m_team_scratch +
           ScratchOffset<team_sizes_pack, BLOCK_ID, COUNTER_ID>::value;
  }

  // Get a block of memory for a given thread and block ID
  template <size_t BLOCK_ID, size_t COUNTER_ID>
  KOKKOS_INLINE_FUNCTION Real *get_thread_scratch(int thread_id) const {
    return m_thread_scratch + thread_id * sum_thread_sizes +
           ScratchOffset<thread_sizes_pack, BLOCK_ID, COUNTER_ID>::value;
  }

  KOKKOS_INLINE_FUNCTION static constexpr int
  reals_needed(const int team_size) {
    return sum_team_sizes + sum_thread_sizes * team_size;
  }

  KOKKOS_INLINE_FUNCTION static constexpr int
  memory_needed(const int team_size) {
    return sizeof(Real) * reals_needed(team_size);
  }

private:
  using ScratchView =
      ViewType<Real *, ScratchMemSpace, Kokkos::MemoryUnmanaged>;
  // The memory buffer
  Real *const m_team_scratch;
  Real *const m_thread_scratch;
};

} // namespace TinMan

#endif // TINMAN_SCRATCH_MANAGER_HPP
