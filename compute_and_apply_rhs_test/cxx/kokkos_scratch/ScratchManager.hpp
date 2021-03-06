#ifndef TINMAN_SCRATCH_MANAGER_HPP
#define TINMAN_SCRATCH_MANAGER_HPP

#include "Types.hpp"

namespace TinMan
{

// A simple holder for a pair <Count,Size>
template<size_t Count, size_t Size>
struct CountAndSize
{
  static constexpr size_t count = Count;
  static constexpr size_t size  = Size;
};

// A pack of pairs <Count,Size> flattened
template<size_t Count, size_t Size, size_t... Tail>
struct CountAndSizePack
{
  typedef CountAndSize<Count,Size>  head;
  typedef CountAndSizePack<Tail...> tail;

  static constexpr size_t num_blocks  = 1 + tail::num_blocks;
  static constexpr size_t total_count = head::count + tail::total_count;
  static constexpr size_t total_size  = head::size*head::count  + tail::total_size;
};

template<size_t Count, size_t Size>
struct CountAndSizePack<Count,Size>
{
  typedef CountAndSize<Count,Size>  head;

  static constexpr size_t num_blocks  = 1;
  static constexpr size_t total_count = head::count;
  static constexpr size_t total_size  = head::size*head::count;
};

// Given a CountAndSizePack, we compute the offset of the COUNTER_ID-th view in the BLOCK_ID-th CountAndSize pair
template<typename CountAndSizePack, size_t BLOCK_ID, size_t COUNTER_ID>
struct ScratchOffset
{
  typedef CountAndSizePack    pack;
  typedef typename pack::head head;
  typedef typename pack::tail tail;

  static constexpr size_t value = head::count*head::size + ScratchOffset<tail,BLOCK_ID-1,COUNTER_ID>::value;
};

template<typename CountAndSizePack, size_t BLOCK_ID>
struct ScratchOffset<CountAndSizePack,BLOCK_ID,0>
{
  typedef CountAndSizePack    pack;
  typedef typename pack::head head;
  typedef typename pack::tail tail;

  static_assert (BLOCK_ID<pack::num_blocks, "Error! The BLOCK_ID parameter is out of bounds.\n");

  static constexpr size_t value = head::count*head::size + ScratchOffset<tail,BLOCK_ID-1,0>::value;
};

template<typename CountAndSizePack, size_t COUNTER_ID>
struct ScratchOffset<CountAndSizePack,0,COUNTER_ID>
{
  typedef CountAndSizePack    pack;
  typedef typename pack::head head;

  static_assert (0<pack::num_blocks, "Error! The pack does not store any block.\n");
  static_assert (COUNTER_ID<head::count, "Error! The COUNTER_ID parameter is out of bounds.\n");

  static constexpr size_t value = COUNTER_ID*head::size;
};

template<typename CountAndSizePack>
struct ScratchOffset<CountAndSizePack,0,0>
{
  typedef CountAndSizePack    pack;
  typedef typename pack::head head;

  static_assert (0<pack::num_blocks, "Error! The pack does not store any block.\n");
  static_assert (0<head::count, "Error! The COUNTER_ID parameter is out of bounds.\n");

  static constexpr size_t value = 0;
};

template<typename TeamSizesPack, typename ThreadSizesPack>
class ScratchManager
{
public:

  typedef TeamSizesPack     team_sizes_pack;
  typedef ThreadSizesPack   thread_sizes_pack;

  static constexpr size_t sum_team_sizes   = team_sizes_pack::total_size;
  static constexpr size_t sum_thread_sizes = thread_sizes_pack::total_size;

  // Get a block of memory for a given block ID
  template<size_t BLOCK_ID, size_t COUNTER_ID>
  Real* get_team_scratch () { return m_team_scratch + ScratchOffset<team_sizes_pack,BLOCK_ID,COUNTER_ID>::value; }

  // Get a block of memory for a given thread and block ID
  template<size_t BLOCK_ID, size_t COUNTER_ID>
  Real* get_thread_scratch (int thread_id) { return m_thread_scratch + thread_id*sum_thread_sizes + ScratchOffset<thread_sizes_pack,BLOCK_ID,COUNTER_ID>::value; }

  static int memory_needed (const int team_size) { return sizeof(Real) * (sum_team_sizes + sum_thread_sizes*team_size); }

  void set_scratch_memory (Real* const scratch)
  {
    m_team_scratch   = scratch;
    // The threads offset is the sum of all the sizes of the team blocks
    m_thread_scratch = scratch + sum_team_sizes;
  }

private:

  // The memory buffer
  Real* m_team_scratch;
  Real* m_thread_scratch;
};

} // namespace TinMan

#endif // TINMAN_SCRATCH_MANAGER_HPP
