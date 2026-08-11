#include "CKKS/Arena.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

// Links fhn_ckks_arena and nothing else — in particular not fhn_ckks_params.
// The arena must stay a leaf: an allocator that needed parameters could not be
// handed to a kernel that does not have them.

using fhenomenon::ckks::Arena;

namespace {

std::uintptr_t addressOf(const void *p) { return reinterpret_cast<std::uintptr_t>(p); }

} // namespace

TEST(CkksArenaTest, HandsOutAlignedStorage) {
  Arena arena(4096);
  EXPECT_EQ(arena.capacity(), 4096u);
  EXPECT_EQ(arena.used(), 0u);

  uint64_t *a = arena.allocate<uint64_t>(8);
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(addressOf(a) % Arena::kAlignment, 0u);

  // 8 * 8 = 64 bytes exactly, so the next block starts one line later.
  EXPECT_EQ(arena.used(), 64u);

  uint64_t *b = arena.allocate<uint64_t>(1);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(addressOf(b) % Arena::kAlignment, 0u);
  EXPECT_EQ(addressOf(b) - addressOf(a), 64u);
  // A single word still consumes a whole line: alignment is a guarantee the
  // caller may rely on, so the padding is real and must show in used().
  EXPECT_EQ(arena.used(), 128u);
}

TEST(CkksArenaTest, WritesDoNotOverlap) {
  Arena arena(4096);
  uint64_t *a = arena.allocate<uint64_t>(4);
  uint64_t *b = arena.allocate<uint64_t>(4);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);

  for (int i = 0; i < 4; ++i) {
    a[i] = static_cast<uint64_t>(i);
    b[i] = static_cast<uint64_t>(100 + i);
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(a[i], static_cast<uint64_t>(i));
    EXPECT_EQ(b[i], static_cast<uint64_t>(100 + i));
  }
}

// Exhaustion is a planning bug, so the arena reports it and stops rather than
// growing — growing would silently invalidate the footprint the caller already
// reported to the movement planner.
TEST(CkksArenaTest, RefusesToGrow) {
  Arena arena(128);
  ASSERT_NE(arena.allocate<uint64_t>(8), nullptr); // 64 bytes
  ASSERT_NE(arena.allocate<uint64_t>(8), nullptr); // 128 bytes, exactly full
  EXPECT_EQ(arena.remaining(), 0u);

  EXPECT_EQ(arena.allocate<uint64_t>(1), nullptr);
  EXPECT_EQ(arena.used(), 128u); // a refused request consumes nothing
}

TEST(CkksArenaTest, RejectsDegenerateRequests) {
  Arena arena(256);
  EXPECT_EQ(arena.allocateBytes(0), nullptr);
  EXPECT_EQ(arena.used(), 0u);

  // Overflowing the byte count must fail rather than wrap into a small offset.
  EXPECT_EQ(arena.allocate<uint64_t>(SIZE_MAX / 4), nullptr);
  EXPECT_EQ(arena.used(), 0u);

  Arena empty(0);
  EXPECT_EQ(empty.capacity(), 0u);
  EXPECT_EQ(empty.allocate<uint64_t>(1), nullptr);
}

TEST(CkksArenaTest, ResetRewindsButKeepsTheSlab) {
  Arena arena(1024);
  const void *first = arena.allocate<uint64_t>(4);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(arena.used(), 64u);

  arena.reset();
  EXPECT_EQ(arena.used(), 0u);
  // Same slab, so the next program reuses the same memory instead of
  // returning it to the system and asking for it back.
  EXPECT_EQ(arena.allocate<uint64_t>(4), first);
}

// High water is how a caller checks that the size it planned was the size it
// needed. It deliberately survives reset, because the interesting quantity is
// the peak over a whole program, not over one instruction.
TEST(CkksArenaTest, HighWaterTracksThePeakAcrossResets) {
  Arena arena(1024);
  ASSERT_NE(arena.allocate<uint64_t>(4), nullptr); // 64
  EXPECT_EQ(arena.highWater(), 64u);

  arena.reset();
  ASSERT_NE(arena.allocate<uint64_t>(16), nullptr); // 128
  EXPECT_EQ(arena.used(), 128u);
  EXPECT_EQ(arena.highWater(), 128u);

  arena.reset();
  ASSERT_NE(arena.allocate<uint64_t>(1), nullptr); // 64
  EXPECT_EQ(arena.used(), 64u);
  EXPECT_EQ(arena.highWater(), 128u);
}

// Scope is what lets composite kernels nest temporaries without every level
// sizing its own buffer.
TEST(CkksArenaTest, ScopeReclaimsNestedTemporaries) {
  Arena arena(1024);
  uint64_t *outer = arena.allocate<uint64_t>(4);
  ASSERT_NE(outer, nullptr);
  const std::size_t after_outer = arena.used();

  const void *inner_first = nullptr;
  {
    Arena::Scope scope(arena);
    inner_first = arena.allocate<uint64_t>(8);
    ASSERT_NE(inner_first, nullptr);
    ASSERT_NE(arena.allocate<uint64_t>(8), nullptr);
    EXPECT_GT(arena.used(), after_outer);
  }
  EXPECT_EQ(arena.used(), after_outer);

  // The outer allocation survived, and the reclaimed space is handed out
  // again.
  outer[0] = 7;
  EXPECT_EQ(outer[0], 7u);
  EXPECT_EQ(arena.allocate<uint64_t>(8), inner_first);

  // The peak reached inside the scope is still remembered.
  EXPECT_GE(arena.highWater(), after_outer + 128);
}

TEST(CkksArenaTest, ScopesNest) {
  Arena arena(1024);
  {
    Arena::Scope outer(arena);
    ASSERT_NE(arena.allocate<uint64_t>(4), nullptr);
    const std::size_t at_outer = arena.used();
    {
      Arena::Scope inner(arena);
      ASSERT_NE(arena.allocate<uint64_t>(4), nullptr);
      EXPECT_GT(arena.used(), at_outer);
    }
    EXPECT_EQ(arena.used(), at_outer);
  }
  EXPECT_EQ(arena.used(), 0u);
}

// Moving must leave the source empty. A defaulted move would keep capacity_
// while the slab went away, and the moved-from arena would hand out pointers
// into nothing.
TEST(CkksArenaTest, MoveLeavesTheSourceEmpty) {
  Arena source(512);
  ASSERT_NE(source.allocate<uint64_t>(4), nullptr);

  Arena moved(std::move(source));
  EXPECT_EQ(moved.capacity(), 512u);
  EXPECT_EQ(moved.used(), 64u);
  ASSERT_NE(moved.allocate<uint64_t>(4), nullptr);

  EXPECT_EQ(source.capacity(), 0u); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.used(), 0u);
  EXPECT_EQ(source.allocate<uint64_t>(1), nullptr);
}

// One arena per thread is the whole concurrency story: no lock, no shared
// allocator, so N threads scale without cloning an evaluator.
TEST(CkksArenaTest, ArenasAreIndependent) {
  std::vector<Arena> arenas;
  arenas.reserve(4);
  for (int i = 0; i < 4; ++i) {
    arenas.emplace_back(256);
  }

  std::vector<uint64_t *> blocks;
  for (std::size_t i = 0; i < arenas.size(); ++i) {
    uint64_t *block = arenas[i].allocate<uint64_t>(8);
    ASSERT_NE(block, nullptr);
    block[0] = i;
    blocks.push_back(block);
  }
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    EXPECT_EQ(blocks[i][0], i);
    EXPECT_EQ(arenas[i].used(), 64u);
  }
}
