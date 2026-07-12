#include "FHN/FhnMovementPlan.h"
#include "FhnTestProgramBuilder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

using namespace fhenomenon;
using fhenomenon::testutil::ProgramBuilder;

// r3 = a1 + b2: inputs prefetched at their first use, result allocated at
// its def, dead operands freed after their last use, pinned result kept.
TEST(FhnMovementPlan, SingleAddSchedulesJustInTime) {
  auto prog = ProgramBuilder().input(1).input(2).inst(FHN_ADD_CC, 3, 1, 2).output(3).build();

  auto plan = FhnMovementPlan::analyze(*prog, /*pinned=*/{3});
  ASSERT_TRUE(plan.has_value());

  const FhnMovementActions &a0 = plan->at(0);
  EXPECT_EQ(a0.alloc, (std::vector<uint32_t>{3}));
  EXPECT_EQ(a0.prefetch, (std::vector<uint32_t>{1, 2}));
  EXPECT_TRUE(a0.evict.empty());
  // Inputs die here (not pinned); the pinned result survives.
  EXPECT_EQ(a0.free, (std::vector<uint32_t>{1, 2}));

  EXPECT_EQ(plan->stats().high_water, 3u);
  EXPECT_EQ(plan->stats().alloc_count, 1u);
  EXPECT_EQ(plan->stats().prefetch_count, 2u);
  EXPECT_EQ(plan->stats().evict_count, 0u);
}

// t3 = a1 + a1; r4 = t3 * b2: t3 dies after its last use, so the live-set
// maximum (high_water) is below the total buffer count.
TEST(FhnMovementPlan, ChainFreesIntermediateEarly) {
  auto prog = ProgramBuilder().input(1).input(2).inst(FHN_ADD_CC, 3, 1, 1).inst(FHN_MULT_CC, 4, 3, 2).output(4).build();

  auto plan = FhnMovementPlan::analyze(*prog, {4});
  ASSERT_TRUE(plan.has_value());

  // a1 dies at inst 0; t3 and b2 die at inst 1.
  EXPECT_EQ(plan->at(0).free, (std::vector<uint32_t>{1}));
  EXPECT_EQ(plan->at(1).free, (std::vector<uint32_t>{2, 3}));
  // Live sets: {1,3} then {2,3,4} -> high_water 3, not 4 total buffers.
  EXPECT_EQ(plan->stats().high_water, 3u);
  // b2 is prefetched at inst 1, its first use — not at inst 0.
  EXPECT_TRUE(plan->at(0).prefetch == (std::vector<uint32_t>{1}));
  EXPECT_EQ(plan->at(1).prefetch, (std::vector<uint32_t>{2}));
}

// Pinned ids are never freed, even when dead.
TEST(FhnMovementPlan, PinnedIdsAreNeverFreed) {
  auto prog = ProgramBuilder().input(1).input(2).inst(FHN_ADD_CC, 3, 1, 2).inst(FHN_ADD_CC, 4, 3, 3).output(4).build();

  auto plan = FhnMovementPlan::analyze(*prog, /*pinned=*/{1, 2, 4});
  ASSERT_TRUE(plan.has_value());
  for (uint32_t i = 0; i < 2; ++i) {
    for (uint32_t id : plan->at(i).free) {
      EXPECT_NE(id, 1u);
      EXPECT_NE(id, 2u);
      EXPECT_NE(id, 4u);
    }
  }
  // The unpinned intermediate still dies.
  EXPECT_EQ(plan->at(1).free, (std::vector<uint32_t>{3}));
}

// An unused, unpinned input is freed in the epilogue (after the last
// instruction) — the plan owns every non-pinned lifetime uniformly.
TEST(FhnMovementPlan, UnusedInputFreedAtEnd) {
  auto prog = ProgramBuilder()
                .input(1)
                .input(2) // never used
                .inst(FHN_NEGATE, 3, 1)
                .output(3)
                .build();

  auto plan = FhnMovementPlan::analyze(*prog, {3});
  ASSERT_TRUE(plan.has_value());
  const auto &f = plan->at(0).free;
  EXPECT_NE(std::find(f.begin(), f.end(), 2u), f.end());
  // Never resident: no prefetch was planned for it.
  EXPECT_EQ(plan->at(0).prefetch, (std::vector<uint32_t>{1}));
}

// Validation: an operand that is never defined refuses to plan.
TEST(FhnMovementPlan, UndefinedOperandIsRejected) {
  auto prog = ProgramBuilder().input(1).inst(FHN_ADD_CC, 3, 1, 99).output(3).build();
  EXPECT_FALSE(FhnMovementPlan::analyze(*prog, {3}).has_value());
}

// Validation: duplicate definitions refuse to plan.
TEST(FhnMovementPlan, DuplicateDefIsRejected) {
  auto prog = ProgramBuilder()
                .input(1)
                .inst(FHN_NEGATE, 3, 1)
                .inst(FHN_NEGATE, 3, 1) // redefines id 3
                .output(3)
                .build();
  EXPECT_FALSE(FhnMovementPlan::analyze(*prog, {3}).has_value());
}

// Budget 4, constructed so LRU and Belady disagree at the eviction point.
// Ids: a1, b2, c8 are inputs; 3..7 are results.
//   i0: 3 = b2 + b2   prefetch b2; resident {2,3}
//   i1: 4 = 3 + a1    prefetch a1; resident {1,2,3,4} = 4 (fits); 3 dies -> {1,2,4}
//   i2: 5 = 4 + c8    prefetch c8 + alloc 5 = 2 incoming, 3 resident -> over
//       budget: candidates {a1, b2}. a1 is MORE recent (i1 > i0) so LRU
//       would evict b2 — but a1's next use (i4) is farther than b2's (i3),
//       so Belady must evict a1. 4 and c8 die after i2 -> {2,5}
//   i3: 6 = 5 + b2    b2 still resident: no re-prefetch (Belady's win)
//   i4: 7 = 6 + a1    a1 comes back just in time
TEST(FhnMovementPlan, BeladyEvictsFarthestNextUseNotLru) {
  auto prog = ProgramBuilder()
                .input(1)                  // a1: used i1, i4
                .input(2)                  // b2: used i0, i3
                .input(8)                  // c8: used i2 only
                .inst(FHN_ADD_CC, 3, 2, 2) // i0
                .inst(FHN_ADD_CC, 4, 3, 1) // i1
                .inst(FHN_ADD_CC, 5, 4, 8) // i2
                .inst(FHN_ADD_CC, 6, 5, 2) // i3
                .inst(FHN_ADD_CC, 7, 6, 1) // i4
                .output(7)
                .build();

  auto plan = FhnMovementPlan::analyze(*prog, /*pinned=*/{7}, /*device_budget=*/4);
  ASSERT_TRUE(plan.has_value());

  // The eviction happens at i2 and Belady picks a1 (farthest next use),
  // even though LRU would pick b2 (least recently used).
  EXPECT_EQ(plan->at(2).evict, (std::vector<uint32_t>{1}));
  // b2 stayed resident across i3: no re-prefetch there.
  EXPECT_TRUE(plan->at(3).prefetch.empty());
  // a1 comes back just in time.
  EXPECT_EQ(plan->at(4).prefetch, (std::vector<uint32_t>{1}));
  // Exactly one eviction; prefetches are b2@i0, a1@i1, c8@i2, a1@i4.
  EXPECT_EQ(plan->stats().evict_count, 1u);
  EXPECT_EQ(plan->stats().prefetch_count, 4u);
  // Budget respected throughout.
  EXPECT_LE(plan->stats().high_water, 4u);
}

// Ties on next-use distance break on the lower id (deterministic plans).
TEST(FhnMovementPlan, BeladyTieBreaksOnLowerId) {
  // a1 and b2 both have NO future use after i0 but are pinned (so not
  // freed); at i1 one must be evicted to fit — the lower id (1) goes.
  auto prog = ProgramBuilder()
                .input(1)
                .input(2)
                .inst(FHN_ADD_CC, 3, 1, 2) // i0: live {1,2,3}
                .inst(FHN_NEGATE, 4, 3)    // i1: working {3,4}
                .output(4)
                .build();

  auto plan = FhnMovementPlan::analyze(*prog, /*pinned=*/{1, 2, 4}, /*device_budget=*/3);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->at(1).evict, (std::vector<uint32_t>{1}));
}

// A budget smaller than one instruction's working set cannot be satisfied.
TEST(FhnMovementPlan, InfeasibleBudgetIsRejected) {
  auto prog = ProgramBuilder().input(1).input(2).inst(FHN_ADD_CC, 3, 1, 2).output(3).build();
  EXPECT_FALSE(FhnMovementPlan::analyze(*prog, {3}, /*device_budget=*/2).has_value());
}

// Unlimited budget (0) plans no evictions regardless of pressure.
TEST(FhnMovementPlan, UnlimitedBudgetNeverEvicts) {
  auto prog = ProgramBuilder()
                .input(1)
                .input(2)
                .inst(FHN_ADD_CC, 3, 1, 2)
                .inst(FHN_ADD_CC, 4, 3, 1)
                .inst(FHN_ADD_CC, 5, 4, 2)
                .output(5)
                .build();
  auto plan = FhnMovementPlan::analyze(*prog, {5}, 0);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->stats().evict_count, 0u);
}

// Same program as BeladyEvictsFarthestNextUseNotLru, planned under the
// benchmarking-only LRU baseline: at i2 the least-recently-touched
// candidate is b2 (last touch i0; a1 was touched at i1), so LRU evicts b2
// where Belady evicts a1 — and pays a re-prefetch at i3 that Belady never
// pays. This pins that the two policies genuinely diverge.
TEST(FhnMovementPlan, LruEvictsLeastRecentlyUsedNotBelady) {
  auto prog = ProgramBuilder()
                .input(1)                  // a1: used i1, i4
                .input(2)                  // b2: used i0, i3
                .input(8)                  // c8: used i2 only
                .inst(FHN_ADD_CC, 3, 2, 2) // i0
                .inst(FHN_ADD_CC, 4, 3, 1) // i1
                .inst(FHN_ADD_CC, 5, 4, 8) // i2
                .inst(FHN_ADD_CC, 6, 5, 2) // i3
                .inst(FHN_ADD_CC, 7, 6, 1) // i4
                .output(7)
                .build();

  auto plan = FhnMovementPlan::analyze(*prog, /*pinned=*/{7}, /*device_budget=*/4, FhnEvictionPolicy::Lru);
  ASSERT_TRUE(plan.has_value());

  EXPECT_EQ(plan->at(2).evict, (std::vector<uint32_t>{2}));
  // b2 must come back just in time at i3 (the cost LRU pays here).
  EXPECT_EQ(plan->at(3).prefetch, (std::vector<uint32_t>{2}));
  // a1 was never evicted: no re-prefetch at i4.
  EXPECT_TRUE(plan->at(4).prefetch.empty());
  EXPECT_EQ(plan->stats().evict_count, 1u);
  // prefetches: b2@i0, a1@i1, c8@i2, b2 again @i3.
  EXPECT_EQ(plan->stats().prefetch_count, 4u);
}

// LRU ties (equal last touch) break on the lower id, like Belady's ties.
TEST(FhnMovementPlan, LruTieBreaksOnLowerId) {
  auto prog = ProgramBuilder()
                .input(1)
                .input(2)
                .inst(FHN_ADD_CC, 3, 1, 2) // i0: touches 1 and 2 equally
                .inst(FHN_NEGATE, 4, 3)    // i1
                .output(4)
                .build();

  auto plan = FhnMovementPlan::analyze(*prog, /*pinned=*/{1, 2, 4}, /*device_budget=*/3, FhnEvictionPolicy::Lru);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->at(1).evict, (std::vector<uint32_t>{1}));
}

// The default policy is Belady: a 3-arg call and an explicit Belady call
// produce identical plans.
TEST(FhnMovementPlan, DefaultPolicyIsBelady) {
  auto prog = ProgramBuilder()
                .input(1)
                .input(2)
                .input(8)
                .inst(FHN_ADD_CC, 3, 2, 2)
                .inst(FHN_ADD_CC, 4, 3, 1)
                .inst(FHN_ADD_CC, 5, 4, 8)
                .inst(FHN_ADD_CC, 6, 5, 2)
                .inst(FHN_ADD_CC, 7, 6, 1)
                .output(7)
                .build();
  auto a = FhnMovementPlan::analyze(*prog, {7}, 4);
  auto b = FhnMovementPlan::analyze(*prog, {7}, 4, FhnEvictionPolicy::Belady);
  ASSERT_TRUE(a.has_value());
  ASSERT_TRUE(b.has_value());
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(a->at(i).evict, b->at(i).evict);
    EXPECT_EQ(a->at(i).prefetch, b->at(i).prefetch);
  }
}
