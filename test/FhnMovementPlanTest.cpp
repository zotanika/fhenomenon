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
