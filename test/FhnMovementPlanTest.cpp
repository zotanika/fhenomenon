#include "FHN/FhnMovementPlan.h"
#include "FhnTestProgramBuilder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <map>
#include <set>
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

namespace {

// Synthetic CKKS-ish model for plan tests: fresh level 2, sizes chosen so
// levels are observable through byte accounting (level 2 = 100 bytes,
// level 1 = 60, level 0 = 30).
FhnLevelModel testModel() {
  FhnLevelModel model;
  model.fresh_level = 2;
  model.bytes_by_level = {30, 60, 100};
  model.effects[FHN_ADD_CC] = FHN_LEVEL_PRESERVE;
  model.effects[FHN_MULT_CC] = FHN_LEVEL_CONSUME;
  model.effects[FHN_NEGATE] = FHN_LEVEL_PRESERVE;
  model.effects[FHN_LEVEL_DOWN] = FHN_LEVEL_SET_PARAM0;
  return model;
}

} // namespace

// Byte accounting reflects inferred levels: inputs at fresh level (100B),
// a CONSUME result one level down (60B), PRESERVE keeps the min level.
TEST(FhnMovementPlan, ByteHighWaterFollowsLevelInference) {
  auto prog = ProgramBuilder()
                .input(1)
                .input(2)
                .inst(FHN_MULT_CC, 3, 1, 2) // level 1 (60B)
                .inst(FHN_ADD_CC, 4, 3, 3)  // level 1 (60B)
                .output(4)
                .build();
  const FhnLevelModel model = testModel();
  auto plan = FhnMovementPlan::analyze(*prog, {4}, 0, FhnEvictionPolicy::Belady, &model);
  ASSERT_TRUE(plan.has_value());
  // Peak residency: i0 holds inputs 1,2 (100+100) + result 3 (60) = 260;
  // after i0 frees nothing yet pinned-wise: 1,2 die at i0 -> {3}=60;
  // i1 adds 4 (60) -> 120. Peak is 260.
  EXPECT_EQ(plan->stats().high_water_bytes, 260u);
  // Count-based high_water keeps meaning in byte mode.
  EXPECT_EQ(plan->stats().high_water, 3u);
}

// Slot mode reports no byte high-water.
TEST(FhnMovementPlan, SlotModeHasZeroByteHighWater) {
  auto prog = ProgramBuilder().input(1).input(2).inst(FHN_ADD_CC, 3, 1, 2).output(3).build();
  auto plan = FhnMovementPlan::analyze(*prog, {3});
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->stats().high_water_bytes, 0u);
}

// A CONSUME chain deeper than the parameter chain underflows -> nullopt.
TEST(FhnMovementPlan, LevelUnderflowIsRejected) {
  auto prog = ProgramBuilder()
                .input(1)
                .inst(FHN_MULT_CC, 3, 1, 1) // level 1
                .inst(FHN_MULT_CC, 4, 3, 3) // level 0
                .inst(FHN_MULT_CC, 5, 4, 4) // level -1: underflow
                .output(5)
                .build();
  const FhnLevelModel model = testModel();
  EXPECT_FALSE(FhnMovementPlan::analyze(*prog, {5}, 0, FhnEvictionPolicy::Belady, &model).has_value());
}

// SET_PARAM0 must not raise the level (v1 bug-catcher; FHN_LEVEL_REFRESH
// is the future additive escape hatch for bootstrap).
TEST(FhnMovementPlan, LevelRaiseIsRejected) {
  auto prog = ProgramBuilder()
                .input(1)
                .inst(FHN_MULT_CC, 3, 1, 1)       // level 1
                .inst_p0(FHN_LEVEL_DOWN, 4, 3, 2) // target 2 > 1: raise
                .output(4)
                .build();
  const FhnLevelModel model = testModel();
  EXPECT_FALSE(FhnMovementPlan::analyze(*prog, {4}, 0, FhnEvictionPolicy::Belady, &model).has_value());
}

// An opcode the model does not declare is rejected.
TEST(FhnMovementPlan, MissingEffectIsRejected) {
  auto prog = ProgramBuilder().input(1).inst(FHN_SUB_CC, 3, 1, 1).output(3).build();
  const FhnLevelModel model = testModel(); // no FHN_SUB_CC entry
  EXPECT_FALSE(FhnMovementPlan::analyze(*prog, {3}, 0, FhnEvictionPolicy::Belady, &model).has_value());
}

// Malformed models are rejected: table shorter than fresh_level+1 or a
// zero byte size.
TEST(FhnMovementPlan, MalformedModelIsRejected) {
  auto prog = ProgramBuilder().input(1).inst(FHN_ADD_CC, 3, 1, 1).output(3).build();
  FhnLevelModel short_model = testModel();
  short_model.bytes_by_level = {30, 60}; // fresh_level 2 needs 3 entries
  EXPECT_FALSE(FhnMovementPlan::analyze(*prog, {3}, 0, FhnEvictionPolicy::Belady, &short_model).has_value());
  FhnLevelModel zero_model = testModel();
  zero_model.bytes_by_level[1] = 0;
  EXPECT_FALSE(FhnMovementPlan::analyze(*prog, {3}, 0, FhnEvictionPolicy::Belady, &zero_model).has_value());
}

// Feasibility and admission are byte-denominated: a budget equal to the
// peak working set is feasible with zero evictions, one byte less is not.
TEST(FhnMovementPlan, ByteBudgetFeasibilityIsByteDenominated) {
  // ids: inputs a1,b2 (level 2, 100B each). i0: 3=1*2 (60B). i1: 4=3*3
  // (30B). i2: 5=3+3 (60B). i3: 6=5+1 -> needs a1 (100B) back.
  auto prog = ProgramBuilder()
                .input(1)
                .input(2)
                .inst(FHN_MULT_CC, 3, 1, 2) // 60B; a1,b2 die after? a1 used at i3
                .inst(FHN_MULT_CC, 4, 3, 3) // 30B
                .inst(FHN_ADD_CC, 5, 3, 3)  // 60B; 3 dies here
                .inst(FHN_ADD_CC, 6, 5, 1)  // needs a1 again
                .output(6)
                .build();
  const FhnLevelModel model = testModel();
  // Peak working set is i0: inputs 1,2 (100+100) + result 3 (60) = 260.
  auto plan = FhnMovementPlan::analyze(*prog, {6}, 260, FhnEvictionPolicy::Belady, &model);
  ASSERT_TRUE(plan.has_value());
  // At 260 everything fits at every step (traced in plan review):
  EXPECT_EQ(plan->stats().evict_count, 0u);
  // ...and at 220 the i0 working set (260) is infeasible:
  EXPECT_FALSE(FhnMovementPlan::analyze(*prog, {6}, 220, FhnEvictionPolicy::Belady, &model).has_value());
}

// Under byte pressure the eviction loop must free enough BYTES, evicting
// TWO small residents to admit a big working set — count-based logic
// would never evict two in one pre-step here.
TEST(FhnMovementPlan, ByteBudgetEvictsMultipleSmallForBigWorkingSet) {
  // Levels/sizes per testModel(): fresh 2=100B, 1=60B, 0=30B.
  // i0: 3=1*1 (60B), 1 dies.        i1: 4=3*3 (30B), 3 dies; 4 used @i5.
  // i2: 5=2*2 (60B), 2 dies.        i3: 6=5*5 (30B), 5 dies; 6 used @i5.
  // i4: 8=7+7 (100B) — working {7,8} = 200B exactly; residents {4,6}
  //     (30+30) are idle with future uses, so budget 200 forces BOTH out.
  // i5: 9=4+6 — both come back (prefetch {4,6}).
  auto prog = ProgramBuilder()
                .input(1)
                .input(2)
                .input(7)
                .inst(FHN_MULT_CC, 3, 1, 1) // i0
                .inst(FHN_MULT_CC, 4, 3, 3) // i1
                .inst(FHN_MULT_CC, 5, 2, 2) // i2
                .inst(FHN_MULT_CC, 6, 5, 5) // i3
                .inst(FHN_ADD_CC, 8, 7, 7)  // i4: pressure generator
                .inst(FHN_ADD_CC, 9, 4, 6)  // i5
                .output(9)
                .build();
  const FhnLevelModel model = testModel();
  auto plan = FhnMovementPlan::analyze(*prog, {9}, 200, FhnEvictionPolicy::Belady, &model);
  ASSERT_TRUE(plan.has_value());
  // Both 30-byte residents leave in one pre-step (Belady tie on next use
  // i5 breaks to the lower id first).
  EXPECT_EQ(plan->at(4).evict, (std::vector<uint32_t>{4, 6}));
  EXPECT_EQ(plan->stats().evict_count, 2u);
  EXPECT_EQ(plan->at(5).prefetch, (std::vector<uint32_t>{4, 6}));
}

// A garbage effect value from a buggy backend must reject the model, not
// silently plan as PRESERVE.
TEST(FhnMovementPlan, OutOfRangeEffectIsRejected) {
  auto prog = ProgramBuilder().input(1).inst(FHN_ADD_CC, 3, 1, 1).output(3).build();
  FhnLevelModel model = testModel();
  for (uint32_t raw : {uint32_t{42}, uint32_t{0x80000000}, std::numeric_limits<uint32_t>::max()}) {
    SCOPED_TRACE(raw);
    model.effects[FHN_ADD_CC] = static_cast<FhnLevelEffect>(raw);
    EXPECT_FALSE(FhnMovementPlan::analyze(*prog, {3}, 0, FhnEvictionPolicy::Belady, &model).has_value());
  }
}

// Regression documentation of the byte-mode policy's limit: knowing all
// future uses does not make greedy farthest-use eviction globally optimal
// when buffer sizes differ. Both schedules start after input normalization.
TEST(FhnMovementPlan, VariableSizeFarthestUseHasCheaperFeasibleAlternative) {
  auto prog = ProgramBuilder()
                .input(2)
                .input(3)
                .inst_p0(FHN_LEVEL_DOWN, 4, 3, 0)
                .inst_p0(FHN_LEVEL_DOWN, 5, 2, 2)
                .inst_p0(FHN_LEVEL_DOWN, 6, 5, 0)
                .inst_p0(FHN_LEVEL_DOWN, 7, 3, 1)
                .inst_p0(FHN_LEVEL_DOWN, 8, 2, 2)
                .inst_p0(FHN_LEVEL_DOWN, 9, 4, 0)
                .inst_p0(FHN_LEVEL_DOWN, 10, 6, 0)
                .inst_p0(FHN_LEVEL_DOWN, 11, 4, 0)
                .inst_p0(FHN_LEVEL_DOWN, 12, 10, 0)
                .output(12)
                .build();
  FhnLevelModel model;
  model.fresh_level = 2;
  model.bytes_by_level = {1, 2, 3};
  model.effects[FHN_LEVEL_DOWN] = FHN_LEVEL_SET_PARAM0;
  auto plan = FhnMovementPlan::analyze(*prog, {12}, 7, FhnEvictionPolicy::Belady, &model);
  ASSERT_TRUE(plan);
  const std::map<uint32_t, uint64_t> bytes = {{2, 3}, {3, 3}, {4, 1},  {5, 3},  {6, 1}, {7, 2},
                                              {8, 3}, {9, 1}, {10, 1}, {11, 1}, {12, 1}};
  uint64_t greedy_bytes = 0;
  for (uint32_t i = 0; i < 9; ++i) {
    for (auto id : plan->at(i).evict)
      greedy_bytes += bytes.at(id);
    for (auto id : plan->at(i).prefetch)
      greedy_bytes += bytes.at(id);
  }
  EXPECT_EQ(plan->stats().evict_count + plan->stats().prefetch_count, 10u);
  EXPECT_EQ(greedy_bytes, 22u);

  // Hand-checked alternative: retain the small buffer 4 at i1, evict only
  // the larger buffer 3; later drop 2 at i2 and 4 at i4. Check every peak
  // and each use, rather than merely asserting a claimed oracle score.
  const std::vector<std::vector<uint32_t>> evictions = {{}, {3}, {2}, {}, {4}, {}, {}, {}, {}};
  const std::vector<std::vector<uint32_t>> frees = {{}, {}, {5}, {3, 7}, {2, 8}, {9}, {6}, {4, 11}, {10}};
  std::set<uint32_t> resident;
  uint64_t alternative_transfers = 0, alternative_bytes = 0;
  for (uint32_t i = 0; i < 9; ++i) {
    const auto &inst = prog->instructions[i];
    for (auto id : evictions[i]) {
      ASSERT_NE(id, inst.operands[0]);
      ASSERT_EQ(resident.erase(id), 1u);
      ++alternative_transfers;
      alternative_bytes += bytes.at(id);
    }
    if (resident.insert(inst.operands[0]).second) {
      ++alternative_transfers;
      alternative_bytes += bytes.at(inst.operands[0]);
    }
    ASSERT_TRUE(resident.insert(inst.result_id).second);
    uint64_t peak_bytes = 0;
    for (auto id : resident)
      peak_bytes += bytes.at(id);
    EXPECT_LE(peak_bytes, 7u);
    for (auto id : frees[i])
      ASSERT_EQ(resident.erase(id), 1u);
  }
  EXPECT_EQ(resident, (std::set<uint32_t>{12}));
  EXPECT_EQ(alternative_transfers, 8u);
  EXPECT_EQ(alternative_bytes, 20u);
  EXPECT_LT(alternative_bytes, greedy_bytes);
}

TEST(FhnMovementPlan, BudgetPreludeAndMetricsDeclareNormalizationSeparately) {
  auto prog = ProgramBuilder().input(1).input(2).input(3).inst(FHN_ADD_CC, 4, 1, 2).output(4).build();
  auto budgeted = FhnMovementPlan::analyze(*prog, {1, 2, 3, 4}, 3);
  auto unlimited = FhnMovementPlan::analyze(*prog, {1, 2, 3, 4});
  ASSERT_TRUE(budgeted);
  ASSERT_TRUE(unlimited);
  EXPECT_EQ(budgeted->initialEvictions(), (std::vector<uint32_t>{1, 2, 3}));
  EXPECT_EQ(budgeted->stats().initial_evict_count, 3u);
  EXPECT_EQ(budgeted->stats().evict_count, 0u);
  EXPECT_EQ(budgeted->stats().high_water, 3u);
  EXPECT_TRUE(unlimited->initialEvictions().empty());
  EXPECT_EQ(unlimited->stats().initial_evict_count, 0u);
}

TEST(FhnMovementPlan, SemanticSnapshotAcceptsEqualCopiesAndDetectsEveryArrayField) {
  auto original = ProgramBuilder().input(1).input(2).inst(FHN_ADD_CC, 3, 1, 2).output(3).build();
  auto copy = ProgramBuilder().input(1).input(2).inst(FHN_ADD_CC, 3, 1, 2).output(3).build();
  auto plan = FhnMovementPlan::analyze(*original, {3});
  ASSERT_TRUE(plan);
  ASSERT_TRUE(plan->matches(*copy));
  for (int j = 0; j < 4; ++j) {
    copy->instructions[0].operands[j] ^= 1;
    EXPECT_FALSE(plan->matches(*copy));
    copy->instructions[0].operands[j] ^= 1;
    copy->instructions[0].params[j] = 123;
    EXPECT_FALSE(plan->matches(*copy));
    copy->instructions[0].params[j] = 0;
  }
  for (int j = 0; j < 2; ++j) {
    copy->instructions[0].fparams[j] = -0.0;
    EXPECT_FALSE(plan->matches(*copy));
    copy->instructions[0].fparams[j] = 0.0;
  }
  EXPECT_TRUE(plan->matches(*copy));
}

TEST(FhnMovementPlan, RejectsUnrepresentableByteWorkingSet) {
  auto prog = ProgramBuilder().input(1).inst(FHN_NEGATE, 2, 1).output(2).build();
  FhnLevelModel model;
  model.fresh_level = 0;
  model.bytes_by_level = {uint64_t{1} << 63};
  model.effects[FHN_NEGATE] = FHN_LEVEL_PRESERVE;
  for (uint64_t budget : {uint64_t{0}, uint64_t{1}, std::numeric_limits<uint64_t>::max()})
    EXPECT_FALSE(FhnMovementPlan::analyze(*prog, {1, 2}, budget, FhnEvictionPolicy::Belady, &model));
}

TEST(FhnMovementPlan, ByteBudgetAdmissionDoesNotWrapAccumulatedResidency) {
  auto prog =
    ProgramBuilder().input(1).inst(FHN_NEGATE, 2, 1).inst(FHN_NEGATE, 3, 1).inst(FHN_NEGATE, 4, 1).output(4).build();
  FhnLevelModel model;
  model.fresh_level = 0;
  model.bytes_by_level = {uint64_t{1} << 62};
  model.effects[FHN_NEGATE] = FHN_LEVEL_PRESERVE;
  auto plan = FhnMovementPlan::analyze(*prog, {1, 2, 3, 4}, std::numeric_limits<uint64_t>::max(),
                                       FhnEvictionPolicy::Belady, &model);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->at(2).evict, (std::vector<uint32_t>{2}));
  EXPECT_EQ(plan->stats().high_water_bytes, (uint64_t{3} << 62));
  // Unlimited mode still reports high-water honestly; a sum exceeding the
  // statistic's representable range must not silently wrap back to zero.
  EXPECT_FALSE(FhnMovementPlan::analyze(*prog, {1, 2, 3, 4}, 0, FhnEvictionPolicy::Belady, &model));
}
