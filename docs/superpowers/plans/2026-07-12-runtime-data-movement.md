# Runtime Data Movement Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A runtime pass that turns an FhnProgram's exact def-use knowledge into a buffer movement schedule — alloc-late/free-early, just-in-time prefetch, Belady-optimal eviction — executed by a plan-aware FhnDefaultExecutor and used by Session.

**Architecture:** `FhnMovementPlan` is a pure analysis over `FhnProgram` producing per-instruction action lists; `FhnDefaultExecutor` gains an overload that applies those actions around unchanged kernel dispatch via a `FhnMovementHooks` struct; Session builds hooks from `FhnRuntime` and delegates all non-input buffer lifetimes to the plan. Two new OPTIONAL data-plane ABI exports (`fhn_buffer_prefetch`/`fhn_buffer_evict`) carry transfers; they are host orchestration, never kernel-table opcodes.

**Tech Stack:** C++20, CMake + ctest + GoogleTest 1.17, existing FHN backend ABI (`include/FHN/fhn_backend_api.h`).

**Spec:** `docs/superpowers/specs/2026-07-12-runtime-data-movement-design.md` — read it first; its liveness/eviction rules are normative.

## Global Constraints

- Branch: `feat/runtime-data-movement` (already exists; work there).
- Build: `cmake -S . -B build && cmake --build build -j$(nproc)`. New sources are picked up by GLOB (`CMakeLists.txt:55`) — a **reconfigure is required** whenever a file is added.
- Test: `ctest --test-dir build` must stay 100% green after every task; run the single relevant binary (`./build/bin/<Target>`) inside TDD cycles.
- Format: every touched `.cpp/.h` must pass `clang-format --dry-run --Werror` with clang-format 18 (CI blocks on it). A pinned venv binary exists at `/tmp/claude-1000/-home-zotanika-workspace-zotanika-fhenomenon/b9ac8324-94be-4a94-b866-0a0f7fef35c9/scratchpad/cfvenv/bin/clang-format`; if absent, `python3 -m venv <dir> && <dir>/bin/pip install clang-format==18.1.8`.
- Buffer ids are single-assignment (lowering emits fresh ids); id 0 means "unused".
- Commit after every task with the trailer lines used in this repo:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01NW9CkxwDsfqYL2yBs9MWsC`.

---

### Task 1: FhnMovementPlan — liveness core (no budget)

**Files:**
- Create: `include/FHN/FhnMovementPlan.h`
- Create: `src/FHN/FhnMovementPlan.cpp`
- Create: `test/FhnMovementPlanTest.cpp`
- Modify: `test/CMakeLists.txt` (register the new test target)

**Interfaces:**
- Consumes: `FhnProgram`, `FhnInstruction`, `fhn_program_alloc/free` from `include/FHN/fhn_program.h`.
- Produces (later tasks rely on these exact names):
  - `struct fhenomenon::FhnMovementActions { std::vector<uint32_t> evict, alloc, prefetch, free; }`
  - `class fhenomenon::FhnMovementPlan` with
    `static std::optional<FhnMovementPlan> analyze(const FhnProgram&, const std::vector<uint32_t>& pinned, uint32_t device_budget = 0)`,
    `const FhnMovementActions& at(uint32_t) const`,
    `struct Stats { uint32_t high_water, alloc_count, prefetch_count, evict_count; }`, `const Stats& stats() const`.

- [ ] **Step 1: Write the failing tests**

Create `test/FhnMovementPlanTest.cpp`:

```cpp
#include "FHN/FhnMovementPlan.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

using namespace fhenomenon;

namespace {

// Owning FhnProgram builder for hand-written test programs.
struct ProgramBuilder {
  std::vector<FhnInstruction> insts;
  std::vector<uint32_t> inputs;
  std::vector<uint32_t> outputs;

  ProgramBuilder &input(uint32_t id) {
    inputs.push_back(id);
    return *this;
  }
  ProgramBuilder &inst(FhnOpCode op, uint32_t result, uint32_t a = 0, uint32_t b = 0) {
    FhnInstruction in{};
    in.opcode = op;
    in.result_id = result;
    in.operands[0] = a;
    in.operands[1] = b;
    insts.push_back(in);
    return *this;
  }
  ProgramBuilder &output(uint32_t id) {
    outputs.push_back(id);
    return *this;
  }
  std::unique_ptr<FhnProgram, decltype(&fhn_program_free)> build() {
    auto *p = fhn_program_alloc(static_cast<uint32_t>(insts.size()), static_cast<uint32_t>(inputs.size()),
                                static_cast<uint32_t>(outputs.size()));
    for (uint32_t i = 0; i < p->num_instructions; ++i)
      p->instructions[i] = insts[i];
    for (uint32_t i = 0; i < p->num_inputs; ++i)
      p->input_ids[i] = inputs[i];
    for (uint32_t i = 0; i < p->num_outputs; ++i)
      p->output_ids[i] = outputs[i];
    return {p, &fhn_program_free};
  }
};

} // namespace

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
  auto prog = ProgramBuilder()
                .input(1)
                .input(2)
                .inst(FHN_ADD_CC, 3, 1, 1)
                .inst(FHN_MULT_CC, 4, 3, 2)
                .output(4)
                .build();

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
  auto prog = ProgramBuilder()
                .input(1)
                .input(2)
                .inst(FHN_ADD_CC, 3, 1, 2)
                .inst(FHN_ADD_CC, 4, 3, 3)
                .output(4)
                .build();

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
```

Create `include/FHN/FhnMovementPlan.h` (declaration only, so the test compiles then fails at link/behavior):

```cpp
#pragma once

#include "FHN/fhn_program.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace fhenomenon {

// One instruction slot's data movement actions.
// Pre-instruction order is evict -> alloc -> prefetch: evictions make room
// before allocations and transfers claim it. free applies post-instruction.
struct FhnMovementActions {
  std::vector<uint32_t> evict;
  std::vector<uint32_t> alloc;
  std::vector<uint32_t> prefetch;
  std::vector<uint32_t> free;
};

// A runtime data movement schedule for one FhnProgram.
//
// FhnPrograms are straight-line and data-oblivious, so the full def-use
// chain of every buffer id is exact at plan time. analyze() turns that into
// alloc-at-def / free-after-last-use / prefetch-before-use actions, and —
// under a device budget — Belady-optimal eviction (evict the resident
// buffer whose next use is farthest; exact here, not a heuristic, because
// the future is fully known).
class FhnMovementPlan {
  public:
  struct Stats {
    uint32_t high_water = 0; // max simultaneously device-resident buffers
    uint32_t alloc_count = 0;
    uint32_t prefetch_count = 0;
    uint32_t evict_count = 0;
  };

  // pinned: ids that must survive execution; the plan never frees them.
  // Non-pinned lifetimes belong to the plan — callers must pin every buffer
  // they own (Session pins its inputs and write-back targets).
  // device_budget: max simultaneously device-resident buffers, 0 = unlimited.
  // nullopt: invalid program (zero/duplicate defs, operand used before or
  // without a def) or infeasible budget (one instruction's working set
  // exceeds it).
  static std::optional<FhnMovementPlan> analyze(const FhnProgram &program, const std::vector<uint32_t> &pinned,
                                                uint32_t device_budget = 0);

  const FhnMovementActions &at(uint32_t inst_index) const { return actions_[inst_index]; }
  const Stats &stats() const { return stats_; }

  private:
  FhnMovementPlan() = default;

  std::vector<FhnMovementActions> actions_;
  Stats stats_;
};

} // namespace fhenomenon
```

Register the test target in `test/CMakeLists.txt`, directly after the `ParameterTest` block:

```cmake
add_executable(FhnMovementPlanTest FhnMovementPlanTest.cpp)
target_link_libraries(FhnMovementPlanTest PRIVATE ${PROJECT_LIB_NAME} gtest_main)
add_gtest_target_to_ctest(FhnMovementPlanTest)
```

Create `src/FHN/FhnMovementPlan.cpp` with only a stub so the target links:

```cpp
#include "FHN/FhnMovementPlan.h"

namespace fhenomenon {

std::optional<FhnMovementPlan> FhnMovementPlan::analyze(const FhnProgram &, const std::vector<uint32_t> &, uint32_t) {
  return std::nullopt;
}

} // namespace fhenomenon
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cmake -S . -B build && cmake --build build --target FhnMovementPlanTest -j$(nproc)
./build/bin/FhnMovementPlanTest
```

Expected: the two rejection tests PASS (stub returns nullopt), all four scheduling tests FAIL on `ASSERT_TRUE(plan.has_value())`. That is the correct RED shape.

- [ ] **Step 3: Implement the analysis**

Replace `src/FHN/FhnMovementPlan.cpp` with:

```cpp
#include "FHN/FhnMovementPlan.h"

#include <algorithm>
#include <limits>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace fhenomenon {

std::optional<FhnMovementPlan> FhnMovementPlan::analyze(const FhnProgram &program,
                                                        const std::vector<uint32_t> &pinned,
                                                        uint32_t device_budget) {
  constexpr int64_t kBeforeProgram = -1;
  constexpr int64_t kNever = std::numeric_limits<int64_t>::max();

  // Definitions: inputs are defined before the program; each instruction
  // defines its (single-assignment) result id.
  std::unordered_map<uint32_t, int64_t> def_pos;
  for (uint32_t i = 0; i < program.num_inputs; ++i) {
    const uint32_t id = program.input_ids[i];
    if (id == 0 || !def_pos.emplace(id, kBeforeProgram).second)
      return std::nullopt;
  }
  for (uint32_t i = 0; i < program.num_instructions; ++i) {
    const uint32_t id = program.instructions[i].result_id;
    if (id == 0 || !def_pos.emplace(id, static_cast<int64_t>(i)).second)
      return std::nullopt;
  }

  // Uses, ascending per id by construction. Every use must be after a def.
  std::unordered_map<uint32_t, std::vector<int64_t>> uses;
  for (uint32_t i = 0; i < program.num_instructions; ++i) {
    const FhnInstruction &inst = program.instructions[i];
    for (int j = 0; j < 4; ++j) {
      const uint32_t id = inst.operands[j];
      if (id == 0)
        continue;
      auto it = def_pos.find(id);
      if (it == def_pos.end() || it->second >= static_cast<int64_t>(i))
        return std::nullopt;
      uses[id].push_back(static_cast<int64_t>(i));
    }
  }

  const std::unordered_set<uint32_t> pinned_set(pinned.begin(), pinned.end());

  // First use strictly after position `pos` (kNever if none).
  auto next_use_after = [&uses, kNever](uint32_t id, int64_t pos) -> int64_t {
    auto it = uses.find(id);
    if (it == uses.end())
      return kNever;
    const auto &v = it->second;
    auto p = std::upper_bound(v.begin(), v.end(), pos);
    return p == v.end() ? kNever : *p;
  };

  FhnMovementPlan plan;
  plan.actions_.resize(program.num_instructions);
  std::set<uint32_t> resident; // ordered: deterministic Belady tie-break on lower id

  for (uint32_t i = 0; i < program.num_instructions; ++i) {
    const FhnInstruction &inst = program.instructions[i];
    FhnMovementActions &act = plan.actions_[i];
    const int64_t pos = static_cast<int64_t>(i);

    std::set<uint32_t> working;
    working.insert(inst.result_id);
    std::set<uint32_t> operand_set;
    for (int j = 0; j < 4; ++j)
      if (inst.operands[j] != 0) {
        working.insert(inst.operands[j]);
        operand_set.insert(inst.operands[j]);
      }

    std::vector<uint32_t> to_prefetch;
    for (uint32_t id : operand_set)
      if (!resident.count(id))
        to_prefetch.push_back(id);

    if (device_budget > 0) {
      if (working.size() > device_budget)
        return std::nullopt; // one instruction cannot fit — infeasible
      // Belady: make room for the result alloc + missing operands by
      // evicting residents (outside the working set) whose next use is
      // farthest; kNever (no future use) sorts farthest of all, and the
      // ordered set breaks ties on the lower id.
      const uint64_t incoming = 1 /*result alloc*/ + to_prefetch.size();
      while (resident.size() + incoming > device_budget) {
        bool found = false;
        uint32_t victim = 0;
        int64_t victim_next = -1;
        for (uint32_t id : resident) {
          if (working.count(id))
            continue;
          const int64_t nu = next_use_after(id, pos - 1);
          if (!found || nu > victim_next) {
            found = true;
            victim = id;
            victim_next = nu;
          }
        }
        if (!found)
          return std::nullopt; // working set already fills the device
        act.evict.push_back(victim);
        resident.erase(victim);
        plan.stats_.evict_count++;
      }
    }

    act.alloc.push_back(inst.result_id);
    resident.insert(inst.result_id);
    plan.stats_.alloc_count++;

    for (uint32_t id : to_prefetch) {
      act.prefetch.push_back(id);
      resident.insert(id);
      plan.stats_.prefetch_count++;
    }

    plan.stats_.high_water = std::max(plan.stats_.high_water, static_cast<uint32_t>(resident.size()));

    // Free everything whose last use has passed: operands with no later
    // use, and a result nothing ever reads.
    for (uint32_t id : working) {
      if (pinned_set.count(id))
        continue;
      const bool dead = (id == inst.result_id) ? (uses.find(id) == uses.end()) : (next_use_after(id, pos) == kNever);
      if (dead) {
        act.free.push_back(id);
        resident.erase(id);
      }
    }
  }

  // Epilogue: unused, unpinned inputs were never resident but their
  // lifetime still belongs to the plan — release them at the end.
  if (program.num_instructions > 0) {
    FhnMovementActions &last = plan.actions_[program.num_instructions - 1];
    for (uint32_t k = 0; k < program.num_inputs; ++k) {
      const uint32_t id = program.input_ids[k];
      if (!pinned_set.count(id) && uses.find(id) == uses.end())
        last.free.push_back(id);
    }
  }

  return plan;
}

} // namespace fhenomenon
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cmake --build build --target FhnMovementPlanTest -j$(nproc) && ./build/bin/FhnMovementPlanTest
```

Expected: all 6 tests PASS. Note the `free` ordering the tests assert comes
from iterating the ordered `working` set (ascending ids).

- [ ] **Step 5: Full suite, format, commit**

```bash
ctest --test-dir build   # 14/14 green (13 existing + FhnMovementPlanTest)
<clang-format> --dry-run --Werror include/FHN/FhnMovementPlan.h src/FHN/FhnMovementPlan.cpp test/FhnMovementPlanTest.cpp
git add include/FHN/FhnMovementPlan.h src/FHN/FhnMovementPlan.cpp test/FhnMovementPlanTest.cpp test/CMakeLists.txt
git commit -m "feat: FhnMovementPlan — liveness-driven runtime movement schedule"
```

---

### Task 2: Belady eviction under a device budget

**Files:**
- Modify: `test/FhnMovementPlanTest.cpp` (append tests)
- Modify: `src/FHN/FhnMovementPlan.cpp` (only if a test exposes a defect — the budget logic shipped in Task 1)

**Interfaces:**
- Consumes/Produces: unchanged from Task 1 (`device_budget` parameter now exercised).

- [ ] **Step 1: Write the failing/pinning tests**

Append to `test/FhnMovementPlanTest.cpp`:

```cpp
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
                .input(1) // a1: used i1, i4
                .input(2) // b2: used i0, i3
                .input(8) // c8: used i2 only
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
```

- [ ] **Step 2: Run to verify state**

```bash
cmake --build build --target FhnMovementPlanTest -j$(nproc) && ./build/bin/FhnMovementPlanTest
```

Expected: Task 1's implementation already contains the budget logic; these
tests are its first exercise. If all four PASS immediately, verify RED
validity by temporarily flipping the Belady comparison (`nu > victim_next` →
`nu < victim_next`) and rerunning: `BeladyEvictsFarthestNextUseNotLru` must
FAIL (proving the test discriminates), then restore the comparison. If any
test FAILS with the correct implementation, fix `FhnMovementPlan.cpp` — the
tests are normative (spec: farthest-next-use, kNever farthest, lower-id tie,
infeasible → nullopt).

- [ ] **Step 3: Full suite, format, commit**

```bash
ctest --test-dir build
<clang-format> --dry-run --Werror test/FhnMovementPlanTest.cpp src/FHN/FhnMovementPlan.cpp
git add test/FhnMovementPlanTest.cpp src/FHN/FhnMovementPlan.cpp
git commit -m "test: pin Belady eviction semantics of FhnMovementPlan"
```

---

### Task 3: ABI exports + FhnRuntime plumbing

**Files:**
- Modify: `include/FHN/fhn_backend_api.h` (two typedefs + doc)
- Modify: `include/Backend/Backend.h:33-43` (FhnRuntime gains two members)
- Modify: `include/Backend/External.h` (vtable struct gains two members — find `FhnBufferAllocFn buffer_alloc;` in the vtable and add below it)
- Modify: `src/Backend/External.cpp:53-59` area (resolve optional symbols) and `:101` (runtime_ aggregate init)
- Modify: `test/FhnExternalBackendTest.cpp` (pin optional-absent resolution)

**Interfaces:**
- Produces:
  - `typedef int (*FhnBufferPrefetchFn)(FhnBackendCtx *ctx, FhnBuffer *buffer);`
  - `typedef int (*FhnBufferEvictFn)(FhnBackendCtx *ctx, FhnBuffer *buffer);`
  - `FhnRuntime::prefetch` / `FhnRuntime::evict` (nullable), used by Task 5.

- [ ] **Step 1: Write the failing test**

Append to `test/FhnExternalBackendTest.cpp` (follow the file's existing fixture pattern for loading the toyfhe_fhn library — reuse whatever helper the `ResolvesDataPlane` test uses):

```cpp
// ToyFHE has a single memory space and exports no movement hooks; the
// runtime must surface them as null so plan execution skips the actions.
TEST(FhnExternalBackend, MovementHooksAbsentResolveToNull) {
  ExternalBackend backend(libPath(), "toyfhe_");
  const FhnRuntime *rt = backend.fhnRuntime();
  ASSERT_NE(rt, nullptr);
  EXPECT_EQ(rt->prefetch, nullptr);
  EXPECT_EQ(rt->evict, nullptr);
}
```

(`libPath()` here stands for however the existing tests in this file compute
the toyfhe_fhn library path and prefix — copy the exact call from the
neighboring `ResolvesDataPlane` test rather than inventing a new helper.)

- [ ] **Step 2: Verify it fails to compile**

```bash
cmake --build build --target FhnExternalBackendTest -j$(nproc)
```

Expected: compile error — `FhnRuntime` has no member `prefetch`. That is the RED.

- [ ] **Step 3: Implement**

In `include/FHN/fhn_backend_api.h`, directly after the `FhnBufferFreeFn` typedef, add:

```c
/* ── Optional movement hooks (data plane) ──
   Host-orchestrated residency transfers for backends with a distinct
   compute memory space (e.g. GPU). Deliberately NOT kernel-table opcodes:
   movement is not compute, and the public instruction stream stays
   compute-only. Optional and additive — absent exports mean a single
   memory space, and their addition does not bump FHN_ABI_VERSION because
   every previously conformant backend remains conformant.

   fhn_buffer_prefetch: ensure the buffer is resident in the backend's
   compute space (H2D). No-op if already resident. 0 on success.
   fhn_buffer_evict: demote the buffer to host memory (D2H), preserving
   contents; a later prefetch restores compute residency. 0 on success.
   fhn_buffer_free must accept a buffer in either residency state. */
typedef int (*FhnBufferPrefetchFn)(FhnBackendCtx *ctx, FhnBuffer *buffer);
typedef int (*FhnBufferEvictFn)(FhnBackendCtx *ctx, FhnBuffer *buffer);
```

In `include/Backend/Backend.h`, extend `FhnRuntime` (insert before `keepalive` and keep the aggregate-init order in sync with the External.cpp site below):

```cpp
struct FhnRuntime {
  FhnBackendCtx *ctx = nullptr;
  FhnDefaultExecutor *executor = nullptr;
  FhnBufferAllocFn buffer_alloc = nullptr;
  FhnBufferFreeFn buffer_free = nullptr;
  // Optional movement hooks; null = single memory space, movement skipped.
  FhnBufferPrefetchFn prefetch = nullptr;
  FhnBufferEvictFn evict = nullptr;
  // (existing keepalive member and its comment stay unchanged below)
  std::shared_ptr<void> keepalive;
};
```

In `include/Backend/External.h`, add to the vtable struct after `buffer_free`:

```cpp
  FhnBufferPrefetchFn prefetch = nullptr;
  FhnBufferEvictFn evict = nullptr;
```

In `src/Backend/External.cpp`, after the encrypt/decrypt optional resolutions (around line 63), add:

```cpp
  // Optional movement hooks: absent means a single memory space.
  vtable_.prefetch = reinterpret_cast<FhnBufferPrefetchFn>(dlsym(dl_handle_, sym("fhn_buffer_prefetch").c_str()));
  vtable_.evict = reinterpret_cast<FhnBufferEvictFn>(dlsym(dl_handle_, sym("fhn_buffer_evict").c_str()));
```

and change the runtime aggregate init at line ~101 to:

```cpp
  runtime_ = {fhn_ctx_, executor_.get(), vtable_.buffer_alloc, vtable_.buffer_free, vtable_.prefetch, vtable_.evict,
              core_};
```

`BuiltinBackend` needs no change: its `runtime_{}` members default to `nullptr`, but CHECK how it fills `runtime_` (search `runtime_` in `src/Backend/Builtin.cpp`); if it uses positional aggregate init like External, update that site the same way.

- [ ] **Step 4: Run tests**

```bash
cmake --build build -j$(nproc) && ./build/bin/FhnExternalBackendTest && ctest --test-dir build
```

Expected: new test PASSES, everything stays green.

- [ ] **Step 5: Format and commit**

```bash
<clang-format> --dry-run --Werror include/FHN/fhn_backend_api.h include/Backend/Backend.h include/Backend/External.h src/Backend/External.cpp test/FhnExternalBackendTest.cpp
git add -A include src/Backend test/FhnExternalBackendTest.cpp
git commit -m "feat: optional fhn_buffer_prefetch/evict movement hooks in the FHN ABI"
```

---

### Task 4: Plan-aware execution in FhnDefaultExecutor

**Files:**
- Modify: `include/FHN/FhnDefaultExecutor.h` (hooks struct + overload declaration)
- Modify: `src/FHN/FhnDefaultExecutor.cpp` (overload implementation)
- Modify: `test/FhnExecutorTest.cpp` (append a self-contained mock-runtime test suite)

**Interfaces:**
- Consumes: `FhnMovementPlan` (Task 1), `FhnBufferPrefetchFn`/`FhnBufferEvictFn` (Task 3).
- Produces (Task 5 relies on):

```cpp
struct FhnMovementHooks {
  FhnBackendCtx *ctx = nullptr;
  FhnBufferAllocFn buffer_alloc = nullptr; // required
  FhnBufferFreeFn buffer_free = nullptr;   // required
  FhnBufferPrefetchFn prefetch = nullptr;  // optional: null skips prefetch actions
  FhnBufferEvictFn evict = nullptr;        // optional: null skips evict actions
};
int FhnDefaultExecutor::execute(const FhnMovementHooks &hooks, const FhnProgram *program, FhnBuffer **buffers,
                                const FhnMovementPlan &plan);
```

Semantics: per instruction, apply `plan.at(i)` as evict → alloc → prefetch,
dispatch exactly like the legacy path (kernel or decompose), then apply
`free` via `hooks.buffer_free`. `buffers` arrives with input ids filled;
the executor allocates every planned id (writing into `buffers`), never
frees pinned ids on success (the plan contains no frees for them), and on
ANY failure frees every plan-allocated id not yet freed (pinned included)
and nulls their `buffers` entries before returning the error code.

- [ ] **Step 1: Write the failing tests**

Append to `test/FhnExecutorTest.cpp` a self-contained section (do not reuse
mock state from other tests in the file; keep this section's globals in an
anonymous namespace suffixed `Movement` to avoid collisions):

```cpp
#include "FHN/FhnMovementPlan.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace {

// A fake single-int "ciphertext" space. Buffers live in device_vals while
// resident; evict moves the value to host_vals (clearing the device slot),
// prefetch moves it back. The kernel reads/writes device_vals only, so a
// missed prefetch computes garbage and fails the value assertions.
struct MovementWorld {
  std::map<FhnBuffer *, long> device_vals;
  std::map<FhnBuffer *, long> host_vals;
  std::vector<std::string> log; // "alloc#N"/"free#N"/"evict#N"/"prefetch#N"/"kernel ADD r"
  std::map<FhnBuffer *, int> names;
  int next_name = 1;
  int allocs = 0, frees = 0;
  int fail_kernel_at_call = -1; // when >=0, the Nth kernel call returns -1
  int kernel_calls = 0;
};
MovementWorld *g_world = nullptr;

FhnBuffer *movementAlloc(FhnBackendCtx *) {
  auto *b = reinterpret_cast<FhnBuffer *>(new char[1]);
  g_world->names[b] = g_world->next_name++;
  g_world->device_vals[b] = 0;
  g_world->allocs++;
  g_world->log.push_back("alloc#" + std::to_string(g_world->names[b]));
  return b;
}
void movementFree(FhnBackendCtx *, FhnBuffer *b) {
  g_world->frees++;
  g_world->log.push_back("free#" + std::to_string(g_world->names[b]));
  g_world->device_vals.erase(b);
  g_world->host_vals.erase(b);
  delete[] reinterpret_cast<char *>(b);
}
int movementEvict(FhnBackendCtx *, FhnBuffer *b) {
  g_world->log.push_back("evict#" + std::to_string(g_world->names[b]));
  g_world->host_vals[b] = g_world->device_vals.at(b);
  g_world->device_vals.erase(b); // physically gone from the device
  return 0;
}
int movementPrefetch(FhnBackendCtx *, FhnBuffer *b) {
  g_world->log.push_back("prefetch#" + std::to_string(g_world->names[b]));
  if (g_world->device_vals.count(b))
    return 0; // already resident
  g_world->device_vals[b] = g_world->host_vals.at(b);
  g_world->host_vals.erase(b);
  return 0;
}
int movementAddKernel(FhnBackendCtx *, FhnBuffer *result, const FhnBuffer *const *ops, const int64_t *,
                      const double *) {
  g_world->kernel_calls++;
  if (g_world->fail_kernel_at_call >= 0 && g_world->kernel_calls == g_world->fail_kernel_at_call)
    return -1;
  // .at() throws if an operand is not device-resident: movement bugs fail loudly.
  const long a = g_world->device_vals.at(const_cast<FhnBuffer *>(ops[0]));
  const long b = g_world->device_vals.at(const_cast<FhnBuffer *>(ops[1]));
  g_world->device_vals.at(result) = a + b;
  g_world->log.push_back("kernel ADD");
  return 0;
}

FhnKernelEntry movementEntries[] = {{FHN_ADD_CC, movementAddKernel, "add_cc"}};
FhnKernelTable movementTable{1, movementEntries};

} // namespace

// Full interleaving on the Belady program from FhnMovementPlanTest:
// values must come out right even though a1 physically leaves the device.
TEST(FhnExecutorMovement, BudgetedExecutionComputesCorrectValues) {
  MovementWorld world;
  g_world = &world;

  // Same program as FhnMovementPlan.BeladyEvictsFarthestNextUseNotLru.
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
  auto plan = FhnMovementPlan::analyze(*prog, {7}, 4);
  ASSERT_TRUE(plan.has_value());

  // Caller-provided inputs: a1=10, b2=1, c8=100. The caller deliberately
  // does not pin them, so the plan frees them after their last use.
  FhnBuffer *a1 = movementAlloc(nullptr);
  FhnBuffer *b2 = movementAlloc(nullptr);
  FhnBuffer *c8 = movementAlloc(nullptr);
  world.device_vals[a1] = 10;
  world.device_vals[b2] = 1;
  world.device_vals[c8] = 100;

  std::vector<FhnBuffer *> buffers(9, nullptr);
  buffers[1] = a1;
  buffers[2] = b2;
  buffers[8] = c8;

  FhnDefaultExecutor executor(&movementTable);
  FhnMovementHooks hooks{nullptr, movementAlloc, movementFree, movementPrefetch, movementEvict};
  ASSERT_EQ(executor.execute(hooks, prog.get(), buffers.data(), *plan), 0);

  // 3=1+1=2; 4=2+10=12; 5=12+100=112; 6=112+1=113; 7=113+10=123.
  EXPECT_EQ(world.device_vals.at(buffers[7]), 123);
  // a1 (buffer name #1) was evicted then prefetched back.
  EXPECT_NE(std::find(world.log.begin(), world.log.end(), "evict#1"), world.log.end());
  // Everything except the pinned output was freed — including the unpinned
  // caller inputs, whose lifetime the plan owns by contract.
  EXPECT_EQ(world.allocs - world.frees, 1); // only pinned id 7 survives
  EXPECT_NE(buffers[7], nullptr);
}

// Null hooks skip movement actions but execution still works when
// everything shares one memory space (host==device in the mock: seed
// device_vals directly and use null prefetch/evict).
TEST(FhnExecutorMovement, NullHooksSkipMovement) {
  MovementWorld world;
  g_world = &world;

  auto prog = ProgramBuilder().input(1).input(2).inst(FHN_ADD_CC, 3, 1, 2).output(3).build();
  auto plan = FhnMovementPlan::analyze(*prog, {3}, 0);
  ASSERT_TRUE(plan.has_value());

  FhnBuffer *a = movementAlloc(nullptr);
  FhnBuffer *b = movementAlloc(nullptr);
  world.device_vals[a] = 4;
  world.device_vals[b] = 5;
  std::vector<FhnBuffer *> buffers(4, nullptr);
  buffers[1] = a;
  buffers[2] = b;

  FhnDefaultExecutor executor(&movementTable);
  FhnMovementHooks hooks{nullptr, movementAlloc, movementFree, nullptr, nullptr};
  ASSERT_EQ(executor.execute(hooks, prog.get(), buffers.data(), *plan), 0);
  EXPECT_EQ(world.device_vals.at(buffers[3]), 9);
  for (const auto &line : world.log) {
    EXPECT_EQ(line.find("prefetch"), std::string::npos);
    EXPECT_EQ(line.find("evict"), std::string::npos);
  }
}

// A mid-program kernel failure must free every plan-allocated buffer —
// pinned included — and null their entries. Caller inputs stay untouched.
TEST(FhnExecutorMovement, FailureFreesAllPlanAllocations) {
  MovementWorld world;
  g_world = &world;
  world.fail_kernel_at_call = 2;

  auto prog = ProgramBuilder()
                .input(1)
                .input(2)
                .inst(FHN_ADD_CC, 3, 1, 2)
                .inst(FHN_ADD_CC, 4, 3, 3)
                .output(4)
                .build();
  // Inputs are pinned here: this test's caller keeps ownership of a and b,
  // and asserts below that the failure path leaves them untouched.
  auto plan = FhnMovementPlan::analyze(*prog, {1, 2, 4}, 0);
  ASSERT_TRUE(plan.has_value());

  FhnBuffer *a = movementAlloc(nullptr);
  FhnBuffer *b = movementAlloc(nullptr);
  world.device_vals[a] = 1;
  world.device_vals[b] = 2;
  std::vector<FhnBuffer *> buffers(5, nullptr);
  buffers[1] = a;
  buffers[2] = b;

  FhnDefaultExecutor executor(&movementTable);
  FhnMovementHooks hooks{nullptr, movementAlloc, movementFree, nullptr, nullptr};
  EXPECT_NE(executor.execute(hooks, prog.get(), buffers.data(), *plan), 0);

  // Every plan alloc (ids 3,4) was freed on the failure path.
  EXPECT_EQ(world.allocs - world.frees, 2); // only caller's a,b remain
  EXPECT_EQ(buffers[3], nullptr);
  EXPECT_EQ(buffers[4], nullptr);
  // Caller-owned inputs untouched.
  EXPECT_EQ(world.device_vals.at(a), 1);
}
```

These tests use `ProgramBuilder`: copy the struct from
`test/FhnMovementPlanTest.cpp` (Task 1) into this section's anonymous
namespace verbatim — test helpers are not shared across targets in this
repo.

- [ ] **Step 2: Verify RED**

```bash
cmake --build build --target FhnExecutorTest -j$(nproc)
```

Expected: compile error — no `FhnMovementHooks`, no such `execute` overload.

- [ ] **Step 3: Implement**

In `include/FHN/FhnDefaultExecutor.h` add after the includes:

```cpp
#include "FHN/FhnMovementPlan.h"
```

and inside the namespace, before the class:

```cpp
// Runtime services for plan-aware execution. ctx is passed through to every
// hook; prefetch/evict may be null (single memory space: actions skipped).
struct FhnMovementHooks {
  FhnBackendCtx *ctx = nullptr;
  FhnBufferAllocFn buffer_alloc = nullptr;
  FhnBufferFreeFn buffer_free = nullptr;
  FhnBufferPrefetchFn prefetch = nullptr;
  FhnBufferEvictFn evict = nullptr;
};
```

and inside the class, after the existing `execute`:

```cpp
  // Plan-aware execution: applies plan.at(i) around each instruction
  // (evict -> alloc -> prefetch before, free after). buffers arrives with
  // input ids filled; planned allocations are written into it. On failure
  // every plan-allocated id not yet freed is freed and nulled.
  int execute(const FhnMovementHooks &hooks, const FhnProgram *program, FhnBuffer **buffers,
              const FhnMovementPlan &plan);
```

In `src/FHN/FhnDefaultExecutor.cpp` add `#include <algorithm>` and `#include <vector>` if missing, then:

```cpp
int FhnDefaultExecutor::execute(const FhnMovementHooks &hooks, const FhnProgram *program, FhnBuffer **buffers,
                                const FhnMovementPlan &plan) {
  if (!program || !buffers || !hooks.buffer_alloc || !hooks.buffer_free)
    return -1;
  if (program->version != FHN_ABI_VERSION)
    return -1;

  std::vector<uint32_t> owned; // plan-allocated ids not yet freed
  auto fail = [&](int rc) {
    for (uint32_t id : owned) {
      if (buffers[id]) {
        hooks.buffer_free(hooks.ctx, buffers[id]);
        buffers[id] = nullptr;
      }
    }
    return rc;
  };

  for (uint32_t i = 0; i < program->num_instructions; ++i) {
    const FhnMovementActions &act = plan.at(i);

    for (uint32_t id : act.evict) {
      if (hooks.evict && hooks.evict(hooks.ctx, buffers[id]) != 0)
        return fail(-1);
    }
    for (uint32_t id : act.alloc) {
      buffers[id] = hooks.buffer_alloc(hooks.ctx);
      if (!buffers[id])
        return fail(-1);
      owned.push_back(id);
    }
    for (uint32_t id : act.prefetch) {
      if (hooks.prefetch && hooks.prefetch(hooks.ctx, buffers[id]) != 0)
        return fail(-1);
    }

    const FhnInstruction &inst = program->instructions[i];
    auto it = dispatch_.find(static_cast<int>(inst.opcode));
    if (it == dispatch_.end()) {
      if (!decompose(hooks.ctx, inst, buffers))
        return fail(-1);
    } else {
      const FhnBuffer *ops[4] = {nullptr, nullptr, nullptr, nullptr};
      for (int j = 0; j < 4; ++j) {
        if (inst.operands[j] != 0)
          ops[j] = buffers[inst.operands[j]];
      }
      const int rc = it->second(hooks.ctx, buffers[inst.result_id], ops, inst.params, inst.fparams);
      if (rc != 0)
        return fail(rc);
    }

    for (uint32_t id : act.free) {
      hooks.buffer_free(hooks.ctx, buffers[id]);
      buffers[id] = nullptr;
      owned.erase(std::remove(owned.begin(), owned.end(), id), owned.end());
    }
  }

  return 0;
}
```

- [ ] **Step 4: Run tests**

```bash
cmake --build build --target FhnExecutorTest -j$(nproc) && ./build/bin/FhnExecutorTest && ctest --test-dir build
```

Expected: all executor tests (old + 3 new) PASS; full suite green.

- [ ] **Step 5: Format and commit**

```bash
<clang-format> --dry-run --Werror include/FHN/FhnDefaultExecutor.h src/FHN/FhnDefaultExecutor.cpp test/FhnExecutorTest.cpp
git add include/FHN/FhnDefaultExecutor.h src/FHN/FhnDefaultExecutor.cpp test/FhnExecutorTest.cpp
git commit -m "feat: plan-aware execution overload in FhnDefaultExecutor"
```

---

### Task 5: Session delegates buffer lifetimes to the plan

**Files:**
- Modify: `src/Session/Session.cpp` (rewrite `executeThroughFhnRuntime`, lines ~25-115)
- Modify: `test/SessionTest.cpp` (append one pinning test)

**Interfaces:**
- Consumes: `FhnMovementPlan::analyze` (Task 1), `FhnMovementHooks` + executor overload (Task 4), `FhnRuntime::prefetch/evict` (Task 3).
- Produces: no new interfaces; behavior contract is "existing suite stays green, memory schedule changes underneath".

- [ ] **Step 1: Write the failing test**

Append to `test/SessionTest.cpp`:

```cpp
// Two entities can end the session bound to the SAME value id (b = a).
// Adoption of plan-allocated buffers must dedupe by id, or the two
// write-backs would double-free one buffer when both entities die.
TEST(SessionTest, AliasedWriteBackTargetsShareOneBuffer) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());

  Fhenon<int> a = 0;
  Fhenon<int> b = 0;
  a.belong(profile);
  b.belong(profile);

  session->run([&]() {
    a = a + 7;
    b = a; // b's latest binding is a's value id
  });

  EXPECT_EQ(a.decrypt(), 7);
  EXPECT_EQ(b.decrypt(), 7);
}
```

- [ ] **Step 2: Verify current behavior**

```bash
cmake --build build --target SessionTest -j$(nproc) && ./build/bin/SessionTest
```

Expected: PASSES today (the current all-shared_ptr table dedupes
naturally). This test is the tripwire for the rewrite — after Step 3 it
guards the exact new failure mode (double-free segfault/ASan report), which
is why it must exist BEFORE the rewrite.

- [ ] **Step 3: Rewrite executeThroughFhnRuntime**

In `src/Session/Session.cpp`, add includes `#include "FHN/FhnMovementPlan.h"` and `#include <unordered_set>`, then replace the body of `executeThroughFhnRuntime` (keep the enclosing anonymous namespace and signature) with:

```cpp
bool executeThroughFhnRuntime(scheduler::Scheduler &scheduler, scheduler::Planner<int> &planner, const Backend &backend,
                              const FhnRuntime &runtime) {
  scheduler::LowerToFhnProgram::EntityBindings<int> bindings;
  std::unique_ptr<FhnProgram, decltype(&fhn_program_free)> program(scheduler.lowerGraph<int>(planner, &bindings),
                                                                   &fhn_program_free);
  if (!program || bindings.empty()) {
    return false;
  }

  // Walking the bindings forward, the last binding per entity wins — that
  // id holds the value the entity must observe after the run.
  std::unordered_map<Fhenon<int> *, std::pair<std::shared_ptr<Fhenon<int>>, uint32_t>> latest;
  for (const auto &[id, entity] : bindings) {
    if (entity) {
      latest[entity.get()] = {entity, id};
    }
  }

  // Pin what must survive execution: program inputs (entity-owned buffers
  // the plan must not free) and every write-back target. Superseded
  // intermediate results are deliberately NOT pinned — the plan frees them
  // at their last use.
  std::vector<uint32_t> pinned(program->input_ids, program->input_ids + program->num_inputs);
  for (const auto &[raw_entity, bound] : latest) {
    (void)raw_entity;
    pinned.push_back(bound.second);
  }

  const auto plan = FhnMovementPlan::analyze(*program, pinned, /*device_budget=*/0);
  if (!plan) {
    throw std::runtime_error("Session: FHN program failed movement analysis (operand used without a definition?)");
  }

  uint32_t max_id = 0;
  for (uint32_t i = 0; i < program->num_instructions; ++i) {
    max_id = std::max(max_id, program->instructions[i].result_id);
  }
  for (uint32_t i = 0; i < program->num_inputs; ++i) {
    max_id = std::max(max_id, program->input_ids[i]);
  }
  for (const auto &binding : bindings) {
    max_id = std::max(max_id, binding.first);
  }

  // Input provisioning only: every other id is allocated by the plan. An
  // encrypted entity bound to an input id contributes its buffer in place
  // (zero copy); a foreign owner is an error, not a skip.
  const std::unordered_set<uint32_t> input_ids(program->input_ids, program->input_ids + program->num_inputs);
  std::vector<std::shared_ptr<FhnBuffer>> input_hold(max_id + 1);
  for (const auto &[id, entity] : bindings) {
    if (!input_ids.count(id) || input_hold[id] || !entity || !entity->isEncrypted_) {
      continue;
    }
    if (const auto *ct = std::any_cast<FhnCiphertext>(&entity->ciphertext_)) {
      if (ct->owner != &backend) {
        throw std::runtime_error("Session: operand was encrypted by a different backend");
      }
      input_hold[id] = ct->buffer;
    }
  }
  for (uint32_t i = 0; i < program->num_inputs; ++i) {
    if (!input_hold[program->input_ids[i]]) {
      throw std::runtime_error("Session: operand is not encrypted — call belong() before using it in a session");
    }
  }

  std::vector<FhnBuffer *> buffers(max_id + 1, nullptr);
  for (uint32_t id = 1; id <= max_id; ++id) {
    if (input_hold[id]) {
      buffers[id] = input_hold[id].get();
    }
  }

  const FhnMovementHooks hooks{runtime.ctx, runtime.buffer_alloc, runtime.buffer_free, runtime.prefetch,
                               runtime.evict};
  const int rc = runtime.executor->execute(hooks, program.get(), buffers.data(), *plan);
  if (rc != 0) {
    throw std::runtime_error("Session: FHN executor failed with rc=" + std::to_string(rc));
  }

  // Adopt surviving pinned buffers and write back. Adoption is deduped by
  // id: two entities bound to the same value id must share one shared_ptr,
  // not wrap the same raw pointer twice (double free). Input ids reuse the
  // entity-owned shared_ptr; plan-allocated ids get a deleter sharing the
  // runtime keepalive, exactly like the old preallocation path.
  auto *ctx = runtime.ctx;
  auto free_fn = runtime.buffer_free;
  auto keepalive = runtime.keepalive;
  std::unordered_map<uint32_t, std::shared_ptr<FhnBuffer>> adopted;
  auto adopt = [&](uint32_t id) -> std::shared_ptr<FhnBuffer> {
    if (input_hold[id]) {
      return input_hold[id];
    }
    auto &slot = adopted[id];
    if (!slot) {
      slot = std::shared_ptr<FhnBuffer>(buffers[id],
                                        [ctx, free_fn, keepalive](FhnBuffer *buffer) { free_fn(ctx, buffer); });
    }
    return slot;
  };
  for (auto &[raw_entity, bound] : latest) {
    (void)raw_entity;
    bound.first->ciphertext_ = FhnCiphertext{adopt(bound.second), &backend};
    bound.first->isEncrypted_ = true;
  }

  return true;
}
```

- [ ] **Step 4: Run the full suite (this is the real gate)**

```bash
cmake --build build -j$(nproc) && ctest --test-dir build
```

Expected: 100% green — SessionTest (incl. the new alias test and the
existing read-after-write/chained/products/scalar/second-run/unencrypted
cases), FhnIntegrationTest, FhnExternalBackendTest, FhnMatvecBenchTest all
exercise this path against real ToyFHE. Debug any failure here before
proceeding; the alias test failing with a crash means adoption dedupe broke.

- [ ] **Step 5: Sanitizer pass**

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan -j$(nproc)
for t in SessionTest NoSessionTest FhnExternalBackendTest FhnIntegrationTest FhnMovementPlanTest FhnExecutorTest; do
  UBSAN_OPTIONS=print_stacktrace=1 ./build-asan/bin/$t || exit 1
done
```

Expected: zero sanitizer reports (double frees and leaks in the new
ownership flow would surface here).

- [ ] **Step 6: Format and commit**

```bash
<clang-format> --dry-run --Werror src/Session/Session.cpp test/SessionTest.cpp
git add src/Session/Session.cpp test/SessionTest.cpp
git commit -m "feat: Session executes FhnPrograms through the movement plan"
```

---

### Task 6: Bench stats and --budget knob

**Files:**
- Modify: `benchmarks/fhn_matvec_bench.cpp`

**Interfaces:**
- Consumes: `FhnMovementPlan::analyze` and `Stats` (Task 1). The bench builds its own FhnPrograms and executes them directly; find where it calls the executor (search `execute(`) and note which program variables exist (fused vs decomposed paths).

- [ ] **Step 1: Add the flag and stats reporting**

In `benchmarks/fhn_matvec_bench.cpp`:
- add `#include "FHN/FhnMovementPlan.h"`,
- extend the arg loop (lines ~101-108) with:

```cpp
    } else if (std::strcmp(argv[i], "--budget") == 0 && i + 1 < argc) {
      budget = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
```

  declaring `uint32_t budget = 0;` next to `n`/`reps` and adding
  `[--budget <max resident buffers, default 0 = unlimited>]` to `usage()`.
- after each program is built (both the fused and the decomposed variant),
  analyze and print — pin the program's outputs (`output_ids`) so the plan
  matches how a caller would run it:

```cpp
  std::vector<uint32_t> pinned(prog->output_ids, prog->output_ids + prog->num_outputs);
  const auto plan = fhenomenon::FhnMovementPlan::analyze(*prog, pinned, budget);
  if (!plan) {
    std::fprintf(stderr, "error: movement analysis failed (budget %u infeasible?)\n", budget);
    return 1;
  }
  std::printf("movement[%s]: high_water=%u allocs=%u prefetches=%u evicts=%u (budget %u)\n", label,
              plan->stats().high_water, plan->stats().alloc_count, plan->stats().prefetch_count,
              plan->stats().evict_count, budget);
```

  where `label` is the same string the bench already uses to tag the fused
  vs decomposed timing lines. Reporting only — the bench keeps executing
  through its existing (non-plan) path, so its correctness checks are
  untouched by the budget value.

- [ ] **Step 2: Run it**

```bash
cmake --build build --target fhn-bench-matvec -j$(nproc)
./build/bin/fhn-bench-matvec --n 8 --reps 1
./build/bin/fhn-bench-matvec --n 8 --reps 1 --budget 4
```

Expected: two `movement[...]` lines per run; with `--budget 4` the evict
count is nonzero and grows as the budget shrinks; exit code 0 both times
(correctness checks green). Also `ctest --test-dir build` stays green
(FhnMatvecBenchTest runs the no-budget form).

- [ ] **Step 3: Format and commit**

```bash
<clang-format> --dry-run --Werror benchmarks/fhn_matvec_bench.cpp
git add benchmarks/fhn_matvec_bench.cpp
git commit -m "feat: movement-plan stats and --budget knob in fhn-bench-matvec"
```

---

## Completion checklist (after Task 6)

- `ctest --test-dir build` fully green; sanitizer run from Task 5 Step 5 clean.
- Repo-wide `clang-format --dry-run --Werror` over `src include examples test` clean.
- Push the branch and open a PR against main titled
  "feat: runtime data movement pass — liveness, JIT prefetch, Belady eviction";
  the PR body should link the spec and note the two optional ABI exports.
