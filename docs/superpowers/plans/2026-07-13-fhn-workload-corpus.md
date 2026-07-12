# FHN Real-Workload Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `fhn-corpus` — 12 IR-level workload shapes with plaintext oracles, planned under Belady vs a new LRU baseline across a budget sweep, executed against any dlopen'd FHN backend, reporting headline transfer savings.

**Architecture:** A benchmarking-only `FhnEvictionPolicy::Lru` joins `FhnMovementPlan::analyze` (shared simulation, victim selection differs). `benchmarks/corpus/` holds a small static lib (shape builder, plaintext slot oracle, pure-C-ABI dlopen loader) plus a runner main; unit tests cover the oracle, the loader, and per-shape invariants; CI runs the binary against ToyFHE.

**Tech Stack:** C++20, CMake + ctest + GoogleTest 1.17, FHN ABI (`include/FHN/fhn_backend_api.h`), `FhnMovementPlan`/`FhnDefaultExecutor`.

**Spec:** `docs/superpowers/specs/2026-07-13-fhn-workload-corpus-design.md` — normative.

## Global Constraints

- Branch: `feat/fhn-corpus` (exists; work there).
- Build: `cmake -S . -B build && cmake --build build -j$(nproc)`; **reconfigure whenever a file is added** (sources are GLOBbed).
- `ctest --test-dir build` 100% green after every task.
- Format: every touched `.cpp/.h` must pass `clang-format --dry-run --Werror` under **BOTH** clang-format 18.1.3 (CI's apt build) and 18.1.8. Pinned venv binaries: `/tmp/claude-1000/-home-zotanika-workspace-zotanika-fhenomenon/b9ac8324-94be-4a94-b866-0a0f7fef35c9/scratchpad/cf1813/bin/clang-format` and `.../cfvenv/bin/clang-format` (if missing: `python3 -m venv <dir> && <dir>/bin/pip install clang-format==18.1.3` / `==18.1.8`).
- Apple clang gotcha: NEVER explicitly capture a constexpr in a lambda (`-Wunused-lambda-capture` + `-Werror` breaks the macOS build); avoid `<<`-concatenated string literals in one statement (clang-format version drift) — use adjacent string literals.
- Determinism: no RNG anywhere in the corpus — closed-form input patterns only.
- Plaintext magnitudes: every shape's oracle values must stay well inside int64, and shapes intended to execute on ToyFHE must stay under its plaintext modulus 2^32 (the sizes in Tasks 4-5 are chosen for this — do not change constants).
- Commit after every task with trailers:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01NW9CkxwDsfqYL2yBs9MWsC`.

---

### Task 1: `FhnEvictionPolicy` (Belady default, benchmarking-only LRU)

**Files:**
- Modify: `include/FHN/FhnMovementPlan.h`
- Modify: `src/FHN/FhnMovementPlan.cpp`
- Modify: `test/FhnMovementPlanTest.cpp`
- Modify: `test/FhnExecutorTest.cpp`

**Interfaces:**
- Consumes: existing `FhnMovementPlan::analyze(program, pinned, budget)`.
- Produces (Task 6 relies on): `enum class FhnEvictionPolicy { Belady, Lru };` and the 4th parameter `FhnEvictionPolicy policy = FhnEvictionPolicy::Belady` on `analyze`. No caller besides tests and the corpus ever passes `Lru`.

- [ ] **Step 1: Write the failing tests**

Append to `test/FhnMovementPlanTest.cpp`:

```cpp
// Same program as BeladyEvictsFarthestNextUseNotLru, planned under the
// benchmarking-only LRU baseline: at i2 the least-recently-touched
// candidate is b2 (last touch i0; a1 was touched at i1), so LRU evicts b2
// where Belady evicts a1 — and pays a re-prefetch at i3 that Belady never
// pays. This pins that the two policies genuinely diverge.
TEST(FhnMovementPlan, LruEvictsLeastRecentlyUsedNotBelady) {
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
```

Append to the `FhnExecutorMovement` section of `test/FhnExecutorTest.cpp` (it already has `MovementWorld`, `movementAlloc/Free/Prefetch/Evict`, `movementTable`, and `ProgramBuilder` via the shared header):

```cpp
// An LRU-planned program must execute to the same values as Belady — the
// policies move buffers differently but never change results.
TEST(FhnExecutorMovement, BudgetedExecutionCorrectUnderLru) {
  MovementWorld world;
  g_world = &world;

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
  auto plan = FhnMovementPlan::analyze(*prog, {7}, 4, FhnEvictionPolicy::Lru);
  ASSERT_TRUE(plan.has_value());

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

  // 3=2; 4=12; 5=112; 6=113; 7=123 — same as the Belady run.
  EXPECT_EQ(world.device_vals.at(buffers[7]), 123);
  // LRU's distinguishing move: b2 (buffer name #2) was evicted.
  EXPECT_NE(std::find(world.log.begin(), world.log.end(), "evict#2"), world.log.end());
  movementFree(nullptr, buffers[7]);
}
```

- [ ] **Step 2: Verify RED**

```bash
cmake --build build --target FhnMovementPlanTest -j$(nproc)
```

Expected: compile error — `FhnEvictionPolicy` is not declared.

- [ ] **Step 3: Implement**

`include/FHN/FhnMovementPlan.h` — before the class:

```cpp
// Eviction policy for budgeted planning. Belady (exact optimal, enabled by
// the data-oblivious IR's perfect future knowledge) is the default and the
// only policy production paths use. Lru exists SOLELY as a benchmarking
// baseline so the corpus can quantify what exact future knowledge saves.
enum class FhnEvictionPolicy { Belady, Lru };
```

Change the `analyze` declaration to:

```cpp
  static std::optional<FhnMovementPlan> analyze(const FhnProgram &program, const std::vector<uint32_t> &pinned,
                                                uint32_t device_budget = 0,
                                                FhnEvictionPolicy policy = FhnEvictionPolicy::Belady);
```

`src/FHN/FhnMovementPlan.cpp` — matching signature, plus:

1. Add a last-touch map next to `resident`:

```cpp
  std::set<uint32_t> resident;
  std::unordered_map<uint32_t, int64_t> last_touch; // position of most recent def/prefetch/use
```

2. In the eviction loop, replace the single victim scan with a policy
   switch (same loop shell; both scans iterate the ordered `resident` set,
   so ties break on the lower id via strict comparison):

```cpp
        bool found = false;
        uint32_t victim = 0;
        int64_t victim_key = 0;
        for (uint32_t id : resident) {
          if (working.count(id))
            continue;
          if (policy == FhnEvictionPolicy::Belady) {
            const int64_t nu = next_use_after(id, pos - 1);
            if (!found || nu > victim_key) {
              found = true;
              victim = id;
              victim_key = nu;
            }
          } else { // Lru: oldest last touch wins; strict < keeps the lower id on ties
            const int64_t lt = last_touch.count(id) ? last_touch.at(id) : -1;
            if (!found || lt < victim_key) {
              found = true;
              victim = id;
              victim_key = lt;
            }
          }
        }
```

3. Record touches: after the alloc push (`resident.insert(inst.result_id)`)
   add `last_touch[inst.result_id] = pos;`; after each prefetch insert add
   `last_touch[id] = pos;`; and after the high-water update, touch every
   operand: `for (uint32_t id : operand_set) last_touch[id] = pos;`.
   (Candidates for eviction at instruction i are never in i's working set,
   so the update order within one instruction cannot affect victim choice.)

- [ ] **Step 4: GREEN + full suite**

```bash
cmake --build build -j$(nproc) && ./build/bin/FhnMovementPlanTest && ./build/bin/FhnExecutorTest && ctest --test-dir build
```

Expected: all new tests pass; suite 14/14.

- [ ] **Step 5: Format (both versions) and commit**

```bash
for CF in cf1813 cfvenv; do /tmp/claude-1000/-home-zotanika-workspace-zotanika-fhenomenon/b9ac8324-94be-4a94-b866-0a0f7fef35c9/scratchpad/$CF/bin/clang-format --dry-run --Werror include/FHN/FhnMovementPlan.h src/FHN/FhnMovementPlan.cpp test/FhnMovementPlanTest.cpp test/FhnExecutorTest.cpp || exit 1; done
git add include/FHN/FhnMovementPlan.h src/FHN/FhnMovementPlan.cpp test/FhnMovementPlanTest.cpp test/FhnExecutorTest.cpp
git commit -m "feat: benchmarking-only LRU eviction baseline in FhnMovementPlan"
```

---

### Task 2: Corpus scaffolding — types, builder, oracle, unit-test target

**Files:**
- Create: `benchmarks/corpus/corpus_types.h`
- Create: `benchmarks/corpus/corpus_builder.h`
- Create: `benchmarks/corpus/corpus_oracle.h`
- Create: `benchmarks/corpus/corpus_oracle.cpp`
- Modify: `benchmarks/CMakeLists.txt` (add `fhn_corpus_lib` static lib)
- Create: `test/CorpusUnitTest.cpp`
- Modify: `test/CMakeLists.txt` (register `CorpusUnitTest`)

**Interfaces:**
- Produces (Tasks 3-6 rely on, exact):
  - `namespace fhenomenon::corpus`; `using Slots = std::vector<int64_t>;`
  - `using FhnProgramPtr = std::unique_ptr<FhnProgram, decltype(&fhn_program_free)>;`
  - `struct Shape { std::string name; std::string axis; uint32_t slot_count; uint32_t ct_mult_depth; FhnProgramPtr program; std::map<uint32_t, Slots> inputs; std::vector<uint32_t> output_ids; }`
  - `class ShapeBuilder` with `uint32_t input(Slots v)`, `uint32_t cc(FhnOpCode, uint32_t a, uint32_t b)`, `uint32_t cs(FhnOpCode, uint32_t a, int64_t scalar)`, `uint32_t rot(uint32_t a, int64_t dist)`, `uint32_t un(FhnOpCode, uint32_t a)`, `Shape finish(std::string name, std::string axis, uint32_t slot_count, uint32_t ct_mult_depth, std::vector<uint32_t> output_ids)`
  - `std::optional<std::map<uint32_t, Slots>> evaluate(const FhnProgram &, const std::map<uint32_t, Slots> &inputs);`

- [ ] **Step 1: Write the failing oracle tests**

Create `test/CorpusUnitTest.cpp`:

```cpp
#include "corpus_builder.h"
#include "corpus_oracle.h"

#include <gtest/gtest.h>

using namespace fhenomenon::corpus;

TEST(CorpusOracle, ElementwiseArithmetic) {
  ShapeBuilder b;
  const uint32_t x = b.input({1, 2, 3, 4});
  const uint32_t y = b.input({10, 20, 30, 40});
  const uint32_t s = b.cc(FHN_ADD_CC, x, y);
  const uint32_t d = b.cc(FHN_SUB_CC, y, x);
  const uint32_t p = b.cc(FHN_MULT_CC, x, y);
  const uint32_t n = b.un(FHN_NEGATE, x);
  Shape shape = b.finish("t", "test", 4, 1, {s, d, p, n});

  auto vals = evaluate(*shape.program, shape.inputs);
  ASSERT_TRUE(vals.has_value());
  EXPECT_EQ(vals->at(s), (Slots{11, 22, 33, 44}));
  EXPECT_EQ(vals->at(d), (Slots{9, 18, 27, 36}));
  EXPECT_EQ(vals->at(p), (Slots{10, 40, 90, 160}));
  EXPECT_EQ(vals->at(n), (Slots{-1, -2, -3, -4}));
}

TEST(CorpusOracle, ScalarOpsUseFparams) {
  ShapeBuilder b;
  const uint32_t x = b.input({5, -3});
  const uint32_t a = b.cs(FHN_ADD_CS, x, 7);
  const uint32_t m = b.cs(FHN_MULT_CS, x, -2);
  Shape shape = b.finish("t", "test", 2, 0, {a, m});

  auto vals = evaluate(*shape.program, shape.inputs);
  ASSERT_TRUE(vals.has_value());
  EXPECT_EQ(vals->at(a), (Slots{12, 4}));
  EXPECT_EQ(vals->at(m), (Slots{-10, 6}));
}

TEST(CorpusOracle, RotateIsCyclicSignedLeft) {
  ShapeBuilder b;
  const uint32_t x = b.input({1, 2, 3, 4});
  const uint32_t l = b.rot(x, 1);  // positive = left
  const uint32_t r = b.rot(x, -1); // negative = right
  const uint32_t w = b.rot(x, 5);  // wraps mod slot count
  Shape shape = b.finish("t", "test", 4, 0, {l, r, w});

  auto vals = evaluate(*shape.program, shape.inputs);
  ASSERT_TRUE(vals.has_value());
  EXPECT_EQ(vals->at(l), (Slots{2, 3, 4, 1}));
  EXPECT_EQ(vals->at(r), (Slots{4, 1, 2, 3}));
  EXPECT_EQ(vals->at(w), (Slots{2, 3, 4, 1}));
}

TEST(CorpusOracle, BooleanAndComparisonOps) {
  ShapeBuilder b;
  const uint32_t x = b.input({3, 5, 7});
  const uint32_t y = b.input({3, 9, 2});
  const uint32_t eq = b.cc(FHN_EQ, x, y);
  const uint32_t lt = b.cc(FHN_LT, x, y);
  const uint32_t le = b.cc(FHN_LE, x, y);
  const uint32_t an = b.cc(FHN_AND, eq, le);
  Shape shape = b.finish("t", "test", 3, 0, {eq, lt, le, an});

  auto vals = evaluate(*shape.program, shape.inputs);
  ASSERT_TRUE(vals.has_value());
  EXPECT_EQ(vals->at(eq), (Slots{1, 0, 0}));
  EXPECT_EQ(vals->at(lt), (Slots{0, 1, 0}));
  EXPECT_EQ(vals->at(le), (Slots{1, 1, 0}));
  EXPECT_EQ(vals->at(an), (Slots{1, 0, 0}));
}

TEST(CorpusOracle, FusedOpcodeIsRejected) {
  ShapeBuilder b;
  const uint32_t x = b.input({1});
  const uint32_t y = b.input({2});
  const uint32_t h = b.cc(FHN_HMULT, x, y); // corpus generators never emit fused ops
  Shape shape = b.finish("t", "test", 1, 1, {h});
  EXPECT_FALSE(evaluate(*shape.program, shape.inputs).has_value());
}

TEST(CorpusOracle, SlotSizeMismatchIsRejected) {
  ShapeBuilder b;
  const uint32_t x = b.input({1, 2});
  const uint32_t y = b.input({1, 2, 3});
  const uint32_t s = b.cc(FHN_ADD_CC, x, y);
  Shape shape = b.finish("t", "test", 2, 0, {s});
  EXPECT_FALSE(evaluate(*shape.program, shape.inputs).has_value());
}
```

- [ ] **Step 2: Create the headers and a stub oracle; verify RED**

`benchmarks/corpus/corpus_types.h`:

```cpp
#pragma once

#include "FHN/fhn_program.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace fhenomenon {
namespace corpus {

using Slots = std::vector<int64_t>;
using FhnProgramPtr = std::unique_ptr<FhnProgram, decltype(&fhn_program_free)>;

// One corpus workload: a generated FhnProgram plus the plaintext world it
// computes over. `expected` values come from the oracle at run time, not
// from the generator.
struct Shape {
  std::string name;
  std::string axis;       // one line: which dataflow axis this stresses
  uint32_t slot_count;    // slots per ciphertext in the plaintext model
  uint32_t ct_mult_depth; // longest chain of ct*ct multiplies (execution
                          // eligibility vs a backend's exactness bound)
  FhnProgramPtr program;
  std::map<uint32_t, Slots> inputs; // input id -> slot values
  std::vector<uint32_t> output_ids;
};

} // namespace corpus
} // namespace fhenomenon
```

`benchmarks/corpus/corpus_builder.h`:

```cpp
#pragma once

#include "corpus_types.h"

namespace fhenomenon {
namespace corpus {

// Incremental FhnProgram builder for shape generators. Ids are allocated
// sequentially (single-assignment); inputs carry their plaintext slots.
class ShapeBuilder {
  public:
  uint32_t input(Slots v) {
    const uint32_t id = next_id_++;
    input_ids_.push_back(id);
    inputs_[id] = std::move(v);
    return id;
  }

  uint32_t cc(FhnOpCode op, uint32_t a, uint32_t b) {
    FhnInstruction in{};
    in.opcode = op;
    in.result_id = next_id_++;
    in.operands[0] = a;
    in.operands[1] = b;
    insts_.push_back(in);
    return in.result_id;
  }

  uint32_t cs(FhnOpCode op, uint32_t a, int64_t scalar) {
    FhnInstruction in{};
    in.opcode = op;
    in.result_id = next_id_++;
    in.operands[0] = a;
    in.fparams[0] = static_cast<double>(scalar);
    insts_.push_back(in);
    return in.result_id;
  }

  uint32_t rot(uint32_t a, int64_t dist) {
    FhnInstruction in{};
    in.opcode = FHN_ROTATE;
    in.result_id = next_id_++;
    in.operands[0] = a;
    in.params[0] = dist;
    insts_.push_back(in);
    return in.result_id;
  }

  uint32_t un(FhnOpCode op, uint32_t a) {
    FhnInstruction in{};
    in.opcode = op;
    in.result_id = next_id_++;
    in.operands[0] = a;
    insts_.push_back(in);
    return in.result_id;
  }

  Shape finish(std::string name, std::string axis, uint32_t slot_count, uint32_t ct_mult_depth,
               std::vector<uint32_t> output_ids) {
    auto *p = fhn_program_alloc(static_cast<uint32_t>(insts_.size()), static_cast<uint32_t>(input_ids_.size()),
                                static_cast<uint32_t>(output_ids.size()));
    for (uint32_t i = 0; i < p->num_instructions; ++i)
      p->instructions[i] = insts_[i];
    for (uint32_t i = 0; i < p->num_inputs; ++i)
      p->input_ids[i] = input_ids_[i];
    for (uint32_t i = 0; i < p->num_outputs; ++i)
      p->output_ids[i] = output_ids[i];
    return Shape{std::move(name), std::move(axis),   slot_count,          ct_mult_depth,
                 FhnProgramPtr(p, &fhn_program_free), std::move(inputs_), std::move(output_ids)};
  }

  private:
  uint32_t next_id_ = 1;
  std::vector<FhnInstruction> insts_;
  std::vector<uint32_t> input_ids_;
  std::map<uint32_t, Slots> inputs_;
};

} // namespace corpus
} // namespace fhenomenon
```

(If the aggregate-order in `finish` fails to compile because `Shape`'s
member order differs, initialize the struct member-by-member instead —
the field values are what matters.)

`benchmarks/corpus/corpus_oracle.h`:

```cpp
#pragma once

#include "corpus_types.h"

#include <optional>

namespace fhenomenon {
namespace corpus {

// Plaintext reference evaluator over slot vectors. Models the primitive
// compute opcodes only (fused composites are rejected — corpus generators
// never emit them). Returns id -> slots for every defined id, or nullopt
// on an unmodeled opcode, an undefined operand, or mismatched slot sizes.
std::optional<std::map<uint32_t, Slots>> evaluate(const FhnProgram &program,
                                                  const std::map<uint32_t, Slots> &inputs);

} // namespace corpus
} // namespace fhenomenon
```

`benchmarks/corpus/corpus_oracle.cpp` stub for RED:

```cpp
#include "corpus_oracle.h"

namespace fhenomenon {
namespace corpus {

std::optional<std::map<uint32_t, Slots>> evaluate(const FhnProgram &, const std::map<uint32_t, Slots> &) {
  return std::nullopt;
}

} // namespace corpus
} // namespace fhenomenon
```

`benchmarks/CMakeLists.txt` — append:

```cmake
# --- Corpus library (shapes, oracle, backend loader) ---
add_library(fhn_corpus_lib STATIC
  corpus/corpus_oracle.cpp)
target_include_directories(fhn_corpus_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/corpus)
target_link_libraries(fhn_corpus_lib PUBLIC ${PROJECT_LIB_NAME})
```

`test/CMakeLists.txt` — after the FhnMovementPlanTest block:

```cmake
add_executable(CorpusUnitTest CorpusUnitTest.cpp)
target_link_libraries(CorpusUnitTest PRIVATE fhn_corpus_lib gtest_main)
add_gtest_target_to_ctest(CorpusUnitTest)
```

```bash
cmake -S . -B build && cmake --build build --target CorpusUnitTest -j$(nproc) && ./build/bin/CorpusUnitTest
```

Expected RED: builds, and the four positive oracle tests FAIL on
`ASSERT_TRUE(vals.has_value())` while the two rejection tests pass.

- [ ] **Step 3: Implement the oracle**

Replace `corpus_oracle.cpp`:

```cpp
#include "corpus_oracle.h"

namespace fhenomenon {
namespace corpus {

namespace {

int64_t scalarOf(const FhnInstruction &inst) { return static_cast<int64_t>(inst.fparams[0]); }

} // namespace

std::optional<std::map<uint32_t, Slots>> evaluate(const FhnProgram &program,
                                                  const std::map<uint32_t, Slots> &inputs) {
  std::map<uint32_t, Slots> vals = inputs;

  auto get = [&vals](uint32_t id) -> const Slots * {
    auto it = vals.find(id);
    return it == vals.end() ? nullptr : &it->second;
  };

  for (uint32_t i = 0; i < program.num_instructions; ++i) {
    const FhnInstruction &inst = program.instructions[i];
    const Slots *a = get(inst.operands[0]);
    const Slots *b = get(inst.operands[1]);
    if (!a)
      return std::nullopt;

    auto binary = [&](auto fn) -> std::optional<Slots> {
      if (!b || b->size() != a->size())
        return std::nullopt;
      Slots out(a->size());
      for (size_t s = 0; s < out.size(); ++s)
        out[s] = fn((*a)[s], (*b)[s]);
      return out;
    };
    auto unary = [&](auto fn) -> Slots {
      Slots out(a->size());
      for (size_t s = 0; s < out.size(); ++s)
        out[s] = fn((*a)[s]);
      return out;
    };

    std::optional<Slots> out;
    switch (inst.opcode) {
    case FHN_ADD_CC:
      out = binary([](int64_t x, int64_t y) { return x + y; });
      break;
    case FHN_SUB_CC:
      out = binary([](int64_t x, int64_t y) { return x - y; });
      break;
    case FHN_MULT_CC:
      out = binary([](int64_t x, int64_t y) { return x * y; });
      break;
    case FHN_ADD_CS:
      out = unary([&inst](int64_t x) { return x + scalarOf(inst); });
      break;
    case FHN_MULT_CS:
      out = unary([&inst](int64_t x) { return x * scalarOf(inst); });
      break;
    case FHN_SUB_SC:
      out = unary([&inst](int64_t x) { return scalarOf(inst) - x; });
      break;
    case FHN_NEGATE:
      out = unary([](int64_t x) { return -x; });
      break;
    case FHN_ROTATE: {
      const int64_t n = static_cast<int64_t>(a->size());
      const int64_t d = ((inst.params[0] % n) + n) % n; // positive = left
      Slots rotated(a->size());
      for (int64_t s = 0; s < n; ++s)
        rotated[static_cast<size_t>(s)] = (*a)[static_cast<size_t>((s + d) % n)];
      out = rotated;
      break;
    }
    case FHN_EQ:
      out = binary([](int64_t x, int64_t y) { return x == y ? 1 : 0; });
      break;
    case FHN_LT:
      out = binary([](int64_t x, int64_t y) { return x < y ? 1 : 0; });
      break;
    case FHN_LE:
      out = binary([](int64_t x, int64_t y) { return x <= y ? 1 : 0; });
      break;
    case FHN_AND:
      out = binary([](int64_t x, int64_t y) { return x & y; });
      break;
    case FHN_OR:
      out = binary([](int64_t x, int64_t y) { return x | y; });
      break;
    case FHN_XOR:
      out = binary([](int64_t x, int64_t y) { return x ^ y; });
      break;
    default:
      return std::nullopt; // fused/lifecycle opcodes are not modeled
    }

    if (!out)
      return std::nullopt;
    vals[inst.result_id] = std::move(*out);
  }

  return vals;
}

} // namespace corpus
} // namespace fhenomenon
```

- [ ] **Step 4: GREEN + full suite**

```bash
cmake --build build -j$(nproc) && ./build/bin/CorpusUnitTest && ctest --test-dir build
```

Expected: 6/6 CorpusUnitTest; suite 15/15.

- [ ] **Step 5: Format (both versions) and commit**

```bash
for CF in cf1813 cfvenv; do /tmp/claude-1000/-home-zotanika-workspace-zotanika-fhenomenon/b9ac8324-94be-4a94-b866-0a0f7fef35c9/scratchpad/$CF/bin/clang-format --dry-run --Werror benchmarks/corpus/*.h benchmarks/corpus/*.cpp test/CorpusUnitTest.cpp || exit 1; done
git add benchmarks/corpus test/CorpusUnitTest.cpp benchmarks/CMakeLists.txt test/CMakeLists.txt
git commit -m "feat: corpus scaffolding — shape builder and plaintext slot oracle"
```

---

### Task 3: Pure-C-ABI backend loader

**Files:**
- Create: `benchmarks/corpus/corpus_backend.h`
- Create: `benchmarks/corpus/corpus_backend.cpp`
- Modify: `benchmarks/CMakeLists.txt` (add source + `dl` link)
- Modify: `test/CorpusUnitTest.cpp` (loader tests)
- Modify: `test/CMakeLists.txt` (TEST_LIB_DIR define + toyfhe_fhn dependency for CorpusUnitTest)

**Interfaces:**
- Produces (Task 6 relies on):

```cpp
namespace fhenomenon::corpus {
// A dlopen'd FHN backend, consumed purely through the public C ABI —
// deliberately independent of the C++ Backend hierarchy so this file
// doubles as a reference ABI consumer for backend integrators.
class CorpusBackend {
  public:
  // nullopt on failure with a human-readable reason in *error.
  // Requires: ABI handshake, create/destroy/get_kernels, buffer plane,
  // and encrypt/decrypt i64 (the corpus cannot verify without them).
  static std::optional<CorpusBackend> load(const std::string &library_path, const std::string &symbol_prefix,
                                           std::string *error);
  FhnBackendCtx *ctx() const;
  FhnKernelTable *kernels() const;
  FhnBufferAllocFn bufferAlloc() const;
  FhnBufferFreeFn bufferFree() const;
  FhnEncryptInt64Fn encryptI64() const;
  FhnDecryptInt64Fn decryptI64() const;
  FhnBufferPrefetchFn prefetch() const; // nullable
  FhnBufferEvictFn evict() const;       // nullable
  // movable, not copyable; destructor destroys the ctx then dlcloses.
};
}
```

- [ ] **Step 1: Write the failing tests**

Append to `test/CorpusUnitTest.cpp` (mirror how `FhnExternalBackendTest.cpp` computes the fixture path from `TEST_LIB_DIR` — copy its exact path-building expression, including the platform extension handling if present):

```cpp
#include "corpus_backend.h"

TEST(CorpusBackend, LoadsToyFheAndResolvesDataPlane) {
  std::string error;
  auto backend = CorpusBackend::load(std::string(TEST_LIB_DIR) + "/libtoyfhe_fhn.so", "toyfhe_", &error);
  ASSERT_TRUE(backend.has_value()) << error;
  EXPECT_NE(backend->ctx(), nullptr);
  EXPECT_NE(backend->kernels(), nullptr);
  EXPECT_NE(backend->encryptI64(), nullptr);
  EXPECT_NE(backend->decryptI64(), nullptr);
  // ToyFHE exports no movement hooks.
  EXPECT_EQ(backend->prefetch(), nullptr);
  EXPECT_EQ(backend->evict(), nullptr);

  // Round-trip one value through the data plane.
  FhnBuffer *buf = backend->bufferAlloc()(backend->ctx());
  ASSERT_NE(buf, nullptr);
  ASSERT_EQ(backend->encryptI64()(backend->ctx(), buf, 42), 0);
  int64_t out = 0;
  ASSERT_EQ(backend->decryptI64()(backend->ctx(), buf, &out), 0);
  EXPECT_EQ(out, 42);
  backend->bufferFree()(backend->ctx(), buf);
}

TEST(CorpusBackend, MissingLibraryReportsError) {
  std::string error;
  auto backend = CorpusBackend::load("/nonexistent/libnope.so", "toyfhe_", &error);
  EXPECT_FALSE(backend.has_value());
  EXPECT_FALSE(error.empty());
}
```

- [ ] **Step 2: Verify RED (compile error), then implement**

`corpus_backend.h` declares the class exactly as the Interfaces block
above (members: `void *dl_ = nullptr;` handle, the function pointers, a
`FhnDestroyFn destroy_`; move ctor/assignment transfer and null the
source; destructor calls `destroy_(ctx_)` then `dlclose(dl_)` when
non-null). `corpus_backend.cpp` implements `load`:

1. `dlopen(library_path.c_str(), RTLD_LAZY)`; on null → `*error = dlerror()`.
2. Resolve `<prefix>fhn_get_abi_version`; missing or `!= FHN_ABI_VERSION`
   → error + dlclose.
3. Resolve required: `fhn_create`, `fhn_destroy`, `fhn_get_kernels`,
   `fhn_buffer_alloc`, `fhn_buffer_free`, `fhn_encrypt_i64`,
   `fhn_decrypt_i64` (all with prefix). Any missing → error + dlclose.
4. Optional: `fhn_buffer_prefetch`, `fhn_buffer_evict`; if exactly one
   resolves, null both (same half-pair rule as ExternalBackend).
5. `ctx = create(nullptr)`; null → error + dlclose. `kernels =
   get_kernels(ctx)`; null → destroy, error, dlclose.

CMake: add `corpus/corpus_backend.cpp` to `fhn_corpus_lib` sources and
`target_link_libraries(fhn_corpus_lib PUBLIC ${PROJECT_LIB_NAME} dl)`.
In `test/CMakeLists.txt`, give `CorpusUnitTest` the same
`TEST_LIB_DIR="${CMAKE_BINARY_DIR}/lib"` compile definition and
`add_dependencies(CorpusUnitTest toyfhe_fhn)` used by
`FhnExternalBackendTest`.

- [ ] **Step 3: GREEN + full suite**

```bash
cmake -S . -B build && cmake --build build -j$(nproc) && ./build/bin/CorpusUnitTest && ctest --test-dir build
```

- [ ] **Step 4: Format (both versions) and commit**

```bash
for CF in cf1813 cfvenv; do /tmp/claude-1000/-home-zotanika-workspace-zotanika-fhenomenon/b9ac8324-94be-4a94-b866-0a0f7fef35c9/scratchpad/$CF/bin/clang-format --dry-run --Werror benchmarks/corpus/corpus_backend.h benchmarks/corpus/corpus_backend.cpp test/CorpusUnitTest.cpp || exit 1; done
git add benchmarks/corpus/corpus_backend.h benchmarks/corpus/corpus_backend.cpp benchmarks/CMakeLists.txt test/CorpusUnitTest.cpp test/CMakeLists.txt
git commit -m "feat: corpus backend loader — a pure C-ABI dlopen consumer"
```

---

### Task 4: Shapes 1-6

**Files:**
- Create: `benchmarks/corpus/corpus_shapes.h`
- Create: `benchmarks/corpus/corpus_shapes.cpp`
- Modify: `benchmarks/CMakeLists.txt` (add source)
- Modify: `test/CorpusUnitTest.cpp` (shape invariant tests)

**Interfaces:**
- Produces: `std::vector<Shape> allShapes();` in `corpus_shapes.h` —
  Task 5 appends generators to the same vector; Task 6 iterates it.

- [ ] **Step 1: Write the failing invariant tests**

Append to `test/CorpusUnitTest.cpp`:

```cpp
#include "FHN/FhnMovementPlan.h"
#include "corpus_shapes.h"

#include <set>

// Every shape must: (1) plan successfully with its outputs pinned,
// (2) carry >= 30 instructions, (3) evaluate through the oracle with
// non-empty output slots of the declared width, (4) have inputs of the
// declared slot width, (5) plan with zero evictions at the liveness
// high-water budget (per-spec sanity).
TEST(CorpusShapes, AllShapesSatisfyInvariants) {
  auto shapes = allShapes();
  ASSERT_GE(shapes.size(), 6u); // Task 5 raises this to 12

  std::set<std::string> names;
  for (const auto &shape : shapes) {
    SCOPED_TRACE(shape.name);
    EXPECT_TRUE(names.insert(shape.name).second) << "duplicate shape name";
    EXPECT_GE(shape.program->num_instructions, 30u);

    for (const auto &[id, slots] : shape.inputs) {
      (void)id;
      EXPECT_EQ(slots.size(), shape.slot_count);
    }

    auto plan = FhnMovementPlan::analyze(*shape.program, shape.output_ids, 0);
    ASSERT_TRUE(plan.has_value());
    auto at_hw = FhnMovementPlan::analyze(*shape.program, shape.output_ids, plan->stats().high_water);
    ASSERT_TRUE(at_hw.has_value());
    EXPECT_EQ(at_hw->stats().evict_count, 0u);

    auto vals = evaluate(*shape.program, shape.inputs);
    ASSERT_TRUE(vals.has_value());
    for (uint32_t out : shape.output_ids) {
      ASSERT_TRUE(vals->count(out));
      EXPECT_EQ(vals->at(out).size(), shape.slot_count);
    }
  }
}
```

- [ ] **Step 2: Implement shapes 1-6**

`benchmarks/corpus/corpus_shapes.h`:

```cpp
#pragma once

#include "corpus_types.h"

namespace fhenomenon {
namespace corpus {

// The 12 workload shapes (Tasks 4 and 5 populate this incrementally).
std::vector<Shape> allShapes();

} // namespace corpus
} // namespace fhenomenon
```

`benchmarks/corpus/corpus_shapes.cpp` — deterministic value helpers and
generators. Every constant below was sized for int64/ToyFHE headroom; do
not change them:

```cpp
#include "corpus_shapes.h"

#include "corpus_builder.h"

namespace fhenomenon {
namespace corpus {

namespace {

// Deterministic small values: |v| <= 8.
int64_t val(uint32_t i) { return static_cast<int64_t>((i * 7 + 3) % 17) - 8; }
Slots vec(uint32_t n, uint32_t seed) {
  Slots v(n);
  for (uint32_t i = 0; i < n; ++i)
    v[i] = val(seed * 131 + i);
  return v;
}

// Rotate-accumulate reduction: s = sum over all slots, in every slot.
uint32_t reduceTree(ShapeBuilder &b, uint32_t x, uint32_t n) {
  uint32_t s = x;
  for (uint32_t k = n / 2; k >= 1; k /= 2)
    s = b.cc(FHN_ADD_CC, s, b.rot(s, static_cast<int64_t>(k)));
  return s;
}

// Diagonal-packed matvec skeleton: one rotate + scalar-mult per diagonal,
// accumulated. Stresses fan-out reuse of the single input ciphertext.
uint32_t matvecOnce(ShapeBuilder &b, uint32_t x, uint32_t n, uint32_t seed) {
  uint32_t acc = 0;
  for (uint32_t d = 0; d < n; ++d) {
    const uint32_t r = (d == 0) ? x : b.rot(x, static_cast<int64_t>(d));
    const uint32_t m = b.cs(FHN_MULT_CS, r, ((static_cast<int64_t>(d) * 3 + seed) % 7) - 3);
    acc = (d == 0) ? m : b.cc(FHN_ADD_CC, acc, m);
  }
  return acc;
}

Shape shapeMatvec() {
  ShapeBuilder b;
  const uint32_t x = b.input(vec(64, 1));
  const uint32_t y = matvecOnce(b, x, 64, 0);
  return b.finish("matvec", "fan-out reuse of one input ct", 64, 0, {y});
}

Shape shapeMatmulTile() {
  ShapeBuilder b;
  const uint32_t x = b.input(vec(32, 2));
  const uint32_t y = matvecOnce(b, x, 32, 1);
  const uint32_t z = matvecOnce(b, y, 32, 2);
  return b.finish("matmul-tile", "chained matvecs, deep and wide", 32, 0, {z});
}

Shape shapeHorner15() {
  ShapeBuilder b;
  const uint32_t x = b.input({2});
  uint32_t acc = b.input({3}); // c15 as a ciphertext input
  for (int k = 14; k >= 0; --k) {
    acc = b.cc(FHN_MULT_CC, acc, x);
    acc = b.cs(FHN_ADD_CS, acc, ((static_cast<int64_t>(k) * 5) % 9) - 4);
  }
  return b.finish("horner15", "long sequential ct*ct chain", 1, 15, {acc});
}

Shape shapeReduceTree() {
  ShapeBuilder b;
  uint32_t combined = 0;
  for (uint32_t j = 0; j < 4; ++j) {
    const uint32_t x = b.input(vec(64, 10 + j));
    const uint32_t s = reduceTree(b, x, 64);
    combined = (j == 0) ? s : b.cc(FHN_ADD_CC, combined, s);
  }
  return b.finish("reduce-tree", "log-depth rotate-add fan-in x4", 64, 0, {combined});
}

Shape shapeWideFront() {
  ShapeBuilder b;
  std::vector<uint32_t> parts;
  for (uint32_t i = 0; i < 24; ++i) {
    const uint32_t a = b.input({val(100 + i)});
    const uint32_t c = b.input({val(200 + i)});
    const uint32_t d = b.input({val(300 + i)});
    parts.push_back(b.cc(FHN_ADD_CC, b.cc(FHN_MULT_CC, a, c), d));
  }
  while (parts.size() > 1) {
    std::vector<uint32_t> next;
    for (size_t i = 0; i + 1 < parts.size(); i += 2)
      next.push_back(b.cc(FHN_ADD_CC, parts[i], parts[i + 1]));
    if (parts.size() % 2 == 1)
      next.push_back(parts.back());
    parts = next;
  }
  return b.finish("wide-front", "maximum simultaneous live set", 1, 1, {parts[0]});
}

Shape shapeLogreg() {
  ShapeBuilder b;
  const uint32_t w = b.input(vec(32, 20)); // shared weights: fan-out x3
  uint32_t combined = 0;
  for (uint32_t j = 0; j < 3; ++j) {
    const uint32_t x = b.input(vec(32, 21 + j));
    const uint32_t d = reduceTree(b, b.cc(FHN_MULT_CC, x, w), 32);
    const uint32_t d3 = b.cc(FHN_MULT_CC, b.cc(FHN_MULT_CC, d, d), d);
    const uint32_t r = b.cc(FHN_ADD_CC, b.cs(FHN_MULT_CS, d, 3), b.cs(FHN_MULT_CS, d3, -1));
    combined = (j == 0) ? r : b.cc(FHN_ADD_CC, combined, r);
  }
  return b.finish("logreg", "shared-weight dot products + polynomial", 32, 3, {combined});
}

} // namespace

std::vector<Shape> allShapes() {
  std::vector<Shape> shapes;
  shapes.push_back(shapeMatvec());
  shapes.push_back(shapeMatmulTile());
  shapes.push_back(shapeHorner15());
  shapes.push_back(shapeReduceTree());
  shapes.push_back(shapeWideFront());
  shapes.push_back(shapeLogreg());
  return shapes;
}

} // namespace corpus
} // namespace fhenomenon
```

Add `corpus/corpus_shapes.cpp` to `fhn_corpus_lib` in `benchmarks/CMakeLists.txt`.

- [ ] **Step 3: RED→GREEN, full suite**

```bash
cmake -S . -B build && cmake --build build -j$(nproc) && ./build/bin/CorpusUnitTest && ctest --test-dir build
```

The invariant test must have been observed failing (missing
`corpus_shapes.h`) before the implementation lands. All CorpusUnitTest
cases green after; full suite green.

- [ ] **Step 4: Format (both versions) and commit**

```bash
for CF in cf1813 cfvenv; do /tmp/claude-1000/-home-zotanika-workspace-zotanika-fhenomenon/b9ac8324-94be-4a94-b866-0a0f7fef35c9/scratchpad/$CF/bin/clang-format --dry-run --Werror benchmarks/corpus/corpus_shapes.h benchmarks/corpus/corpus_shapes.cpp test/CorpusUnitTest.cpp || exit 1; done
git add benchmarks/corpus/corpus_shapes.h benchmarks/corpus/corpus_shapes.cpp benchmarks/CMakeLists.txt test/CorpusUnitTest.cpp
git commit -m "feat: corpus shapes 1-6 — matvec, matmul-tile, horner15, reduce-tree, wide-front, logreg"
```

---

### Task 5: Shapes 7-12

**Files:**
- Modify: `benchmarks/corpus/corpus_shapes.cpp`
- Modify: `test/CorpusUnitTest.cpp` (raise the count assertion to 12 and add the executable-subset depth check)

**Interfaces:** unchanged (`allShapes()` now returns 12).

- [ ] **Step 1: Extend the tests first**

In `AllShapesSatisfyInvariants`, change `ASSERT_GE(shapes.size(), 6u)` to
`ASSERT_EQ(shapes.size(), 12u)`, and append a new test:

```cpp
// The ToyFHE-executable subset must stay depth-safe and rotate-free:
// slot_count 1, ct_mult_depth <= 3, no FHN_ROTATE or boolean opcodes.
TEST(CorpusShapes, ExecutableSubsetIsDepthSafe) {
  const std::set<std::string> executable = {"wide-front", "iter-update", "weighted-sum", "diamond"};
  for (const auto &shape : allShapes()) {
    if (!executable.count(shape.name))
      continue;
    SCOPED_TRACE(shape.name);
    EXPECT_EQ(shape.slot_count, 1u);
    EXPECT_LE(shape.ct_mult_depth, 3u);
    for (uint32_t i = 0; i < shape.program->num_instructions; ++i) {
      const FhnOpCode op = shape.program->instructions[i].opcode;
      EXPECT_NE(op, FHN_ROTATE);
      EXPECT_TRUE(op == FHN_ADD_CC || op == FHN_SUB_CC || op == FHN_MULT_CC || op == FHN_ADD_CS ||
                  op == FHN_MULT_CS || op == FHN_NEGATE);
    }
  }
}
```

Run: `ASSERT_EQ(..., 12u)` FAILS (RED).

- [ ] **Step 2: Implement shapes 7-12**

Append to the anonymous namespace in `corpus_shapes.cpp` and extend
`allShapes()` in the same order:

```cpp
Shape shapeConv1d() {
  ShapeBuilder b;
  uint32_t p = b.input(vec(64, 30));
  for (uint32_t layer = 0; layer < 3; ++layer) {
    uint32_t acc = b.cs(FHN_MULT_CS, p, 2); // center tap
    const int64_t taps[4] = {-2, -1, 1, 2};
    for (int64_t t : taps) {
      const uint32_t m = b.cs(FHN_MULT_CS, b.rot(p, t), (t % 3) + 1);
      acc = b.cc(FHN_ADD_CC, acc, m);
    }
    p = acc;
  }
  return b.finish("conv1d", "overlapping stencil window reuse x3 layers", 64, 0, {p});
}

Shape shapeStats() {
  ShapeBuilder b;
  std::vector<uint32_t> outs;
  for (uint32_t j = 0; j < 2; ++j) {
    const uint32_t x = b.input(vec(64, 40 + j));
    const uint32_t s = reduceTree(b, x, 64);
    const uint32_t s2 = reduceTree(b, b.cc(FHN_MULT_CC, x, x), 64);
    // 64*Var proxy: 64*sum(x^2) - sum(x)^2
    const uint32_t v = b.cc(FHN_SUB_CC, b.cs(FHN_MULT_CS, s2, 64), b.cc(FHN_MULT_CC, s, s));
    outs.push_back(s);
    outs.push_back(v);
  }
  return b.finish("stats", "two datasets, four live outputs", 64, 2, outs);
}

Shape shapePsiEq() {
  ShapeBuilder b;
  std::vector<uint32_t> bits;
  for (uint32_t i = 0; i < 32; ++i) {
    const uint32_t a = b.input({val(500 + i)});
    const uint32_t c = b.input({(i % 4 == 0) ? val(500 + i) : val(700 + i)});
    bits.push_back(b.cc(FHN_EQ, a, c));
  }
  while (bits.size() > 1) {
    std::vector<uint32_t> next;
    for (size_t i = 0; i + 1 < bits.size(); i += 2)
      next.push_back(b.cc(FHN_AND, bits[i], bits[i + 1]));
    if (bits.size() % 2 == 1)
      next.push_back(bits.back());
    bits = next;
  }
  return b.finish("psi-eq", "boolean equality-AND tree (plan-only today)", 1, 0, {bits[0]});
}

Shape shapeIterUpdate() {
  ShapeBuilder b;
  uint32_t x = b.input({1});
  for (uint32_t k = 0; k < 32; ++k)
    x = (k % 2 == 0) ? b.cs(FHN_MULT_CS, x, 2) : b.cs(FHN_ADD_CS, x, 3);
  return b.finish("iter-update", "rebinding-heavy sequential chain", 1, 0, {x});
}

Shape shapeWeightedSum() {
  ShapeBuilder b;
  uint32_t acc = 0;
  for (uint32_t i = 0; i < 24; ++i) {
    const uint32_t x = b.input({val(600 + i)});
    const uint32_t m = b.cs(FHN_MULT_CS, x, ((static_cast<int64_t>(i) * 5) % 7) - 3);
    acc = (i == 0) ? m : b.cc(FHN_ADD_CC, acc, m);
  }
  return b.finish("weighted-sum", "scalar-multiply accumulate", 1, 0, {acc});
}

Shape shapeDiamond() {
  ShapeBuilder b;
  const uint32_t root = b.input({5});
  std::vector<uint32_t> tips;
  for (uint32_t d = 0; d < 2; ++d) {
    uint32_t p = root;
    for (uint32_t level = 0; level < 8; ++level) {
      const uint32_t l = b.cs(FHN_ADD_CS, p, static_cast<int64_t>(level) + 1);
      const uint32_t r = b.cc(FHN_ADD_CC, p, root); // long-gap root reuse
      p = b.cc(FHN_ADD_CC, l, r);
    }
    tips.push_back(p);
  }
  return b.finish("diamond", "fork-join with long-gap root reuse", 1, 0, {b.cc(FHN_ADD_CC, tips[0], tips[1])});
}
```

`allShapes()` push order: conv1d, stats, psi-eq, iter-update,
weighted-sum, diamond (after the Task 4 six).

- [ ] **Step 3: GREEN + full suite**

```bash
cmake --build build -j$(nproc) && ./build/bin/CorpusUnitTest && ctest --test-dir build
```

If `AllShapesSatisfyInvariants` fails on the >=30-instruction bound for a
shape, the generator deviates from this plan — fix the transcription, not
the bound.

- [ ] **Step 4: Format (both versions) and commit**

```bash
for CF in cf1813 cfvenv; do /tmp/claude-1000/-home-zotanika-workspace-zotanika-fhenomenon/b9ac8324-94be-4a94-b866-0a0f7fef35c9/scratchpad/$CF/bin/clang-format --dry-run --Werror benchmarks/corpus/corpus_shapes.cpp test/CorpusUnitTest.cpp || exit 1; done
git add benchmarks/corpus/corpus_shapes.cpp test/CorpusUnitTest.cpp
git commit -m "feat: corpus shapes 7-12 — conv1d, stats, psi-eq, iter-update, weighted-sum, diamond"
```

---

### Task 6: Runner, report, CI registration

**Files:**
- Create: `benchmarks/corpus/fhn_corpus_main.cpp`
- Modify: `benchmarks/CMakeLists.txt` (add `fhn-corpus` executable)
- Modify: `test/CMakeLists.txt` (register `FhnCorpusTest`)

**Interfaces:**
- Consumes everything above plus `FhnDefaultExecutor` and
  `FhnMovementHooks` (`include/FHN/FhnDefaultExecutor.h`).

- [ ] **Step 1: Implement the runner**

`benchmarks/corpus/fhn_corpus_main.cpp`:

```cpp
#include "FHN/FhnDefaultExecutor.h"
#include "FHN/FhnMovementPlan.h"
#include "corpus_backend.h"
#include "corpus_oracle.h"
#include "corpus_shapes.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

using namespace fhenomenon;
using namespace fhenomenon::corpus;

namespace {

struct SweepPoint {
  uint32_t budget;
  FhnMovementPlan::Stats belady;
  FhnMovementPlan::Stats lru;
};

uint64_t transfers(const FhnMovementPlan::Stats &s) {
  return static_cast<uint64_t>(s.prefetch_count) + static_cast<uint64_t>(s.evict_count);
}

uint32_t maxWorkingSet(const FhnProgram &program) {
  uint32_t max_ws = 0;
  for (uint32_t i = 0; i < program.num_instructions; ++i) {
    const FhnInstruction &inst = program.instructions[i];
    std::vector<uint32_t> ids{inst.result_id};
    for (int j = 0; j < 4; ++j)
      if (inst.operands[j] != 0)
        ids.push_back(inst.operands[j]);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    max_ws = std::max(max_ws, static_cast<uint32_t>(ids.size()));
  }
  return max_ws;
}

void usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s [--backend <lib.so>] [--prefix <sym, default toyfhe_>] "
               "[--shape <name>] [--max-depth <N>] [--list]\n",
               argv0);
}

} // namespace

int main(int argc, char **argv) {
  std::string backend_path;
  std::string prefix = "toyfhe_";
  std::string only_shape;
  uint32_t max_depth = UINT32_MAX;
  bool list_only = false;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
      backend_path = argv[++i];
    } else if (std::strcmp(argv[i], "--prefix") == 0 && i + 1 < argc) {
      prefix = argv[++i];
    } else if (std::strcmp(argv[i], "--shape") == 0 && i + 1 < argc) {
      only_shape = argv[++i];
    } else if (std::strcmp(argv[i], "--max-depth") == 0 && i + 1 < argc) {
      max_depth = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (std::strcmp(argv[i], "--list") == 0) {
      list_only = true;
    } else {
      usage(argv[0]);
      return 2;
    }
  }

  auto shapes = allShapes();
  if (list_only) {
    for (const auto &s : shapes)
      std::printf("%-14s slots=%-3u depth=%-2u insts=%-4u %s\n", s.name.c_str(), s.slot_count, s.ct_mult_depth,
                  s.program->num_instructions, s.axis.c_str());
    return 0;
  }

  std::optional<CorpusBackend> backend;
  if (!backend_path.empty()) {
    std::string error;
    backend = CorpusBackend::load(backend_path, prefix, &error);
    if (!backend) {
      std::fprintf(stderr, "error: cannot load backend: %s\n", error.c_str());
      return 1;
    }
  }

  bool failed = false;
  uint64_t total_belady = 0;
  uint64_t total_lru = 0;
  std::printf("%-14s %6s | %8s %14s %14s %8s\n", "shape", "budget", "hw", "belady p/e", "lru p/e", "saved");

  for (const auto &shape : shapes) {
    if (!only_shape.empty() && shape.name != only_shape)
      continue;

    auto unlimited = FhnMovementPlan::analyze(*shape.program, shape.output_ids, 0);
    if (!unlimited) {
      std::fprintf(stderr, "FAIL %s: movement analysis rejected the program\n", shape.name.c_str());
      failed = true;
      continue;
    }
    const uint32_t hw = unlimited->stats().high_water;
    const uint32_t b_min = maxWorkingSet(*shape.program);
    const uint32_t b_mid = std::max(b_min, static_cast<uint32_t>((hw * 6 + 9) / 10));
    const uint32_t points[3] = {b_min, b_mid, hw};

    for (uint32_t budget : points) {
      auto belady = FhnMovementPlan::analyze(*shape.program, shape.output_ids, budget, FhnEvictionPolicy::Belady);
      auto lru = FhnMovementPlan::analyze(*shape.program, shape.output_ids, budget, FhnEvictionPolicy::Lru);
      if (!belady || !lru) {
        std::fprintf(stderr, "FAIL %s: budget %u infeasible\n", shape.name.c_str(), budget);
        failed = true;
        continue;
      }
      if (budget == hw && belady->stats().evict_count != 0) {
        std::fprintf(stderr, "FAIL %s: evictions at the high-water budget\n", shape.name.c_str());
        failed = true;
      }
      const uint64_t tb = transfers(belady->stats());
      const uint64_t tl = transfers(lru->stats());
      const double saved = tl == 0 ? 0.0 : 100.0 * (static_cast<double>(tl) - static_cast<double>(tb)) /
                                             static_cast<double>(tl);
      std::printf("%-14s %6u | %8u %7u/%-6u %7u/%-6u %7.1f%%\n", shape.name.c_str(), budget, hw,
                  belady->stats().prefetch_count, belady->stats().evict_count, lru->stats().prefetch_count,
                  lru->stats().evict_count, saved);
      if (budget == b_mid) {
        total_belady += tb;
        total_lru += tl;
      }
    }

    // Execution pass: backend loaded, opcodes supported, single-slot
    // instantiation, and within the operator-declared exactness depth.
    if (backend && shape.slot_count == 1 && shape.ct_mult_depth <= max_depth) {
      FhnDefaultExecutor executor(backend->kernels());
      bool supported = true;
      for (uint32_t i = 0; i < shape.program->num_instructions; ++i)
        supported = supported && executor.supports(shape.program->instructions[i].opcode);
      if (supported) {
        std::vector<uint32_t> pinned(shape.program->input_ids,
                                     shape.program->input_ids + shape.program->num_inputs);
        pinned.insert(pinned.end(), shape.output_ids.begin(), shape.output_ids.end());
        auto plan = FhnMovementPlan::analyze(*shape.program, pinned, b_mid, FhnEvictionPolicy::Belady);
        if (!plan) {
          std::fprintf(stderr, "FAIL %s: execution plan infeasible at B_mid\n", shape.name.c_str());
          failed = true;
          continue;
        }

        uint32_t max_id = 0;
        for (uint32_t i = 0; i < shape.program->num_instructions; ++i)
          max_id = std::max(max_id, shape.program->instructions[i].result_id);
        for (uint32_t i = 0; i < shape.program->num_inputs; ++i)
          max_id = std::max(max_id, shape.program->input_ids[i]);

        std::vector<FhnBuffer *> buffers(max_id + 1, nullptr);
        bool exec_ok = true;
        for (const auto &[id, slots] : shape.inputs) {
          buffers[id] = backend->bufferAlloc()(backend->ctx());
          if (!buffers[id] || backend->encryptI64()(backend->ctx(), buffers[id], slots[0]) != 0) {
            exec_ok = false;
            break;
          }
        }

        FhnMovementHooks hooks{backend->ctx(), backend->bufferAlloc(), backend->bufferFree(), backend->prefetch(),
                               backend->evict()};
        if (exec_ok && executor.execute(hooks, shape.program.get(), buffers.data(), *plan) != 0)
          exec_ok = false;

        auto expected = evaluate(*shape.program, shape.inputs);
        if (exec_ok && expected) {
          for (uint32_t out : shape.output_ids) {
            int64_t got = 0;
            if (backend->decryptI64()(backend->ctx(), buffers[out], &got) != 0 || got != expected->at(out)[0]) {
              std::fprintf(stderr, "FAIL %s: output id %u got %" PRId64 " want %" PRId64 "\n", shape.name.c_str(),
                           out, got, expected->at(out)[0]);
              exec_ok = false;
            }
          }
        } else if (exec_ok) {
          exec_ok = false;
        }

        for (uint32_t id = 1; id <= max_id; ++id)
          if (buffers[id])
            backend->bufferFree()(backend->ctx(), buffers[id]);

        if (!exec_ok) {
          std::fprintf(stderr, "FAIL %s: execution/verification failed\n", shape.name.c_str());
          failed = true;
        } else {
          std::printf("%-14s executed and verified on backend\n", shape.name.c_str());
        }
      }
    }
  }

  if (total_lru > 0) {
    std::printf("aggregate @B_mid: belady=%" PRIu64 " lru=%" PRIu64 " transfers -> belady saves %.1f%%\n",
                total_belady, total_lru,
                100.0 * (static_cast<double>(total_lru) - static_cast<double>(total_belady)) /
                  static_cast<double>(total_lru));
  }
  return failed ? 1 : 0;
}
```

Note the buffer cleanup loop frees every non-null table entry exactly
once: the plan never frees pinned ids (inputs + outputs are pinned) and
the executor nulls what it frees, so surviving entries are exactly the
caller-owned inputs plus adopted outputs plus nothing else.

`benchmarks/CMakeLists.txt`:

```cmake
add_executable(fhn-corpus corpus/fhn_corpus_main.cpp)
target_link_libraries(fhn-corpus PRIVATE fhn_corpus_lib)
add_dependencies(fhn-corpus toyfhe_fhn)
```

`test/CMakeLists.txt` — after the FhnMatvecBenchTest line:

```cmake
# The corpus binary self-checks (plan sanity per shape, oracle-verified
# execution of the depth-safe subset on ToyFHE) and exits nonzero on any
# failure, so it doubles as a CI test.
add_test(NAME FhnCorpusTest
         COMMAND fhn-corpus --backend $<TARGET_FILE:toyfhe_fhn> --prefix toyfhe_ --max-depth 3)
```

(ctest resolves `$<TARGET_FILE:...>`; the `add_dependencies` above ensures
the fixture library is built before the test runs.)

- [ ] **Step 2: Verify end-to-end**

```bash
cmake -S . -B build && cmake --build build -j$(nproc)
./build/bin/fhn-corpus --list                       # 12 rows
./build/bin/fhn-corpus                              # plan-only: table + aggregate, exit 0
./build/bin/fhn-corpus --backend build/lib/libtoyfhe_fhn.so --prefix toyfhe_ --max-depth 3
# expect: 4 "executed and verified on backend" lines (wide-front,
# iter-update, weighted-sum, diamond), exit 0
./build/bin/fhn-corpus --shape diamond              # single-shape filter works
ctest --test-dir build                              # all green incl. FhnCorpusTest
./build/bin/fhn-corpus > /tmp/a.txt && ./build/bin/fhn-corpus > /tmp/b.txt && diff /tmp/a.txt /tmp/b.txt  # deterministic
```

Also run the sanitizer build over the corpus:

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -g" -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan --target fhn-corpus toyfhe_fhn CorpusUnitTest -j$(nproc)
./build-asan/bin/CorpusUnitTest && ./build-asan/bin/fhn-corpus --backend build-asan/lib/libtoyfhe_fhn.so --prefix toyfhe_ --max-depth 3
```

Expected: zero sanitizer reports, exit 0.

- [ ] **Step 3: Format (both versions) and commit**

```bash
for CF in cf1813 cfvenv; do /tmp/claude-1000/-home-zotanika-workspace-zotanika-fhenomenon/b9ac8324-94be-4a94-b866-0a0f7fef35c9/scratchpad/$CF/bin/clang-format --dry-run --Werror benchmarks/corpus/fhn_corpus_main.cpp || exit 1; done
git add benchmarks/corpus/fhn_corpus_main.cpp benchmarks/CMakeLists.txt test/CMakeLists.txt
git commit -m "feat: fhn-corpus runner — budget sweep, Belady-vs-LRU report, backend execution"
```

---

## Completion checklist (after Task 6)

- `ctest --test-dir build` fully green (16 targets incl. CorpusUnitTest + FhnCorpusTest); sanitizer runs clean.
- Repo-wide clang-format clean under BOTH 18.1.3 and 18.1.8.
- `fhn-corpus` aggregate line shows a nonzero Belady-vs-LRU saving.
- Push the branch, open a PR titled "feat: FHN real-workload corpus — 12 shapes, Belady-vs-LRU evidence"; link the spec; note the ToyFHE-executable subset and the Cheddar drop-in path.
