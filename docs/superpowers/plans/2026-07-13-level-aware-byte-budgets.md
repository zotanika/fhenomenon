# Level-Aware Byte Budgets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backends declare level semantics through three all-or-nothing optional ABI exports; `FhnMovementPlan` gains level inference and byte-denominated budgets; ToyFHE exports a flat model so CI exercises the byte path end-to-end; the corpus gains `--budget-bytes`.

**Architecture:** A caller-built `FhnLevelModel` switches `analyze()`'s budget unit from slots to bytes via a single `cost(id)` function (slot mode cost = 1, making the no-model path structurally bit-identical). Level inference runs once up front (ids are single-assignment, so levels are static) and doubles as plan-time validation. Loaders resolve the export trio all-or-nothing, mirroring the movement-hook half-pair rule.

**Tech Stack:** C++20, CMake + ctest + GoogleTest 1.17, FHN C ABI.

**Spec:** `docs/superpowers/specs/2026-07-13-level-aware-byte-budgets-design.md` — normative (note: Cheddar exports are DEFERRED per the spec; do not touch cheddar files).

## Global Constraints

- Branch: `feat/byte-budgets` (exists; work there).
- Build: `cmake -S . -B build && cmake --build build -j$(nproc)`; **reconfigure whenever a file is added** (GLOBbed sources).
- `ctest --test-dir build` 100% green after every task (16 test targets throughout — Task 3's stub is a library, not a test).
- Format: every touched `.cpp/.h` clean under BOTH clang-format 18.1.3 (`/tmp/claude-1000/-home-zotanika-workspace-zotanika-fhenomenon/b9ac8324-94be-4a94-b866-0a0f7fef35c9/scratchpad/cf1813/bin/clang-format`) and 18.1.8 (`.../cfvenv/bin/clang-format`); recreate venvs via `python3 -m venv <dir> && <dir>/bin/pip install clang-format==<ver>` if missing.
- Apple clang gotchas: never explicitly capture a constexpr in a lambda; adjacent string literals, no `<<`-chains in one statement.
- `FhnMovementPlan::analyze`'s `device_budget` parameter widens `uint32_t` → `uint64_t` in Task 1 (byte budgets exceed 4 GB); this is source-compatible for every existing caller — do not change call sites.
- Commit after every task with trailers:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01NW9CkxwDsfqYL2yBs9MWsC`.

---

### Task 1: `FhnLevelModel` + byte-mode planning in `FhnMovementPlan`

**Files:**
- Modify: `include/FHN/FhnMovementPlan.h`
- Modify: `src/FHN/FhnMovementPlan.cpp`
- Modify: `test/FhnMovementPlanTest.cpp`

**Interfaces:**
- Consumes: existing analyze(program, pinned, budget, policy); `FhnLevelEffect` does NOT exist yet in the ABI header — Task 1 therefore defines the enum in the ABI header FIRST (see Step 3; Task 2 only adds the function typedefs/exports around it).
- Produces (Tasks 3-4 rely on):
  - `enum FhnLevelEffect { FHN_LEVEL_PRESERVE = 0, FHN_LEVEL_CONSUME = 1, FHN_LEVEL_SET_PARAM0 = 2 }` (C enum in `include/FHN/fhn_backend_api.h`)
  - `struct fhenomenon::FhnLevelModel { int64_t fresh_level; std::vector<uint64_t> bytes_by_level; std::unordered_map<int, FhnLevelEffect> effects; }`
  - `analyze(const FhnProgram&, const std::vector<uint32_t>& pinned, uint64_t device_budget = 0, FhnEvictionPolicy policy = FhnEvictionPolicy::Belady, const FhnLevelModel *model = nullptr)`
  - `FhnMovementPlan::Stats::high_water_bytes` (uint64_t, 0 in slot mode)

- [ ] **Step 1: Write the failing tests**

Append to `test/FhnMovementPlanTest.cpp`:

```cpp
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
                .inst(FHN_MULT_CC, 3, 1, 1)      // level 1
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

NOTE to the implementer: `ProgramBuilder` (test/FhnTestProgramBuilder.h)
has no LEVEL_DOWN helper. Add one to the shared header as part of Step 1:

```cpp
  ProgramBuilder &inst_p0(FhnOpCode op, uint32_t result, uint32_t a, int64_t p0) {
    FhnInstruction in{};
    in.opcode = op;
    in.result_id = result;
    in.operands[0] = a;
    in.params[0] = p0;
    insts.push_back(in);
    return *this;
  }
```

and write the LevelRaiseIsRejected test with
`.inst_p0(FHN_LEVEL_DOWN, 4, 3, 2)` instead of the placeholder
`build_level_down(...)` shown above.

- [ ] **Step 2: Verify RED**

```bash
cmake --build build --target FhnMovementPlanTest -j$(nproc)
```

Expected: compile errors — no `FhnLevelModel`, no `FhnLevelEffect`, no 5-arg `analyze`, no `inst_p0`.

- [ ] **Step 3: Implement**

1. `include/FHN/fhn_backend_api.h` — add the C enum (typedefs come in
   Task 2), placed with the other data-plane declarations:

```c
/* Level effect of a compute opcode, declared by CKKS-family backends for
   byte-accurate movement planning (see fhn_fresh_level/fhn_level_bytes/
   fhn_opcode_level_effect, resolved as an all-or-nothing trio). */
typedef enum FhnLevelEffect {
  FHN_LEVEL_PRESERVE = 0,   /* result level = min over operand levels    */
  FHN_LEVEL_CONSUME = 1,    /* result level = (min over operands) - 1    */
  FHN_LEVEL_SET_PARAM0 = 2, /* result level = params[0] (must not raise) */
} FhnLevelEffect;
```

2. `include/FHN/FhnMovementPlan.h`:

```cpp
// A backend's declared level semantics, pre-queried by the caller — the
// planner never calls the ABI. bytes_by_level is indexed by level and
// must cover 0..fresh_level; effects must cover every opcode the program
// uses. With a model, analyze() interprets device_budget as BYTES.
struct FhnLevelModel {
  int64_t fresh_level = 0;
  std::vector<uint64_t> bytes_by_level;
  std::unordered_map<int, FhnLevelEffect> effects; // key: FhnOpCode
};
```

`Stats` gains `uint64_t high_water_bytes = 0;`. `analyze` becomes:

```cpp
  static std::optional<FhnMovementPlan> analyze(const FhnProgram &program, const std::vector<uint32_t> &pinned,
                                                uint64_t device_budget = 0,
                                                FhnEvictionPolicy policy = FhnEvictionPolicy::Belady,
                                                const FhnLevelModel *model = nullptr);
```

Add `#include "FHN/fhn_backend_api.h"` and `<unordered_map>`.

3. `src/FHN/FhnMovementPlan.cpp`:

   a. **Level inference pass** (after the existing def/use validation,
   before the simulation). Ids are single-assignment so levels are
   static:

```cpp
  std::unordered_map<uint32_t, int64_t> level_of;
  if (model) {
    if (model->fresh_level < 0 ||
        model->bytes_by_level.size() < static_cast<size_t>(model->fresh_level) + 1)
      return std::nullopt;
    for (uint64_t b : model->bytes_by_level)
      if (b == 0)
        return std::nullopt;
    for (uint32_t i = 0; i < program.num_inputs; ++i)
      level_of[program.input_ids[i]] = model->fresh_level;
    for (uint32_t i = 0; i < program.num_instructions; ++i) {
      const FhnInstruction &inst = program.instructions[i];
      auto eff = model->effects.find(static_cast<int>(inst.opcode));
      if (eff == model->effects.end())
        return std::nullopt;
      int64_t min_level = model->fresh_level;
      for (int j = 0; j < 4; ++j)
        if (inst.operands[j] != 0)
          min_level = std::min(min_level, level_of.at(inst.operands[j]));
      int64_t result_level = min_level;
      switch (eff->second) {
      case FHN_LEVEL_PRESERVE:
        break;
      case FHN_LEVEL_CONSUME:
        result_level = min_level - 1;
        break;
      case FHN_LEVEL_SET_PARAM0:
        result_level = inst.params[0];
        if (result_level > min_level)
          return std::nullopt; // levels never rise (v1 bug-catcher)
        break;
      }
      if (result_level < 0 || result_level > model->fresh_level)
        return std::nullopt; // underflow / out of declared range
      level_of[inst.result_id] = result_level;
    }
  }

  // Unit cost of holding an id resident: bytes in byte mode, 1 in slot
  // mode — this single function keeps the no-model path bit-identical.
  auto cost = [&](uint32_t id) -> uint64_t {
    return model ? model->bytes_by_level[static_cast<size_t>(level_of.at(id))] : 1;
  };
```

   b. **Unify the budget accounting on `cost`.** Maintain
   `uint64_t resident_units = 0;` alongside the existing `resident` set:
   add `resident_units += cost(id)` at every `resident.insert`, subtract
   at every erase (evict and free). Replace the count-based checks:

   - infeasibility: `working_units > device_budget` where
     `working_units = Σ cost(id) for id in working`;
   - eviction loop condition: `resident_units + incoming_units >
     device_budget` with `incoming_units = cost(result) + Σ
     cost(prefetch ids)`;
   - stats: `high_water` stays `resident.size()`-based;
     `high_water_bytes = max(high_water_bytes, resident_units)` only
     when `model != nullptr` (else it stays 0).

   Victim selection (Belady/LRU scans) is untouched.

- [ ] **Step 4: GREEN + full suite**

```bash
cmake --build build -j$(nproc) && ./build/bin/FhnMovementPlanTest && ctest --test-dir build
```

All new tests pass; every pre-existing test (slot mode) passes unchanged.

- [ ] **Step 5: Format both, commit**

```bash
for CF in cf1813 cfvenv; do /tmp/claude-1000/-home-zotanika-workspace-zotanika-fhenomenon/b9ac8324-94be-4a94-b866-0a0f7fef35c9/scratchpad/$CF/bin/clang-format --dry-run --Werror include/FHN/fhn_backend_api.h include/FHN/FhnMovementPlan.h src/FHN/FhnMovementPlan.cpp test/FhnMovementPlanTest.cpp test/FhnTestProgramBuilder.h || exit 1; done
git add include/FHN/fhn_backend_api.h include/FHN/FhnMovementPlan.h src/FHN/FhnMovementPlan.cpp test/FhnMovementPlanTest.cpp test/FhnTestProgramBuilder.h
git commit -m "feat: level inference and byte-denominated budgets in FhnMovementPlan"
```

---

### Task 2: ABI export trio + ToyFHE flat model + runtime plumbing

**Files:**
- Modify: `include/FHN/fhn_backend_api.h` (three typedefs + vtable members + doc)
- Modify: `include/Backend/Backend.h` (FhnRuntime members)
- Modify: `src/Backend/External.cpp` (all-or-nothing resolution + runtime init)
- Modify: `src/Backend/Builtin.cpp` (runtime aggregate-init site gains the members)
- Modify: `include/FHN/ToyFheKernels.h`, `src/FHN/ToyFheKernels.cpp` (flat model exports)
- Modify: `test/FhnExternalBackendTest.cpp` (positive resolution + values)

**Interfaces:**
- Produces: typedefs `FhnFreshLevelFn`, `FhnLevelBytesFn`, `FhnOpcodeLevelEffectFn`; export names `fhn_fresh_level`, `fhn_level_bytes`, `fhn_opcode_level_effect`; `FhnRuntime::{fresh_level, level_bytes, opcode_level_effect}` (nullable, all-or-nothing); ToyFHE exports `toyfhe_fhn_fresh_level` (returns 0), `toyfhe_fhn_level_bytes` (returns `sizeof(toyfhe::Ciphertext)` for level 0, else 0), `toyfhe_fhn_opcode_level_effect` (returns `FHN_LEVEL_PRESERVE` for every opcode).

- [ ] **Step 1: Write the failing test**

Append to `test/FhnExternalBackendTest.cpp` (reuse the file's existing construction pattern):

```cpp
// ToyFHE declares a flat level model: the trio resolves (positive dlsym
// coverage), fresh level 0, a nonzero flat size, PRESERVE everywhere.
TEST(FhnExternalBackend, LevelModelResolvesFlatForToyFhe) {
  ExternalBackend backend(getTestLibPath(), "toyfhe_");
  const FhnRuntime *rt = backend.fhnRuntime();
  ASSERT_NE(rt, nullptr);
  ASSERT_NE(rt->fresh_level, nullptr);
  ASSERT_NE(rt->level_bytes, nullptr);
  ASSERT_NE(rt->opcode_level_effect, nullptr);
  EXPECT_EQ(rt->fresh_level(rt->ctx), 0);
  EXPECT_GT(rt->level_bytes(rt->ctx, 0), 0u);
  EXPECT_EQ(rt->level_bytes(rt->ctx, 1), 0u); // invalid level
  EXPECT_EQ(rt->opcode_level_effect(rt->ctx, FHN_ADD_CC), FHN_LEVEL_PRESERVE);
  EXPECT_EQ(rt->opcode_level_effect(rt->ctx, FHN_HMULT), FHN_LEVEL_PRESERVE);
}
```

(Adapt the constructor/path expression to the file's existing helpers —
copy from the neighboring tests, do not invent new ones.)

- [ ] **Step 2: RED (compile error: FhnRuntime has no fresh_level), then implement**

1. `fhn_backend_api.h` — after the FhnLevelEffect enum:

```c
/* ── Optional level model (data plane) ──
   All three exports appear TOGETHER or the group is ignored with a
   warning (mirroring the movement-hook half-pair rule). Additive and
   optional: no FHN_ABI_VERSION bump; existing backends stay conformant. */
typedef int64_t (*FhnFreshLevelFn)(FhnBackendCtx *ctx);
typedef uint64_t (*FhnLevelBytesFn)(FhnBackendCtx *ctx, int64_t level);
typedef FhnLevelEffect (*FhnOpcodeLevelEffectFn)(FhnBackendCtx *ctx, FhnOpCode opcode);
```

and three members at the end of `FhnBackendVTable` (after `evict`):
`FhnFreshLevelFn fresh_level; FhnLevelBytesFn level_bytes; FhnOpcodeLevelEffectFn opcode_level_effect;`

2. `Backend.h` — `FhnRuntime` gains, before `keepalive`:

```cpp
  // Optional level model (all-or-nothing trio); null = no byte budgets.
  FhnFreshLevelFn fresh_level = nullptr;
  FhnLevelBytesFn level_bytes = nullptr;
  FhnOpcodeLevelEffectFn opcode_level_effect = nullptr;
```

3. `External.cpp` — resolve the trio after the movement hooks; if the
   three are not all-present-or-all-absent, null all three with a
   LOG_MESSAGE warning (adjacent string literals); extend the
   `runtime_ = {...}` aggregate init with the three members in order.
4. `Builtin.cpp` — extend its `runtime_ = {...}` positional init with
   `toyfhe_fhn_fresh_level, toyfhe_fhn_level_bytes,
   toyfhe_fhn_opcode_level_effect` (the builtin uses ToyFHE directly).
5. `ToyFheKernels.h/.cpp` — declare and implement the three exports with
   C linkage next to the existing data-plane exports:

```cpp
extern "C" int64_t toyfhe_fhn_fresh_level(FhnBackendCtx *) { return 0; }
extern "C" uint64_t toyfhe_fhn_level_bytes(FhnBackendCtx *, int64_t level) {
  return level == 0 ? sizeof(toyfhe::Ciphertext) : 0;
}
extern "C" FhnLevelEffect toyfhe_fhn_opcode_level_effect(FhnBackendCtx *, FhnOpCode) {
  // ToyFHE is flat: its Engine rescales internally, sizes never change.
  return FHN_LEVEL_PRESERVE;
}
```

(match the file's existing export style — some exports take the ctx and
ignore it; keep signatures exactly as the typedefs.)

- [ ] **Step 3: GREEN + full suite; format both; commit**

```bash
cmake --build build -j$(nproc) && ./build/bin/FhnExternalBackendTest && ctest --test-dir build
for CF in cf1813 cfvenv; do /tmp/claude-1000/-home-zotanika-workspace-zotanika-fhenomenon/b9ac8324-94be-4a94-b866-0a0f7fef35c9/scratchpad/$CF/bin/clang-format --dry-run --Werror include/FHN/fhn_backend_api.h include/Backend/Backend.h src/Backend/External.cpp src/Backend/Builtin.cpp include/FHN/ToyFheKernels.h src/FHN/ToyFheKernels.cpp test/FhnExternalBackendTest.cpp || exit 1; done
git add -A include/FHN include/Backend src/Backend src/FHN test/FhnExternalBackendTest.cpp
git commit -m "feat: level-model ABI trio with ToyFHE's flat declaration"
```

---

### Task 3: Partial-export fixture + all-or-nothing in both loaders

**Files:**
- Create: `test/partial_fhn_backend.cpp` (stub backend library)
- Modify: `test/CMakeLists.txt` (SHARED lib target `partial_fhn`, output to `${CMAKE_BINARY_DIR}/lib`, dependency for the two test targets)
- Modify: `benchmarks/corpus/corpus_backend.h`, `benchmarks/corpus/corpus_backend.cpp` (trio getters + all-or-nothing resolution)
- Modify: `test/CorpusUnitTest.cpp`, `test/FhnExternalBackendTest.cpp` (fixture tests)

**Interfaces:**
- Produces: `CorpusBackend::{freshLevel(), levelBytes(), opcodeLevelEffect()}` getters (nullable) — Task 4 consumes them; fixture library `libpartial_fhn.so` with symbol prefix `ptl_` exporting the handshake, required data plane, and ONLY `ptl_fhn_fresh_level` from the trio.

- [ ] **Step 1: Write the stub backend**

`test/partial_fhn_backend.cpp` — a minimal conformant backend whose only
purpose is exporting a PARTIAL level-model trio. Implement with plain
static state (no real crypto):

```cpp
#include "FHN/fhn_backend_api.h"

#include <cstdint>
#include <cstdlib>

// A deliberately minimal FHN backend: valid handshake, empty kernel
// table, malloc-backed buffers, identity "encryption" of one int64 —
// and, the point of the fixture, a PARTIAL level-model trio (only
// fhn_fresh_level), which conformant loaders must ignore whole.
namespace {
struct PartialCtx {
  int unused;
};
FhnKernelTable g_table{0, nullptr};
} // namespace

extern "C" uint32_t ptl_fhn_get_abi_version(void) { return FHN_ABI_VERSION; }
extern "C" FhnBackendInfo *ptl_fhn_get_info(void) {
  static FhnBackendInfo info{"partial-fixture", "0", FHN_DEVICE_CPU, 0};
  return &info;
}
extern "C" FhnBackendCtx *ptl_fhn_create(const char *) {
  return reinterpret_cast<FhnBackendCtx *>(new PartialCtx{0});
}
extern "C" void ptl_fhn_destroy(FhnBackendCtx *ctx) { delete reinterpret_cast<PartialCtx *>(ctx); }
extern "C" FhnKernelTable *ptl_fhn_get_kernels(FhnBackendCtx *) { return &g_table; }
extern "C" FhnBuffer *ptl_fhn_buffer_alloc(FhnBackendCtx *) {
  return reinterpret_cast<FhnBuffer *>(new int64_t(0));
}
extern "C" void ptl_fhn_buffer_free(FhnBackendCtx *, FhnBuffer *buffer) {
  delete reinterpret_cast<int64_t *>(buffer);
}
extern "C" int ptl_fhn_encrypt_i64(FhnBackendCtx *, FhnBuffer *out, int64_t value) {
  *reinterpret_cast<int64_t *>(out) = value;
  return 0;
}
extern "C" int ptl_fhn_decrypt_i64(FhnBackendCtx *, const FhnBuffer *in, int64_t *value_out) {
  *value_out = *reinterpret_cast<const int64_t *>(in);
  return 0;
}
// The partial trio: fresh_level WITHOUT level_bytes/opcode_level_effect.
extern "C" int64_t ptl_fhn_fresh_level(FhnBackendCtx *) { return 3; }
```

`test/CMakeLists.txt` — next to the toyfhe_fhn block:

```cmake
# Deliberately exports a PARTIAL level-model trio: loaders must ignore it.
add_library(partial_fhn SHARED partial_fhn_backend.cpp)
target_include_directories(partial_fhn PRIVATE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
set_target_properties(partial_fhn PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
add_dependencies(FhnExternalBackendTest partial_fhn)
add_dependencies(CorpusUnitTest partial_fhn)
```

- [ ] **Step 2: Write the failing tests**

`test/FhnExternalBackendTest.cpp`:

```cpp
// A backend exporting only part of the level-model trio has the whole
// group ignored (all-or-nothing), like the movement-hook half-pair rule.
TEST(FhnExternalBackend, PartialLevelModelIsIgnoredWhole) {
  ExternalBackend backend(std::string(TEST_LIB_DIR) + "/libpartial_fhn.so", "ptl_");
  const FhnRuntime *rt = backend.fhnRuntime();
  ASSERT_NE(rt, nullptr);
  EXPECT_EQ(rt->fresh_level, nullptr);
  EXPECT_EQ(rt->level_bytes, nullptr);
  EXPECT_EQ(rt->opcode_level_effect, nullptr);
}
```

`test/CorpusUnitTest.cpp` (mirror with `CorpusBackend::load` + the three
getters null; also assert the ToyFHE positive path once the getters
exist):

```cpp
TEST(CorpusBackend, PartialLevelModelIsIgnoredWhole) {
  std::string error;
  auto backend = CorpusBackend::load(std::string(TEST_LIB_DIR) + "/libpartial_fhn.so", "ptl_", &error);
  ASSERT_TRUE(backend.has_value()) << error;
  EXPECT_EQ(backend->freshLevel(), nullptr);
  EXPECT_EQ(backend->levelBytes(), nullptr);
  EXPECT_EQ(backend->opcodeLevelEffect(), nullptr);
}

TEST(CorpusBackend, ToyFheLevelModelResolves) {
  std::string error;
  auto backend = CorpusBackend::load(getTestLibPath(), "toyfhe_", &error);
  ASSERT_TRUE(backend.has_value()) << error;
  ASSERT_NE(backend->freshLevel(), nullptr);
  ASSERT_NE(backend->levelBytes(), nullptr);
  ASSERT_NE(backend->opcodeLevelEffect(), nullptr);
  EXPECT_EQ(backend->freshLevel()(backend->ctx()), 0);
  EXPECT_GT(backend->levelBytes()(backend->ctx(), 0), 0u);
}
```

- [ ] **Step 3: RED, implement, GREEN**

Implement the three members + getters + all-or-nothing resolution in
`CorpusBackend` (same structure as its prefetch/evict handling: resolve
all three, and if the present/absent pattern is mixed, null all three).
Reconfigure (new file), build, run both test binaries, full ctest.

- [ ] **Step 4: Format both, commit**

```bash
for CF in cf1813 cfvenv; do /tmp/claude-1000/-home-zotanika-workspace-zotanika-fhenomenon/b9ac8324-94be-4a94-b866-0a0f7fef35c9/scratchpad/$CF/bin/clang-format --dry-run --Werror test/partial_fhn_backend.cpp benchmarks/corpus/corpus_backend.h benchmarks/corpus/corpus_backend.cpp test/CorpusUnitTest.cpp test/FhnExternalBackendTest.cpp || exit 1; done
git add test/partial_fhn_backend.cpp test/CMakeLists.txt benchmarks/corpus test/CorpusUnitTest.cpp test/FhnExternalBackendTest.cpp
git commit -m "feat: all-or-nothing level-model resolution, pinned by a partial-export fixture"
```

---

### Task 4: Corpus `--budget-bytes` + CI byte exercise

**Files:**
- Modify: `benchmarks/corpus/fhn_corpus_main.cpp`
- Modify: `test/CMakeLists.txt` (FhnCorpusTest gains `--budget-bytes min`)

**Interfaces:**
- Consumes: Task 1's model-taking `analyze`, Task 3's `CorpusBackend` trio getters.

- [ ] **Step 1: Implement**

In `fhn_corpus_main.cpp`:

1. Flag: `--budget-bytes <N|min>` parsed alongside the others (`min` is
   the literal keyword; a number parses via strtoull into `uint64_t
   budget_bytes`; a `bool budget_bytes_min` flag records the keyword).
   Update `usage()`.
2. After backend load: if the flag was given and (no backend OR any trio
   getter is null) → `error: --budget-bytes requires a backend with a
   level model` to stderr, exit 2.
3. Build the model once:

```cpp
FhnLevelModel queryLevelModel(const CorpusBackend &backend) {
  FhnLevelModel model;
  model.fresh_level = backend.freshLevel()(backend.ctx());
  for (int64_t l = 0; l <= model.fresh_level; ++l)
    model.bytes_by_level.push_back(backend.levelBytes()(backend.ctx(), l));
  for (int op = FHN_NOP; op < FHN_OPCODE_COUNT; ++op)
    model.effects[op] = backend.opcodeLevelEffect()(backend.ctx(), static_cast<FhnOpCode>(op));
  return model;
}
```

4. Resolve `min`: `budget_bytes = 3 * model.bytes_by_level[model.fresh_level]`
   (corpus generators emit unary/binary primitives → working sets ≤ 3
   buffers; with ToyFHE's flat model this is the minimum feasible byte
   budget). Note in a comment.
5. Per shape (inside the existing loop, after the slot sweep): when byte
   mode is active, run `analyze(..., budget=0, Belady, &model)` for
   `hw_bytes`, then both policies at `budget_bytes`; on nullopt print
   `FAIL <shape>: byte-budget infeasible/model invalid` and set failed;
   else print exactly:

```
movement-bytes[<name>]: hw_bytes=<N> belady p/e=<p>/<e> lru p/e=<p>/<e> (budget-bytes <B>)
```

6. `test/CMakeLists.txt`: FhnCorpusTest command gains `--budget-bytes min`.

- [ ] **Step 2: Verify**

```bash
cmake --build build -j$(nproc)
./build/bin/fhn-corpus --backend build/lib/libtoyfhe_fhn.so --prefix toyfhe_ --max-depth 3 --budget-bytes min
# expect: 12 movement-bytes lines; nonzero evicts on fan-out shapes
# (matvec-family) whose count high-water exceeds 3; the 4 executable
# shapes still verify; exit 0.
./build/bin/fhn-corpus --budget-bytes 1000            # no backend -> exit 2
./build/bin/fhn-corpus --backend build/lib/libpartial_fhn.so --prefix ptl_ --budget-bytes 1000  # trio ignored -> exit 2
ctest --test-dir build                                 # all green
# determinism: two identical byte-mode runs diff-identical
./build/bin/fhn-corpus --backend build/lib/libtoyfhe_fhn.so --prefix toyfhe_ --max-depth 3 --budget-bytes min > /tmp/a.txt
./build/bin/fhn-corpus --backend build/lib/libtoyfhe_fhn.so --prefix toyfhe_ --max-depth 3 --budget-bytes min > /tmp/b.txt
diff /tmp/a.txt /tmp/b.txt
```

Note: the partial-fixture invocation loads a backend with an EMPTY kernel
table — the corpus must handle "backend loaded but nothing executable"
gracefully (it already gates execution on supports()); the exit-2 here
comes from the missing level model, before any execution attempt.

Sanitizers:

```bash
cmake --build build-asan --target fhn-corpus toyfhe_fhn partial_fhn CorpusUnitTest FhnMovementPlanTest -j$(nproc) 2>/dev/null || cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -g" -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" && cmake --build build-asan --target fhn-corpus toyfhe_fhn partial_fhn CorpusUnitTest FhnMovementPlanTest -j$(nproc)
./build-asan/bin/CorpusUnitTest && ./build-asan/bin/FhnMovementPlanTest && ./build-asan/bin/fhn-corpus --backend build-asan/lib/libtoyfhe_fhn.so --prefix toyfhe_ --max-depth 3 --budget-bytes min
```

Zero reports, exit 0.

- [ ] **Step 3: Format both, commit**

```bash
for CF in cf1813 cfvenv; do /tmp/claude-1000/-home-zotanika-workspace-zotanika-fhenomenon/b9ac8324-94be-4a94-b866-0a0f7fef35c9/scratchpad/$CF/bin/clang-format --dry-run --Werror benchmarks/corpus/fhn_corpus_main.cpp || exit 1; done
git add benchmarks/corpus/fhn_corpus_main.cpp test/CMakeLists.txt
git commit -m "feat: corpus --budget-bytes mode with CI byte exercise on ToyFHE"
```

---

## Completion checklist (after Task 4)

- `ctest --test-dir build` fully green; sanitizer runs clean; repo-wide
  clang-format clean under BOTH versions.
- Byte-mode corpus lines deterministic; slot-mode output byte-identical
  to pre-branch behavior.
- Push branch; open PR "feat: level-aware byte budgets — backends declare
  their level semantics"; body links the spec, notes the all-or-nothing
  trio, the Cheddar deferral, and the recorded bootstrap path
  (FHN_BOOTSTRAP + FHN_LEVEL_REFRESH, future).
