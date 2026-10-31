# Audit corrections implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Correct the audit's reproducible runtime and CKKS defects, and provide an honest portable packed-workload verification path.
**Architecture:** Preserve the existing flat IR and layered CKKS design. Separate recorded internal values from observable outputs, strengthen execution contracts, and add optional packed data-plane operations.
**Tech Stack:** C++17, CMake, GoogleTest, optional CUDA/Cheddar.
**Spec:** `docs/superpowers/specs/2026-10-28-audit-corrections-design.md`

## Global Constraints

- Preserve C++17 and ABI version 1; new data-plane exports remain optional.
- Add no third-party dependency and preserve user untracked files.
- Meaningful failing regression tests precede product corrections.
- CKKS dependencies point downward; no hidden allocation/global context.
- Distinguish internal Session temporaries from caller-observable values.
- Reject known unsafe requests before the offending operation mutates caller buffers; earlier successful instructions are not rolled back; no unplanned fallback scratch.
- Keep planning-only results, execution, skips and failures distinguishable.
- No publish/push/merge. Work in the existing writable checkout after isolated-worktree execution was blocked by the automatic review service error.

## Review Focus

- Shared-owned user variables and multiple output assignments must survive Session pruning (Task 1).
- Distinct SSA ids may alias the same physical buffer (Task 2).
- A same-length mutated program must not pass a stale-plan check (Task 2).
- Near-equal 62-bit prime products, malformed layout ranges and active Arena guards must preserve invariants (Task 3).
- A selected packed workload must execute or explicitly fail required verification; skipped work cannot count as success (Task 5).

### Task 1: Session semantics, diagnostics and liveness

**Files:** src/Fhenon.cpp, include/Fhenon.h, include/Session/*, src/Session/Session.cpp, include/Scheduler/*, src/Scheduler/Operation.cpp, include/Utils/log.h, test/SessionTest.cpp, test/NoSessionTest.cpp, test/LowerToFhnProgramTest.cpp.
**Interfaces:** Keep existing public run/operators. Explicitly identify synthetic recording temporaries; preserve external/shared-owned values. If planner stats are surfaced, expose them as useful execution diagnostics rather than test-only internals. No changes to include/FHN or src/FHN.
- [x] Add and observe regressions: c=(a+b)*2 gives 10 for a=2,b=3; (a*b)+3 gives9; chained assignments and externally retained intermediates remain correct.
- [x] Add real lowering/execution liveness coverage: 12 internal scalar-add results must not all remain pinned; preserve only caller-observable results and inputs.
- [x] Remove automatic PrintAST; make debug logging opt-in and explicit AST printing DAG-aware. Test bounded output for repeated a=a+a.
- [x] Fix current-session selection, reject nested recording before corrupting outer state, and reject decrypt during recording instead of returning stale data.
- [x] Resolve missing Operation<float/double>::execute definitions while explicitly rejecting unsupported Session types before graph evaluation.
- [x] Run focused tests, record evidence, and request independent review.

### Task 2: Runtime and ToyFHE contracts

**Files:** include/FHN/FhnDefaultExecutor.h, include/FHN/FhnMovementPlan.h, src/FHN/FhnDefaultExecutor.cpp, src/FHN/FhnMovementPlan.cpp, src/FHN/ToyFheKernels.cpp, test/FhnExecutorTest.cpp, test/FhnMovementPlanTest.cpp, test/FhnToyFheTest.cpp. Do not change fhn_backend_api.h or packed export declarations yet; Task5 will own those after Task2 finishes.
**Interfaces:** Existing executor and planner calls continue to compile. New safe overloads or plan metadata may be added; preserve Session's nonbudgeted use. Bind plans to all relevant program content, not just instruction count.
- [x] Reproduce HROT_ADD/HCONJ_ADD/MAD physical alias hazards; reject unsupported decomposition aliasing before any mutation, or use only explicitly planned scratch. Native disjoint/fallback results must agree.
- [x] Require HMULT's MULT_CC, RELINEARIZE and RESCALE primitives unless a complete native HMULT is registered; update the minimal backend test to reflect the public opcode contract.
- [x] Add slot-wise vector ADD_CS/MULT_CS/SUB_CC/NEGATE with validation and alias-safe results; test literal expected vectors and malformed kinds.
- [x] Reject same-length stale plans before any hooks/kernel calls; test changed result/operand/input/output ids and in-place mutation of the original program.
- [x] Define and implement budgeted execution's initial-residency contract: normalize input residency before allocations or explicitly account for it. The audit's 3-resident-input/budget3 ADD must not overallocate. Do not silently assume an already resident input is host-only.
- [x] Bound 'optimal' claims to the equal-size model; retain a byte-mode counterexample test/documentation without promising an unimplemented globally optimal policy.
- [x] Run focused tests, record evidence, and request independent review.

### Task 3: CKKS invariants

**Files:** src/CKKS/params/*, src/CKKS/storage/*, src/CKKS/arena/*, test/CKKS/*.cpp. Any shared number-theory helper must obey downward layer isolation.
**Interfaces:** Params::create and layout validation return failure for invalid input. Arena move behavior must be safe and have an explicit contract with active Scope objects.
- [x] Add both exact prime-product audit counterexamples and fix comparisons with exact arithmetic.
- [x] Reject composite 2049 for logN10 in main/aux/terminal prime lists without introducing L0→L1 includes.
- [x] Reject limb-count overflow, invalid NttForm and empty-main ciphertext layouts; check bounds at256/257.
- [x] Reproduce active-scope Arena move; prevent invalid offsets/pointers for source and destination guards. Test move constructor/assignment and reset interactions.
- [x] Preserve the existing ParamsId bytes unless deliberately documented; pin a literal known-answer identity and describe the current custom-seeded FNV variant.
- [x] Run layer tests plus ASan/UBSan repros, record evidence, and request independent review.

### Task 4: Reusable build and truthful experiment runner

**Files:** CMakeLists.txt, src/FHN/CMakeLists.txt or benchmarks/CMakeLists.txt, test/CMakeLists.txt, cmake/Config.cmake.in, .github/workflows/ci.yml, experiments/run.sh, test/FhnCheddarGpuTest.cpp.
**Interfaces:** BUILD_TEST=OFF configures/builds. Installation supplies public headers and transitive dependencies so find_package consumer links. Existing test target names preserved.
- [x] Reproduce test-off configure and install-consumer failures; relocate shared Toy target or conditionalize dependencies correctly.
- [x] Fix installed headers/export dependencies and verify a tiny out-of-tree consumer.
- [x] Add --no-tests=error to GPU test selection; provide correct Cheddar symbol prefix to corpus runs.
- [x] Synchronize completed GPU work around timing and distinguish total/device/host costs in labels. Replace private buffer layout access when Task5's packed exports exist.
- [x] Add economical build/install verification to CI, run local checks and record evidence.

### Task 5: Portable packed corpus verification

**Files:** include/FHN/fhn_backend_api.h, include/FHN/ToyFheKernels.h, src/FHN/ToyFheKernels.cpp (after Task2), src/FHN/cheddar/CheddarFhnBackend.cpp, benchmarks/corpus/*, test/CorpusUnitTest.cpp, test/FhnCheddarGpuTest.cpp, relevant CMake/docs.
**Interfaces:** Add optional vector f64 encrypt/decrypt export typedefs with explicit element count/capacity behavior. Existing scalar-only backends remain loadable. CorpusBackend::load gains optional configuration while preserving existing call sites.
- [x] Add public packed-I/O conformance tests with literal vectors, size validation, errors and scalar-only compatibility.
- [x] Implement Toy and Cheddar vector exports without private-layout access from consumers; use the new exports in GPU tests.
- [x] Let corpus pass backend config, execute packed shapes through vectors and verify f64 tolerances against independent oracles. Preserve integer-only backend support and depth controls.
- [x] Report executed/planned/skipped/failed counts and explicit skip reasons. Add a required-execution option that fails if any selected required workload cannot execute.
- [x] Verify at least shallow packed matvec, both native and primitive fallback, against Toy; preserve honest distinction between CPU reference validation and unrun GPU validation.
- [x] Document runnable Cheddar command and remaining hardware-dependent checks; independent integration review.

### Task 6: Integration and final review

- [x] Run full build/CTest, no-LTO build, full ASan/UBSan, install-consumer and packed required-execution checks.
- [x] Re-run critical audit repros; inspect diff and user-file preservation.
- [x] Whole-change review by Astra; fix material findings, recheck affected tests, and record remaining limits.
- [x] Deliver reviewable local changes and concise evidence. Do not claim GPU measurements or native CKKS completion.
