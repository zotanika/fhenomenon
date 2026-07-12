# Runtime Data Movement Pass — Design

Date: 2026-07-12
Status: approved (design discussion in session; scope, eviction policy, and
architecture each confirmed explicitly)

## Premise

An `FhnProgram` is straight-line and data-oblivious: no instruction depends on
ciphertext contents for control flow. At runtime, immediately before
execution, the complete def-use chain of every buffer id is therefore *exact*
— not an estimate. Work that ahead-of-time compilers must do with heuristics
(placement, prefetching, eviction) can be done at runtime here with perfect
future knowledge. In particular, optimal cache replacement (Belady/MIN,
"evict the buffer whose next use is farthest away") is directly computable,
not merely approximated.

This makes data movement a *runtime* responsibility in the FHENOMENON
programming model, which is the conclusion of the earlier design discussion
this spec implements.

## Goals

- Replace the Session runtime's allocate-everything-up-front buffer handling
  with a liveness-driven schedule: allocate at first def, free after last use.
- Plan host↔device transfers (prefetch before use, evict under memory
  pressure) for backends with a distinct device memory space.
- Implement Belady-optimal eviction under a configurable device budget.
- Keep the public instruction stream compute-only: movement is host-side
  orchestration, never a kernel-table opcode.
- Fully testable without a GPU (mock runtime); ToyFHE behavior unchanged.

## Non-goals (v1)

- Byte-accurate budget accounting. v1 budgets count *buffer slots*; byte
  sizes join when per-kernel/level size metadata exists (separate roadmap
  item "kernel constraint/cost/residency metadata").
- Async compute/transfer overlap (submit/poll/wait ABI is its own backlog
  item; this plan structure is designed to feed it later).
- Cross-backend movement/serialization (blocked on the FHN wire format item).
- Changing the legacy `evaluateGraph` fallback path.

## Component 1 — `FhnMovementPlan`

New files: `include/FHN/FhnMovementPlan.h`, `src/FHN/FhnMovementPlan.cpp`.
A pure analysis pass over an `FhnProgram`; backend-agnostic, no I/O, no
allocation of FHN buffers itself — it only produces a schedule.

```cpp
struct FhnMovementActions {
  // Applied BEFORE the instruction, in this order: evict → alloc → prefetch.
  std::vector<uint32_t> evict;    // demote to host (D2H), contents preserved
  std::vector<uint32_t> alloc;    // create device-resident buffer (first def)
  std::vector<uint32_t> prefetch; // ensure device residency (H2D) before use
  // Applied AFTER the instruction.
  std::vector<uint32_t> free;     // last use passed and id is not pinned
};

class FhnMovementPlan {
  public:
  struct Stats {
    uint32_t high_water = 0;     // max simultaneously device-resident buffers
    uint32_t alloc_count = 0;
    uint32_t prefetch_count = 0; // H2D transfers planned
    uint32_t evict_count = 0;    // D2H transfers planned
  };

  // pinned: ids that must survive execution (never freed): program inputs,
  // program outputs, and the Session's write-back targets.
  // device_budget: max simultaneously device-resident buffers; 0 = unlimited
  // (no evictions planned).
  // Returns nullopt for invalid programs (see Validation) or infeasible
  // budgets (budget < largest single-instruction working set).
  static std::optional<FhnMovementPlan> analyze(const FhnProgram &program,
                                                const std::vector<uint32_t> &pinned,
                                                uint32_t device_budget = 0);

  const FhnMovementActions &at(uint32_t inst_index) const;
  const Stats &stats() const;
};
```

### Liveness rules

- Ids are single-assignment: lowering allocates a fresh id per instruction
  result, and input ids are distinct. `analyze` validates this.
- Input ids are defined "before the program" (def position −1). They are NOT
  prefetched up front — each is prefetched immediately before its first use.
  This is the just-in-time win over today's behavior.
- A non-input id is `alloc`ed immediately before its defining instruction.
  Allocation produces a device-resident buffer (backends allocate in their
  native space; Cheddar allocates device ciphertexts).
- An id's last use is the last instruction referencing it as an operand.
  Non-pinned ids are freed immediately after their last use. Pinned ids are
  never freed by the plan.
- The plan simulates residency throughout: a buffer evicted earlier and used
  again gets a `prefetch` before that use.

### Eviction (Belady)

At each instruction, the working set (result id + operand ids) must be
resident. If satisfying the instruction's allocs/prefetches would exceed
`device_budget`, the plan evicts resident buffers — excluding the current
instruction's working set — choosing the buffer whose next use is farthest
in the future (no future use, e.g. a pinned output already past its last
use, sorts as farthest of all). Ties break on the lower buffer id, keeping
plans deterministic for tests. This is exactly Belady/MIN, exact here
because future uses are fully known. If the current instruction's own
working set exceeds the budget, `analyze` returns nullopt (infeasible).

### Validation (free safety win)

`analyze` rejects programs where an operand id is never defined (neither an
input nor a prior result), and duplicate defs. Today such programs reach
kernels as null buffer pointers; with the plan in the Session path they fail
loudly before execution.

## Component 2 — ABI additions (`fhn_backend_api.h`)

Two new OPTIONAL host-side data-plane exports (dlsym-resolved, may be
absent/NULL):

```c
/* Ensure the buffer is resident in the backend's compute space (H2D).
   No-op if already resident. Returns 0 on success. */
typedef int (*FhnBufferPrefetchFn)(FhnBackendCtx *ctx, FhnBuffer *buffer);
/* Demote the buffer to host memory (D2H), preserving contents; a later
   prefetch must restore it for compute. Returns 0 on success. */
typedef int (*FhnBufferEvictFn)(FhnBackendCtx *ctx, FhnBuffer *buffer);
```

- Export names: `fhn_buffer_prefetch`, `fhn_buffer_evict`.
- These are data-plane exports, not kernel-table opcodes: movement is host
  orchestration; the public instruction stream stays compute-only, keeping
  the trust-boundary structure intact.
- Backends without a distinct memory space (ToyFHE) simply do not export
  them; the plan's prefetch/evict actions are skipped at execution.
- `fhn_buffer_free` must accept a buffer in either residency state (a pinned
  buffer can finish the program evicted to host).
- No `FHN_ABI_VERSION` bump: the addition is purely additive and optional —
  every existing backend remains fully conformant. Document this rationale
  in the header next to the version-bump policy note.

`FhnRuntime` (in `include/Backend/Backend.h`) gains two nullable members
(`prefetch`, `evict`); `ExternalBackend` resolves the optional symbols;
`BuiltinBackend`'s runtime leaves them null.

## Component 3 — Plan-aware execution (`FhnDefaultExecutor`)

The existing `execute(ctx, program, buffers)` stays unchanged. New overload
taking a small hooks struct rather than `FhnRuntime` (which lives in
`Backend/Backend.h` and itself points at the executor — passing it into the
executor's own header would invert the layering):

```cpp
struct FhnMovementHooks {
  FhnBackendCtx *ctx = nullptr;
  FhnBufferAllocFn buffer_alloc = nullptr; // required
  FhnBufferFreeFn buffer_free = nullptr;   // required
  FhnBufferPrefetchFn prefetch = nullptr;  // optional (null = skip)
  FhnBufferEvictFn evict = nullptr;        // optional (null = skip)
};
int execute(const FhnMovementHooks &hooks, const FhnProgram *program,
            FhnBuffer **buffers, const FhnMovementPlan &plan);
```

Session builds the hooks from its `FhnRuntime`.

Per instruction i: apply `plan.at(i)` pre-actions in order evict → alloc →
prefetch (evict/prefetch skipped when the runtime hooks are null), dispatch
the instruction (including decompose paths, unchanged), then apply `free`
post-actions via `runtime.buffer_free`.

- `buffers` arrives with *input* entries filled by the caller; the executor
  allocates every other id per plan — including pinned defs (program
  outputs, write-back targets), which it never frees.
- Ownership handoff: after a successful run the caller adopts the pinned
  non-input buffers it needs directly from the `buffers` table (Session
  wraps them in the existing shared_ptr/keepalive deleter pattern before
  write-back).
- Error safety: on any failure (kernel rc≠0, alloc/prefetch/evict failure),
  the executor frees every plan-allocated id not yet freed — pinned included,
  since nothing was adopted — before returning the error. No leaks either way.

## Component 4 — Session integration

`executeThroughFhnRuntime` changes:

1. Fill input buffers from entity ciphertexts (as today; foreign-owner check
   unchanged).
2. Pin set = input ids ∪ the latest-binding id per entity (write-back
   targets). Program output ids are NOT pinned as such: an output superseded
   by a later assignment is not any entity's latest binding, so nobody would
   adopt it — pinning it would leak it, and freeing it at last use is
   correct. Every output the caller actually needs IS some entity's latest
   binding and gets pinned through that.
3. `FhnMovementPlan::analyze(program, pinned, /*budget=*/0)`; on nullopt,
   throw (the program is malformed — this replaces today's silent null
   buffer path).
4. Call the plan-aware executor overload. Session no longer preallocates
   intermediates; its shared_ptr ownership covers only input/pinned buffers
   and write-back results (unchanged deleter/keepalive pattern).

Default budget in the Session path is unlimited: production behavior gains
alloc-late/free-early and JIT prefetch with zero eviction risk. Budgeted
Belady is exercised by tests and the bench knob; wiring a real budget waits
for byte-size metadata.

## Component 5 — Tests (TDD throughout)

New target `FhnMovementPlanTest` (plan analysis, no backend):
- alloc appears immediately before first def; free immediately after last
  use; pinned ids never freed.
- inputs prefetched before first use, not before.
- chained program: high-water equals live-set maximum, lower than total
  buffer count.
- forced budget: Belady evicts the resident buffer with farthest next use
  (construct a program where LRU and Belady disagree; assert Belady's pick).
- evicted-then-reused id gets a re-prefetch before the reuse.
- infeasible budget → nullopt; undefined operand id → nullopt; duplicate
  def → nullopt.

`FhnExecutorTest` additions (mock kernel table + recording mock runtime):
- full call sequence (evict/alloc/prefetch/kernel/free interleaving) matches
  the plan for a representative program.
- movement correctness, not just ordering: a mock runtime whose evict
  physically moves buffer contents to a host-side map (and clears the
  "device" copy) and whose prefetch restores it — final decrypted results
  must still be correct under a tiny budget.
- null prefetch/evict hooks: actions skipped, execution still correct.
- mid-program kernel failure: no leaked plan-allocated buffers (mock
  alloc/free counters balance).

Session/e2e:
- entire existing suite stays green (ToyFHE, external backend, sessions).
- new SessionTest case pinning that write-back targets survive planning
  (e.g. chained assignments where an entity's final id is an intermediate).
- sanitizer suite (ASan/UBSan) clean on the new path.

## Component 6 — Bench

`fhn-bench-matvec` prints `plan.stats()` (high-water, alloc/prefetch/evict
counts) for both fused and decomposed paths, and gains `--budget N` to show
eviction counts scaling as the budget shrinks. Correctness checks in the
bench must stay green under any budget.

## Error handling summary

- `analyze` → nullopt on malformed program or infeasible budget; Session
  throws with a descriptive message.
- Runtime hook failures (alloc/prefetch/evict rc≠0) → executor cleans up
  plan-allocated buffers, returns rc; Session throws (existing pattern).

## Future work enabled by this design

- Byte-accurate budgets once kernel/level size metadata lands.
- Async overlap: the plan's action lists are the natural input to a
  submit/poll scheduler (prefetch N+1 while computing N).
- Serialized buffer import/export can reuse evict/prefetch semantics for
  cross-process movement.
