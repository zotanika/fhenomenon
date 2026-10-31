#pragma once

#include "FHN/FhnMovementPlan.h"
#include "FHN/fhn_backend_api.h"
#include <unordered_map>

namespace fhenomenon {

// Runtime services for plan-aware execution. ctx is passed through to every
// hook. prefetch/evict must be both present or both null. Both null declares
// a single memory space: movement is skipped and plan residency is modeled.
// For budgeted execution, evict must safely accept an already host-resident
// input (a no-op): the normalization prelude visits every declared input.
struct FhnMovementHooks {
  FhnBackendCtx *ctx = nullptr;
  FhnBufferAllocFn buffer_alloc = nullptr;
  FhnBufferFreeFn buffer_free = nullptr;
  FhnBufferPrefetchFn prefetch = nullptr;
  FhnBufferEvictFn evict = nullptr;
};

class FhnDefaultExecutor {
  public:
  explicit FhnDefaultExecutor(FhnKernelTable *table);

  bool supports(FhnOpCode opcode) const;

  // Execute a program by dispatching each instruction to kernel functions.
  // buffers is indexed by result_id.
  // Returns 0 on success, non-zero on error. Fallback HROT_ADD, HCONJ_ADD
  // and MAD reject physical result/addend aliases before invoking primitives;
  // the decomposition has no planned scratch to preserve the addend.
  int execute(FhnBackendCtx *ctx, const FhnProgram *program, FhnBuffer **buffers);

  // Plan-aware execution: applies plan.at(i) around each instruction
  // (evict -> alloc -> prefetch before, free after). buffers arrives with
  // input ids filled with distinct, non-null physical handles; aliases are
  // rejected before any hook. Planned allocations are written into it. On failure
  // every plan-allocated id not yet freed is freed and nulled. Budgeted
  // plans first evict all inputs (when movement hooks are present), before
  // any allocation. Entry residency is caller-owned and outside the budget;
  // scheduled high-water starts after normalization. Input contents survive
  // normalization even if a later hook fails; residency may have changed.
  int execute(const FhnMovementHooks &hooks, const FhnProgram *program, FhnBuffer **buffers,
              const FhnMovementPlan &plan);

  private:
  std::unordered_map<int, FhnKernelFn> dispatch_;

  // Attempt to decompose a fused opcode into primitives.
  // Returns true if decomposition succeeded.
  bool decompose(FhnBackendCtx *ctx, const FhnInstruction &inst, FhnBuffer **buffers);
};

} // namespace fhenomenon
