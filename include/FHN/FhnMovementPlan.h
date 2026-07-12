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
  // A zero-instruction program has no action slots, so unused unpinned
  // inputs are not freed in that (degenerate) case; callers that can
  // produce such programs must pin or free their inputs themselves.
  static std::optional<FhnMovementPlan> analyze(const FhnProgram &program, const std::vector<uint32_t> &pinned,
                                                uint32_t device_budget = 0);

  const FhnMovementActions &at(uint32_t inst_index) const { return actions_[inst_index]; }
  const Stats &stats() const { return stats_; }
  // Used to reject executing a plan against a different program.
  uint32_t instructionCount() const { return static_cast<uint32_t>(actions_.size()); }

  private:
  FhnMovementPlan() = default;

  std::vector<FhnMovementActions> actions_;
  Stats stats_;
};

} // namespace fhenomenon
