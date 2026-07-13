#pragma once

#include "FHN/fhn_backend_api.h"
#include "FHN/fhn_program.h"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace fhenomenon {

// Eviction policy for budgeted planning. Belady (exact optimal, enabled by
// the data-oblivious IR's perfect future knowledge) is the default and the
// only policy production paths use. Lru exists SOLELY as a benchmarking
// baseline so the corpus can quantify what exact future knowledge saves.
enum class FhnEvictionPolicy { Belady, Lru };

// A backend's declared level semantics, pre-queried by the caller — the
// planner never calls the ABI. bytes_by_level is indexed by level and
// must cover 0..fresh_level; effects must cover every opcode the program
// uses. With a model, analyze() interprets device_budget as BYTES.
struct FhnLevelModel {
  int64_t fresh_level = 0;
  std::vector<uint64_t> bytes_by_level;
  std::unordered_map<int, FhnLevelEffect> effects; // key: FhnOpCode
};

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
    uint64_t high_water_bytes = 0; // max simultaneously resident bytes (model only, else 0)
  };

  // pinned: ids that must survive execution; the plan never frees them.
  // Non-pinned lifetimes belong to the plan — callers must pin every buffer
  // they own (Session pins its inputs and write-back targets).
  // device_budget: max simultaneously device-resident buffers (slot mode)
  // or bytes (byte mode, model != nullptr), 0 = unlimited.
  // model: a backend's declared level semantics (see FhnLevelModel). When
  // non-null, device_budget is byte-denominated and the plan is rejected
  // (nullopt) if the model is malformed or the program's level trace is
  // invalid (underflow, a raise, or an opcode the model does not declare).
  // When null (the default), planning is slot-count based and behavior is
  // bit-identical to before this parameter existed.
  // nullopt: invalid program (zero/duplicate defs, operand used before or
  // without a def), infeasible budget (one instruction's working set
  // exceeds it), or (byte mode) an invalid model/level trace.
  // A zero-instruction program has no action slots, so unused unpinned
  // inputs are not freed in that (degenerate) case; callers that can
  // produce such programs must pin or free their inputs themselves.
  static std::optional<FhnMovementPlan> analyze(const FhnProgram &program, const std::vector<uint32_t> &pinned,
                                                uint64_t device_budget = 0,
                                                FhnEvictionPolicy policy = FhnEvictionPolicy::Belady,
                                                const FhnLevelModel *model = nullptr);

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
