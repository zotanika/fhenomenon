#pragma once

#include "FHN/fhn_backend_api.h"
#include "FHN/fhn_program.h"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace fhenomenon {

// Eviction policy for budgeted planning. Belady uses farthest next use,
// optimal for equal-size paging with uniform transfer costs. With variable
// buffer sizes it is a heuristic, not a minimum-transfer/byte guarantee.
// Lru is a benchmarking baseline.
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
// under a device budget — farthest-next-use eviction. Perfect future
// knowledge does not make this greedy policy optimal for variable sizes.
class FhnMovementPlan {
  public:
  struct Stats {
    // Scheduled residency after input normalization; excludes the caller
    // state on entry, backend scratch and reserved allocation capacity.
    // With absent movement hooks these are modeled, not measured, values.
    uint32_t high_water = 0; // max simultaneously scheduled resident buffers
    uint32_t alloc_count = 0;
    uint32_t prefetch_count = 0;
    uint32_t evict_count = 0; // instruction-scheduled evictions only
    uint64_t high_water_bytes = 0; // max simultaneously resident bytes (model only, else 0)
    uint32_t initial_evict_count = 0; // normalization calls, separate from transfers
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
  // When null (the default), planning is slot-count based.
  // Budgeted plans include initialEvictions(): every input is demoted before
  // any allocation, establishing the empty initial residency used below.
  // The budget applies after this prelude, not to caller residency on entry.
  // Pinned means lifetime preservation, not permanent device residency.
  // nullopt: invalid program (zero/duplicate defs, operand used before or
  // without a def), infeasible budget (one instruction's working set
  // exceeds it), or (byte mode) an invalid model/level trace or cost sum
  // that cannot be represented in uint64_t.
  // A zero-instruction program has no action slots, so unused unpinned
  // inputs are not freed in that (degenerate) case; callers that can
  // produce such programs must pin or free their inputs themselves.
  static std::optional<FhnMovementPlan> analyze(const FhnProgram &program, const std::vector<uint32_t> &pinned,
                                                uint64_t device_budget = 0,
                                                FhnEvictionPolicy policy = FhnEvictionPolicy::Belady,
                                                const FhnLevelModel *model = nullptr);

  const FhnMovementActions &at(uint32_t inst_index) const { return actions_[inst_index]; }
  const Stats &stats() const { return stats_; }
  uint32_t instructionCount() const { return static_cast<uint32_t>(actions_.size()); }
  // An owned semantic snapshot detects changes even to the original object.
  bool matches(const FhnProgram &program) const;
  bool budgeted() const { return budgeted_; }
  const std::vector<uint32_t> &initialEvictions() const { return initial_evictions_; }

  private:
  FhnMovementPlan() = default;

  std::vector<FhnMovementActions> actions_;
  uint32_t version_ = FHN_ABI_VERSION;
  std::vector<FhnInstruction> instructions_;
  std::vector<uint32_t> inputs_, outputs_;
  bool budgeted_ = false;
  std::vector<uint32_t> initial_evictions_;
  Stats stats_;
};

} // namespace fhenomenon
