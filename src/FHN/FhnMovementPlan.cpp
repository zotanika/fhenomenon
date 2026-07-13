#include "FHN/FhnMovementPlan.h"

#include <algorithm>
#include <limits>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace fhenomenon {

std::optional<FhnMovementPlan> FhnMovementPlan::analyze(const FhnProgram &program, const std::vector<uint32_t> &pinned,
                                                        uint64_t device_budget, FhnEvictionPolicy policy,
                                                        const FhnLevelModel *model) {
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

  // Level inference: ids are single-assignment, so every id's level is
  // static and can be computed once, up front. Only runs in byte mode.
  std::unordered_map<uint32_t, int64_t> level_of;
  if (model) {
    if (model->fresh_level < 0 || model->bytes_by_level.size() < static_cast<size_t>(model->fresh_level) + 1)
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
      default:
        // An out-of-range FhnLevelEffect from a buggy/future backend must
        // reject the model, not silently plan as PRESERVE. A future
        // FHN_LEVEL_REFRESH lands as an explicit case above.
        return std::nullopt;
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

  const std::unordered_set<uint32_t> pinned_set(pinned.begin(), pinned.end());

  // First use strictly after position `pos` (kNever if none).
  auto next_use_after = [&uses](uint32_t id, int64_t pos) -> int64_t {
    auto it = uses.find(id);
    if (it == uses.end())
      return kNever;
    const auto &v = it->second;
    auto p = std::upper_bound(v.begin(), v.end(), pos);
    return p == v.end() ? kNever : *p;
  };

  FhnMovementPlan plan;
  plan.actions_.resize(program.num_instructions);
  std::set<uint32_t> resident;                      // ordered: deterministic Belady tie-break on lower id
  uint64_t resident_units = 0;                      // Σ cost(id) for id in resident — bytes in byte mode, else count
  std::unordered_map<uint32_t, int64_t> last_touch; // position of most recent def/prefetch/use

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
      uint64_t working_units = 0;
      for (uint32_t id : working)
        working_units += cost(id);
      if (working_units > device_budget)
        return std::nullopt; // one instruction cannot fit — infeasible
      // Belady: make room for the result alloc + missing operands by
      // evicting residents (outside the working set) whose next use is
      // farthest; kNever (no future use) sorts farthest of all, and the
      // ordered set breaks ties on the lower id.
      uint64_t incoming_units = cost(inst.result_id);
      for (uint32_t id : to_prefetch)
        incoming_units += cost(id);
      while (resident_units + incoming_units > device_budget) {
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
        if (!found)
          return std::nullopt; // working set already fills the device
        act.evict.push_back(victim);
        resident.erase(victim);
        resident_units -= cost(victim);
        plan.stats_.evict_count++;
      }
    }

    act.alloc.push_back(inst.result_id);
    resident.insert(inst.result_id);
    resident_units += cost(inst.result_id);
    last_touch[inst.result_id] = pos;
    plan.stats_.alloc_count++;

    for (uint32_t id : to_prefetch) {
      act.prefetch.push_back(id);
      resident.insert(id);
      resident_units += cost(id);
      last_touch[id] = pos;
      plan.stats_.prefetch_count++;
    }

    plan.stats_.high_water = std::max(plan.stats_.high_water, static_cast<uint32_t>(resident.size()));
    if (model)
      plan.stats_.high_water_bytes = std::max(plan.stats_.high_water_bytes, resident_units);

    // Touch every operand for LRU tracking.
    for (uint32_t id : operand_set)
      last_touch[id] = pos;

    // Free everything whose last use has passed: operands with no later
    // use, and a result nothing ever reads.
    for (uint32_t id : working) {
      if (pinned_set.count(id))
        continue;
      const bool dead = (id == inst.result_id) ? (uses.find(id) == uses.end()) : (next_use_after(id, pos) == kNever);
      if (dead) {
        act.free.push_back(id);
        resident.erase(id);
        resident_units -= cost(id);
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
