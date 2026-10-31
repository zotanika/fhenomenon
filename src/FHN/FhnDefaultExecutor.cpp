#include "FHN/FhnDefaultExecutor.h"

#include <algorithm>
#include <vector>
#include <unordered_set>

namespace fhenomenon {

FhnDefaultExecutor::FhnDefaultExecutor(FhnKernelTable *table) {
  if (!table)
    return;
  for (uint32_t i = 0; i < table->num_kernels; ++i) {
    const FhnKernelEntry &entry = table->kernels[i];
    if (entry.fn != nullptr) {
      dispatch_[static_cast<int>(entry.opcode)] = entry.fn;
    }
  }
}

bool FhnDefaultExecutor::supports(FhnOpCode opcode) const {
  return dispatch_.find(static_cast<int>(opcode)) != dispatch_.end();
}

int FhnDefaultExecutor::execute(FhnBackendCtx *ctx, const FhnProgram *program, FhnBuffer **buffers) {
  if (!program || !buffers)
    return -1;
  // A program built against a different ABI revision may carry renumbered
  // opcodes or a different instruction layout; refuse to dispatch it.
  if (program->version != FHN_ABI_VERSION)
    return -1;

  for (uint32_t i = 0; i < program->num_instructions; ++i) {
    const FhnInstruction &inst = program->instructions[i];

    auto it = dispatch_.find(static_cast<int>(inst.opcode));
    if (it == dispatch_.end()) {
      if (!decompose(ctx, inst, buffers)) {
        return -1;
      }
      continue;
    }

    // Build operand array from buffers indexed by operand ids.
    const FhnBuffer *ops[4] = {nullptr, nullptr, nullptr, nullptr};
    for (int j = 0; j < 4; ++j) {
      if (inst.operands[j] != 0) {
        ops[j] = buffers[inst.operands[j]];
      }
    }

    FhnKernelFn fn = it->second;
    int rc = fn(ctx, buffers[inst.result_id], ops, inst.params, inst.fparams);
    if (rc != 0)
      return rc;
  }

  return 0;
}

int FhnDefaultExecutor::execute(const FhnMovementHooks &hooks, const FhnProgram *program, FhnBuffer **buffers,
                                const FhnMovementPlan &plan) {
  if (!program || !buffers || !hooks.buffer_alloc || !hooks.buffer_free)
    return -1;
  if (program->version != FHN_ABI_VERSION)
    return -1;
  // A stale plan analyzed against a different (e.g. re-lowered) program
  // would index this buffer table out of bounds; reject the mismatch
  // instead of dispatching against it.
  if (!plan.matches(*program))
    return -1;
  if (static_cast<bool>(hooks.prefetch) != static_cast<bool>(hooks.evict))
    return -1;
  // Validate the complete prelude before changing any input's residency.
  // Plans model distinct ids as distinct residency and lifetime units.
  // Reject aliases rather than evict/free a handle still used by another id.
  std::unordered_set<FhnBuffer *> input_handles;
  for (uint32_t i = 0; i < program->num_inputs; ++i) {
    auto *input = buffers[program->input_ids[i]];
    if (!input || !input_handles.insert(input).second)
      return -1;
  }
  for (uint32_t id : plan.initialEvictions())
    if (hooks.evict && hooks.evict(hooks.ctx, buffers[id]) != 0)
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

bool FhnDefaultExecutor::decompose(FhnBackendCtx *ctx, const FhnInstruction &inst, FhnBuffer **buffers) {
  auto call = [&](FhnOpCode op, FhnBuffer *res, const FhnBuffer *const *ops, const int64_t *params,
                  const double *fparams) -> int {
    auto it = dispatch_.find(static_cast<int>(op));
    if (it == dispatch_.end())
      return -1;
    return it->second(ctx, res, ops, params, fparams);
  };

  switch (inst.opcode) {
  case FHN_HMULT: {
    // HMULT promises all three stages. Validate the complete decomposition
    // before MULT_CC can change the result.
    if (!supports(FHN_MULT_CC) || !supports(FHN_RELINEARIZE) || !supports(FHN_RESCALE))
      return false;

    const FhnBuffer *ops[] = {buffers[inst.operands[0]], buffers[inst.operands[1]], nullptr, nullptr};
    if (call(FHN_MULT_CC, buffers[inst.result_id], ops, inst.params, inst.fparams) != 0)
      return false;

    const FhnBuffer *self_ops[] = {buffers[inst.result_id], nullptr, nullptr, nullptr};
    if (call(FHN_RELINEARIZE, buffers[inst.result_id], self_ops, inst.params, inst.fparams) != 0)
      return false;
    if (call(FHN_RESCALE, buffers[inst.result_id], self_ops, inst.params, inst.fparams) != 0)
      return false;

    return true;
  }
  case FHN_HROT: {
    // FHN_ROTATE is a complete key-switched rotation by convention, so the
    // fused HROT decomposes to a single ROTATE. (A MULT_KEY + ROTATE
    // sequence would apply the evaluation key twice and silently corrupt
    // the ciphertext.)
    if (!dispatch_.count(static_cast<int>(FHN_ROTATE)))
      return false;
    const FhnBuffer *ops[] = {buffers[inst.operands[0]], nullptr, nullptr, nullptr};
    return call(FHN_ROTATE, buffers[inst.result_id], ops, inst.params, inst.fparams) == 0;
  }
  case FHN_HROT_ADD: {
    if (inst.operands[1] == 0 || !buffers[inst.operands[1]] ||
        buffers[inst.result_id] == buffers[inst.operands[1]] || !supports(FHN_ADD_CC))
      return false;
    // Decompose HROT part first
    FhnInstruction hrot_inst = inst;
    hrot_inst.opcode = FHN_HROT;
    bool ok;
    if (dispatch_.count(static_cast<int>(FHN_HROT))) {
      const FhnBuffer *ops[] = {buffers[inst.operands[0]], nullptr, nullptr, nullptr};
      ok = call(FHN_HROT, buffers[inst.result_id], ops, inst.params, inst.fparams) == 0;
    } else {
      ok = decompose(ctx, hrot_inst, buffers);
    }
    if (!ok)
      return false;
    const FhnBuffer *add_ops[] = {buffers[inst.result_id], buffers[inst.operands[1]], nullptr, nullptr};
    return call(FHN_ADD_CC, buffers[inst.result_id], add_ops, inst.params, inst.fparams) == 0;
  }
  case FHN_HCONJ_ADD: {
    if (inst.operands[1] == 0 || !buffers[inst.operands[1]] ||
        buffers[inst.result_id] == buffers[inst.operands[1]])
      return false;
    if (!dispatch_.count(static_cast<int>(FHN_CONJUGATE)) || !dispatch_.count(static_cast<int>(FHN_ADD_CC)))
      return false;
    const FhnBuffer *ops[] = {buffers[inst.operands[0]], nullptr, nullptr, nullptr};
    if (call(FHN_CONJUGATE, buffers[inst.result_id], ops, inst.params, inst.fparams) != 0)
      return false;
    const FhnBuffer *add_ops[] = {buffers[inst.result_id], buffers[inst.operands[1]], nullptr, nullptr};
    return call(FHN_ADD_CC, buffers[inst.result_id], add_ops, inst.params, inst.fparams) == 0;
  }
  case FHN_MAD: {
    // res = a * fparams[0] + b. The multiply lands in the result buffer
    // before b is read, so b must be present and must not alias res.
    if (inst.operands[1] == 0 || !buffers[inst.operands[1]] ||
        buffers[inst.result_id] == buffers[inst.operands[1]])
      return false;
    if (!dispatch_.count(static_cast<int>(FHN_MULT_CS)) || !dispatch_.count(static_cast<int>(FHN_ADD_CC)))
      return false;
    const FhnBuffer *mult_ops[] = {buffers[inst.operands[0]], nullptr, nullptr, nullptr};
    if (call(FHN_MULT_CS, buffers[inst.result_id], mult_ops, inst.params, inst.fparams) != 0)
      return false;
    const FhnBuffer *add_ops[] = {buffers[inst.result_id], buffers[inst.operands[1]], nullptr, nullptr};
    return call(FHN_ADD_CC, buffers[inst.result_id], add_ops, inst.params, inst.fparams) == 0;
  }
  default:
    return false;
  }
}

} // namespace fhenomenon
