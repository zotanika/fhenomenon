#include "FHN/FhnDefaultExecutor.h"

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
    // Must have MULT_CC at minimum
    if (dispatch_.find(static_cast<int>(FHN_MULT_CC)) == dispatch_.end())
      return false;

    const FhnBuffer *ops[] = {buffers[inst.operands[0]], buffers[inst.operands[1]], nullptr, nullptr};
    if (call(FHN_MULT_CC, buffers[inst.result_id], ops, inst.params, inst.fparams) != 0)
      return false;

    // RELINEARIZE and RESCALE are optional (applied if supported)
    const FhnBuffer *self_ops[] = {buffers[inst.result_id], nullptr, nullptr, nullptr};
    if (dispatch_.count(static_cast<int>(FHN_RELINEARIZE)))
      call(FHN_RELINEARIZE, buffers[inst.result_id], self_ops, inst.params, inst.fparams);
    if (dispatch_.count(static_cast<int>(FHN_RESCALE)))
      call(FHN_RESCALE, buffers[inst.result_id], self_ops, inst.params, inst.fparams);

    return true;
  }
  case FHN_HROT: {
    if (!dispatch_.count(static_cast<int>(FHN_MULT_KEY)) || !dispatch_.count(static_cast<int>(FHN_ROTATE)))
      return false;
    const FhnBuffer *ops[] = {buffers[inst.operands[0]], nullptr, nullptr, nullptr};
    if (call(FHN_MULT_KEY, buffers[inst.result_id], ops, inst.params, inst.fparams) != 0)
      return false;
    const FhnBuffer *self_ops[] = {buffers[inst.result_id], nullptr, nullptr, nullptr};
    return call(FHN_ROTATE, buffers[inst.result_id], self_ops, inst.params, inst.fparams) == 0;
  }
  case FHN_HROT_ADD: {
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
    if (!ok || !dispatch_.count(static_cast<int>(FHN_ADD_CC)))
      return false;
    const FhnBuffer *add_ops[] = {buffers[inst.result_id], buffers[inst.operands[1]], nullptr, nullptr};
    return call(FHN_ADD_CC, buffers[inst.result_id], add_ops, inst.params, inst.fparams) == 0;
  }
  case FHN_HCONJ_ADD: {
    if (!dispatch_.count(static_cast<int>(FHN_CONJUGATE)) || !dispatch_.count(static_cast<int>(FHN_ADD_CC)))
      return false;
    const FhnBuffer *ops[] = {buffers[inst.operands[0]], nullptr, nullptr, nullptr};
    if (call(FHN_CONJUGATE, buffers[inst.result_id], ops, inst.params, inst.fparams) != 0)
      return false;
    const FhnBuffer *add_ops[] = {buffers[inst.result_id], buffers[inst.operands[1]], nullptr, nullptr};
    return call(FHN_ADD_CC, buffers[inst.result_id], add_ops, inst.params, inst.fparams) == 0;
  }
  case FHN_MAD: {
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
