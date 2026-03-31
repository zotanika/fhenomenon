#include "FHN/FhnDefaultExecutor.h"

namespace fhenomenon {

FhnDefaultExecutor::FhnDefaultExecutor(FhnKernelTable *table) {
    if (!table) return;
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

int FhnDefaultExecutor::execute(FhnBackendCtx *ctx, const FhnProgram *program,
                                FhnBuffer **buffers) {
    if (!program || !buffers) return -1;

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
        int rc = fn(ctx, buffers[inst.result_id], ops, inst.params,
                    inst.fparams);
        if (rc != 0) return rc;
    }

    return 0;
}

bool FhnDefaultExecutor::decompose(FhnBackendCtx * /*ctx*/,
                                   const FhnInstruction & /*inst*/,
                                   FhnBuffer ** /*buffers*/) {
    return false;
}

} // namespace fhenomenon
