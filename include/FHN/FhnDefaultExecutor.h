#pragma once

#include "FHN/fhn_backend_api.h"
#include <unordered_map>

namespace fhenomenon {

class FhnDefaultExecutor {
  public:
  explicit FhnDefaultExecutor(FhnKernelTable *table);

  bool supports(FhnOpCode opcode) const;

  // Execute a program by dispatching each instruction to kernel functions.
  // buffers is indexed by result_id.
  // Returns 0 on success, non-zero on error.
  int execute(FhnBackendCtx *ctx, const FhnProgram *program, FhnBuffer **buffers);

  private:
  std::unordered_map<int, FhnKernelFn> dispatch_;

  // Attempt to decompose a fused opcode into primitives.
  // Returns true if decomposition succeeded.
  bool decompose(FhnBackendCtx *ctx, const FhnInstruction &inst, FhnBuffer **buffers);
};

} // namespace fhenomenon
