#pragma once

#include "FHN/fhn_program.h"

#include <memory>
#include <vector>

namespace fhenomenon {
namespace testutil {

// Owning FhnProgram builder for hand-written test programs.
struct ProgramBuilder {
  std::vector<FhnInstruction> insts;
  std::vector<uint32_t> inputs;
  std::vector<uint32_t> outputs;

  ProgramBuilder &input(uint32_t id) {
    inputs.push_back(id);
    return *this;
  }
  ProgramBuilder &inst(FhnOpCode op, uint32_t result, uint32_t a = 0, uint32_t b = 0) {
    FhnInstruction in{};
    in.opcode = op;
    in.result_id = result;
    in.operands[0] = a;
    in.operands[1] = b;
    insts.push_back(in);
    return *this;
  }
  ProgramBuilder &output(uint32_t id) {
    outputs.push_back(id);
    return *this;
  }
  std::unique_ptr<FhnProgram, decltype(&fhn_program_free)> build() {
    auto *p = fhn_program_alloc(static_cast<uint32_t>(insts.size()), static_cast<uint32_t>(inputs.size()),
                                static_cast<uint32_t>(outputs.size()));
    for (uint32_t i = 0; i < p->num_instructions; ++i)
      p->instructions[i] = insts[i];
    for (uint32_t i = 0; i < p->num_inputs; ++i)
      p->input_ids[i] = inputs[i];
    for (uint32_t i = 0; i < p->num_outputs; ++i)
      p->output_ids[i] = outputs[i];
    return {p, &fhn_program_free};
  }
};

} // namespace testutil
} // namespace fhenomenon
