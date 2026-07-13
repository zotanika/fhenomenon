#pragma once

#include "corpus_types.h"

namespace fhenomenon {
namespace corpus {

// Incremental FhnProgram builder for shape generators. Ids are allocated
// sequentially (single-assignment); inputs carry their plaintext slots.
class ShapeBuilder {
  public:
  uint32_t input(Slots v) {
    const uint32_t id = next_id_++;
    input_ids_.push_back(id);
    inputs_[id] = std::move(v);
    return id;
  }

  uint32_t cc(FhnOpCode op, uint32_t a, uint32_t b) {
    FhnInstruction in{};
    in.opcode = op;
    in.result_id = next_id_++;
    in.operands[0] = a;
    in.operands[1] = b;
    insts_.push_back(in);
    return in.result_id;
  }

  uint32_t cs(FhnOpCode op, uint32_t a, int64_t scalar) {
    FhnInstruction in{};
    in.opcode = op;
    in.result_id = next_id_++;
    in.operands[0] = a;
    in.fparams[0] = static_cast<double>(scalar);
    insts_.push_back(in);
    return in.result_id;
  }

  uint32_t rot(uint32_t a, int64_t dist) {
    FhnInstruction in{};
    in.opcode = FHN_ROTATE;
    in.result_id = next_id_++;
    in.operands[0] = a;
    in.params[0] = dist;
    insts_.push_back(in);
    return in.result_id;
  }

  uint32_t un(FhnOpCode op, uint32_t a) {
    FhnInstruction in{};
    in.opcode = op;
    in.result_id = next_id_++;
    in.operands[0] = a;
    insts_.push_back(in);
    return in.result_id;
  }

  Shape finish(std::string name, std::string axis, uint32_t slot_count, uint32_t ct_mult_depth,
               std::vector<uint32_t> output_ids) {
    auto *p = fhn_program_alloc(static_cast<uint32_t>(insts_.size()), static_cast<uint32_t>(input_ids_.size()),
                                static_cast<uint32_t>(output_ids.size()));
    for (uint32_t i = 0; i < p->num_instructions; ++i)
      p->instructions[i] = insts_[i];
    for (uint32_t i = 0; i < p->num_inputs; ++i)
      p->input_ids[i] = input_ids_[i];
    for (uint32_t i = 0; i < p->num_outputs; ++i)
      p->output_ids[i] = output_ids[i];
    return Shape{
      std::move(name),    std::move(axis),      slot_count, ct_mult_depth, FhnProgramPtr(p, &fhn_program_free),
      std::move(inputs_), std::move(output_ids)};
  }

  private:
  uint32_t next_id_ = 1;
  std::vector<FhnInstruction> insts_;
  std::vector<uint32_t> input_ids_;
  std::map<uint32_t, Slots> inputs_;
};

} // namespace corpus
} // namespace fhenomenon
