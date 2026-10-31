#include "corpus_circuit.h"
#include <limits>
#include <unordered_set>

namespace fhenomenon::corpus {
FhnCircuit Circuit::view() const {
  return {
    FHN_CIRCUIT_VERSION, static_cast<uint32_t>(nodes.size()),   nodes.data(),  static_cast<uint32_t>(inputs.size()),
    inputs.data(),       static_cast<uint32_t>(outputs.size()), outputs.data()};
}
std::optional<Circuit> importCircuit(const FhnProgram &program, std::string *error) {
  if (error)
    error->clear();
  auto fail = [&](const std::string &message) -> std::optional<Circuit> {
    if (error)
      *error = message;
    return std::nullopt;
  };
  if (program.version != FHN_ABI_VERSION || (program.num_instructions && !program.instructions) ||
      (program.num_inputs && !program.input_ids) || (program.num_outputs && !program.output_ids))
    return fail("invalid source program version or arrays");
  Circuit result;
  if (program.num_inputs)
    result.inputs.assign(program.input_ids, program.input_ids + program.num_inputs);
  if (program.num_outputs)
    result.outputs.assign(program.output_ids, program.output_ids + program.num_outputs);
  // Reserve every source ID before inventing temporaries, including future
  // definitions and sparse UINT32_MAX IDs. No dense max-ID allocation.
  std::unordered_set<uint32_t> used(result.inputs.begin(), result.inputs.end());
  used.insert(result.outputs.begin(), result.outputs.end());
  for (uint32_t i = 0; i < program.num_instructions; ++i) {
    const auto &in = program.instructions[i];
    used.insert(in.result_id);
    for (auto operand : in.operands)
      used.insert(operand);
  }
  uint64_t candidate = 1;
  auto temporary = [&]() -> uint32_t {
    while (candidate <= UINT32_MAX && used.count(static_cast<uint32_t>(candidate)))
      ++candidate;
    if (candidate > UINT32_MAX)
      return 0;
    const auto id = static_cast<uint32_t>(candidate++);
    used.insert(id);
    return id;
  };
  for (uint32_t i = 0; i < program.num_instructions; ++i) {
    const auto &in = program.instructions[i];
    FhnCircuitNode node{};
    node.result_id = in.result_id;
    node.operands[0] = in.operands[0];
    node.operands[1] = in.operands[1];
    node.rotation = in.params[0];
    node.scalar = in.fparams[0];
    switch (in.opcode) {
    case FHN_ADD_CC:
      node.op = FHN_CIRCUIT_ADD;
      break;
    case FHN_SUB_CC:
      node.op = FHN_CIRCUIT_SUBTRACT;
      break;
    case FHN_MULT_CC:
    case FHN_HMULT:
      node.op = FHN_CIRCUIT_MULTIPLY;
      break;
    case FHN_MULT_CS:
      node.op = FHN_CIRCUIT_SCALE;
      break;
    case FHN_ADD_CS:
      node.op = FHN_CIRCUIT_ADD_CONSTANT;
      break;
    case FHN_NEGATE:
      node.op = FHN_CIRCUIT_NEGATE;
      break;
    case FHN_ROTATE:
    case FHN_HROT:
      node.op = FHN_CIRCUIT_ROTATE;
      break;
    case FHN_CONJUGATE:
      node.op = FHN_CIRCUIT_CONJUGATE;
      break;
    case FHN_HROT_ADD:
    case FHN_HCONJ_ADD:
    case FHN_MAD:
    case FHN_SUB_SC: {
      node.result_id = temporary();
      if (!node.result_id)
        return fail("no free circuit ID for composite expansion");
      node.operands[1] = 0;
      node.op = in.opcode == FHN_HROT_ADD    ? FHN_CIRCUIT_ROTATE
                : in.opcode == FHN_HCONJ_ADD ? FHN_CIRCUIT_CONJUGATE
                : in.opcode == FHN_MAD       ? FHN_CIRCUIT_SCALE
                                             : FHN_CIRCUIT_NEGATE;
      result.nodes.push_back(node);
      node.operands[0] = node.result_id;
      node.operands[1] = in.opcode == FHN_SUB_SC ? 0 : in.operands[1];
      node.result_id = in.result_id;
      node.op = in.opcode == FHN_SUB_SC ? FHN_CIRCUIT_ADD_CONSTANT : FHN_CIRCUIT_ADD;
      break;
    }
    default:
      return fail("source instruction " + std::to_string(i) + " (opcode " +
                  std::to_string(static_cast<unsigned>(in.opcode)) +
                  ") has no corpus value-circuit mapping (physical or Boolean operation)");
    }
    result.nodes.push_back(node);
    if (result.nodes.size() > UINT32_MAX)
      return fail("expanded circuit exceeds node-count representation");
  }
  return result;
}
} // namespace fhenomenon::corpus
