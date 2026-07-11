#pragma once

#include "FHN/fhn_program.h"
#include "Scheduler/ASTNode.h"
#include "Scheduler/Planner.h"

#include <unordered_map>
#include <vector>

namespace fhenomenon {
namespace scheduler {

class LowerToFhnProgram {
  public:
  // One (buffer id, entity) association, in lowering order. Inputs bind
  // OperandNode entities to their input ids; instructions bind result
  // entities to their result ids; assignments re-bind the assigned variable
  // to the value's id. Walking the list forward therefore yields each
  // entity's final id (later entries win).
  template <typename T> using EntityBindings = std::vector<std::pair<uint32_t, std::shared_ptr<Fhenon<T>>>>;

  // Lower an AST plan to an FhnProgram.
  // Caller owns the returned FhnProgram and must call fhn_program_free().
  // When `bindings` is non-null it receives the id/entity associations the
  // caller needs to provision input buffers and write results back.
  template <typename T> FhnProgram *lower(const Planner<T> &plan, EntityBindings<T> *bindings = nullptr) const;

  private:
  template <typename T>
  void lowerNode(ASTNode *node, std::vector<FhnInstruction> &instructions, std::vector<uint32_t> &inputs,
                 uint32_t &next_id, std::unordered_map<ASTNode *, uint32_t> &node_ids, EntityBindings<T> *bindings,
                 bool &supported) const;

  static FhnOpCode mapOpType(OperationType type);
};

// ---------------------------------------------------------------------------
// Template implementations
// ---------------------------------------------------------------------------

template <typename T> FhnProgram *LowerToFhnProgram::lower(const Planner<T> &plan, EntityBindings<T> *bindings) const {
  std::vector<FhnInstruction> instructions;
  std::vector<uint32_t> inputs;
  uint32_t next_id = 1;
  std::unordered_map<ASTNode *, uint32_t> node_ids;
  bool supported = true;

  // Post-order traversal of all roots
  for (const auto &root : plan.getRoots()) {
    lowerNode<T>(root.get(), instructions, inputs, next_id, node_ids, bindings, supported);
  }

  // A plan containing a node kind this lowering cannot express (e.g. a
  // FusedKernelNode) must fail whole: a partial program would silently skip
  // the unsupported work.
  if (!supported) {
    if (bindings)
      bindings->clear();
    return nullptr;
  }

  // Collect output ids (root results)
  std::vector<uint32_t> outputs;
  for (const auto &root : plan.getRoots()) {
    auto it = node_ids.find(root.get());
    if (it != node_ids.end()) {
      outputs.push_back(it->second);
    }
  }

  // Allocate FhnProgram
  auto *prog = fhn_program_alloc(static_cast<uint32_t>(instructions.size()), static_cast<uint32_t>(inputs.size()),
                                 static_cast<uint32_t>(outputs.size()));
  if (!prog)
    return nullptr;

  // Copy data
  for (uint32_t i = 0; i < prog->num_instructions; i++)
    prog->instructions[i] = instructions[i];
  for (uint32_t i = 0; i < prog->num_inputs; i++)
    prog->input_ids[i] = inputs[i];
  for (uint32_t i = 0; i < prog->num_outputs; i++)
    prog->output_ids[i] = outputs[i];

  return prog;
}

template <typename T>
void LowerToFhnProgram::lowerNode(ASTNode *node, std::vector<FhnInstruction> &instructions,
                                  std::vector<uint32_t> &inputs, uint32_t &next_id,
                                  std::unordered_map<ASTNode *, uint32_t> &node_ids, EntityBindings<T> *bindings,
                                  bool &supported) const {
  if (node_ids.count(node))
    return; // Already visited (DAG)

  if (auto *op_node = dynamic_cast<OperatorNode<T> *>(node)) {
    // A scalar right operand folds into the instruction itself (FHN_*_CS
    // with the value in fparams[0]) instead of lowering to a ciphertext
    // program input; it gets no id, no buffer, and no binding.
    std::shared_ptr<Fhenon<T>> scalar;
    if ((op_node->getType() == OperationType::Add || op_node->getType() == OperationType::Multiply) &&
        op_node->getRight()) {
      if (auto *rhs = dynamic_cast<OperandNode<T> *>(op_node->getRight().get())) {
        if (rhs->getEntity() && rhs->getEntity()->isScalar())
          scalar = rhs->getEntity();
      }
    }

    // Visit children first (post-order)
    if (op_node->getLeft())
      lowerNode<T>(op_node->getLeft().get(), instructions, inputs, next_id, node_ids, bindings, supported);
    if (op_node->getRight() && !scalar)
      lowerNode<T>(op_node->getRight().get(), instructions, inputs, next_id, node_ids, bindings, supported);

    // Assignment nodes don't produce FHN instructions: reads of the assigned
    // variable must resolve to the *assigned value* — the right child.
    if (op_node->getType() == OperationType::Assignment) {
      ASTNode *value = op_node->getRight() ? op_node->getRight().get() : op_node->getLeft().get();
      auto it = node_ids.find(value);
      if (it != node_ids.end()) {
        node_ids[node] = it->second;
        if (bindings)
          bindings->emplace_back(it->second, op_node->getResult());
      }
      return;
    }

    FhnInstruction inst{};
    inst.opcode = mapOpType(op_node->getType());
    inst.result_id = next_id++;

    if (scalar) {
      inst.opcode = (op_node->getType() == OperationType::Add) ? FHN_ADD_CS : FHN_MULT_CS;
      inst.fparams[0] = static_cast<double>(scalar->getValue());
    }

    // Operation-specific parameters. FHN_ROTATE encodes a signed rotation
    // distance in params[0] (positive = left); RightRotate negates.
    if (op_node->getType() == OperationType::LeftRotate || op_node->getType() == OperationType::RightRotate) {
      int64_t distance = 0;
      if (auto operation = op_node->getOperation()) {
        distance = operation->getParam();
      }
      inst.params[0] = (op_node->getType() == OperationType::RightRotate) ? -distance : distance;
    }

    if (op_node->getLeft()) {
      auto it = node_ids.find(op_node->getLeft().get());
      if (it != node_ids.end())
        inst.operands[0] = it->second;
    }
    if (op_node->getRight()) {
      auto it = node_ids.find(op_node->getRight().get());
      if (it != node_ids.end())
        inst.operands[1] = it->second;
    }

    instructions.push_back(inst);
    node_ids[node] = inst.result_id;
    if (bindings)
      bindings->emplace_back(inst.result_id, op_node->getResult());

  } else if (auto *operand_node = dynamic_cast<OperandNode<T> *>(node)) {
    uint32_t id = next_id++;
    node_ids[node] = id;
    inputs.push_back(id);
    if (bindings)
      bindings->emplace_back(id, operand_node->getEntity());
  } else {
    // Unknown node kind (e.g. FusedKernelNode) — this plan cannot be
    // expressed as an FhnProgram.
    supported = false;
  }
}

} // namespace scheduler
} // namespace fhenomenon
