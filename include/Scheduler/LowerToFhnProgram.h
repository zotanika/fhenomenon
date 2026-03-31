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
  // Lower an AST plan to an FhnProgram.
  // Caller owns the returned FhnProgram and must call fhn_program_free().
  template <typename T> FhnProgram *lower(const Planner<T> &plan) const;

  private:
  template <typename T>
  void lowerNode(ASTNode *node, std::vector<FhnInstruction> &instructions, std::vector<uint32_t> &inputs,
                 uint32_t &next_id, std::unordered_map<ASTNode *, uint32_t> &node_ids) const;

  static FhnOpCode mapOpType(OperationType type);
};

// ---------------------------------------------------------------------------
// Template implementations
// ---------------------------------------------------------------------------

template <typename T> FhnProgram *LowerToFhnProgram::lower(const Planner<T> &plan) const {
  std::vector<FhnInstruction> instructions;
  std::vector<uint32_t> inputs;
  uint32_t next_id = 1;
  std::unordered_map<ASTNode *, uint32_t> node_ids;

  // Post-order traversal of all roots
  for (const auto &root : plan.getRoots()) {
    lowerNode<T>(root.get(), instructions, inputs, next_id, node_ids);
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
                                  std::unordered_map<ASTNode *, uint32_t> &node_ids) const {
  if (node_ids.count(node))
    return; // Already visited (DAG)

  if (auto *op_node = dynamic_cast<OperatorNode<T> *>(node)) {
    // Visit children first (post-order)
    if (op_node->getLeft())
      lowerNode<T>(op_node->getLeft().get(), instructions, inputs, next_id, node_ids);
    if (op_node->getRight())
      lowerNode<T>(op_node->getRight().get(), instructions, inputs, next_id, node_ids);

    // Skip Assignment nodes (they don't produce FHN instructions)
    if (op_node->getType() == OperationType::Assignment) {
      // Map this node to its left child's id
      if (op_node->getLeft()) {
        auto it = node_ids.find(op_node->getLeft().get());
        if (it != node_ids.end())
          node_ids[node] = it->second;
      }
      return;
    }

    FhnInstruction inst{};
    inst.opcode = mapOpType(op_node->getType());
    inst.result_id = next_id++;

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

  } else if (dynamic_cast<OperandNode<T> *>(node)) {
    uint32_t id = next_id++;
    node_ids[node] = id;
    inputs.push_back(id);
  }
}

} // namespace scheduler
} // namespace fhenomenon
