#include "Scheduler/LowerToFhnProgram.h"
#include "Compuon.h"
#include "FHN/fhn_program.h"
#include "Scheduler/ASTNode.h"
#include "Scheduler/Operation.h"
#include "Scheduler/Planner.h"
#include <gtest/gtest.h>

using namespace fhenomenon;
using namespace fhenomenon::scheduler;

TEST(LowerToFhnProgram, SingleAdd) {
  auto a = std::make_shared<Compuon<int>>(10);
  auto b = std::make_shared<Compuon<int>>(20);
  auto result = std::make_shared<Compuon<int>>(0);

  auto op = std::make_shared<Operation<int>>(OperationType::Add, a, b, result);
  auto left = std::make_shared<OperandNode<int>>(a);
  auto right = std::make_shared<OperandNode<int>>(b);
  auto root = std::make_shared<OperatorNode<int>>(op, OperationType::Add, left, right, result);

  Planner<int> plan;
  plan.addRoot(root);

  LowerToFhnProgram lowering;
  FhnProgram *prog = lowering.lower(plan);

  ASSERT_NE(prog, nullptr);
  EXPECT_EQ(prog->num_inputs, 2u);
  EXPECT_EQ(prog->num_instructions, 1u);
  EXPECT_EQ(prog->instructions[0].opcode, FHN_ADD_CC);
  EXPECT_EQ(prog->num_outputs, 1u);

  fhn_program_free(prog);
}

TEST(LowerToFhnProgram, ChainedAddMult) {
  // AST: t = a + b; result = t * c
  auto a = std::make_shared<Compuon<int>>(1);
  auto b = std::make_shared<Compuon<int>>(2);
  auto c = std::make_shared<Compuon<int>>(3);
  auto t = std::make_shared<Compuon<int>>(0);
  auto result = std::make_shared<Compuon<int>>(0);

  auto add_op = std::make_shared<Operation<int>>(OperationType::Add, a, b, t);
  auto mul_op = std::make_shared<Operation<int>>(OperationType::Multiply, t, c, result);

  auto leaf_a = std::make_shared<OperandNode<int>>(a);
  auto leaf_b = std::make_shared<OperandNode<int>>(b);
  auto leaf_c = std::make_shared<OperandNode<int>>(c);

  auto add_node = std::make_shared<OperatorNode<int>>(add_op, OperationType::Add, leaf_a, leaf_b, t);
  auto mul_node = std::make_shared<OperatorNode<int>>(mul_op, OperationType::Multiply, add_node, leaf_c, result);

  Planner<int> plan;
  plan.addRoot(mul_node);

  LowerToFhnProgram lowering;
  FhnProgram *prog = lowering.lower(plan);

  ASSERT_NE(prog, nullptr);
  EXPECT_EQ(prog->num_inputs, 3u);      // a, b, c
  EXPECT_EQ(prog->num_instructions, 2u); // add then mult
  EXPECT_EQ(prog->instructions[0].opcode, FHN_ADD_CC);
  EXPECT_EQ(prog->instructions[1].opcode, FHN_HMULT);
  // mult's first operand = add's result
  EXPECT_EQ(prog->instructions[1].operands[0], prog->instructions[0].result_id);
  EXPECT_EQ(prog->num_outputs, 1u);

  fhn_program_free(prog);
}

TEST(LowerToFhnProgram, SharedOperand) {
  // AST: r1 = a + a (same operand node used twice)
  auto a = std::make_shared<Compuon<int>>(5);
  auto result = std::make_shared<Compuon<int>>(0);

  auto op = std::make_shared<Operation<int>>(OperationType::Add, a, a, result);
  auto left = std::make_shared<OperandNode<int>>(a);
  // Use SAME node for both left and right
  auto root = std::make_shared<OperatorNode<int>>(op, OperationType::Add, left, left, result);

  Planner<int> plan;
  plan.addRoot(root);

  LowerToFhnProgram lowering;
  FhnProgram *prog = lowering.lower(plan);

  ASSERT_NE(prog, nullptr);
  EXPECT_EQ(prog->num_inputs, 1u);      // just 'a' once
  EXPECT_EQ(prog->num_instructions, 1u);
  // Both operands reference the same input id
  EXPECT_EQ(prog->instructions[0].operands[0], prog->instructions[0].operands[1]);

  fhn_program_free(prog);
}
