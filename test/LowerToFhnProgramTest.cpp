#include "Scheduler/LowerToFhnProgram.h"
#include "FHN/fhn_program.h"
#include "Fhenon.h"
#include "Scheduler/ASTNode.h"
#include "Scheduler/Operation.h"
#include "Scheduler/Planner.h"
#include <gtest/gtest.h>

using namespace fhenomenon;
using namespace fhenomenon::scheduler;

TEST(LowerToFhnProgram, SingleAdd) {
  auto a = std::make_shared<Fhenon<int>>(10);
  auto b = std::make_shared<Fhenon<int>>(20);
  auto result = std::make_shared<Fhenon<int>>(0);

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
  auto a = std::make_shared<Fhenon<int>>(1);
  auto b = std::make_shared<Fhenon<int>>(2);
  auto c = std::make_shared<Fhenon<int>>(3);
  auto t = std::make_shared<Fhenon<int>>(0);
  auto result = std::make_shared<Fhenon<int>>(0);

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
  EXPECT_EQ(prog->num_inputs, 3u);       // a, b, c
  EXPECT_EQ(prog->num_instructions, 2u); // add then mult
  EXPECT_EQ(prog->instructions[0].opcode, FHN_ADD_CC);
  EXPECT_EQ(prog->instructions[1].opcode, FHN_HMULT);
  // mult's first operand = add's result
  EXPECT_EQ(prog->instructions[1].operands[0], prog->instructions[0].result_id);
  EXPECT_EQ(prog->num_outputs, 1u);

  fhn_program_free(prog);
}

TEST(LowerToFhnProgram, RotationDistanceIsEncoded) {
  auto a = std::make_shared<Fhenon<int>>(5);
  auto result = std::make_shared<Fhenon<int>>(0);

  auto op = std::make_shared<Operation<int>>(OperationType::LeftRotate, a, nullptr, result, nullptr, 3);
  auto leaf = std::make_shared<OperandNode<int>>(a);
  auto root = std::make_shared<OperatorNode<int>>(op, OperationType::LeftRotate, leaf, nullptr, result);

  Planner<int> plan;
  plan.addRoot(root);

  LowerToFhnProgram lowering;
  FhnProgram *prog = lowering.lower(plan);

  ASSERT_NE(prog, nullptr);
  ASSERT_EQ(prog->num_instructions, 1u);
  EXPECT_EQ(prog->instructions[0].opcode, FHN_ROTATE);
  EXPECT_EQ(prog->instructions[0].params[0], 3);
  EXPECT_EQ(prog->instructions[0].operands[1], 0u);

  fhn_program_free(prog);
}

TEST(LowerToFhnProgram, RightRotationEncodesNegativeDistance) {
  auto a = std::make_shared<Fhenon<int>>(5);
  auto result = std::make_shared<Fhenon<int>>(0);

  auto op = std::make_shared<Operation<int>>(OperationType::RightRotate, a, nullptr, result, nullptr, 2);
  auto leaf = std::make_shared<OperandNode<int>>(a);
  auto root = std::make_shared<OperatorNode<int>>(op, OperationType::RightRotate, leaf, nullptr, result);

  Planner<int> plan;
  plan.addRoot(root);

  LowerToFhnProgram lowering;
  FhnProgram *prog = lowering.lower(plan);

  ASSERT_NE(prog, nullptr);
  ASSERT_EQ(prog->num_instructions, 1u);
  EXPECT_EQ(prog->instructions[0].opcode, FHN_ROTATE);
  EXPECT_EQ(prog->instructions[0].params[0], -2);

  fhn_program_free(prog);
}

TEST(LowerToFhnProgram, ScalarAddLowersToAddCs) {
  // a + 5: the scalar folds into the instruction (fparams[0]) instead of
  // becoming a ciphertext program input.
  auto a = std::make_shared<Fhenon<int>>(10);
  auto s = std::make_shared<Fhenon<int>>(5);
  s->setScalar();
  auto result = std::make_shared<Fhenon<int>>(0);

  auto op = std::make_shared<Operation<int>>(OperationType::Add, a, s, result);
  auto left = std::make_shared<OperandNode<int>>(a);
  auto right = std::make_shared<OperandNode<int>>(s);
  auto root = std::make_shared<OperatorNode<int>>(op, OperationType::Add, left, right, result);

  Planner<int> plan;
  plan.addRoot(root);

  LowerToFhnProgram lowering;
  LowerToFhnProgram::EntityBindings<int> bindings;
  FhnProgram *prog = lowering.lower(plan, &bindings);

  ASSERT_NE(prog, nullptr);
  EXPECT_EQ(prog->num_inputs, 1u); // just 'a' — the scalar is not an input
  ASSERT_EQ(prog->num_instructions, 1u);
  EXPECT_EQ(prog->instructions[0].opcode, FHN_ADD_CS);
  EXPECT_EQ(prog->instructions[0].operands[0], prog->input_ids[0]);
  EXPECT_EQ(prog->instructions[0].operands[1], 0u);
  EXPECT_DOUBLE_EQ(prog->instructions[0].fparams[0], 5.0);

  // The scalar entity must not appear in the bindings: it needs no buffer
  // and must not be written back.
  for (const auto &binding : bindings) {
    EXPECT_NE(binding.second, s);
  }

  fhn_program_free(prog);
}

TEST(LowerToFhnProgram, ScalarMultiplyLowersToMultCs) {
  auto a = std::make_shared<Fhenon<int>>(10);
  auto s = std::make_shared<Fhenon<int>>(3);
  s->setScalar();
  auto result = std::make_shared<Fhenon<int>>(0);

  auto op = std::make_shared<Operation<int>>(OperationType::Multiply, a, s, result);
  auto left = std::make_shared<OperandNode<int>>(a);
  auto right = std::make_shared<OperandNode<int>>(s);
  auto root = std::make_shared<OperatorNode<int>>(op, OperationType::Multiply, left, right, result);

  Planner<int> plan;
  plan.addRoot(root);

  LowerToFhnProgram lowering;
  FhnProgram *prog = lowering.lower(plan);

  ASSERT_NE(prog, nullptr);
  EXPECT_EQ(prog->num_inputs, 1u);
  ASSERT_EQ(prog->num_instructions, 1u);
  EXPECT_EQ(prog->instructions[0].opcode, FHN_MULT_CS);
  EXPECT_DOUBLE_EQ(prog->instructions[0].fparams[0], 3.0);

  fhn_program_free(prog);
}

TEST(LowerToFhnProgram, ChainedScalarOpsShareOneInput) {
  // result = (a + 5) * 3: two CS instructions, one ciphertext input.
  auto a = std::make_shared<Fhenon<int>>(10);
  auto s5 = std::make_shared<Fhenon<int>>(5);
  s5->setScalar();
  auto s3 = std::make_shared<Fhenon<int>>(3);
  s3->setScalar();
  auto t = std::make_shared<Fhenon<int>>(0);
  auto result = std::make_shared<Fhenon<int>>(0);

  auto add_op = std::make_shared<Operation<int>>(OperationType::Add, a, s5, t);
  auto mul_op = std::make_shared<Operation<int>>(OperationType::Multiply, t, s3, result);

  auto leaf_a = std::make_shared<OperandNode<int>>(a);
  auto leaf_s5 = std::make_shared<OperandNode<int>>(s5);
  auto leaf_s3 = std::make_shared<OperandNode<int>>(s3);

  auto add_node = std::make_shared<OperatorNode<int>>(add_op, OperationType::Add, leaf_a, leaf_s5, t);
  auto mul_node = std::make_shared<OperatorNode<int>>(mul_op, OperationType::Multiply, add_node, leaf_s3, result);

  Planner<int> plan;
  plan.addRoot(mul_node);

  LowerToFhnProgram lowering;
  FhnProgram *prog = lowering.lower(plan);

  ASSERT_NE(prog, nullptr);
  EXPECT_EQ(prog->num_inputs, 1u); // just 'a'
  ASSERT_EQ(prog->num_instructions, 2u);
  EXPECT_EQ(prog->instructions[0].opcode, FHN_ADD_CS);
  EXPECT_DOUBLE_EQ(prog->instructions[0].fparams[0], 5.0);
  EXPECT_EQ(prog->instructions[1].opcode, FHN_MULT_CS);
  EXPECT_DOUBLE_EQ(prog->instructions[1].fparams[0], 3.0);
  EXPECT_EQ(prog->instructions[1].operands[0], prog->instructions[0].result_id);

  fhn_program_free(prog);
}

TEST(LowerToFhnProgram, SharedOperand) {
  // AST: r1 = a + a (same operand node used twice)
  auto a = std::make_shared<Fhenon<int>>(5);
  auto result = std::make_shared<Fhenon<int>>(0);

  auto op = std::make_shared<Operation<int>>(OperationType::Add, a, a, result);
  auto left = std::make_shared<OperandNode<int>>(a);
  // Use SAME node for both left and right
  auto root = std::make_shared<OperatorNode<int>>(op, OperationType::Add, left, left, result);

  Planner<int> plan;
  plan.addRoot(root);

  LowerToFhnProgram lowering;
  FhnProgram *prog = lowering.lower(plan);

  ASSERT_NE(prog, nullptr);
  EXPECT_EQ(prog->num_inputs, 1u); // just 'a' once
  EXPECT_EQ(prog->num_instructions, 1u);
  // Both operands reference the same input id
  EXPECT_EQ(prog->instructions[0].operands[0], prog->instructions[0].operands[1]);

  fhn_program_free(prog);
}
