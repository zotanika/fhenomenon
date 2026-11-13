#include "FHN/fhn_program.h"

#include <gtest/gtest.h>

TEST(FhnProgram, AllocFree) {
  FhnProgram *prog = fhn_program_alloc(3, 2, 1);
  ASSERT_NE(prog, nullptr);

  EXPECT_EQ(prog->version, 1u);
  EXPECT_EQ(prog->num_instructions, 3u);
  EXPECT_EQ(prog->num_inputs, 2u);
  EXPECT_EQ(prog->num_outputs, 1u);
  EXPECT_NE(prog->instructions, nullptr);
  EXPECT_NE(prog->input_ids, nullptr);
  EXPECT_NE(prog->output_ids, nullptr);

  fhn_program_free(prog);
}

TEST(FhnProgram, InstructionFields) {
  FhnProgram *prog = fhn_program_alloc(1, 0, 0);
  ASSERT_NE(prog, nullptr);

  FhnInstruction &inst = prog->instructions[0];

  /* Verify zero-initialized by calloc */
  EXPECT_EQ(inst.opcode, FHN_NOP);
  EXPECT_EQ(inst.result_id, 0u);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(inst.operands[i], 0u);
    EXPECT_EQ(inst.params[i], 0);
  }
  EXPECT_DOUBLE_EQ(inst.fparams[0], 0.0);
  EXPECT_DOUBLE_EQ(inst.fparams[1], 0.0);

  /* Set all fields */
  inst.opcode = FHN_ADD_CC;
  inst.result_id = 10;
  inst.operands[0] = 1;
  inst.operands[1] = 2;
  inst.operands[2] = 3;
  inst.operands[3] = 4;
  inst.params[0] = -100;
  inst.params[1] = 200;
  inst.params[2] = -300;
  inst.params[3] = 400;
  inst.fparams[0] = 1.5;
  inst.fparams[1] = -2.5;

  /* Verify set values */
  EXPECT_EQ(inst.opcode, FHN_ADD_CC);
  EXPECT_EQ(inst.result_id, 10u);
  EXPECT_EQ(inst.operands[0], 1u);
  EXPECT_EQ(inst.operands[1], 2u);
  EXPECT_EQ(inst.operands[2], 3u);
  EXPECT_EQ(inst.operands[3], 4u);
  EXPECT_EQ(inst.params[0], -100);
  EXPECT_EQ(inst.params[1], 200);
  EXPECT_EQ(inst.params[2], -300);
  EXPECT_EQ(inst.params[3], 400);
  EXPECT_DOUBLE_EQ(inst.fparams[0], 1.5);
  EXPECT_DOUBLE_EQ(inst.fparams[1], -2.5);

  fhn_program_free(prog);
}

TEST(FhnProgram, BuildSimpleAddProgram) {
  /* Build a program: result = input0 + input1 */
  FhnProgram *prog = fhn_program_alloc(1, 2, 1);
  ASSERT_NE(prog, nullptr);

  /* Define inputs */
  prog->input_ids[0] = 100;
  prog->input_ids[1] = 101;

  /* Define output */
  prog->output_ids[0] = 200;

  /* Define instruction: ADD_CC result=200, operands={100, 101} */
  FhnInstruction &add = prog->instructions[0];
  add.opcode = FHN_ADD_CC;
  add.result_id = 200;
  add.operands[0] = 100;
  add.operands[1] = 101;

  /* Verify the program structure */
  EXPECT_EQ(prog->version, 1u);
  EXPECT_EQ(prog->num_instructions, 1u);
  EXPECT_EQ(prog->num_inputs, 2u);
  EXPECT_EQ(prog->num_outputs, 1u);

  EXPECT_EQ(prog->input_ids[0], 100u);
  EXPECT_EQ(prog->input_ids[1], 101u);
  EXPECT_EQ(prog->output_ids[0], 200u);

  EXPECT_EQ(prog->instructions[0].opcode, FHN_ADD_CC);
  EXPECT_EQ(prog->instructions[0].result_id, 200u);
  EXPECT_EQ(prog->instructions[0].operands[0], 100u);
  EXPECT_EQ(prog->instructions[0].operands[1], 101u);

  fhn_program_free(prog);
}

TEST(FhnProgram, OpcodeCount) {
  /* FHN_NOP must be 0 */
  EXPECT_EQ(FHN_NOP, 0);

  /* Sentinel must be greater than 0, meaning we have actual opcodes */
  EXPECT_GT(FHN_OPCODE_COUNT, 0);
}
