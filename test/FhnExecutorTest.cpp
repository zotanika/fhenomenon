#include "FHN/FhnDefaultExecutor.h"
#include "FHN/fhn_program.h"

#include <gtest/gtest.h>

namespace {

// Fake buffer type for testing
struct TestBuffer {
  int64_t value;
};

// Kernel functions that work on TestBuffer
int test_add_cc(FhnBackendCtx *, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *, const double *) {
  auto *r = reinterpret_cast<TestBuffer *>(result);
  auto *a = reinterpret_cast<const TestBuffer *>(operands[0]);
  auto *b = reinterpret_cast<const TestBuffer *>(operands[1]);
  r->value = a->value + b->value;
  return 0;
}

int test_mult_cc(FhnBackendCtx *, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                 const double *) {
  auto *r = reinterpret_cast<TestBuffer *>(result);
  auto *a = reinterpret_cast<const TestBuffer *>(operands[0]);
  auto *b = reinterpret_cast<const TestBuffer *>(operands[1]);
  r->value = a->value * b->value;
  return 0;
}

int test_negate(FhnBackendCtx *, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *, const double *) {
  auto *r = reinterpret_cast<TestBuffer *>(result);
  auto *a = reinterpret_cast<const TestBuffer *>(operands[0]);
  r->value = -a->value;
  return 0;
}

int test_noop(FhnBackendCtx *, FhnBuffer *, const FhnBuffer *const *, const int64_t *, const double *) { return 0; }

} // namespace

TEST(FhnExecutor, SupportsRegisteredOpcodes) {
  FhnKernelEntry entries[3] = {
    {FHN_ADD_CC, test_add_cc, "add_cc"},
    {FHN_MULT_CC, test_mult_cc, "mult_cc"},
    {FHN_NEGATE, test_negate, "negate"},
  };
  FhnKernelTable table = {3, entries};

  fhenomenon::FhnDefaultExecutor executor(&table);

  EXPECT_TRUE(executor.supports(FHN_ADD_CC));
  EXPECT_TRUE(executor.supports(FHN_MULT_CC));
  EXPECT_TRUE(executor.supports(FHN_NEGATE));
  EXPECT_FALSE(executor.supports(FHN_ROTATE));
  EXPECT_FALSE(executor.supports(FHN_HMULT));
}

TEST(FhnExecutor, ExecuteSimpleAdd) {
  FhnKernelEntry entries[1] = {
    {FHN_ADD_CC, test_add_cc, "add_cc"},
  };
  FhnKernelTable table = {1, entries};

  fhenomenon::FhnDefaultExecutor executor(&table);

  // Program: r3 = r1 + r2
  FhnProgram *prog = fhn_program_alloc(1, 2, 1);
  ASSERT_NE(prog, nullptr);

  prog->input_ids[0] = 1;
  prog->input_ids[1] = 2;
  prog->output_ids[0] = 3;

  FhnInstruction &inst = prog->instructions[0];
  inst.opcode = FHN_ADD_CC;
  inst.result_id = 3;
  inst.operands[0] = 1;
  inst.operands[1] = 2;

  // Set up buffers: indices 0..3
  TestBuffer bufs[4] = {{0}, {10}, {20}, {0}};
  FhnBuffer *ptrs[4] = {
    reinterpret_cast<FhnBuffer *>(&bufs[0]),
    reinterpret_cast<FhnBuffer *>(&bufs[1]),
    reinterpret_cast<FhnBuffer *>(&bufs[2]),
    reinterpret_cast<FhnBuffer *>(&bufs[3]),
  };

  int rc = executor.execute(nullptr, prog, ptrs);
  EXPECT_EQ(rc, 0);
  EXPECT_EQ(bufs[3].value, 30);

  fhn_program_free(prog);
}

TEST(FhnExecutor, ExecuteChain) {
  FhnKernelEntry entries[2] = {
    {FHN_ADD_CC, test_add_cc, "add_cc"},
    {FHN_MULT_CC, test_mult_cc, "mult_cc"},
  };
  FhnKernelTable table = {2, entries};

  fhenomenon::FhnDefaultExecutor executor(&table);

  // Program: r3 = r1 + r2; r4 = r3 * r1
  FhnProgram *prog = fhn_program_alloc(2, 2, 2);
  ASSERT_NE(prog, nullptr);

  prog->input_ids[0] = 1;
  prog->input_ids[1] = 2;
  prog->output_ids[0] = 3;
  prog->output_ids[1] = 4;

  FhnInstruction &add = prog->instructions[0];
  add.opcode = FHN_ADD_CC;
  add.result_id = 3;
  add.operands[0] = 1;
  add.operands[1] = 2;

  FhnInstruction &mult = prog->instructions[1];
  mult.opcode = FHN_MULT_CC;
  mult.result_id = 4;
  mult.operands[0] = 3;
  mult.operands[1] = 1;

  // Buffers: indices 0..4
  TestBuffer bufs[5] = {{0}, {5}, {3}, {0}, {0}};
  FhnBuffer *ptrs[5] = {
    reinterpret_cast<FhnBuffer *>(&bufs[0]), reinterpret_cast<FhnBuffer *>(&bufs[1]),
    reinterpret_cast<FhnBuffer *>(&bufs[2]), reinterpret_cast<FhnBuffer *>(&bufs[3]),
    reinterpret_cast<FhnBuffer *>(&bufs[4]),
  };

  int rc = executor.execute(nullptr, prog, ptrs);
  EXPECT_EQ(rc, 0);
  EXPECT_EQ(bufs[3].value, 8);  // 5 + 3
  EXPECT_EQ(bufs[4].value, 40); // 8 * 5

  fhn_program_free(prog);
}

TEST(FhnExecutor, UnsupportedOpcodeReturnsError) {
  FhnKernelEntry entries[1] = {
    {FHN_ADD_CC, test_add_cc, "add_cc"},
  };
  FhnKernelTable table = {1, entries};

  fhenomenon::FhnDefaultExecutor executor(&table);

  // Program with FHN_ROTATE (not registered)
  FhnProgram *prog = fhn_program_alloc(1, 1, 1);
  ASSERT_NE(prog, nullptr);

  prog->input_ids[0] = 1;
  prog->output_ids[0] = 2;

  FhnInstruction &inst = prog->instructions[0];
  inst.opcode = FHN_ROTATE;
  inst.result_id = 2;
  inst.operands[0] = 1;

  TestBuffer bufs[3] = {{0}, {10}, {0}};
  FhnBuffer *ptrs[3] = {
    reinterpret_cast<FhnBuffer *>(&bufs[0]),
    reinterpret_cast<FhnBuffer *>(&bufs[1]),
    reinterpret_cast<FhnBuffer *>(&bufs[2]),
  };

  int rc = executor.execute(nullptr, prog, ptrs);
  EXPECT_NE(rc, 0);

  fhn_program_free(prog);
}

TEST(FhnExecutor, DecomposeHMult) {
  // Register MULT_CC, RELINEARIZE, RESCALE but NOT HMULT.
  FhnKernelEntry entries[3] = {
    {FHN_MULT_CC, test_mult_cc, "mult_cc"},
    {FHN_RELINEARIZE, test_noop, "relin"},
    {FHN_RESCALE, test_noop, "rescale"},
  };
  FhnKernelTable table = {3, entries};

  fhenomenon::FhnDefaultExecutor executor(&table);

  // Program: r3 = HMULT(r1, r2)
  FhnProgram *prog = fhn_program_alloc(1, 2, 1);
  ASSERT_NE(prog, nullptr);

  prog->input_ids[0] = 1;
  prog->input_ids[1] = 2;
  prog->output_ids[0] = 3;

  FhnInstruction &inst = prog->instructions[0];
  inst.opcode = FHN_HMULT;
  inst.result_id = 3;
  inst.operands[0] = 1;
  inst.operands[1] = 2;

  TestBuffer bufs[4] = {{0}, {7}, {6}, {0}};
  FhnBuffer *ptrs[4] = {
    reinterpret_cast<FhnBuffer *>(&bufs[0]),
    reinterpret_cast<FhnBuffer *>(&bufs[1]),
    reinterpret_cast<FhnBuffer *>(&bufs[2]),
    reinterpret_cast<FhnBuffer *>(&bufs[3]),
  };

  int rc = executor.execute(nullptr, prog, ptrs);
  EXPECT_EQ(rc, 0);
  EXPECT_EQ(bufs[3].value, 42); // 7 * 6

  fhn_program_free(prog);
}

TEST(FhnExecutor, DecomposeHMultMinimal) {
  // Register only MULT_CC (no RELIN, no RESCALE).
  FhnKernelEntry entries[1] = {
    {FHN_MULT_CC, test_mult_cc, "mult_cc"},
  };
  FhnKernelTable table = {1, entries};

  fhenomenon::FhnDefaultExecutor executor(&table);

  // Program: r3 = HMULT(r1, r2)
  FhnProgram *prog = fhn_program_alloc(1, 2, 1);
  ASSERT_NE(prog, nullptr);

  prog->input_ids[0] = 1;
  prog->input_ids[1] = 2;
  prog->output_ids[0] = 3;

  FhnInstruction &inst = prog->instructions[0];
  inst.opcode = FHN_HMULT;
  inst.result_id = 3;
  inst.operands[0] = 1;
  inst.operands[1] = 2;

  TestBuffer bufs[4] = {{0}, {5}, {9}, {0}};
  FhnBuffer *ptrs[4] = {
    reinterpret_cast<FhnBuffer *>(&bufs[0]),
    reinterpret_cast<FhnBuffer *>(&bufs[1]),
    reinterpret_cast<FhnBuffer *>(&bufs[2]),
    reinterpret_cast<FhnBuffer *>(&bufs[3]),
  };

  int rc = executor.execute(nullptr, prog, ptrs);
  EXPECT_EQ(rc, 0);
  EXPECT_EQ(bufs[3].value, 45); // 5 * 9

  fhn_program_free(prog);
}

TEST(FhnExecutor, DecomposeFailsWhenPrimitiveMissing) {
  // Register only RELINEARIZE (no MULT_CC).
  FhnKernelEntry entries[1] = {
    {FHN_RELINEARIZE, test_noop, "relin"},
  };
  FhnKernelTable table = {1, entries};

  fhenomenon::FhnDefaultExecutor executor(&table);

  // Program: r3 = HMULT(r1, r2) — should fail because MULT_CC is missing.
  FhnProgram *prog = fhn_program_alloc(1, 2, 1);
  ASSERT_NE(prog, nullptr);

  prog->input_ids[0] = 1;
  prog->input_ids[1] = 2;
  prog->output_ids[0] = 3;

  FhnInstruction &inst = prog->instructions[0];
  inst.opcode = FHN_HMULT;
  inst.result_id = 3;
  inst.operands[0] = 1;
  inst.operands[1] = 2;

  TestBuffer bufs[4] = {{0}, {7}, {6}, {0}};
  FhnBuffer *ptrs[4] = {
    reinterpret_cast<FhnBuffer *>(&bufs[0]),
    reinterpret_cast<FhnBuffer *>(&bufs[1]),
    reinterpret_cast<FhnBuffer *>(&bufs[2]),
    reinterpret_cast<FhnBuffer *>(&bufs[3]),
  };

  int rc = executor.execute(nullptr, prog, ptrs);
  EXPECT_NE(rc, 0);

  fhn_program_free(prog);
}
