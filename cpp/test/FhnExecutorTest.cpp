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

int test_fail(FhnBackendCtx *, FhnBuffer *, const FhnBuffer *const *, const int64_t *, const double *) { return -7; }

// "Rotation" over scalar test buffers: multiply by 10 so single and double
// application are distinguishable.
int test_rotate(FhnBackendCtx *, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *, const double *) {
  auto *r = reinterpret_cast<TestBuffer *>(result);
  auto *a = reinterpret_cast<const TestBuffer *>(operands[0]);
  r->value = a->value * 10;
  return 0;
}

int test_mult_key(FhnBackendCtx *, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                  const double *) {
  auto *r = reinterpret_cast<TestBuffer *>(result);
  auto *a = reinterpret_cast<const TestBuffer *>(operands[0]);
  r->value = a->value * 1000; // must NOT be applied by the HROT fallback
  return 0;
}

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

TEST(FhnExecutor, RejectsMismatchedProgramVersion) {
  FhnKernelEntry entries[1] = {
    {FHN_ADD_CC, test_add_cc, "add_cc"},
  };
  FhnKernelTable table = {1, entries};

  fhenomenon::FhnDefaultExecutor executor(&table);

  FhnProgram *prog = fhn_program_alloc(1, 2, 1);
  ASSERT_NE(prog, nullptr);
  EXPECT_EQ(prog->version, FHN_ABI_VERSION);

  prog->input_ids[0] = 1;
  prog->input_ids[1] = 2;
  prog->output_ids[0] = 3;
  prog->instructions[0].opcode = FHN_ADD_CC;
  prog->instructions[0].result_id = 3;
  prog->instructions[0].operands[0] = 1;
  prog->instructions[0].operands[1] = 2;

  TestBuffer bufs[4] = {{0}, {10}, {20}, {0}};
  FhnBuffer *ptrs[4] = {
    reinterpret_cast<FhnBuffer *>(&bufs[0]),
    reinterpret_cast<FhnBuffer *>(&bufs[1]),
    reinterpret_cast<FhnBuffer *>(&bufs[2]),
    reinterpret_cast<FhnBuffer *>(&bufs[3]),
  };

  // A program stamped with a different ABI revision must be refused, even
  // when every opcode it contains happens to be dispatchable.
  prog->version = FHN_ABI_VERSION + 1;
  EXPECT_NE(executor.execute(nullptr, prog, ptrs), 0);

  prog->version = FHN_ABI_VERSION;
  EXPECT_EQ(executor.execute(nullptr, prog, ptrs), 0);
  EXPECT_EQ(bufs[3].value, 30);

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

TEST(FhnExecutor, DecomposeHMultRejectsIncompletePrimitiveSet) {
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
  EXPECT_NE(rc, 0);
  EXPECT_EQ(bufs[3].value, 0); // missing stages must reject before MULT_CC writes

  fhn_program_free(prog);
}

TEST(FhnExecutor, DecomposeHRotIsASingleRotate) {
  // ROTATE is a complete key-switched rotation, so HROT must decompose to
  // exactly one ROTATE — never MULT_KEY + ROTATE (double key application).
  FhnKernelEntry entries[2] = {
    {FHN_ROTATE, test_rotate, "rotate"},
    {FHN_MULT_KEY, test_mult_key, "mult_key"},
  };
  FhnKernelTable table = {2, entries};

  fhenomenon::FhnDefaultExecutor executor(&table);

  FhnProgram *prog = fhn_program_alloc(1, 1, 1);
  ASSERT_NE(prog, nullptr);
  prog->input_ids[0] = 1;
  prog->output_ids[0] = 2;
  prog->instructions[0].opcode = FHN_HROT;
  prog->instructions[0].result_id = 2;
  prog->instructions[0].operands[0] = 1;

  TestBuffer bufs[3] = {{0}, {7}, {0}};
  FhnBuffer *ptrs[3] = {
    reinterpret_cast<FhnBuffer *>(&bufs[0]),
    reinterpret_cast<FhnBuffer *>(&bufs[1]),
    reinterpret_cast<FhnBuffer *>(&bufs[2]),
  };

  EXPECT_EQ(executor.execute(nullptr, prog, ptrs), 0);
  EXPECT_EQ(bufs[2].value, 70); // one rotate; 70000 would mean mult_key ran too

  fhn_program_free(prog);
}

TEST(FhnExecutor, DecomposeHMultPropagatesRelinFailure) {
  FhnKernelEntry entries[3] = {
    {FHN_MULT_CC, test_mult_cc, "mult_cc"},
    {FHN_RELINEARIZE, test_fail, "relin"},
    {FHN_RESCALE, test_noop, "rescale"},
  };
  FhnKernelTable table = {3, entries};

  fhenomenon::FhnDefaultExecutor executor(&table);

  FhnProgram *prog = fhn_program_alloc(1, 2, 1);
  ASSERT_NE(prog, nullptr);
  prog->input_ids[0] = 1;
  prog->input_ids[1] = 2;
  prog->output_ids[0] = 3;
  prog->instructions[0].opcode = FHN_HMULT;
  prog->instructions[0].result_id = 3;
  prog->instructions[0].operands[0] = 1;
  prog->instructions[0].operands[1] = 2;

  TestBuffer bufs[4] = {{0}, {7}, {6}, {0}};
  FhnBuffer *ptrs[4] = {
    reinterpret_cast<FhnBuffer *>(&bufs[0]),
    reinterpret_cast<FhnBuffer *>(&bufs[1]),
    reinterpret_cast<FhnBuffer *>(&bufs[2]),
    reinterpret_cast<FhnBuffer *>(&bufs[3]),
  };

  // A registered-but-failing RELINEARIZE must fail the whole instruction.
  EXPECT_NE(executor.execute(nullptr, prog, ptrs), 0);

  fhn_program_free(prog);
}

TEST(FhnExecutor, MadRequiresDistinctAddend) {
  FhnKernelEntry entries[2] = {
    {FHN_MULT_CS, test_mult_cc, "mult_cs"}, // signature-compatible stand-in
    {FHN_ADD_CC, test_add_cc, "add_cc"},
  };
  FhnKernelTable table = {2, entries};

  fhenomenon::FhnDefaultExecutor executor(&table);

  FhnProgram *prog = fhn_program_alloc(1, 2, 1);
  ASSERT_NE(prog, nullptr);
  prog->input_ids[0] = 1;
  prog->input_ids[1] = 2;
  prog->output_ids[0] = 3;
  prog->instructions[0].opcode = FHN_MAD;
  prog->instructions[0].result_id = 3;
  prog->instructions[0].operands[0] = 1;
  prog->instructions[0].operands[1] = 0; // missing addend: must be rejected

  TestBuffer bufs[4] = {{0}, {5}, {4}, {0}};
  FhnBuffer *ptrs[4] = {
    reinterpret_cast<FhnBuffer *>(&bufs[0]),
    reinterpret_cast<FhnBuffer *>(&bufs[1]),
    reinterpret_cast<FhnBuffer *>(&bufs[2]),
    reinterpret_cast<FhnBuffer *>(&bufs[3]),
  };

  EXPECT_NE(executor.execute(nullptr, prog, ptrs), 0);

  // Aliasing the addend with the result is equally invalid.
  prog->instructions[0].operands[1] = 3;
  EXPECT_NE(executor.execute(nullptr, prog, ptrs), 0);

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

#include "FHN/FhnMovementPlan.h"
#include "FhnTestProgramBuilder.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

using namespace fhenomenon;
using fhenomenon::testutil::ProgramBuilder;

namespace {

// A fake single-int "ciphertext" space. Buffers live in device_vals while
// resident; evict moves the value to host_vals (clearing the device slot),
// prefetch moves it back. The kernel reads/writes device_vals only, so a
// missed prefetch computes garbage and fails the value assertions.
struct MovementWorld {
  std::map<FhnBuffer *, long> device_vals;
  std::map<FhnBuffer *, long> host_vals;
  std::vector<std::string> log; // "alloc#N"/"free#N"/"evict#N"/"prefetch#N"/"kernel ADD r"
  std::map<FhnBuffer *, int> names;
  int next_name = 1;
  int allocs = 0, frees = 0;
  int fail_kernel_at_call = -1; // when >=0, the Nth kernel call returns -1
  int kernel_calls = 0;
  size_t budget = 0, peak = 0;
};
MovementWorld *g_world = nullptr;

FhnBuffer *movementAlloc(FhnBackendCtx *) {
  if (g_world->budget && g_world->device_vals.size() >= g_world->budget)
    return nullptr;
  auto *b = reinterpret_cast<FhnBuffer *>(new char[1]);
  g_world->names[b] = g_world->next_name++;
  g_world->device_vals[b] = 0;
  g_world->peak = std::max(g_world->peak, g_world->device_vals.size());
  g_world->allocs++;
  g_world->log.push_back("alloc#" + std::to_string(g_world->names[b]));
  return b;
}
void movementFree(FhnBackendCtx *, FhnBuffer *b) {
  g_world->frees++;
  g_world->log.push_back("free#" + std::to_string(g_world->names[b]));
  g_world->device_vals.erase(b);
  g_world->host_vals.erase(b);
  delete[] reinterpret_cast<char *>(b);
}
int movementEvict(FhnBackendCtx *, FhnBuffer *b) {
  g_world->log.push_back("evict#" + std::to_string(g_world->names[b]));
  if (!g_world->device_vals.count(b))
    return 0; // normalization may include an already host-resident input
  g_world->host_vals[b] = g_world->device_vals.at(b);
  g_world->device_vals.erase(b); // physically gone from the device
  return 0;
}
int movementPrefetch(FhnBackendCtx *, FhnBuffer *b) {
  g_world->log.push_back("prefetch#" + std::to_string(g_world->names[b]));
  if (g_world->device_vals.count(b))
    return 0; // already resident
  if (g_world->budget && g_world->device_vals.size() >= g_world->budget)
    return -1;
  g_world->device_vals[b] = g_world->host_vals.at(b);
  g_world->peak = std::max(g_world->peak, g_world->device_vals.size());
  g_world->host_vals.erase(b);
  return 0;
}
int movementAddKernel(FhnBackendCtx *, FhnBuffer *result, const FhnBuffer *const *ops, const int64_t *,
                      const double *) {
  g_world->kernel_calls++;
  if (g_world->fail_kernel_at_call >= 0 && g_world->kernel_calls == g_world->fail_kernel_at_call)
    return -1;
  // .at() throws if an operand is not device-resident: movement bugs fail loudly.
  const long a = g_world->device_vals.at(const_cast<FhnBuffer *>(ops[0]));
  const long b = g_world->device_vals.at(const_cast<FhnBuffer *>(ops[1]));
  g_world->device_vals.at(result) = a + b;
  g_world->log.push_back("kernel ADD");
  return 0;
}

FhnKernelEntry movementEntries[] = {{FHN_ADD_CC, movementAddKernel, "add_cc"}};
FhnKernelTable movementTable{1, movementEntries};

} // namespace

// Full interleaving on the Belady program from FhnMovementPlanTest:
// values must come out right even though a1 physically leaves the device.
TEST(FhnExecutorMovement, BudgetedExecutionComputesCorrectValues) {
  MovementWorld world;
  g_world = &world;

  // Same program as FhnMovementPlan.BeladyEvictsFarthestNextUseNotLru.
  auto prog = ProgramBuilder()
                .input(1)
                .input(2)
                .input(8)
                .inst(FHN_ADD_CC, 3, 2, 2)
                .inst(FHN_ADD_CC, 4, 3, 1)
                .inst(FHN_ADD_CC, 5, 4, 8)
                .inst(FHN_ADD_CC, 6, 5, 2)
                .inst(FHN_ADD_CC, 7, 6, 1)
                .output(7)
                .build();
  auto plan = FhnMovementPlan::analyze(*prog, {7}, 4);
  ASSERT_TRUE(plan.has_value());

  // Caller-provided inputs: a1=10, b2=1, c8=100. The caller deliberately
  // does not pin them, so the plan frees them after their last use.
  FhnBuffer *a1 = movementAlloc(nullptr);
  FhnBuffer *b2 = movementAlloc(nullptr);
  FhnBuffer *c8 = movementAlloc(nullptr);
  world.device_vals[a1] = 10;
  world.device_vals[b2] = 1;
  world.device_vals[c8] = 100;

  std::vector<FhnBuffer *> buffers(9, nullptr);
  buffers[1] = a1;
  buffers[2] = b2;
  buffers[8] = c8;

  FhnDefaultExecutor executor(&movementTable);
  FhnMovementHooks hooks{nullptr, movementAlloc, movementFree, movementPrefetch, movementEvict};
  ASSERT_EQ(executor.execute(hooks, prog.get(), buffers.data(), *plan), 0);

  // 3=1+1=2; 4=2+10=12; 5=12+100=112; 6=112+1=113; 7=113+10=123.
  EXPECT_EQ(world.device_vals.at(buffers[7]), 123);
  // a1 (buffer name #1) was evicted then prefetched back.
  EXPECT_NE(std::find(world.log.begin(), world.log.end(), "evict#1"), world.log.end());
  // Everything except the pinned output was freed — including the unpinned
  // caller inputs, whose lifetime the plan owns by contract.
  EXPECT_EQ(world.allocs - world.frees, 1); // only pinned id 7 survives
  EXPECT_NE(buffers[7], nullptr);
  movementFree(nullptr, buffers[7]); // adopt-and-release: the pinned output survives for the caller
}

// Null hooks skip movement actions but execution still works when
// everything shares one memory space (host==device in the mock: seed
// device_vals directly and use null prefetch/evict).
TEST(FhnExecutorMovement, NullHooksSkipMovement) {
  MovementWorld world;
  g_world = &world;

  auto prog = ProgramBuilder().input(1).input(2).inst(FHN_ADD_CC, 3, 1, 2).output(3).build();
  auto plan = FhnMovementPlan::analyze(*prog, {3}, 0);
  ASSERT_TRUE(plan.has_value());

  FhnBuffer *a = movementAlloc(nullptr);
  FhnBuffer *b = movementAlloc(nullptr);
  world.device_vals[a] = 4;
  world.device_vals[b] = 5;
  std::vector<FhnBuffer *> buffers(4, nullptr);
  buffers[1] = a;
  buffers[2] = b;

  FhnDefaultExecutor executor(&movementTable);
  FhnMovementHooks hooks{nullptr, movementAlloc, movementFree, nullptr, nullptr};
  ASSERT_EQ(executor.execute(hooks, prog.get(), buffers.data(), *plan), 0);
  EXPECT_EQ(world.device_vals.at(buffers[3]), 9);
  for (const auto &line : world.log) {
    EXPECT_EQ(line.find("prefetch"), std::string::npos);
    EXPECT_EQ(line.find("evict"), std::string::npos);
  }
  movementFree(nullptr, buffers[3]); // pinned output; inputs a,b are plan-freed
}

// A mid-program kernel failure must free every plan-allocated buffer —
// pinned included — and null their entries. Caller inputs stay untouched.
TEST(FhnExecutorMovement, FailureFreesAllPlanAllocations) {
  MovementWorld world;
  g_world = &world;
  world.fail_kernel_at_call = 2;

  auto prog = ProgramBuilder().input(1).input(2).inst(FHN_ADD_CC, 3, 1, 2).inst(FHN_ADD_CC, 4, 3, 3).output(4).build();
  // Inputs are pinned here: this test's caller keeps ownership of a and b,
  // and asserts below that the failure path leaves them untouched.
  auto plan = FhnMovementPlan::analyze(*prog, {1, 2, 4}, 0);
  ASSERT_TRUE(plan.has_value());

  FhnBuffer *a = movementAlloc(nullptr);
  FhnBuffer *b = movementAlloc(nullptr);
  world.device_vals[a] = 1;
  world.device_vals[b] = 2;
  std::vector<FhnBuffer *> buffers(5, nullptr);
  buffers[1] = a;
  buffers[2] = b;

  FhnDefaultExecutor executor(&movementTable);
  FhnMovementHooks hooks{nullptr, movementAlloc, movementFree, nullptr, nullptr};
  EXPECT_NE(executor.execute(hooks, prog.get(), buffers.data(), *plan), 0);

  // Every plan alloc (ids 3,4) was freed on the failure path.
  EXPECT_EQ(world.allocs - world.frees, 2); // only caller's a,b remain
  EXPECT_EQ(buffers[3], nullptr);
  EXPECT_EQ(buffers[4], nullptr);
  // Caller-owned inputs untouched.
  EXPECT_EQ(world.device_vals.at(a), 1);
  movementFree(nullptr, a); // caller-owned inputs, pinned and untouched by failure
  movementFree(nullptr, b);
}

// A plan analyzed for a different program must be rejected, not let its
// action ids index the buffer table out of bounds.
TEST(FhnExecutorMovement, MismatchedPlanIsRejected) {
  MovementWorld world;
  g_world = &world;

  auto prog1 = ProgramBuilder().input(1).input(2).inst(FHN_ADD_CC, 3, 1, 2).output(3).build();
  auto prog2 = ProgramBuilder().input(1).input(2).inst(FHN_ADD_CC, 3, 1, 2).inst(FHN_ADD_CC, 4, 3, 3).output(4).build();
  auto plan1 = FhnMovementPlan::analyze(*prog1, {3}, 0);
  ASSERT_TRUE(plan1.has_value());

  FhnBuffer *a = movementAlloc(nullptr);
  FhnBuffer *b = movementAlloc(nullptr);
  world.device_vals[a] = 1;
  world.device_vals[b] = 2;
  std::vector<FhnBuffer *> buffers(5, nullptr);
  buffers[1] = a;
  buffers[2] = b;

  FhnDefaultExecutor executor(&movementTable);
  FhnMovementHooks hooks{nullptr, movementAlloc, movementFree, nullptr, nullptr};
  EXPECT_NE(executor.execute(hooks, prog2.get(), buffers.data(), *plan1), 0);

  movementFree(nullptr, a);
  movementFree(nullptr, b);
}

// An LRU-planned program must execute to the same values as Belady — the
// policies move buffers differently but never change results.
TEST(FhnExecutorMovement, BudgetedExecutionCorrectUnderLru) {
  MovementWorld world;
  g_world = &world;

  auto prog = ProgramBuilder()
                .input(1)
                .input(2)
                .input(8)
                .inst(FHN_ADD_CC, 3, 2, 2)
                .inst(FHN_ADD_CC, 4, 3, 1)
                .inst(FHN_ADD_CC, 5, 4, 8)
                .inst(FHN_ADD_CC, 6, 5, 2)
                .inst(FHN_ADD_CC, 7, 6, 1)
                .output(7)
                .build();
  auto plan = FhnMovementPlan::analyze(*prog, {7}, 4, FhnEvictionPolicy::Lru);
  ASSERT_TRUE(plan.has_value());

  FhnBuffer *a1 = movementAlloc(nullptr);
  FhnBuffer *b2 = movementAlloc(nullptr);
  FhnBuffer *c8 = movementAlloc(nullptr);
  world.device_vals[a1] = 10;
  world.device_vals[b2] = 1;
  world.device_vals[c8] = 100;
  std::vector<FhnBuffer *> buffers(9, nullptr);
  buffers[1] = a1;
  buffers[2] = b2;
  buffers[8] = c8;

  FhnDefaultExecutor executor(&movementTable);
  FhnMovementHooks hooks{nullptr, movementAlloc, movementFree, movementPrefetch, movementEvict};
  ASSERT_EQ(executor.execute(hooks, prog.get(), buffers.data(), *plan), 0);

  // 3=2; 4=12; 5=112; 6=113; 7=123 — same as the Belady run.
  EXPECT_EQ(world.device_vals.at(buffers[7]), 123);
  // LRU's distinguishing move: b2 (buffer name #2) was evicted.
  EXPECT_NE(std::find(world.log.begin(), world.log.end(), "evict#2"), world.log.end());
  movementFree(nullptr, buffers[7]);
}

namespace {
int test_mult_cs(FhnBackendCtx *, FhnBuffer *result, const FhnBuffer *const *ops, const int64_t *, const double *fp) {
  reinterpret_cast<TestBuffer *>(result)->value =
    reinterpret_cast<const TestBuffer *>(ops[0])->value * static_cast<int64_t>(fp[0]);
  return 0;
}
int test_native_hrot_add(FhnBackendCtx *, FhnBuffer *result, const FhnBuffer *const *ops, const int64_t *,
                         const double *) {
  reinterpret_cast<TestBuffer *>(result)->value =
    reinterpret_cast<const TestBuffer *>(ops[0])->value * 10 + reinterpret_cast<const TestBuffer *>(ops[1])->value;
  return 0;
}
int test_native_hconj_add(FhnBackendCtx *, FhnBuffer *result, const FhnBuffer *const *ops, const int64_t *,
                          const double *) {
  reinterpret_cast<TestBuffer *>(result)->value =
    -reinterpret_cast<const TestBuffer *>(ops[0])->value + reinterpret_cast<const TestBuffer *>(ops[1])->value;
  return 0;
}
int test_native_mad(FhnBackendCtx *, FhnBuffer *result, const FhnBuffer *const *ops, const int64_t *,
                    const double *fp) {
  reinterpret_cast<TestBuffer *>(result)->value =
    reinterpret_cast<const TestBuffer *>(ops[0])->value * static_cast<int64_t>(fp[0]) +
    reinterpret_cast<const TestBuffer *>(ops[1])->value;
  return 0;
}

} // namespace

TEST(FhnExecutor, FusedFallbackRejectsPhysicalAddendAliasBeforeWriting) {
  FhnKernelEntry entries[] = {{FHN_ROTATE, test_rotate, "rotate"},
                              {FHN_CONJUGATE, test_negate, "conjugate"},
                              {FHN_MULT_CS, test_mult_cs, "mult_cs"},
                              {FHN_ADD_CC, test_add_cc, "add"}};
  FhnKernelTable table{4, entries};
  FhnDefaultExecutor executor(&table);
  FhnKernelEntry native_entries[] = {{FHN_HROT_ADD, test_native_hrot_add, "hrot_add"},
                                     {FHN_HCONJ_ADD, test_native_hconj_add, "hconj_add"},
                                     {FHN_MAD, test_native_mad, "mad"}};
  FhnKernelTable native_table{3, native_entries};
  FhnDefaultExecutor native(&native_table);
  for (auto opcode : {FHN_HROT_ADD, FHN_HCONJ_ADD, FHN_MAD}) {
    SCOPED_TRACE(static_cast<int>(opcode));
    auto prog = ProgramBuilder().input(1).input(2).inst(opcode, 3, 1, 2).output(3).build();
    prog->instructions[0].fparams[0] = 3;
    TestBuffer a{5}, b{7}, result{-99};
    FhnBuffer *buffers[] = {nullptr, reinterpret_cast<FhnBuffer *>(&a), reinterpret_cast<FhnBuffer *>(&b),
                            reinterpret_cast<FhnBuffer *>(&result)};
    ASSERT_EQ(executor.execute(nullptr, prog.get(), buffers), 0);
    EXPECT_EQ(result.value, opcode == FHN_HROT_ADD ? 57 : opcode == FHN_HCONJ_ADD ? 2 : 22);
    const int64_t fallback_value = result.value;
    result.value = -99;
    ASSERT_EQ(native.execute(nullptr, prog.get(), buffers), 0);
    EXPECT_EQ(result.value, fallback_value);
    // Distinct SSA ids (2 and 3) refer to the very same physical buffer.
    buffers[3] = buffers[2];
    EXPECT_NE(executor.execute(nullptr, prog.get(), buffers), 0);
    EXPECT_EQ(a.value, 5);
    EXPECT_EQ(b.value, 7);
  }
}

TEST(FhnExecutor, FusedFallbackMissingAddRejectsBeforeTransformWrites) {
  FhnKernelEntry entries[] = {{FHN_HROT, test_rotate, "hrot"}};
  FhnKernelTable table{1, entries};
  FhnDefaultExecutor executor(&table);
  auto prog = ProgramBuilder().input(1).input(2).inst(FHN_HROT_ADD, 3, 1, 2).output(3).build();
  TestBuffer a{5}, b{7}, result{-99};
  FhnBuffer *buffers[] = {nullptr, reinterpret_cast<FhnBuffer *>(&a), reinterpret_cast<FhnBuffer *>(&b),
                          reinterpret_cast<FhnBuffer *>(&result)};
  EXPECT_NE(executor.execute(nullptr, prog.get(), buffers), 0);
  EXPECT_EQ(result.value, -99);
}

TEST(FhnExecutorMovement, SameLengthStalePlanRejectsBeforeHooksOrKernels) {
  for (int field = 0; field < 7; ++field) {
    SCOPED_TRACE(field);
    MovementWorld world;
    g_world = &world;
    auto original = ProgramBuilder().input(1).input(2).inst(FHN_ADD_CC, 3, 1, 2).output(3).build();
    auto plan = FhnMovementPlan::analyze(*original, {1, 2, 3});
    ASSERT_TRUE(plan);
    // Mutate the original storage, not merely a second program pointer.
    switch (field) {
    case 0:
      original->instructions[0].result_id = 4;
      break;
    case 1:
      original->instructions[0].operands[1] = 1;
      break;
    case 2:
      original->input_ids[1] = 4;
      break;
    case 3:
      original->output_ids[0] = 4;
      break;
    case 4:
      original->instructions[0].opcode = FHN_NEGATE;
      break;
    case 5:
      original->instructions[0].params[3] = 42;
      break;
    case 6:
      original->instructions[0].fparams[1] = -0.0;
      break;
    }
    FhnBuffer *buffers[5] = {};
    buffers[1] = movementAlloc(nullptr);
    buffers[2] = movementAlloc(nullptr);
    buffers[4] = movementAlloc(nullptr);
    auto *extra = buffers[4];
    world.device_vals[buffers[1]] = 10;
    world.device_vals[buffers[2]] = 20;
    world.log.clear();
    FhnDefaultExecutor executor(&movementTable);
    FhnMovementHooks hooks{nullptr, movementAlloc, movementFree, movementPrefetch, movementEvict};
    EXPECT_NE(executor.execute(hooks, original.get(), buffers, *plan), 0);
    EXPECT_TRUE(world.log.empty());
    EXPECT_EQ(world.kernel_calls, 0);
    EXPECT_EQ(world.device_vals.at(buffers[1]), 10);
    EXPECT_EQ(world.device_vals.at(buffers[2]), 20);
    if (buffers[3])
      movementFree(nullptr, buffers[3]);
    movementFree(nullptr, buffers[1]);
    movementFree(nullptr, buffers[2]);
    movementFree(nullptr, extra);
  }
}

TEST(FhnExecutorMovement, BudgetNormalizesResidentInputsBeforeFirstAllocation) {
  for (bool mixed_residency : {false, true}) {
    SCOPED_TRACE(mixed_residency);
    MovementWorld world;
    g_world = &world;
    world.budget = 3;
    auto prog = ProgramBuilder().input(1).input(2).input(3).inst(FHN_ADD_CC, 4, 1, 2).output(4).build();
    auto plan = FhnMovementPlan::analyze(*prog, {1, 2, 3, 4}, 3);
    ASSERT_TRUE(plan);
    FhnBuffer *buffers[5] = {};
    for (int i = 1; i <= 3; ++i) {
      buffers[i] = movementAlloc(nullptr);
      world.device_vals[buffers[i]] = i * 10;
    }
    if (mixed_residency)
      movementEvict(nullptr, buffers[2]);
    world.log.clear();
    FhnDefaultExecutor executor(&movementTable);
    FhnMovementHooks hooks{nullptr, movementAlloc, movementFree, movementPrefetch, movementEvict};
    EXPECT_EQ(executor.execute(hooks, prog.get(), buffers, *plan), 0);
    if (buffers[4])
      EXPECT_EQ(world.device_vals.at(buffers[4]), 30);
    EXPECT_LE(world.peak, 3u);
    EXPECT_TRUE(world.host_vals.count(buffers[3]));
    if (world.host_vals.count(buffers[3]))
      EXPECT_EQ(world.host_vals.at(buffers[3]), 30);
    for (int i = 1; i <= 4; ++i)
      if (buffers[i])
        movementFree(nullptr, buffers[i]);
  }
}

TEST(FhnExecutorMovement, BudgetRejectsPartialMovementHooksBeforeAnyWork) {
  for (int missing = 0; missing < 2; ++missing) {
    MovementWorld world;
    g_world = &world;
    auto prog = ProgramBuilder().input(1).inst(FHN_ADD_CC, 2, 1, 1).output(2).build();
    auto plan = FhnMovementPlan::analyze(*prog, {1, 2}, 2);
    ASSERT_TRUE(plan);
    FhnBuffer *buffers[] = {nullptr, movementAlloc(nullptr), nullptr};
    world.log.clear();
    FhnDefaultExecutor executor(&movementTable);
    FhnMovementHooks hooks{nullptr, movementAlloc, movementFree, missing == 0 ? nullptr : movementPrefetch,
                           missing == 1 ? nullptr : movementEvict};
    EXPECT_NE(executor.execute(hooks, prog.get(), buffers, *plan), 0);
    EXPECT_TRUE(world.log.empty());
    EXPECT_EQ(world.kernel_calls, 0);
    movementFree(nullptr, buffers[1]);
    if (buffers[2])
      movementFree(nullptr, buffers[2]);
  }
}

TEST(FhnExecutorMovement, BudgetValidatesAllInputsBeforeNormalization) {
  MovementWorld world;
  g_world = &world;
  auto prog = ProgramBuilder().input(1).input(2).inst(FHN_ADD_CC, 3, 1, 2).output(3).build();
  auto plan = FhnMovementPlan::analyze(*prog, {1, 2, 3}, 3);
  ASSERT_TRUE(plan);
  FhnBuffer *buffers[] = {nullptr, movementAlloc(nullptr), nullptr, nullptr};
  world.device_vals[buffers[1]] = 11;
  world.log.clear();
  FhnDefaultExecutor executor(&movementTable);
  FhnMovementHooks hooks{nullptr, movementAlloc, movementFree, movementPrefetch, movementEvict};
  EXPECT_NE(executor.execute(hooks, prog.get(), buffers, *plan), 0);
  EXPECT_TRUE(world.log.empty());
  EXPECT_EQ(world.device_vals.at(buffers[1]), 11);
  movementFree(nullptr, buffers[1]);
}

TEST(FhnExecutorMovement, BudgetWithoutMovementHooksRetainsSingleSpaceBehavior) {
  MovementWorld world;
  g_world = &world;
  auto prog = ProgramBuilder().input(1).input(2).inst(FHN_ADD_CC, 3, 1, 2).output(3).build();
  auto plan = FhnMovementPlan::analyze(*prog, {3}, 3);
  ASSERT_TRUE(plan);
  FhnBuffer *buffers[] = {nullptr, movementAlloc(nullptr), movementAlloc(nullptr), nullptr};
  world.device_vals[buffers[1]] = 11;
  world.device_vals[buffers[2]] = 22;
  world.log.clear();
  FhnDefaultExecutor executor(&movementTable);
  FhnMovementHooks hooks{nullptr, movementAlloc, movementFree, nullptr, nullptr};
  ASSERT_EQ(executor.execute(hooks, prog.get(), buffers, *plan), 0);
  EXPECT_EQ(world.device_vals.at(buffers[3]), 33);
  for (const auto &line : world.log) {
    EXPECT_EQ(line.find("prefetch"), std::string::npos);
    EXPECT_EQ(line.find("evict"), std::string::npos);
  }
  movementFree(nullptr, buffers[3]);
}

TEST(FhnExecutor, HmultRejectsEitherMissingStageBeforeMultiply) {
  auto prog = ProgramBuilder().input(1).input(2).inst(FHN_HMULT, 3, 1, 2).output(3).build();
  for (auto only_stage : {FHN_RELINEARIZE, FHN_RESCALE}) {
    FhnKernelEntry entries[] = {{FHN_MULT_CC, test_mult_cc, "multiply"}, {only_stage, test_noop, "stage"}};
    FhnKernelTable table{2, entries};
    FhnDefaultExecutor executor(&table);
    TestBuffer a{5}, b{7}, result{-99};
    FhnBuffer *buffers[] = {nullptr, reinterpret_cast<FhnBuffer *>(&a), reinterpret_cast<FhnBuffer *>(&b),
                            reinterpret_cast<FhnBuffer *>(&result)};
    EXPECT_NE(executor.execute(nullptr, prog.get(), buffers), 0);
    EXPECT_EQ(result.value, -99);
  }
}

TEST(FhnExecutorMovement, RejectsPhysicalInputAliasesBeforeAnyHook) {
  MovementWorld world;
  g_world = &world;
  auto prog = ProgramBuilder().input(1).input(2).inst(FHN_ADD_CC, 3, 1, 2).inst(FHN_ADD_CC, 4, 1, 1).output(4).build();
  auto plan = FhnMovementPlan::analyze(*prog, {1, 2, 3, 4}, 3);
  ASSERT_TRUE(plan);
  auto *input = movementAlloc(nullptr);
  world.device_vals[input] = 11;
  FhnBuffer *buffers[] = {nullptr, input, input, nullptr, nullptr};
  world.log.clear();
  FhnDefaultExecutor executor(&movementTable);
  FhnMovementHooks hooks{nullptr, movementAlloc, movementFree, movementPrefetch, movementEvict};
  int rc = 0;
  EXPECT_NO_THROW(rc = executor.execute(hooks, prog.get(), buffers, *plan));
  EXPECT_NE(rc, 0);
  EXPECT_TRUE(world.log.empty());
  EXPECT_EQ(world.kernel_calls, 0);
  EXPECT_TRUE(world.device_vals.count(input));
  if (world.device_vals.count(input))
    EXPECT_EQ(world.device_vals.at(input), 11);
  movementFree(nullptr, input);
  if (buffers[3])
    movementFree(nullptr, buffers[3]);
  if (buffers[4])
    movementFree(nullptr, buffers[4]);
}
