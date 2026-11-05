#include "FHN/FhnDefaultExecutor.h"
#include "FHN/ToyFheKernels.h"
#include "FHN/fhn_program.h"

#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <vector>

class FhnToyFheTest : public ::testing::Test {
  protected:
  void SetUp() override {
    info_ = toyfhe_fhn_get_info();
    ctx_ = toyfhe_fhn_create(nullptr);
    table_ = toyfhe_fhn_get_kernels(ctx_);
    executor_ = std::make_unique<fhenomenon::FhnDefaultExecutor>(table_);
  }
  void TearDown() override { toyfhe_fhn_destroy(ctx_); }
  FhnBackendInfo *info_;
  FhnBackendCtx *ctx_;
  FhnKernelTable *table_;
  std::unique_ptr<fhenomenon::FhnDefaultExecutor> executor_;
};

TEST_F(FhnToyFheTest, AbiVersion) { EXPECT_EQ(toyfhe_fhn_get_abi_version(), FHN_ABI_VERSION); }

TEST_F(FhnToyFheTest, BackendInfo) {
  EXPECT_STREQ(info_->name, "toyfhe-reference");
  EXPECT_STREQ(info_->version, "0.1");
  EXPECT_EQ(info_->device_type, FHN_DEVICE_CPU);
  EXPECT_EQ(info_->device_memory, 0u);
}

TEST_F(FhnToyFheTest, SupportsComputeOpsOnly) {
  EXPECT_TRUE(executor_->supports(FHN_ADD_CC));
  EXPECT_TRUE(executor_->supports(FHN_HMULT));
  EXPECT_TRUE(executor_->supports(FHN_MULT_CS));
  EXPECT_TRUE(executor_->supports(FHN_ROTATE));
  EXPECT_TRUE(executor_->supports(FHN_HROT_ADD));
  EXPECT_TRUE(executor_->supports(FHN_RELINEARIZE));
  EXPECT_TRUE(executor_->supports(FHN_RESCALE));
  EXPECT_FALSE(executor_->supports(FHN_CONJUGATE));
  EXPECT_FALSE(executor_->supports(FHN_AND));
}

// Encryption and decryption live in the host-side data plane, not in the
// instruction stream: encrypt into buffers, run a compute-only program,
// decrypt the output buffer.
TEST_F(FhnToyFheTest, DataPlaneRoundTrip) {
  FhnBuffer *buf = toyfhe_fhn_buffer_alloc(ctx_);

  ASSERT_EQ(toyfhe_fhn_encrypt_i64(ctx_, buf, 42), 0);
  int64_t value = 0;
  ASSERT_EQ(toyfhe_fhn_decrypt_i64(ctx_, buf, &value), 0);
  EXPECT_EQ(value, 42);

  ASSERT_EQ(toyfhe_fhn_encrypt_f64(ctx_, buf, 2.5), 0);
  double dvalue = 0.0;
  ASSERT_EQ(toyfhe_fhn_decrypt_f64(ctx_, buf, &dvalue), 0);
  EXPECT_NEAR(dvalue, 2.5, 1e-3);

  // Decrypting an empty (non-ciphertext) buffer is an error, not a zero.
  FhnBuffer *empty = toyfhe_fhn_buffer_alloc(ctx_);
  EXPECT_NE(toyfhe_fhn_decrypt_i64(ctx_, empty, &value), 0);

  toyfhe_fhn_buffer_free(ctx_, empty);
  toyfhe_fhn_buffer_free(ctx_, buf);
}

TEST_F(FhnToyFheTest, AddProgram) {
  // Program: buf[3] = buf[1] + buf[2] (inputs encrypted host-side)
  FhnProgram *prog = fhn_program_alloc(1, 2, 1);
  ASSERT_NE(prog, nullptr);

  prog->input_ids[0] = 1;
  prog->input_ids[1] = 2;
  prog->output_ids[0] = 3;

  auto &add = prog->instructions[0];
  std::memset(&add, 0, sizeof(add));
  add.opcode = FHN_ADD_CC;
  add.result_id = 3;
  add.operands[0] = 1;
  add.operands[1] = 2;

  FhnBuffer *bufs[4];
  for (int i = 0; i < 4; ++i) {
    bufs[i] = toyfhe_fhn_buffer_alloc(ctx_);
  }
  ASSERT_EQ(toyfhe_fhn_encrypt_i64(ctx_, bufs[1], 10), 0);
  ASSERT_EQ(toyfhe_fhn_encrypt_i64(ctx_, bufs[2], 20), 0);

  int rc = executor_->execute(ctx_, prog, bufs);
  EXPECT_EQ(rc, 0);

  int64_t result = 0;
  ASSERT_EQ(toyfhe_fhn_decrypt_i64(ctx_, bufs[3], &result), 0);
  EXPECT_EQ(result, 30);

  for (int i = 0; i < 4; ++i) {
    toyfhe_fhn_buffer_free(ctx_, bufs[i]);
  }
  fhn_program_free(prog);
}

TEST_F(FhnToyFheTest, ScalarMultiplyProgram) {
  // MULT_CS: ciphertext * scalar with an integral scalar stays scale-free.
  // Program: buf[2] = buf[1] * 6.0
  FhnProgram *prog = fhn_program_alloc(1, 1, 1);
  ASSERT_NE(prog, nullptr);

  prog->input_ids[0] = 1;
  prog->output_ids[0] = 2;

  auto &mult = prog->instructions[0];
  std::memset(&mult, 0, sizeof(mult));
  mult.opcode = FHN_MULT_CS;
  mult.result_id = 2;
  mult.operands[0] = 1;
  mult.fparams[0] = 6.0;

  FhnBuffer *bufs[3];
  for (int i = 0; i < 3; ++i) {
    bufs[i] = toyfhe_fhn_buffer_alloc(ctx_);
  }
  ASSERT_EQ(toyfhe_fhn_encrypt_i64(ctx_, bufs[1], 7), 0);

  int rc = executor_->execute(ctx_, prog, bufs);
  EXPECT_EQ(rc, 0);

  int64_t result = 0;
  ASSERT_EQ(toyfhe_fhn_decrypt_i64(ctx_, bufs[2], &result), 0);
  EXPECT_EQ(result, 42);

  for (int i = 0; i < 3; ++i) {
    toyfhe_fhn_buffer_free(ctx_, bufs[i]);
  }
  fhn_program_free(prog);
}

TEST_F(FhnToyFheTest, CtCtMultiplyProgram) {
  // Ciphertext * ciphertext through the fused HMULT kernel must decrypt to
  // the exact integer product (mult + relin + rescale).
  // Program: buf[3] = HMULT(buf[1], buf[2])
  FhnProgram *prog = fhn_program_alloc(1, 2, 1);
  ASSERT_NE(prog, nullptr);

  prog->input_ids[0] = 1;
  prog->input_ids[1] = 2;
  prog->output_ids[0] = 3;

  auto &mult = prog->instructions[0];
  std::memset(&mult, 0, sizeof(mult));
  mult.opcode = FHN_HMULT;
  mult.result_id = 3;
  mult.operands[0] = 1;
  mult.operands[1] = 2;

  FhnBuffer *bufs[4];
  for (int i = 0; i < 4; ++i) {
    bufs[i] = toyfhe_fhn_buffer_alloc(ctx_);
  }
  ASSERT_EQ(toyfhe_fhn_encrypt_i64(ctx_, bufs[1], 13), 0);
  ASSERT_EQ(toyfhe_fhn_encrypt_i64(ctx_, bufs[2], 20), 0);

  int rc = executor_->execute(ctx_, prog, bufs);
  EXPECT_EQ(rc, 0);

  int64_t result = 0;
  ASSERT_EQ(toyfhe_fhn_decrypt_i64(ctx_, bufs[3], &result), 0);
  EXPECT_EQ(result, 260);

  for (int i = 0; i < 4; ++i) {
    toyfhe_fhn_buffer_free(ctx_, bufs[i]);
  }
  fhn_program_free(prog);
}

TEST_F(FhnToyFheTest, FixedPointDecryptReturnsScaledValue) {
  // A fractional scalar moves the ciphertext to fixed-point encoding; the
  // data-plane decrypt must decode through the scale-aware path.
  // Program: buf[2] = buf[1] * 0.5
  FhnProgram *prog = fhn_program_alloc(1, 1, 1);
  ASSERT_NE(prog, nullptr);

  prog->input_ids[0] = 1;
  prog->output_ids[0] = 2;

  auto &mult = prog->instructions[0];
  std::memset(&mult, 0, sizeof(mult));
  mult.opcode = FHN_MULT_CS;
  mult.result_id = 2;
  mult.operands[0] = 1;
  mult.fparams[0] = 0.5;

  FhnBuffer *bufs[3];
  for (int i = 0; i < 3; ++i) {
    bufs[i] = toyfhe_fhn_buffer_alloc(ctx_);
  }
  ASSERT_EQ(toyfhe_fhn_encrypt_i64(ctx_, bufs[1], 10), 0);

  int rc = executor_->execute(ctx_, prog, bufs);
  EXPECT_EQ(rc, 0);

  double dresult = 0.0;
  ASSERT_EQ(toyfhe_fhn_decrypt_f64(ctx_, bufs[2], &dresult), 0);
  EXPECT_NEAR(dresult, 5.0, 1e-3);

  int64_t iresult = 0;
  ASSERT_EQ(toyfhe_fhn_decrypt_i64(ctx_, bufs[2], &iresult), 0);
  EXPECT_EQ(iresult, 5);

  for (int i = 0; i < 3; ++i) {
    toyfhe_fhn_buffer_free(ctx_, bufs[i]);
  }
  fhn_program_free(prog);
}

// --- Multi-slot (CiphertextVec) tests ---------------------------------------

TEST_F(FhnToyFheTest, VecDataPlaneRoundTrip) {
  FhnBuffer *buf = toyfhe_fhn_buffer_alloc(ctx_);
  const int64_t values[4] = {5, -3, 0, 123456};
  ASSERT_EQ(toyfhe_fhn_encrypt_vec_i64(ctx_, buf, values, 4), 0);

  int64_t decrypted[4] = {0, 0, 0, 0};
  ASSERT_EQ(toyfhe_fhn_decrypt_vec_i64(ctx_, buf, decrypted, 4), 0);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(decrypted[i], values[i]);
  }

  // Slot-count mismatch is an error.
  EXPECT_NE(toyfhe_fhn_decrypt_vec_i64(ctx_, buf, decrypted, 3), 0);

  // Wrong buffer kind (scalar ciphertext) is an error.
  FhnBuffer *scalar = toyfhe_fhn_buffer_alloc(ctx_);
  ASSERT_EQ(toyfhe_fhn_encrypt_i64(ctx_, scalar, 7), 0);
  EXPECT_NE(toyfhe_fhn_decrypt_vec_i64(ctx_, scalar, decrypted, 1), 0);

  toyfhe_fhn_buffer_free(ctx_, scalar);
  toyfhe_fhn_buffer_free(ctx_, buf);
}

TEST_F(FhnToyFheTest, VecAddAndHmultPrograms) {
  // Program: buf[3] = ADD_CC(buf[1], buf[2]); buf[4] = HMULT(buf[1], buf[2])
  FhnProgram *prog = fhn_program_alloc(2, 2, 2);
  ASSERT_NE(prog, nullptr);

  prog->input_ids[0] = 1;
  prog->input_ids[1] = 2;
  prog->output_ids[0] = 3;
  prog->output_ids[1] = 4;

  auto &add = prog->instructions[0];
  add.opcode = FHN_ADD_CC;
  add.result_id = 3;
  add.operands[0] = 1;
  add.operands[1] = 2;

  auto &mult = prog->instructions[1];
  mult.opcode = FHN_HMULT;
  mult.result_id = 4;
  mult.operands[0] = 1;
  mult.operands[1] = 2;

  FhnBuffer *bufs[5];
  for (int i = 0; i < 5; ++i) {
    bufs[i] = toyfhe_fhn_buffer_alloc(ctx_);
  }
  const int64_t a[4] = {1, 2, 3, 4};
  const int64_t b[4] = {10, 20, 30, 40};
  ASSERT_EQ(toyfhe_fhn_encrypt_vec_i64(ctx_, bufs[1], a, 4), 0);
  ASSERT_EQ(toyfhe_fhn_encrypt_vec_i64(ctx_, bufs[2], b, 4), 0);

  ASSERT_EQ(executor_->execute(ctx_, prog, bufs), 0);

  int64_t sum[4] = {0};
  int64_t product[4] = {0};
  ASSERT_EQ(toyfhe_fhn_decrypt_vec_i64(ctx_, bufs[3], sum, 4), 0);
  ASSERT_EQ(toyfhe_fhn_decrypt_vec_i64(ctx_, bufs[4], product, 4), 0);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(sum[i], a[i] + b[i]);
    EXPECT_EQ(product[i], a[i] * b[i]);
  }

  // Mixed scalar/vector operands must fail, as must mismatched slot counts.
  ASSERT_EQ(toyfhe_fhn_encrypt_i64(ctx_, bufs[2], 5), 0);
  EXPECT_NE(executor_->execute(ctx_, prog, bufs), 0);
  ASSERT_EQ(toyfhe_fhn_encrypt_vec_i64(ctx_, bufs[2], b, 2), 0);
  EXPECT_NE(executor_->execute(ctx_, prog, bufs), 0);

  for (int i = 0; i < 5; ++i) {
    toyfhe_fhn_buffer_free(ctx_, bufs[i]);
  }
  fhn_program_free(prog);
}

TEST_F(FhnToyFheTest, VecRotateProgram) {
  // ROTATE is cyclic-left for positive distances: result[i] = src[(i+d) % n].
  // Program: buf[2] = ROTATE(buf[1], +2); buf[3] = ROTATE(buf[1], -1)
  FhnProgram *prog = fhn_program_alloc(2, 1, 2);
  ASSERT_NE(prog, nullptr);

  prog->input_ids[0] = 1;
  prog->output_ids[0] = 2;
  prog->output_ids[1] = 3;

  auto &left2 = prog->instructions[0];
  left2.opcode = FHN_ROTATE;
  left2.result_id = 2;
  left2.operands[0] = 1;
  left2.params[0] = 2;

  auto &right1 = prog->instructions[1];
  right1.opcode = FHN_ROTATE;
  right1.result_id = 3;
  right1.operands[0] = 1;
  right1.params[0] = -1;

  FhnBuffer *bufs[4];
  for (int i = 0; i < 4; ++i) {
    bufs[i] = toyfhe_fhn_buffer_alloc(ctx_);
  }
  const int64_t src[4] = {1, 2, 3, 4};
  ASSERT_EQ(toyfhe_fhn_encrypt_vec_i64(ctx_, bufs[1], src, 4), 0);

  ASSERT_EQ(executor_->execute(ctx_, prog, bufs), 0);

  int64_t rotated[4] = {0};
  ASSERT_EQ(toyfhe_fhn_decrypt_vec_i64(ctx_, bufs[2], rotated, 4), 0);
  const int64_t expected_left2[4] = {3, 4, 1, 2};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(rotated[i], expected_left2[i]);
  }

  ASSERT_EQ(toyfhe_fhn_decrypt_vec_i64(ctx_, bufs[3], rotated, 4), 0);
  const int64_t expected_right1[4] = {4, 1, 2, 3};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(rotated[i], expected_right1[i]);
  }

  for (int i = 0; i < 4; ++i) {
    toyfhe_fhn_buffer_free(ctx_, bufs[i]);
  }
  fhn_program_free(prog);
}

TEST_F(FhnToyFheTest, VecHrotAddProgram) {
  // Fused rotate-and-add: result[i] = a[(i+d) % n] + b[i] with d = 1.
  // Program: buf[3] = HROT_ADD(buf[1], buf[2])
  FhnProgram *prog = fhn_program_alloc(1, 2, 1);
  ASSERT_NE(prog, nullptr);

  prog->input_ids[0] = 1;
  prog->input_ids[1] = 2;
  prog->output_ids[0] = 3;

  auto &hra = prog->instructions[0];
  hra.opcode = FHN_HROT_ADD;
  hra.result_id = 3;
  hra.operands[0] = 1;
  hra.operands[1] = 2;
  hra.params[0] = 1;

  FhnBuffer *bufs[4];
  for (int i = 0; i < 4; ++i) {
    bufs[i] = toyfhe_fhn_buffer_alloc(ctx_);
  }
  const int64_t a[4] = {1, 2, 3, 4};
  const int64_t b[4] = {100, 200, 300, 400};
  ASSERT_EQ(toyfhe_fhn_encrypt_vec_i64(ctx_, bufs[1], a, 4), 0);
  ASSERT_EQ(toyfhe_fhn_encrypt_vec_i64(ctx_, bufs[2], b, 4), 0);

  ASSERT_EQ(executor_->execute(ctx_, prog, bufs), 0);

  int64_t result[4] = {0};
  ASSERT_EQ(toyfhe_fhn_decrypt_vec_i64(ctx_, bufs[3], result, 4), 0);
  const int64_t expected[4] = {102, 203, 304, 401};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(result[i], expected[i]);
  }

  for (int i = 0; i < 4; ++i) {
    toyfhe_fhn_buffer_free(ctx_, bufs[i]);
  }
  fhn_program_free(prog);
}

TEST_F(FhnToyFheTest, ReductionFusedMatchesDecomposed) {
  // Dot-product reduction in SSA form:
  //   buf[3] = HMULT(buf[1], buf[2])          (slot-wise products)
  //   buf[4] = HROT_ADD(buf[3], buf[3]) d=2
  //   buf[5] = HROT_ADD(buf[4], buf[4]) d=1
  // after which every slot of buf[5] holds dot(a, b). Executing the SAME
  // program over the full table (fused kernels) and over a primitives-only
  // filtered table (executor decomposition: HMULT -> MULT_CC + RELINEARIZE +
  // RESCALE, HROT_ADD -> ROTATE + ADD_CC) must decrypt identically.
  FhnProgram *prog = fhn_program_alloc(3, 2, 1);
  ASSERT_NE(prog, nullptr);

  prog->input_ids[0] = 1;
  prog->input_ids[1] = 2;
  prog->output_ids[0] = 5;

  auto &mult = prog->instructions[0];
  mult.opcode = FHN_HMULT;
  mult.result_id = 3;
  mult.operands[0] = 1;
  mult.operands[1] = 2;

  auto &red1 = prog->instructions[1];
  red1.opcode = FHN_HROT_ADD;
  red1.result_id = 4;
  red1.operands[0] = 3;
  red1.operands[1] = 3;
  red1.params[0] = 2;

  auto &red2 = prog->instructions[2];
  red2.opcode = FHN_HROT_ADD;
  red2.result_id = 5;
  red2.operands[0] = 4;
  red2.operands[1] = 4;
  red2.params[0] = 1;

  FhnBuffer *bufs[6];
  for (int i = 0; i < 6; ++i) {
    bufs[i] = toyfhe_fhn_buffer_alloc(ctx_);
  }
  const int64_t a[4] = {3, 1, 4, 1};
  const int64_t b[4] = {2, 7, 1, 8};
  const int64_t dot = 3 * 2 + 1 * 7 + 4 * 1 + 1 * 8; // 25
  ASSERT_EQ(toyfhe_fhn_encrypt_vec_i64(ctx_, bufs[1], a, 4), 0);
  ASSERT_EQ(toyfhe_fhn_encrypt_vec_i64(ctx_, bufs[2], b, 4), 0);

  // (a) full table: HMULT and HROT_ADD dispatch to the fused kernels.
  ASSERT_EQ(executor_->execute(ctx_, prog, bufs), 0);
  int64_t fused[4] = {0};
  ASSERT_EQ(toyfhe_fhn_decrypt_vec_i64(ctx_, bufs[5], fused, 4), 0);

  // (b) primitives-only table: drop the fused entries so the default
  // executor's decomposition rules run. The filtered storage must stay
  // alive as long as the executor built over it.
  std::vector<FhnKernelEntry> primitives;
  for (uint32_t i = 0; i < table_->num_kernels; ++i) {
    const FhnKernelEntry &entry = table_->kernels[i];
    if (entry.opcode == FHN_HMULT || entry.opcode == FHN_HROT_ADD) {
      continue;
    }
    primitives.push_back(entry);
  }
  FhnKernelTable primitives_table = {static_cast<uint32_t>(primitives.size()), primitives.data()};
  fhenomenon::FhnDefaultExecutor decomposed_executor(&primitives_table);
  EXPECT_FALSE(decomposed_executor.supports(FHN_HMULT));
  EXPECT_FALSE(decomposed_executor.supports(FHN_HROT_ADD));

  ASSERT_EQ(decomposed_executor.execute(ctx_, prog, bufs), 0);
  int64_t decomposed[4] = {0};
  ASSERT_EQ(toyfhe_fhn_decrypt_vec_i64(ctx_, bufs[5], decomposed, 4), 0);

  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(fused[i], dot);
    EXPECT_EQ(decomposed[i], fused[i]);
  }

  for (int i = 0; i < 6; ++i) {
    toyfhe_fhn_buffer_free(ctx_, bufs[i]);
  }
  fhn_program_free(prog);
}

TEST_F(FhnToyFheTest, VectorScalarSubtractNegateAreSlotwiseAndAliasSafe) {
  const int64_t a[] = {2, -4, 6, 0};
  const int64_t b[] = {1, 3, -2, 5};
  struct Case {
    FhnOpCode opcode;
    std::vector<int64_t> expected;
  };
  const Case cases[] = {{FHN_ADD_CS, {5, -1, 9, 3}},
                        {FHN_MULT_CS, {6, -12, 18, 0}},
                        {FHN_SUB_CC, {1, -7, 8, -5}},
                        {FHN_NEGATE, {-2, 4, -6, 0}}};
  for (const auto &test : cases) {
    for (int alias = 0; alias < 3; ++alias) {
      SCOPED_TRACE(static_cast<int>(test.opcode));
      SCOPED_TRACE(alias);
      auto *prog = fhn_program_alloc(1, 2, 1);
      prog->input_ids[0] = 1;
      prog->input_ids[1] = 2;
      prog->output_ids[0] = 3;
      auto &inst = prog->instructions[0];
      inst.opcode = test.opcode;
      inst.result_id = 3;
      inst.operands[0] = 1;
      inst.operands[1] = test.opcode == FHN_SUB_CC ? 2 : 0;
      inst.fparams[0] = 3;
      auto *av = toyfhe_fhn_buffer_alloc(ctx_);
      auto *bv = toyfhe_fhn_buffer_alloc(ctx_);
      auto *out = toyfhe_fhn_buffer_alloc(ctx_);
      ASSERT_EQ(toyfhe_fhn_encrypt_vec_i64(ctx_, av, a, 4), 0);
      ASSERT_EQ(toyfhe_fhn_encrypt_vec_i64(ctx_, bv, b, 4), 0);
      FhnBuffer *buffers[] = {nullptr, av, bv, alias == 1 ? av : alias == 2 ? bv : out};
      EXPECT_EQ(executor_->execute(ctx_, prog, buffers), 0);
      std::vector<int64_t> actual(4, -999);
      EXPECT_EQ(toyfhe_fhn_decrypt_vec_i64(ctx_, buffers[3], actual.data(), 4), 0);
      EXPECT_EQ(actual, test.expected);
      toyfhe_fhn_buffer_free(ctx_, av);
      toyfhe_fhn_buffer_free(ctx_, bv);
      toyfhe_fhn_buffer_free(ctx_, out);
      fhn_program_free(prog);
    }
  }
}

TEST_F(FhnToyFheTest, ArithmeticRejectsMalformedKindsWithoutChangingResult) {
  const int64_t v[] = {3, 4};
  for (auto opcode : {FHN_ADD_CS, FHN_MULT_CS, FHN_SUB_CC, FHN_NEGATE, FHN_ADD_CC, FHN_MULT_CC}) {
    SCOPED_TRACE(static_cast<int>(opcode));
    FhnKernelFn kernel = nullptr;
    for (uint32_t i = 0; i < table_->num_kernels; ++i)
      if (table_->kernels[i].opcode == opcode)
        kernel = table_->kernels[i].fn;
    ASSERT_NE(kernel, nullptr);
    auto *empty = toyfhe_fhn_buffer_alloc(ctx_);
    auto *vec = toyfhe_fhn_buffer_alloc(ctx_);
    auto *scalar = toyfhe_fhn_buffer_alloc(ctx_);
    auto *out = toyfhe_fhn_buffer_alloc(ctx_);
    ASSERT_EQ(toyfhe_fhn_encrypt_vec_i64(ctx_, vec, v, 2), 0);
    ASSERT_EQ(toyfhe_fhn_encrypt_i64(ctx_, scalar, 5), 0);
    ASSERT_EQ(toyfhe_fhn_encrypt_i64(ctx_, out, 99), 0);
    const int64_t params[4] = {};
    const double fparams[2] = {2, 0};
    const FhnBuffer *ops[] = {empty, scalar, nullptr, nullptr};
    EXPECT_NE(kernel(ctx_, out, ops, params, fparams), 0);
    int64_t result = 0;
    EXPECT_EQ(toyfhe_fhn_decrypt_i64(ctx_, out, &result), 0);
    EXPECT_EQ(result, 99);
    if (opcode == FHN_SUB_CC || opcode == FHN_ADD_CC || opcode == FHN_MULT_CC) {
      ops[0] = vec;
      EXPECT_NE(kernel(ctx_, out, ops, params, fparams), 0);
      ASSERT_EQ(toyfhe_fhn_encrypt_vec_i64(ctx_, scalar, v, 1), 0);
      EXPECT_NE(kernel(ctx_, out, ops, params, fparams), 0);
      EXPECT_EQ(toyfhe_fhn_decrypt_i64(ctx_, out, &result), 0);
      EXPECT_EQ(result, 99);
    }
    toyfhe_fhn_buffer_free(ctx_, empty);
    toyfhe_fhn_buffer_free(ctx_, vec);
    toyfhe_fhn_buffer_free(ctx_, scalar);
    toyfhe_fhn_buffer_free(ctx_, out);
  }
}

TEST_F(FhnToyFheTest, PackedDoubleRoundTripAndScalarVectorOperations) {
  const std::vector<double> input = {1.25, -2.5, 0.125, 3.75};
  const std::vector<double> other = {2.0, 1.5, -0.25, 0.5};
  auto *a = toyfhe_fhn_buffer_alloc(ctx_);
  auto *b = toyfhe_fhn_buffer_alloc(ctx_);
  auto *out = toyfhe_fhn_buffer_alloc(ctx_);
  ASSERT_EQ(toyfhe_fhn_encrypt_vec_f64(ctx_, a, input.data(), 4), 0);
  ASSERT_EQ(toyfhe_fhn_encrypt_vec_f64(ctx_, b, other.data(), 4), 0);
  std::vector<double> actual(4);
  ASSERT_EQ(toyfhe_fhn_decrypt_vec_f64(ctx_, a, actual.data(), 4), 0);
  for (size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(actual[i], input[i], 1e-4);
  struct Case {
    FhnOpCode opcode;
    std::vector<double> expected;
  };
  const Case cases[] = {{FHN_ADD_CS, {1.75, -2.0, 0.625, 4.25}},    {FHN_MULT_CS, {0.625, -1.25, 0.0625, 1.875}},
                        {FHN_ADD_CC, {3.25, -1.0, -0.125, 4.25}},   {FHN_SUB_CC, {-0.75, -4.0, 0.375, 3.25}},
                        {FHN_HMULT, {2.5, -3.75, -0.03125, 1.875}}, {FHN_NEGATE, {-1.25, 2.5, -0.125, -3.75}}};
  for (const auto &test : cases) {
    SCOPED_TRACE(static_cast<int>(test.opcode));
    auto *prog = fhn_program_alloc(1, 2, 1);
    prog->input_ids[0] = 1;
    prog->input_ids[1] = 2;
    prog->output_ids[0] = 3;
    auto &inst = prog->instructions[0];
    inst.opcode = test.opcode;
    inst.result_id = 3;
    inst.operands[0] = 1;
    inst.operands[1] = 2;
    inst.fparams[0] = 0.5;
    FhnBuffer *buffers[] = {nullptr, a, b, out};
    EXPECT_EQ(executor_->execute(ctx_, prog, buffers), 0);
    EXPECT_EQ(toyfhe_fhn_decrypt_vec_f64(ctx_, out, actual.data(), 4), 0);
    for (size_t i = 0; i < 4; ++i)
      EXPECT_NEAR(actual[i], test.expected[i], 1e-4);
    fhn_program_free(prog);
  }
  toyfhe_fhn_buffer_free(ctx_, a);
  toyfhe_fhn_buffer_free(ctx_, b);
  toyfhe_fhn_buffer_free(ctx_, out);
}

TEST_F(FhnToyFheTest, PackedDoubleInvalidRequestsPreserveBuffersAndReadback) {
  auto *buf = toyfhe_fhn_buffer_alloc(ctx_);
  auto *empty = toyfhe_fhn_buffer_alloc(ctx_);
  const double original[] = {1.25, -2.5, 0.125};
  ASSERT_EQ(toyfhe_fhn_encrypt_vec_f64(ctx_, buf, original, 3), 0);
  EXPECT_NE(toyfhe_fhn_encrypt_vec_f64(nullptr, buf, original, 3), 0);
  EXPECT_NE(toyfhe_fhn_encrypt_vec_f64(ctx_, nullptr, original, 3), 0);
  EXPECT_NE(toyfhe_fhn_encrypt_vec_f64(ctx_, buf, nullptr, 3), 0);
  EXPECT_NE(toyfhe_fhn_encrypt_vec_f64(ctx_, buf, original, 0), 0);
  for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                     -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::max(), 32768.0, -32768.0}) {
    const double invalid[] = {42.0, bad, 13.0};
    EXPECT_NE(toyfhe_fhn_encrypt_vec_f64(ctx_, buf, invalid, 3), 0);
    double restored[3] = {};
    ASSERT_EQ(toyfhe_fhn_decrypt_vec_f64(ctx_, buf, restored, 3), 0);
    for (int i = 0; i < 3; ++i)
      EXPECT_NEAR(restored[i], original[i], 1e-4);
  }
  std::vector<double> sentinel(4, 123.0);
  EXPECT_NE(toyfhe_fhn_decrypt_vec_f64(nullptr, buf, sentinel.data(), 3), 0);
  EXPECT_NE(toyfhe_fhn_decrypt_vec_f64(ctx_, nullptr, sentinel.data(), 3), 0);
  EXPECT_NE(toyfhe_fhn_decrypt_vec_f64(ctx_, buf, nullptr, 3), 0);
  EXPECT_NE(toyfhe_fhn_decrypt_vec_f64(ctx_, empty, sentinel.data(), 3), 0);
  for (uint32_t count : {0u, 2u, 4u, std::numeric_limits<uint32_t>::max()})
    EXPECT_NE(toyfhe_fhn_decrypt_vec_f64(ctx_, buf, sentinel.data(), count), 0);
  EXPECT_EQ(sentinel, (std::vector<double>{123.0, 123.0, 123.0, 123.0}));
  ASSERT_EQ(toyfhe_fhn_encrypt_f64(ctx_, empty, 1.5), 0);
  EXPECT_NE(toyfhe_fhn_decrypt_vec_f64(ctx_, empty, sentinel.data(), 1), 0);
  toyfhe_fhn_buffer_free(ctx_, buf);
  toyfhe_fhn_buffer_free(ctx_, empty);
}
