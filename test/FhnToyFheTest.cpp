#include "FHN/FhnDefaultExecutor.h"
#include "FHN/ToyFheKernels.h"
#include "FHN/fhn_program.h"

#include <cstring>
#include <gtest/gtest.h>
#include <memory>

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
  EXPECT_FALSE(executor_->supports(FHN_ROTATE));
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
