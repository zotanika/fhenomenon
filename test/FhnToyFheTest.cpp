#include "FHN/ToyFheKernels.h"
#include "FHN/FhnDefaultExecutor.h"
#include "FHN/fhn_program.h"

#include <gtest/gtest.h>
#include <cstring>
#include <memory>

class FhnToyFheTest : public ::testing::Test {
protected:
    void SetUp() override {
        info_ = toyfhe_fhn_get_info();
        ctx_ = toyfhe_fhn_create(nullptr);
        table_ = toyfhe_fhn_get_kernels(ctx_);
        executor_ = std::make_unique<fhenomenon::FhnDefaultExecutor>(table_);
    }
    void TearDown() override {
        toyfhe_fhn_destroy(ctx_);
    }
    FhnBackendInfo *info_;
    FhnBackendCtx *ctx_;
    FhnKernelTable *table_;
    std::unique_ptr<fhenomenon::FhnDefaultExecutor> executor_;
};

TEST_F(FhnToyFheTest, BackendInfo) {
    EXPECT_STREQ(info_->name, "toyfhe-reference");
    EXPECT_STREQ(info_->version, "0.1");
    EXPECT_EQ(info_->device_type, FHN_DEVICE_CPU);
    EXPECT_EQ(info_->device_memory, 0u);
}

TEST_F(FhnToyFheTest, SupportsBasicOps) {
    EXPECT_TRUE(executor_->supports(FHN_ADD_CC));
    EXPECT_TRUE(executor_->supports(FHN_HMULT));
    EXPECT_TRUE(executor_->supports(FHN_ENCRYPT));
    EXPECT_TRUE(executor_->supports(FHN_DECRYPT));
    EXPECT_FALSE(executor_->supports(FHN_ROTATE));
    EXPECT_FALSE(executor_->supports(FHN_AND));
}

TEST_F(FhnToyFheTest, EncryptAddDecrypt) {
    // Program:
    //   inst[0]: FHN_ENCRYPT result_id=1, params[0]=10
    //   inst[1]: FHN_ENCRYPT result_id=2, params[0]=20
    //   inst[2]: FHN_ADD_CC  result_id=3, operands={1,2}
    //   inst[3]: FHN_DECRYPT result_id=4, operands={3}
    FhnProgram *prog = fhn_program_alloc(4, 0, 1);
    ASSERT_NE(prog, nullptr);

    prog->output_ids[0] = 4;

    // inst 0: encrypt 10
    auto &enc1 = prog->instructions[0];
    std::memset(&enc1, 0, sizeof(enc1));
    enc1.opcode = FHN_ENCRYPT;
    enc1.result_id = 1;
    enc1.params[0] = 10;

    // inst 1: encrypt 20
    auto &enc2 = prog->instructions[1];
    std::memset(&enc2, 0, sizeof(enc2));
    enc2.opcode = FHN_ENCRYPT;
    enc2.result_id = 2;
    enc2.params[0] = 20;

    // inst 2: add
    auto &add = prog->instructions[2];
    std::memset(&add, 0, sizeof(add));
    add.opcode = FHN_ADD_CC;
    add.result_id = 3;
    add.operands[0] = 1;
    add.operands[1] = 2;

    // inst 3: decrypt
    auto &dec = prog->instructions[3];
    std::memset(&dec, 0, sizeof(dec));
    dec.opcode = FHN_DECRYPT;
    dec.result_id = 4;
    dec.operands[0] = 3;

    // Allocate buffers (index 0 unused, 1-4 for result_ids)
    FhnBuffer *bufs[5];
    for (int i = 0; i < 5; ++i) {
        bufs[i] = toyfhe_fhn_buffer_alloc(ctx_);
    }

    int rc = executor_->execute(ctx_, prog, bufs);
    EXPECT_EQ(rc, 0);

    int64_t result = toyfhe_fhn_buffer_read_int(ctx_, bufs[4]);
    EXPECT_EQ(result, 30);

    for (int i = 0; i < 5; ++i) {
        toyfhe_fhn_buffer_free(ctx_, bufs[i]);
    }
    fhn_program_free(prog);
}

TEST_F(FhnToyFheTest, EncryptMultiplyDecrypt) {
    // Use MULT_CS (ciphertext * scalar) because ToyFHE's ciphertext-ciphertext
    // multiply doesn't preserve exact integer arithmetic (delta^2 scaling).
    // Program:
    //   inst[0]: FHN_ENCRYPT result_id=1, params[0]=7
    //   inst[1]: FHN_MULT_CS result_id=2, operands={1}, fparams[0]=6.0
    //   inst[2]: FHN_DECRYPT result_id=3, operands={2}
    FhnProgram *prog = fhn_program_alloc(3, 0, 1);
    ASSERT_NE(prog, nullptr);

    prog->output_ids[0] = 3;

    auto &enc1 = prog->instructions[0];
    std::memset(&enc1, 0, sizeof(enc1));
    enc1.opcode = FHN_ENCRYPT;
    enc1.result_id = 1;
    enc1.params[0] = 7;

    auto &mult = prog->instructions[1];
    std::memset(&mult, 0, sizeof(mult));
    mult.opcode = FHN_MULT_CS;
    mult.result_id = 2;
    mult.operands[0] = 1;
    mult.fparams[0] = 6.0;

    auto &dec = prog->instructions[2];
    std::memset(&dec, 0, sizeof(dec));
    dec.opcode = FHN_DECRYPT;
    dec.result_id = 3;
    dec.operands[0] = 2;

    FhnBuffer *bufs[4];
    for (int i = 0; i < 4; ++i) {
        bufs[i] = toyfhe_fhn_buffer_alloc(ctx_);
    }

    int rc = executor_->execute(ctx_, prog, bufs);
    EXPECT_EQ(rc, 0);

    int64_t result = toyfhe_fhn_buffer_read_int(ctx_, bufs[3]);
    EXPECT_EQ(result, 42);

    for (int i = 0; i < 4; ++i) {
        toyfhe_fhn_buffer_free(ctx_, bufs[i]);
    }
    fhn_program_free(prog);
}
