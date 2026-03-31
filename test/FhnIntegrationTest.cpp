#include "Backend/Backend.h"
#include "Backend/Builtin.h"
#include "Compuon.h"
#include "FHN/FhnDefaultExecutor.h"
#include "FHN/ToyFheKernels.h"
#include "FHN/fhn_program.h"
#include "Scheduler/ASTNode.h"
#include "Scheduler/LowerToFhnProgram.h"
#include "Scheduler/Operation.h"
#include "Scheduler/Planner.h"
#include "Scheduler/Scheduler.h"
#include <gtest/gtest.h>
#include <vector>

using namespace fhenomenon;
using namespace fhenomenon::scheduler;

class FhnIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx_ = toyfhe_fhn_create(nullptr);
        auto *table = toyfhe_fhn_get_kernels(ctx_);
        executor_ = std::make_unique<FhnDefaultExecutor>(table);
    }
    void TearDown() override {
        toyfhe_fhn_destroy(ctx_);
    }

    FhnBackendCtx *ctx_;
    std::unique_ptr<FhnDefaultExecutor> executor_;
};

TEST_F(FhnIntegrationTest, OperationsToASTToFhnProgramToExecution) {
    // 1. Build operations: result = a + b
    auto a = std::make_shared<Compuon<int>>(10);
    auto b = std::make_shared<Compuon<int>>(20);
    auto result = std::make_shared<Compuon<int>>(0);

    auto add_op = std::make_shared<Operation<int>>(OperationType::Add, a, b, result);

    // 2. Build AST manually (simulating what Scheduler::buildGraph does)
    auto leaf_a = std::make_shared<OperandNode<int>>(a);
    auto leaf_b = std::make_shared<OperandNode<int>>(b);
    auto root = std::make_shared<OperatorNode<int>>(
        add_op, OperationType::Add, leaf_a, leaf_b, result);

    Planner<int> plan;
    plan.addRoot(root);

    // 3. Lower to FhnProgram
    LowerToFhnProgram lowering;
    FhnProgram *prog = lowering.lower(plan);
    ASSERT_NE(prog, nullptr);
    ASSERT_EQ(prog->num_instructions, 1u);
    ASSERT_EQ(prog->num_inputs, 2u);
    ASSERT_EQ(prog->num_outputs, 1u);

    // 4. Find the max result_id to allocate buffer array
    uint32_t max_id = 0;
    for (uint32_t i = 0; i < prog->num_inputs; i++)
        if (prog->input_ids[i] > max_id) max_id = prog->input_ids[i];
    for (uint32_t i = 0; i < prog->num_instructions; i++)
        if (prog->instructions[i].result_id > max_id) max_id = prog->instructions[i].result_id;

    // 5. Allocate buffers
    std::vector<FhnBuffer*> buffers(max_id + 1, nullptr);
    for (auto &buf : buffers)
        buf = toyfhe_fhn_buffer_alloc(ctx_);

    // 6. Encrypt inputs into buffers
    // Build tiny encrypt programs for each input
    for (uint32_t i = 0; i < prog->num_inputs; i++) {
        uint32_t id = prog->input_ids[i];
        auto *enc_prog = fhn_program_alloc(1, 0, 1);
        enc_prog->output_ids[0] = id;
        enc_prog->instructions[0].opcode = FHN_ENCRYPT;
        enc_prog->instructions[0].result_id = id;
        // Determine which Compuon this input corresponds to:
        // input 0 = a (value 10), input 1 = b (value 20)
        enc_prog->instructions[0].params[0] = (i == 0) ? 10 : 20;
        int err = executor_->execute(ctx_, enc_prog, buffers.data());
        ASSERT_EQ(err, 0);
        fhn_program_free(enc_prog);
    }

    // 7. Execute the main program
    int err = executor_->execute(ctx_, prog, buffers.data());
    EXPECT_EQ(err, 0);

    // 8. Decrypt output
    uint32_t out_id = prog->output_ids[0];
    auto *dec_prog = fhn_program_alloc(1, 1, 1);
    uint32_t dec_result_id = max_id + 1;
    buffers.push_back(toyfhe_fhn_buffer_alloc(ctx_));
    dec_prog->input_ids[0] = out_id;
    dec_prog->output_ids[0] = dec_result_id;
    dec_prog->instructions[0].opcode = FHN_DECRYPT;
    dec_prog->instructions[0].result_id = dec_result_id;
    dec_prog->instructions[0].operands[0] = out_id;
    err = executor_->execute(ctx_, dec_prog, buffers.data());
    EXPECT_EQ(err, 0);

    int64_t decrypted = toyfhe_fhn_buffer_read_int(ctx_, buffers[dec_result_id]);
    EXPECT_EQ(decrypted, 30);

    // Cleanup
    for (auto *buf : buffers)
        toyfhe_fhn_buffer_free(ctx_, buf);
    fhn_program_free(dec_prog);
    fhn_program_free(prog);
}

TEST_F(FhnIntegrationTest, ChainedOperations) {
    // Build: t = a + b; result = t + a
    // Values: a=5, b=3 -> t=8 -> result=13
    auto a = std::make_shared<Compuon<int>>(5);
    auto b = std::make_shared<Compuon<int>>(3);
    auto t = std::make_shared<Compuon<int>>(0);
    auto result = std::make_shared<Compuon<int>>(0);

    auto add1 = std::make_shared<Operation<int>>(OperationType::Add, a, b, t);
    auto add2 = std::make_shared<Operation<int>>(OperationType::Add, t, a, result);

    auto leaf_a = std::make_shared<OperandNode<int>>(a);
    auto leaf_b = std::make_shared<OperandNode<int>>(b);
    auto node1 = std::make_shared<OperatorNode<int>>(
        add1, OperationType::Add, leaf_a, leaf_b, t);
    auto node2 = std::make_shared<OperatorNode<int>>(
        add2, OperationType::Add, node1, leaf_a, result);

    Planner<int> plan;
    plan.addRoot(node2);

    LowerToFhnProgram lowering;
    FhnProgram *prog = lowering.lower(plan);
    ASSERT_NE(prog, nullptr);

    // Allocate buffers
    uint32_t max_id = 0;
    for (uint32_t i = 0; i < prog->num_inputs; i++)
        if (prog->input_ids[i] > max_id) max_id = prog->input_ids[i];
    for (uint32_t i = 0; i < prog->num_instructions; i++)
        if (prog->instructions[i].result_id > max_id) max_id = prog->instructions[i].result_id;

    uint32_t dec_id = max_id + 1;
    std::vector<FhnBuffer*> buffers(dec_id + 1, nullptr);
    for (auto &buf : buffers)
        buf = toyfhe_fhn_buffer_alloc(ctx_);

    // Encrypt inputs: we need to map input_ids to values
    // input_ids correspond to leaf_a and leaf_b. Since leaf_a is shared (used twice),
    // there should be 2 unique inputs. Let's encrypt them.
    // We know a=5, b=3
    int values[] = {5, 3};
    for (uint32_t i = 0; i < prog->num_inputs; i++) {
        auto *enc = fhn_program_alloc(1, 0, 1);
        enc->output_ids[0] = prog->input_ids[i];
        enc->instructions[0].opcode = FHN_ENCRYPT;
        enc->instructions[0].result_id = prog->input_ids[i];
        enc->instructions[0].params[0] = values[i];
        executor_->execute(ctx_, enc, buffers.data());
        fhn_program_free(enc);
    }

    // Execute
    int err = executor_->execute(ctx_, prog, buffers.data());
    EXPECT_EQ(err, 0);

    // Decrypt
    auto *dec = fhn_program_alloc(1, 1, 1);
    dec->input_ids[0] = prog->output_ids[0];
    dec->output_ids[0] = dec_id;
    dec->instructions[0].opcode = FHN_DECRYPT;
    dec->instructions[0].result_id = dec_id;
    dec->instructions[0].operands[0] = prog->output_ids[0];
    executor_->execute(ctx_, dec, buffers.data());

    int64_t result_val = toyfhe_fhn_buffer_read_int(ctx_, buffers[dec_id]);
    EXPECT_EQ(result_val, 13);

    for (auto *buf : buffers)
        toyfhe_fhn_buffer_free(ctx_, buf);
    fhn_program_free(dec);
    fhn_program_free(prog);
}
