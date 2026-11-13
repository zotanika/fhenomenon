#include "CKKS/Circuit.h"
#include "CKKS/Backend.h"
#include <array>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <new>
#include <vector>

namespace circuit_allocations {
thread_local bool track = false;
thread_local size_t count = 0;
thread_local size_t fail_at = 0;
void allocation() {
  if (track && ++count == fail_at)
    throw std::bad_alloc();
}
} // namespace circuit_allocations
void *operator new(size_t size) {
  circuit_allocations::allocation();
  if (void *p = std::malloc(size ? size : 1))
    return p;
  throw std::bad_alloc();
}
void *operator new[](size_t size) { return ::operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { ::operator delete(p); }
void operator delete(void *p, size_t) noexcept { ::operator delete(p); }
void operator delete[](void *p, size_t) noexcept { ::operator delete(p); }
void *operator new(size_t size, std::align_val_t alignment) {
  circuit_allocations::allocation();
  void *p = nullptr;
  if (posix_memalign(&p, static_cast<size_t>(alignment), size ? size : 1) != 0)
    throw std::bad_alloc();
  return p;
}
void *operator new[](size_t size, std::align_val_t a) { return ::operator new(size, a); }
void operator delete(void *p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void *p, std::align_val_t a) noexcept { ::operator delete(p, a); }
void operator delete(void *p, size_t, std::align_val_t a) noexcept { ::operator delete(p, a); }
void operator delete[](void *p, size_t, std::align_val_t a) noexcept { ::operator delete(p, a); }

namespace {
using Context = std::unique_ptr<FhnBackendCtx, decltype(&ckks_fhn_destroy)>;
using Client = std::unique_ptr<FhnClientCtx, decltype(&ckks_fhn_client_destroy)>;
using Executable = std::unique_ptr<FhnExecutable, decltype(&ckks_fhn_executable_free)>;
struct Buffer {
  FhnBackendCtx *ctx;
  FhnBuffer *ptr;
  explicit Buffer(FhnBackendCtx *c) : ctx(c), ptr(ckks_fhn_buffer_alloc(c)) {}
  ~Buffer() { ckks_fhn_buffer_free(ctx, ptr); }
};

TEST(CkksCircuit, IntegerScaleAndAffineCancellationPreserveLevel) {
  Context ctx(ckks_fhn_create(nullptr), ckks_fhn_destroy);
  ASSERT_TRUE(ctx);
  Client client(ckks_fhn_client_create(ctx.get()), ckks_fhn_client_destroy);
  ASSERT_TRUE(client);
  Buffer x(ctx.get()), out(ctx.get());
  const std::array<double, 4> values{0.5, -1.25, 2.0, 0.125};
  ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), x.ptr, values.data(), 4), 0);
  FhnCircuitNode nodes[]{{FHN_CIRCUIT_SCALE, 2, {1, 0}, 0, 3},
                         {FHN_CIRCUIT_SUBTRACT, 3, {2, 1}, 0, 0},
                         {FHN_CIRCUIT_ADD_CONSTANT, 4, {3, 0}, 0, 1.5}};
  const uint32_t input = 1, output = 4;
  FhnCircuit circuit{FHN_CIRCUIT_VERSION, 3, nodes, 1, &input, 1, &output};
  const FhnBuffer *inputs[]{x.ptr};
  FhnBuffer *outputs[]{out.ptr};
  char error[256]{};
  Executable exec(ckks_fhn_compile_circuit(ctx.get(), &circuit, inputs, 1, nullptr, error, sizeof(error)),
                  ckks_fhn_executable_free);
  ASSERT_TRUE(exec) << error;
  ASSERT_EQ(ckks_fhn_run_executable(exec.get(), inputs, 1, outputs, 1), 0);
  std::array<double, 4> actual{};
  ASSERT_EQ(ckks_fhn_client_decrypt_vec_f64(client.get(), out.ptr, actual.data(), 4), 0);
  for (size_t i = 0; i < actual.size(); ++i)
    EXPECT_NEAR(actual[i], 2 * values[i] + 1.5, 1e-6);
  FhnCkksBufferInfo info{};
  ASSERT_EQ(ckks_fhn_buffer_info(ctx.get(), out.ptr, &info), 0);
  EXPECT_EQ(info.level, 2);
  FhnExecutableStats stats{};
  stats.struct_size = sizeof(stats);
  ASSERT_EQ(ckks_fhn_executable_stats(exec.get(), &stats), 0);
  EXPECT_EQ(stats.regions, 1u);
}

class Circuit : public ::testing::Test {
  protected:
  Context ctx{nullptr, ckks_fhn_destroy};
  Client client{nullptr, ckks_fhn_client_destroy};
  std::unique_ptr<Buffer> x, y, out, other;
  void SetUp() override {
    ctx.reset(ckks_fhn_create(nullptr));
    ASSERT_TRUE(ctx);
    client.reset(ckks_fhn_client_create(ctx.get()));
    ASSERT_TRUE(client);
    x = std::make_unique<Buffer>(ctx.get());
    y = std::make_unique<Buffer>(ctx.get());
    out = std::make_unique<Buffer>(ctx.get());
    other = std::make_unique<Buffer>(ctx.get());
    encrypt(x->ptr, 2);
    encrypt(y->ptr, 3);
    encrypt(out->ptr, 99);
    encrypt(other->ptr, -99);
  }
  void encrypt(FhnBuffer *b, double v) { ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), b, &v, 1), 0); }
  double decrypt(FhnBuffer *b) {
    double value = 0;
    EXPECT_EQ(ckks_fhn_client_decrypt_vec_f64(client.get(), b, &value, 1), 0);
    return value;
  }
  Executable compile(const std::vector<FhnCircuitNode> &nodes, const std::vector<uint32_t> &outputs,
                     const FhnCompileOptions *options = nullptr, char *error = nullptr) {
    const uint32_t ids[]{1, 2};
    const FhnBuffer *inputs[]{x->ptr, y->ptr};
    FhnCircuit c{FHN_CIRCUIT_VERSION,
                 static_cast<uint32_t>(nodes.size()),
                 nodes.data(),
                 2,
                 ids,
                 static_cast<uint32_t>(outputs.size()),
                 outputs.data()};
    return Executable(ckks_fhn_compile_circuit(ctx.get(), &c, inputs, 2, options, error, error ? 256 : 0),
                      ckks_fhn_executable_free);
  }
};

TEST_F(Circuit, CopiedGraphRebindAndInputOutputAliasAreAllocationFree) {
  std::vector<FhnCircuitNode> nodes{{FHN_CIRCUIT_ADD, 3, {1, 2}, 0, 0}};
  auto exec = compile(nodes, {3});
  ASSERT_TRUE(exec);
  nodes[0].op = FHN_CIRCUIT_SUBTRACT;
  const FhnBuffer *inputs[]{x->ptr, x->ptr};
  FhnBuffer *outputs[]{x->ptr};
  circuit_allocations::count = 0;
  circuit_allocations::track = true;
  int rc = ckks_fhn_run_executable(exec.get(), inputs, 2, outputs, 1);
  circuit_allocations::track = false;
  ASSERT_EQ(rc, 0);
  EXPECT_EQ(circuit_allocations::count, 0u);
  EXPECT_NEAR(decrypt(x->ptr), 4, 1e-6);
  encrypt(y->ptr, 7);
  inputs[0] = y->ptr;
  outputs[0] = out->ptr;
  ASSERT_EQ(ckks_fhn_run_executable(exec.get(), inputs, 2, outputs, 1), 0);
  EXPECT_NEAR(decrypt(out->ptr), 11, 1e-6);
}

TEST_F(Circuit, ProductSumPreparesOnlyNeededKeysAndExpires) {
  std::vector<FhnCircuitNode> nodes{{FHN_CIRCUIT_MULTIPLY, 3, {1, 2}, 0, 0},
                                    {FHN_CIRCUIT_MULTIPLY, 4, {1, 1}, 0, 0},
                                    {FHN_CIRCUIT_ADD, 5, {3, 4}, 0, 0},
                                    {FHN_CIRCUIT_ADD, 6, {5, 2}, 0, 0},
                                    {FHN_CIRCUIT_ADD_CONSTANT, 7, {6, 0}, 0, 1.25}};
  auto exec = compile(nodes, {7});
  ASSERT_TRUE(exec);
  const FhnBuffer *inputs[]{x->ptr, y->ptr};
  FhnBuffer *outputs[]{out->ptr};
  EXPECT_NE(ckks_fhn_run_executable(exec.get(), inputs, 2, outputs, 1), 0);
  EXPECT_NEAR(decrypt(out->ptr), 99, 1e-6);
  ASSERT_EQ(ckks_fhn_client_prepare_executable(client.get(), exec.get()), 0);
  circuit_allocations::count = 0;
  circuit_allocations::track = true;
  int rc = ckks_fhn_run_executable(exec.get(), inputs, 2, outputs, 1);
  circuit_allocations::track = false;
  ASSERT_EQ(rc, 0);
  EXPECT_EQ(circuit_allocations::count, 0u);
  EXPECT_NEAR(decrypt(out->ptr), 14.25, 1e-5);
  FhnExecutableStats stats{};
  stats.struct_size = sizeof(stats);
  ASSERT_EQ(ckks_fhn_executable_stats(exec.get(), &stats), 0);
  EXPECT_EQ(stats.product_regions, 1u);
  EXPECT_EQ(stats.regions, 1u);
  client.reset();
  EXPECT_NE(ckks_fhn_run_executable(exec.get(), inputs, 2, outputs, 1), 0);
}

TEST_F(Circuit, RejectsMalformedGraphsCoefficientOverflowAndInsufficientBudget) {
  char error[256]{};
  EXPECT_FALSE(compile({{FHN_CIRCUIT_ADD, 1, {1, 2}, 0, 0}}, {1}, nullptr, error));
  EXPECT_NE(error[0], '\0');
  EXPECT_FALSE(compile({{FHN_CIRCUIT_ADD, 3, {1, 99}, 0, 0}}, {3}));
  EXPECT_FALSE(compile({{FHN_CIRCUIT_ADD, 3, {1, 0}, 0, 0}}, {3}));
  EXPECT_FALSE(compile({{FHN_CIRCUIT_SCALE, 3, {1, 0}, 0, std::numeric_limits<double>::infinity()}}, {3}));
  EXPECT_FALSE(
    compile({{FHN_CIRCUIT_SCALE, 3, {1, 0}, 0, 4294967296.0}, {FHN_CIRCUIT_SCALE, 4, {3, 0}, 0, 4294967296.0}}, {4}));
  FhnCompileOptions options{sizeof(FhnCompileOptions), 1};
  EXPECT_FALSE(compile({{FHN_CIRCUIT_ADD, 3, {1, 2}, 0, 0}}, {3}, &options));
}

TEST_F(Circuit, AllOutputValidationPrecedesPublication) {
  auto exec = compile({{FHN_CIRCUIT_ADD, 3, {1, 2}, 0, 0}, {FHN_CIRCUIT_SUBTRACT, 4, {1, 2}, 0, 0}}, {3, 4});
  ASSERT_TRUE(exec);
  const FhnBuffer *inputs[]{x->ptr, y->ptr};
  FhnBuffer *outputs[]{out->ptr, out->ptr};
  EXPECT_NE(ckks_fhn_run_executable(exec.get(), inputs, 2, outputs, 2), 0);
  EXPECT_NEAR(decrypt(out->ptr), 99, 1e-6);
  outputs[1] = nullptr;
  EXPECT_NE(ckks_fhn_run_executable(exec.get(), inputs, 2, outputs, 2), 0);
  EXPECT_NEAR(decrypt(out->ptr), 99, 1e-6);
  outputs[1] = other->ptr;
  ASSERT_EQ(ckks_fhn_run_executable(exec.get(), inputs, 2, outputs, 2), 0);
  EXPECT_NEAR(decrypt(out->ptr), 5, 1e-6);
  EXPECT_NEAR(decrypt(other->ptr), -1, 1e-6);
}

TEST_F(Circuit, PassthroughOutputsSwapAndDuplicateIdsSafely) {
  auto swap = compile({}, {1, 2});
  ASSERT_TRUE(swap);
  const FhnBuffer *inputs[]{x->ptr, y->ptr};
  FhnBuffer *outputs[]{y->ptr, x->ptr};
  ASSERT_EQ(ckks_fhn_run_executable(swap.get(), inputs, 2, outputs, 2), 0);
  EXPECT_NEAR(decrypt(x->ptr), 3, 1e-6);
  EXPECT_NEAR(decrypt(y->ptr), 2, 1e-6);
  auto duplicate = compile({}, {1, 1});
  ASSERT_TRUE(duplicate);
  outputs[0] = out->ptr;
  outputs[1] = out->ptr;
  ASSERT_EQ(ckks_fhn_run_executable(duplicate.get(), inputs, 2, outputs, 2), 0);
  EXPECT_NEAR(decrypt(out->ptr), 3, 1e-6);
}

TEST_F(Circuit, SharedRotateReductionDoesNotDuplicateExpensiveLeaves) {
  const double values[]{1, 2, 3, 4};
  ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), x->ptr, values, 4), 0);
  ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), y->ptr, values, 4), 0);
  auto exec = compile({{FHN_CIRCUIT_ROTATE, 3, {1, 0}, 2, 0},
                       {FHN_CIRCUIT_ADD, 4, {1, 3}, 0, 0},
                       {FHN_CIRCUIT_ROTATE, 5, {4, 0}, 1, 0},
                       {FHN_CIRCUIT_ADD, 6, {4, 5}, 0, 0}},
                      {6});
  ASSERT_TRUE(exec);
  ASSERT_EQ(ckks_fhn_client_prepare_executable(client.get(), exec.get()), 0);
  FhnExecutableStats stats{};
  stats.struct_size = sizeof(stats);
  ASSERT_EQ(ckks_fhn_executable_stats(exec.get(), &stats), 0);
  EXPECT_EQ(stats.rotation_regions, 2u);
  EXPECT_EQ(stats.regions, 2u);
  EXPECT_EQ(stats.storage_slots, 2u);
  const FhnBuffer *inputs[]{x->ptr, y->ptr};
  FhnBuffer *outputs[]{out->ptr};
  circuit_allocations::count = 0;
  circuit_allocations::track = true;
  const int rc = ckks_fhn_run_executable(exec.get(), inputs, 2, outputs, 1);
  circuit_allocations::track = false;
  ASSERT_EQ(rc, 0);
  EXPECT_EQ(circuit_allocations::count, 0u);
  std::array<double, 4> actual{};
  ASSERT_EQ(ckks_fhn_client_decrypt_vec_f64(client.get(), out->ptr, actual.data(), 4), 0);
  for (double v : actual)
    EXPECT_NEAR(v, 10, 1e-5);
}

TEST_F(Circuit, ExactBudgetAndWrongPackingAreChecked) {
  const std::vector<FhnCircuitNode> nodes{{FHN_CIRCUIT_ADD, 3, {1, 2}, 0, 0}};
  auto exec = compile(nodes, {3});
  ASSERT_TRUE(exec);
  FhnExecutableStats stats{};
  stats.struct_size = sizeof(stats);
  ASSERT_EQ(ckks_fhn_executable_stats(exec.get(), &stats), 0);
  // The budget counts the Arena's actual aligned allocation, not its request.
  EXPECT_EQ(stats.scratch_bytes % 64, 0u);
  FhnCompileOptions options{sizeof(options), stats.storage_bytes + stats.scratch_bytes};
  EXPECT_TRUE(compile(nodes, {3}, &options));
  --options.workspace_limit_bytes;
  EXPECT_FALSE(compile(nodes, {3}, &options));
  const double packed[]{1, 2};
  ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), y->ptr, packed, 2), 0);
  const FhnBuffer *inputs[]{x->ptr, y->ptr};
  FhnBuffer *outputs[]{out->ptr};
  EXPECT_NE(ckks_fhn_run_executable(exec.get(), inputs, 2, outputs, 1), 0);
  EXPECT_NEAR(decrypt(out->ptr), 99, 1e-6);
}

TEST_F(Circuit, CancellationDoesNotHideIncompatiblePacking) {
  const double packed[]{1, 2};
  ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), y->ptr, packed, 2), 0);
  // The algebraic cancellation leaves only x, but x+y is not a valid packed
  // value operation and must fail before optimization removes the bad edge.
  EXPECT_FALSE(compile({{FHN_CIRCUIT_ADD, 3, {1, 2}, 0, 0}, {FHN_CIRCUIT_SUBTRACT, 4, {3, 2}, 0, 0}}, {4}));
}

TEST_F(Circuit, CancelledProductNeedsNoEvaluationKeyAndConstantEncodingIsChecked) {
  auto exec = compile({{FHN_CIRCUIT_MULTIPLY, 3, {1, 2}, 0, 0},
                       {FHN_CIRCUIT_SUBTRACT, 4, {3, 3}, 0, 0},
                       {FHN_CIRCUIT_ADD_CONSTANT, 5, {4, 0}, 0, 1.25}},
                      {5});
  ASSERT_TRUE(exec);
  const FhnBuffer *inputs[]{x->ptr, y->ptr};
  FhnBuffer *outputs[]{out->ptr};
  ASSERT_EQ(ckks_fhn_run_executable(exec.get(), inputs, 2, outputs, 1), 0);
  EXPECT_NEAR(decrypt(out->ptr), 1.25, 1e-6);
  EXPECT_FALSE(compile({{FHN_CIRCUIT_ADD_CONSTANT, 3, {1, 0}, 0, 1e30}}, {3}));
}

TEST_F(Circuit, FractionalScalingAndUnequalLevelProductsHaveActualMetadata) {
  auto exec = compile({{FHN_CIRCUIT_SCALE, 3, {1, 0}, 0, 0.5}, {FHN_CIRCUIT_MULTIPLY, 4, {3, 2}, 0, 0}}, {3, 4});
  ASSERT_TRUE(exec);
  ASSERT_EQ(ckks_fhn_client_prepare_executable(client.get(), exec.get()), 0);
  const FhnBuffer *inputs[]{x->ptr, y->ptr};
  FhnBuffer *outputs[]{out->ptr, other->ptr};
  ASSERT_EQ(ckks_fhn_run_executable(exec.get(), inputs, 2, outputs, 2), 0);
  EXPECT_NEAR(decrypt(out->ptr), 1, 1e-5);
  EXPECT_NEAR(decrypt(other->ptr), 3, 1e-5);
  FhnCkksBufferInfo fractional{}, product{};
  ASSERT_EQ(ckks_fhn_buffer_info(ctx.get(), out->ptr, &fractional), 0);
  ASSERT_EQ(ckks_fhn_buffer_info(ctx.get(), other->ptr, &product), 0);
  EXPECT_EQ(fractional.level, 1);
  EXPECT_EQ(product.level, 0);
  inputs[0] = out->ptr;
  EXPECT_NE(ckks_fhn_run_executable(exec.get(), inputs, 2, outputs, 2), 0);
  EXPECT_NEAR(decrypt(other->ptr), 3, 1e-5);
}

TEST(CkksCircuit, ExecutablePreparationRetriesEveryAllocationFailure) {
  bool reached_success = false;
  for (size_t failure = 1; failure <= 64; ++failure) {
    Context ctx(ckks_fhn_create(nullptr), ckks_fhn_destroy);
    ASSERT_TRUE(ctx);
    Client client(ckks_fhn_client_create(ctx.get()), ckks_fhn_client_destroy);
    ASSERT_TRUE(client);
    Buffer x(ctx.get()), out(ctx.get());
    const double value = 2;
    ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), x.ptr, &value, 1), 0);
    const FhnCircuitNode nodes[]{{FHN_CIRCUIT_MULTIPLY, 2, {1, 1}, 0, 0}, {FHN_CIRCUIT_ROTATE, 3, {2, 0}, 1, 0}};
    const uint32_t input = 1, output = 3;
    const FhnCircuit c{FHN_CIRCUIT_VERSION, 2, nodes, 1, &input, 1, &output};
    const FhnBuffer *inputs[]{x.ptr};
    FhnBuffer *outputs[]{out.ptr};
    Executable exec(ckks_fhn_compile_circuit(ctx.get(), &c, inputs, 1, nullptr, nullptr, 0), ckks_fhn_executable_free);
    ASSERT_TRUE(exec);
    circuit_allocations::count = 0;
    circuit_allocations::fail_at = failure;
    circuit_allocations::track = true;
    const int first = ckks_fhn_client_prepare_executable(client.get(), exec.get());
    circuit_allocations::track = false;
    const bool injected = circuit_allocations::count >= failure;
    circuit_allocations::fail_at = 0;
    EXPECT_EQ(first, injected ? -1 : 0) << failure;
    ASSERT_EQ(ckks_fhn_client_prepare_executable(client.get(), exec.get()), 0) << failure;
    ASSERT_EQ(ckks_fhn_run_executable(exec.get(), inputs, 1, outputs, 1), 0) << failure;
    double actual = 0;
    ASSERT_EQ(ckks_fhn_client_decrypt_vec_f64(client.get(), out.ptr, &actual, 1), 0);
    EXPECT_NEAR(actual, 4, 1e-5);
    if (!injected) {
      reached_success = true;
      break;
    }
  }
  EXPECT_TRUE(reached_success);
}
} // namespace
