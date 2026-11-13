#include "FHN/contrib/StatsKernels.h"
#include <algorithm>
#include <array>
#include <dlfcn.h>
#include <gtest/gtest.h>

namespace {
using namespace fhenomenon::runtime;
using fhenomenon::contrib::registerStatsKernels;

// Allocation counters wrap the real native allocator; ciphertext arithmetic
// and cleanup remain unchanged. Each test uses one synchronous runtime.
thread_local FhnBufferAllocFn nativeAllocate = nullptr;
thread_local FhnBufferFreeFn nativeFree = nullptr;
thread_local std::size_t allocations = 0;
thread_local std::size_t frees = 0;
FhnBuffer *countedAllocate(FhnBackendCtx *context) {
  ++allocations;
  return nativeAllocate(context);
}
void countedFree(FhnBackendCtx *context, FhnBuffer *buffer) {
  ++frees;
  nativeFree(context, buffer);
}

class StatsContributionTest : public ::testing::Test {
  protected:
  void *library = nullptr;
  NativeServices services;
  FhnClientCtx *client = nullptr;
  FhnBuffer *input = nullptr;
  FhnClientDestroyFn destroyClient = nullptr;
  FhnClientPrepareFn prepare = nullptr;
  FhnClientEncryptVectorDoubleFn encrypt = nullptr;
  FhnClientDecryptVectorDoubleFn decrypt = nullptr;
  const std::array<double, 4> data{0.5, -1.25, 2, 0.125};

  template <class T> T resolve(const char *name) { return reinterpret_cast<T>(dlsym(library, name)); }
  void SetUp() override {
    library = dlopen(FHN_NATIVE_TEST_LIBRARY, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(library, nullptr) << dlerror();
    services = loadNativeServices(FHN_NATIVE_TEST_LIBRARY);
    // Contributions must work with compiler services explicitly unavailable.
    services.compile = nullptr;
    services.run = nullptr;
    services.executable_stats = nullptr;
    services.executable_free = nullptr;
    auto create = resolve<FhnClientCreateFn>("ckks_fhn_client_create");
    destroyClient = resolve<FhnClientDestroyFn>("ckks_fhn_client_destroy");
    prepare = resolve<FhnClientPrepareFn>("ckks_fhn_client_prepare");
    encrypt = resolve<FhnClientEncryptVectorDoubleFn>("ckks_fhn_client_encrypt_vec_f64");
    decrypt = resolve<FhnClientDecryptVectorDoubleFn>("ckks_fhn_client_decrypt_vec_f64");
    ASSERT_TRUE(create && destroyClient && prepare && encrypt && decrypt);
    client = create(services.context.get());
    ASSERT_NE(client, nullptr);
    input = services.buffer_alloc(services.context.get());
    ASSERT_NE(input, nullptr);
    ASSERT_EQ(encrypt(client, input, data.data(), data.size()), 0);
  }
  void releaseHost() {
    if (input)
      services.buffer_free(services.context.get(), input);
    input = nullptr;
    if (client)
      destroyClient(client);
    client = nullptr;
  }
  void TearDown() override {
    releaseHost();
    services = {};
    if (library)
      dlclose(library);
  }
  void keys() {
    std::array<FhnInstruction, 3> instructions{};
    instructions[0].opcode = FHN_HMULT;
    instructions[1].opcode = FHN_ROTATE;
    instructions[1].params[0] = 1;
    instructions[2].opcode = FHN_ROTATE;
    instructions[2].params[0] = 2;
    FhnProgram program{FHN_ABI_VERSION, instructions.size(), instructions.data(), 0, nullptr, 0, nullptr};
    ASSERT_EQ(prepare(client, &program), 0);
  }
  Value sum(NativeRuntime &native, Value value) {
    value = native.op(FHN_ADD_CC, {value, native.rotate(value, 2)});
    return native.op(FHN_ADD_CC, {value, native.rotate(value, 1)});
  }
  void expectValues(NativeRuntime &native, const Value &value, const std::array<double, 4> &expected) {
    std::array<double, 4> actual{};
    ASSERT_EQ(decrypt(client, native.buffer(value), actual.data(), actual.size()), 0);
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i], 1e-5) << "slot " << i;
  }
  std::size_t bytes() {
    return services.level_bytes(services.context.get(), services.fresh_level(services.context.get()));
  }
  NativeServices countedServices() {
    nativeAllocate = services.buffer_alloc;
    nativeFree = services.buffer_free;
    allocations = frees = 0;
    auto counted = services;
    counted.buffer_alloc = countedAllocate;
    counted.buffer_free = countedFree;
    return counted;
  }
  void expectError(Runtime &runtime, const Request &request, ErrorCode expected) {
    try {
      runtime.submit(request);
      FAIL() << "expected runtime error";
    } catch (const Error &error) {
      EXPECT_EQ(error.code(), expected) << error.what();
    }
  }
};

// Omitting the registration, reusing the wrong operand, or losing the ordinary
// sum's metadata breaks this independently calculated encrypted result.
TEST_F(StatsContributionTest, OrdinarySumFeedsVarianceAndSharedInputsFeedIndependentDot) {
  keys();
  Runtime runtime;
  NativeRuntime native(runtime, services);
  registerStatsKernels(runtime, services, FHN_NATIVE_TEST_LIBRARY);
  auto x = native.importCopy(input);
  auto total = sum(native, x);
  expectValues(native, total, {1.375, 1.375, 1.375, 1.375});
  const auto baseline = runtime.stats().live_bytes;
  auto variance = runtime.submit({"contrib.stats.variance", {x, total}, std::uint32_t{4}}).at(0);
  auto doubled = native.op(FHN_ADD_CC, {x, x});
  auto dot = runtime.submit({"contrib.packed.dot", {x, doubled}, std::uint32_t{4}}).at(0);
  expectValues(native, variance, {21.421875, 21.421875, 21.421875, 21.421875});
  expectValues(native, dot, {11.65625, 11.65625, 11.65625, 11.65625});
  expectValues(native, x, data);
  expectValues(native, total, {1.375, 1.375, 1.375, 1.375});
  FhnCkksBufferInfo metadata{};
  ASSERT_EQ(services.buffer_info(native.context(), native.buffer(variance), &metadata), 0);
  EXPECT_EQ(metadata.level, services.fresh_level(native.context()) - 1);
  EXPECT_EQ(metadata.components, 2u);
  EXPECT_EQ(metadata.logical_slots, 4u);
  FhnCkksBufferInfo dotMetadata{};
  ASSERT_EQ(services.buffer_info(native.context(), native.buffer(dot), &dotMetadata), 0);
  EXPECT_DOUBLE_EQ(metadata.scale, 4 * dotMetadata.scale);
  // Both branches are native Values, but their actual scales differ. Joining
  // them must fail rather than silently relabeling centered statistics.
  EXPECT_THROW(native.semantic(FHN_CIRCUIT_ADD, {variance, dot}), Error);
  EXPECT_TRUE(variance.dependencies().empty());
  EXPECT_EQ(runtime.stats().live_bytes, baseline + variance.bytes() + doubled.bytes() + dot.bytes());
  variance = {};
  doubled = {};
  dot = {};
  EXPECT_EQ(runtime.stats().live_bytes, baseline);
  x = {};
  total = {};
  EXPECT_EQ(runtime.stats().live_bytes, 0u);
}

// The published energy must be a real native Value, and a partial reduction
// must preserve four logical slots instead of relabeling them as width two.
TEST_F(StatsContributionTest, EnergyFlowsThroughOrdinaryRotateAddThenPartialReduction) {
  keys();
  Runtime runtime;
  NativeRuntime native(runtime, services);
  registerStatsKernels(runtime, services, FHN_NATIVE_TEST_LIBRARY);
  auto x = native.importCopy(input);
  auto total = sum(native, x);
  auto energy = runtime.submit({"contrib.stats.energy", {x, total}, std::uint32_t{4}}).at(0);
  expectValues(native, energy, {0.09765625, 10.16015625, 10.97265625, 0.19140625});
  auto paired = native.op(FHN_ADD_CC, {energy, native.rotate(energy, 2)});
  expectValues(native, paired, {11.0703125, 10.3515625, 11.0703125, 10.3515625});
  auto reduced = runtime.submit({"contrib.packed.reduce", {paired}, std::uint32_t{2}}).at(0);
  expectValues(native, reduced, {21.421875, 21.421875, 21.421875, 21.421875});
  FhnCkksBufferInfo metadata{};
  ASSERT_EQ(services.buffer_info(native.context(), native.buffer(reduced), &metadata), 0);
  EXPECT_EQ(metadata.logical_slots, 4u);
  EXPECT_EQ(metadata.level, services.fresh_level(native.context()) - 1);
}

TEST_F(StatsContributionTest, PublicationChargesBothBuffersThenReleasesStaging) {
  Runtime runtime({3 * bytes(), 256});
  auto counted = countedServices();
  NativeRuntime native(runtime, counted);
  registerStatsKernels(runtime, counted, FHN_NATIVE_TEST_LIBRARY);
  auto x = native.importCopy(input);
  auto copied = runtime.submit({"contrib.packed.reduce", {x}, std::uint32_t{1}}).at(0);
  expectValues(native, copied, data);
  EXPECT_NE(copied.identity(), x.identity());
  EXPECT_EQ(runtime.stats().peak_live_bytes, 3 * bytes());
  EXPECT_EQ(runtime.stats().live_bytes, 2 * bytes());
  EXPECT_EQ(allocations, 3u);
  EXPECT_EQ(frees, 1u);
  copied = {};
  EXPECT_EQ(runtime.stats().live_bytes, bytes());
  EXPECT_EQ(frees, 2u);
}

TEST_F(StatsContributionTest, PublicationBudgetFailureReleasesUnpublishedStaging) {
  Runtime runtime({2 * bytes(), 256});
  auto counted = countedServices();
  NativeRuntime native(runtime, counted);
  registerStatsKernels(runtime, counted, FHN_NATIVE_TEST_LIBRARY);
  auto x = native.importCopy(input);
  expectError(runtime, {"contrib.packed.reduce", {x}, std::uint32_t{1}}, ErrorCode::BudgetExceeded);
  EXPECT_EQ(runtime.stats().peak_live_bytes, 2 * bytes());
  EXPECT_EQ(runtime.stats().live_bytes, bytes());
  EXPECT_EQ(runtime.stats().live_resources, 1u);
  EXPECT_EQ(allocations, 2u);
  EXPECT_EQ(frees, 1u);
  expectValues(native, x, data);
}

TEST_F(StatsContributionTest, StageBudgetFailureLeavesInputAndAccountingUntouched) {
  Runtime runtime({bytes(), 256});
  auto counted = countedServices();
  NativeRuntime native(runtime, counted);
  registerStatsKernels(runtime, counted, FHN_NATIVE_TEST_LIBRARY);
  auto x = native.importCopy(input);
  expectError(runtime, {"contrib.packed.reduce", {x}, std::uint32_t{1}}, ErrorCode::BudgetExceeded);
  EXPECT_EQ(runtime.stats().peak_live_bytes, bytes());
  EXPECT_EQ(runtime.stats().live_bytes, bytes());
  EXPECT_EQ(allocations, 1u);
  EXPECT_EQ(frees, 0u);
  expectValues(native, x, data);
}

TEST_F(StatsContributionTest, ForeignContextsAndPlainValuesDeclineBeforeNativeExecution) {
  Runtime runtime;
  NativeRuntime native(runtime, services);
  auto other = loadNativeServices(FHN_NATIVE_TEST_LIBRARY);
  registerStatsKernels(runtime, other, FHN_NATIVE_TEST_LIBRARY);
  registerStatsKernels(runtime, services, FHN_NATIVE_TEST_LIBRARY);
  auto x = native.importCopy(input);
  // The foreign context declines before the matching provider is selected.
  auto result = runtime.submit({"contrib.packed.reduce", {x}, std::uint32_t{1}}).at(0);
  expectValues(native, result, data);
  ASSERT_FALSE(runtime.trace().empty());
  EXPECT_EQ(runtime.trace().front().kind, TraceKind::Declined);
  auto plain = runtime.makeValue("test.plain", ResourceKind::Value, sizeof(double), [] { return std::any{2.0}; });
  const auto baseline = runtime.stats().live_bytes;
  expectError(runtime, {"contrib.packed.reduce", {plain}, std::uint32_t{1}}, ErrorCode::Unsupported);
  EXPECT_EQ(runtime.stats().live_bytes, baseline);
  Runtime otherRuntime;
  NativeRuntime otherNative(otherRuntime, services);
  auto foreign = otherNative.importCopy(input);
  expectError(runtime, {"contrib.packed.reduce", {foreign}, std::uint32_t{1}}, ErrorCode::InvalidRequest);
}

TEST_F(StatsContributionTest, InvalidWidthAttributesAndArityDeclineBeforeExecution) {
  Runtime runtime;
  NativeRuntime native(runtime, services);
  registerStatsKernels(runtime, services, FHN_NATIVE_TEST_LIBRARY);
  auto x = native.importCopy(input);
  const auto baseline = runtime.stats().live_bytes;
  for (std::uint32_t width : {0u, 3u, 8u})
    expectError(runtime, {"contrib.packed.reduce", {x}, width}, ErrorCode::Unsupported);
  expectError(runtime, {"contrib.packed.reduce", {x}, std::int64_t{4}}, ErrorCode::Unsupported);
  expectError(runtime, {"contrib.packed.reduce", {x, x}, std::uint32_t{4}}, ErrorCode::Unsupported);
  expectError(runtime, {"contrib.stats.energy", {x}, std::uint32_t{4}}, ErrorCode::Unsupported);
  EXPECT_EQ(runtime.stats().live_bytes, baseline);
  const auto trace = runtime.trace();
  EXPECT_EQ(std::count_if(trace.begin(), trace.end(),
                          [](const TraceEvent &event) { return event.kind == TraceKind::Declined; }),
            6);
  EXPECT_EQ(std::count_if(trace.begin(), trace.end(),
                          [](const TraceEvent &event) { return event.kind == TraceKind::Selected; }),
            0);
}

TEST_F(StatsContributionTest, NoncanonicalAndMismatchedOperandsDeclineToExplicitFallback) {
  keys();
  Runtime runtime;
  NativeRuntime native(runtime, services);
  registerStatsKernels(runtime, services, FHN_NATIVE_TEST_LIBRARY);
  auto x = native.importCopy(input);
  auto tensor = native.op(FHN_MULT_CC, {x, x});
  auto product = native.semantic(FHN_CIRCUIT_MULTIPLY, {x, x});
  auto sameLevel = native.op(FHN_LEVEL_DOWN, {x}, {services.fresh_level(native.context()) - 1, 0, 0, 0});
  auto lastLevel = native.op(FHN_LEVEL_DOWN, {x}, {0, 0, 0, 0});
  ASSERT_EQ(encrypt(client, input, data.data(), 2), 0);
  auto shorter = native.importCopy(input);
  FhnCkksBufferInfo before{}, after{};
  ASSERT_EQ(services.buffer_info(native.context(), native.buffer(sameLevel), &before), 0);
  ASSERT_EQ(services.buffer_info(native.context(), native.buffer(product), &after), 0);
  ASSERT_EQ(before.level, after.level);
  ASSERT_NE(before.scale, after.scale);
  int fallbacks = 0;
  runtime.addRecipe({"contrib.packed.dot", "test.metadata-fallback", -10,
                     [](const Request &, std::string &) { return true; },
                     [&](Runtime &, const Request &) {
                       ++fallbacks;
                       return Values{x};
                     }});
  auto noncanonical = runtime.submit({"contrib.packed.dot", {tensor, tensor}, std::uint32_t{4}}).at(0);
  auto mismatch = runtime.submit({"contrib.packed.dot", {x, product}, std::uint32_t{4}}).at(0);
  auto scaleMismatch = runtime.submit({"contrib.packed.dot", {sameLevel, product}, std::uint32_t{4}}).at(0);
  auto slotMismatch = runtime.submit({"contrib.packed.dot", {x, shorter}, std::uint32_t{2}}).at(0);
  auto exhausted = runtime.submit({"contrib.packed.dot", {lastLevel, lastLevel}, std::uint32_t{4}}).at(0);
  EXPECT_EQ(noncanonical.identity(), x.identity());
  EXPECT_EQ(mismatch.identity(), x.identity());
  EXPECT_EQ(scaleMismatch.identity(), x.identity());
  EXPECT_EQ(slotMismatch.identity(), x.identity());
  EXPECT_EQ(exhausted.identity(), x.identity());
  EXPECT_EQ(fallbacks, 5);
  const auto trace = runtime.trace();
  EXPECT_EQ(std::count_if(trace.begin(), trace.end(),
                          [](const TraceEvent &event) {
                            return event.operation == "contrib.packed.dot" && event.kind == TraceKind::Declined;
                          }),
            5);
}

// These fixtures expose cancellation around a zero variance result rather
// than permitting a relative tolerance to hide an absolute error.
TEST_F(StatsContributionTest, ConstantAndNearlyConstantInputsPreserveVarianceAccuracy) {
  keys();
  Runtime runtime;
  NativeRuntime native(runtime, services);
  registerStatsKernels(runtime, services, FHN_NATIVE_TEST_LIBRARY);
  const std::array<std::array<double, 4>, 2> fixtures{{{3, 3, 3, 3}, {3, 3.0001, 2.9999, 3}}};
  const std::array<double, 2> variances{0, 0.00000008};
  for (std::size_t i = 0; i < fixtures.size(); ++i) {
    ASSERT_EQ(encrypt(client, input, fixtures[i].data(), fixtures[i].size()), 0);
    auto x = native.importCopy(input);
    auto total = sum(native, x);
    auto result = runtime.submit({"contrib.stats.variance", {x, total}, std::uint32_t{4}}).at(0);
    expectValues(native, result, {variances[i], variances[i], variances[i], variances[i]});
  }
}

TEST_F(StatsContributionTest, WidthOneEnergyUsesTheSuppliedSumWithoutInferringItsMeaning) {
  keys();
  Runtime runtime;
  NativeRuntime native(runtime, services);
  registerStatsKernels(runtime, services, FHN_NATIVE_TEST_LIBRARY);
  auto x = native.importCopy(input);
  auto result = runtime.submit({"contrib.stats.energy", {x, x}, std::uint32_t{1}}).at(0);
  expectValues(native, result, {0, 0, 0, 0});
  auto twice = native.op(FHN_ADD_CC, {x, x});
  auto supplied = runtime.submit({"contrib.stats.variance", {x, twice}, std::uint32_t{1}}).at(0);
  expectValues(native, supplied, {0.25, 1.5625, 4, 0.015625});
}

TEST_F(StatsContributionTest, CenteredStatisticsDeclineWhenFewerThanThreeQPrimesRemain) {
  Runtime runtime;
  NativeRuntime native(runtime, services);
  registerStatsKernels(runtime, services, FHN_NATIVE_TEST_LIBRARY);
  auto x = native.importCopy(input);
  for (std::int64_t level : {0, 1}) {
    auto low = native.op(FHN_LEVEL_DOWN, {x}, {level, 0, 0, 0});
    const auto baseline = runtime.stats().live_bytes;
    expectError(runtime, {"contrib.stats.energy", {low, low}, std::uint32_t{4}}, ErrorCode::Unsupported);
    expectError(runtime, {"contrib.stats.variance", {low, low}, std::uint32_t{4}}, ErrorCode::Unsupported);
    EXPECT_EQ(runtime.stats().live_bytes, baseline);
  }
}

TEST_F(StatsContributionTest, MissingCapabilityDeclinesToExplicitFallback) {
  Runtime runtime;
  NativeRuntime native(runtime, services);
  registerStatsKernels(runtime, services, FHN_NON_COARSE_TEST_LIBRARY);
  auto x = native.importCopy(input);
  int fallbacks = 0;
  runtime.addRecipe({"contrib.packed.reduce", "test.capability-fallback", -10,
                     [](const Request &, std::string &) { return true; },
                     [&](Runtime &, const Request &) {
                       ++fallbacks;
                       return Values{x};
                     }});
  auto output = runtime.submit({"contrib.packed.reduce", {x}, std::uint32_t{4}}).at(0);
  EXPECT_EQ(output.identity(), x.identity());
  EXPECT_EQ(fallbacks, 1);
  const auto trace = runtime.trace();
  ASSERT_GE(trace.size(), 3u);
  EXPECT_EQ(trace[0].kind, TraceKind::Declined);
}

TEST_F(StatsContributionTest, ExpiredKeysFailOnceWithoutRetryAndReleaseStaging) {
  keys();
  Runtime runtime;
  NativeRuntime native(runtime, services);
  registerStatsKernels(runtime, services, FHN_NATIVE_TEST_LIBRARY);
  auto x = native.importCopy(input);
  auto total = sum(native, x);
  destroyClient(client);
  client = nullptr;
  int fallbacks = 0;
  runtime.addRecipe({"contrib.stats.variance", "test.execution-fallback", -10,
                     [](const Request &, std::string &) { return true; },
                     [&](Runtime &, const Request &) {
                       ++fallbacks;
                       return Values{x};
                     }});
  const auto baseline = runtime.stats().live_bytes;
  expectError(runtime, {"contrib.stats.variance", {x, total}, std::uint32_t{4}}, ErrorCode::ExecutionFailed);
  EXPECT_EQ(fallbacks, 0);
  EXPECT_EQ(runtime.stats().live_bytes, baseline);
  EXPECT_EQ(runtime.stats().live_resources, 2u);
  EXPECT_NO_THROW(native.op(FHN_ADD_CC, {x, x}));
}

TEST_F(StatsContributionTest, RecipesAndPublishedValueRetainContextAfterHostAndAdapterDie) {
  std::weak_ptr<FhnBackendCtx> context = services.context;
  const auto inspect = services.buffer_info;
  Value survivor;
  const FhnBuffer *escaped = nullptr;
  {
    Runtime runtime;
    Value x;
    {
      NativeRuntime native(runtime, services);
      x = native.importCopy(input);
      registerStatsKernels(runtime, services, FHN_NATIVE_TEST_LIBRARY);
    }
    releaseHost();
    services = {};
    dlclose(library);
    library = nullptr;
    survivor = runtime.submit({"contrib.packed.reduce", {x}, std::uint32_t{1}}).at(0);
    auto retained = context.lock();
    ASSERT_TRUE(retained);
    // NativeRuntime's public access is sufficient to inspect the published Value.
    NativeServices retainedServices = loadNativeServices(FHN_NATIVE_TEST_LIBRARY);
    retainedServices.context = retained;
    NativeRuntime reader(runtime, retainedServices);
    escaped = reader.buffer(survivor);
    EXPECT_EQ(runtime.stats().live_bytes, 2 * survivor.bytes());
  }
  ASSERT_FALSE(context.expired());
  FhnCkksBufferInfo metadata{};
  ASSERT_EQ(inspect(context.lock().get(), escaped, &metadata), 0);
  EXPECT_EQ(metadata.logical_slots, 4u);
  survivor = {};
  EXPECT_TRUE(context.expired());
}
} // namespace
