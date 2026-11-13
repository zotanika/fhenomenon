#include "FHN/contrib/MatvecKernels.h"
#include <algorithm>
#include <array>
#include <dlfcn.h>
#include <gtest/gtest.h>

namespace {
using namespace fhenomenon::runtime;
using fhenomenon::contrib::MatvecPrepareRequest;
using fhenomenon::contrib::registerMatvecKernels;

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

class MatvecContributionTest : public ::testing::Test {
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

TEST_F(MatvecContributionTest, SharedPreparationFeedsDotAndOrdinaryIntermediate) {
  keys();
  Runtime runtime;
  NativeRuntime native(runtime, services);
  registerMatvecKernels(runtime, services, FHN_NATIVE_TEST_LIBRARY);
  auto x = native.importCopy(input);
  auto prepared = runtime.submit({"contrib.matvec.prepare", {x}, MatvecPrepareRequest{4}}).at(0);
  auto result = runtime.submit({"contrib.matvec.dot", {x, x, prepared}, {}}).at(0);
  expectValues(native, result, {5.828125, 5.828125, 5.828125, 5.828125});
  auto row = native.op(FHN_ADD_CC, {x, x});
  auto second = runtime.submit({"contrib.matvec.dot", {x, row, prepared}, {}}).at(0);
  expectValues(native, second, {11.65625, 11.65625, 11.65625, 11.65625});
  auto product = runtime.submit({"contrib.matvec.product", {x, row, prepared}, {}}).at(0);
  expectValues(native, product, {0.5, 3.125, 8, 0.03125});
  auto ordinary = native.op(FHN_HMULT, {x, row});
  auto paired = native.op(FHN_ADD_CC, {ordinary, native.rotate(ordinary, 2)});
  auto reduced = runtime.submit({"contrib.matvec.reduce", {paired, prepared}, std::uint32_t{2}}).at(0);
  expectValues(native, reduced, {11.65625, 11.65625, 11.65625, 11.65625});
  FhnCkksBufferInfo a{}, b{};
  ASSERT_EQ(services.buffer_info(native.context(), native.buffer(reduced), &a), 0);
  ASSERT_EQ(services.buffer_info(native.context(), native.buffer(second), &b), 0);
  EXPECT_EQ(a.level, b.level);
  EXPECT_EQ(a.scale, b.scale);
  EXPECT_EQ(a.logical_slots, 4u);
  EXPECT_TRUE(reduced.dependencies().empty());
  EXPECT_EQ(prepared.kind(), ResourceKind::Artifact);
  ASSERT_EQ(prepared.dependencies().size(), 1u);
  EXPECT_EQ(prepared.dependencies()[0].identity(), x.identity());
}

TEST_F(MatvecContributionTest, SameShapedFreshSourceAndForgedArtifactDeclineBeforeAllocation) {
  keys();
  Runtime runtime;
  auto counted = countedServices();
  NativeRuntime native(runtime, counted);
  registerMatvecKernels(runtime, counted, FHN_NATIVE_TEST_LIBRARY);
  auto x = native.importCopy(input), fresh = native.importCopy(input);
  auto prep = runtime.submit({"contrib.matvec.prepare", {x}, MatvecPrepareRequest{4}}).at(0);
  const auto baseline = runtime.stats().live_bytes, allocated = allocations;
  expectError(runtime, {"contrib.matvec.dot", {fresh, x, prep}, {}}, ErrorCode::Unsupported);
  expectError(runtime, {"contrib.matvec.product", {fresh, x, prep}, {}}, ErrorCode::Unsupported);
  auto forged = runtime.makeValue(prep.format(), ResourceKind::Artifact, 0, [] { return std::any{73}; });
  expectError(runtime, {"contrib.matvec.dot", {x, x, forged}, {}}, ErrorCode::Unsupported);
  EXPECT_EQ(allocations, allocated);
  EXPECT_EQ(runtime.stats().live_bytes, baseline);
}
TEST_F(MatvecContributionTest, PreparationBudgetReservesExactSlabBeforeNativeSetup) {
  keys();
  auto required = resolve<FhnMatvecPreparationRequiredBytesFn>("ckks_fhn_matvec_preparation_required_bytes");
  ASSERT_TRUE(required);
  const auto reserve = required(services.context.get(), input, 4, nullptr);
  ASSERT_GT(reserve, 0u);
  Runtime runtime({bytes() + reserve - 1, 256});
  auto counted = countedServices();
  NativeRuntime native(runtime, counted);
  registerMatvecKernels(runtime, counted, FHN_NATIVE_TEST_LIBRARY);
  auto x = native.importCopy(input);
  expectError(runtime, {"contrib.matvec.prepare", {x}, MatvecPrepareRequest{4}}, ErrorCode::BudgetExceeded);
  EXPECT_EQ(runtime.stats().live_bytes, bytes());
  EXPECT_EQ(runtime.stats().peak_live_bytes, bytes());
  EXPECT_EQ(allocations, 1u);
}
TEST_F(MatvecContributionTest, StageAndPublicationBudgetsRollBackTemporaryCharges) {
  keys();
  auto required = resolve<FhnMatvecPreparationRequiredBytesFn>("ckks_fhn_matvec_preparation_required_bytes");
  ASSERT_TRUE(required);
  const auto reserve = required(services.context.get(), input, 4, nullptr);
  for (bool stage_failure : {true, false}) {
    Runtime runtime({2 * bytes() + reserve - (stage_failure ? 1 : 0), 256});
    auto counted = countedServices();
    NativeRuntime native(runtime, counted);
    registerMatvecKernels(runtime, counted, FHN_NATIVE_TEST_LIBRARY);
    auto x = native.importCopy(input);
    auto prep = runtime.submit({"contrib.matvec.prepare", {x}, MatvecPrepareRequest{4}}).at(0);
    EXPECT_EQ(prep.bytes(), reserve);
    const auto baseline = runtime.stats().live_bytes;
    expectError(runtime, {"contrib.matvec.dot", {x, x, prep}, {}}, ErrorCode::BudgetExceeded);
    EXPECT_EQ(runtime.stats().live_bytes, baseline);
    EXPECT_EQ(allocations, stage_failure ? 1u : 2u);
    EXPECT_EQ(frees, stage_failure ? 0u : 1u);
    expectValues(native, x, data);
  }
}
TEST_F(MatvecContributionTest, SuccessfulPublicationChargesBothBuffersAndRetainsSourceOnlyOnce) {
  keys();
  auto required = resolve<FhnMatvecPreparationRequiredBytesFn>("ckks_fhn_matvec_preparation_required_bytes");
  ASSERT_TRUE(required);
  const auto reserve = required(services.context.get(), input, 4, nullptr);
  Runtime runtime({3 * bytes() + reserve, 256});
  auto counted = countedServices();
  NativeRuntime native(runtime, counted);
  registerMatvecKernels(runtime, counted, FHN_NATIVE_TEST_LIBRARY);
  auto x = native.importCopy(input);
  auto prep = runtime.submit({"contrib.matvec.prepare", {x}, MatvecPrepareRequest{4}}).at(0);
  auto dot = runtime.submit({"contrib.matvec.dot", {x, x, prep}, {}}).at(0);
  EXPECT_EQ(runtime.stats().peak_live_bytes, 3 * bytes() + reserve);
  EXPECT_EQ(runtime.stats().live_bytes, 2 * bytes() + reserve);
  EXPECT_EQ(allocations, 3u);
  EXPECT_EQ(frees, 1u);
  dot = {};
  x = {};
  EXPECT_EQ(runtime.stats().live_bytes, bytes() + reserve);
  prep = {};
  EXPECT_EQ(runtime.stats().live_bytes, 0u);
}
TEST_F(MatvecContributionTest, ContextMismatchMalformedRequestsAndMissingOptionalApiDecline) {
  keys();
  Runtime runtime;
  NativeRuntime native(runtime, services);
  auto x = native.importCopy(input);
  auto other = loadNativeServices(FHN_NATIVE_TEST_LIBRARY);
  registerMatvecKernels(runtime, other, FHN_NATIVE_TEST_LIBRARY);
  registerMatvecKernels(runtime, services, FHN_NATIVE_TEST_LIBRARY);
  auto prep = runtime.submit({"contrib.matvec.prepare", {x}, MatvecPrepareRequest{4}}).at(0);
  ASSERT_FALSE(runtime.trace().empty());
  EXPECT_EQ(runtime.trace()[0].kind, TraceKind::Declined);
  for (uint32_t width : {0u, 3u, 8u})
    expectError(runtime, {"contrib.matvec.prepare", {x}, MatvecPrepareRequest{width}}, ErrorCode::Unsupported);
  expectError(runtime, {"contrib.matvec.prepare", {x}, MatvecPrepareRequest{4, 8}}, ErrorCode::Unsupported);
  expectError(runtime, {"contrib.matvec.prepare", {x}, uint32_t{4}}, ErrorCode::Unsupported);
  expectError(runtime, {"contrib.matvec.dot", {x, prep}, {}}, ErrorCode::Unsupported);
  auto product = runtime.submit({"contrib.matvec.product", {x, x, prep}, {}}).at(0);
  for (uint32_t width : {0u, 3u, 8u})
    expectError(runtime, {"contrib.matvec.reduce", {product, prep}, width}, ErrorCode::Unsupported);
  expectError(runtime, {"contrib.matvec.reduce", {x, prep}, uint32_t{4}}, ErrorCode::Unsupported);
  Runtime unavailable;
  NativeRuntime unavailableNative(unavailable, services);
  auto u = unavailableNative.importCopy(input);
  registerMatvecKernels(unavailable, services, FHN_NON_MATVEC_TEST_LIBRARY);
  int fallbacks = 0;
  unavailable.addRecipe({"contrib.matvec.prepare", "test.fallback", -1,
                         [](const Request &, std::string &) { return true; },
                         [&](Runtime &, const Request &) {
                           ++fallbacks;
                           return Values{u};
                         }});
  EXPECT_EQ(unavailable.submit({"contrib.matvec.prepare", {u}, MatvecPrepareRequest{4}})[0].identity(), u.identity());
  EXPECT_EQ(fallbacks, 1);
}
TEST_F(MatvecContributionTest, MismatchedCoreSymbolsDeclineBeforePassingOpaqueContextToPlugin) {
  Runtime runtime;
  NativeRuntime native(runtime, services);
  auto x = native.importCopy(input);
  auto mismatched = services;
  mismatched.fresh_level = [](FhnBackendCtx *) -> int64_t { return 2; };
  registerMatvecKernels(runtime, mismatched, FHN_NATIVE_TEST_LIBRARY);
  expectError(runtime, {"contrib.matvec.prepare", {x}, MatvecPrepareRequest{4}}, ErrorCode::Unsupported);
}
TEST_F(MatvecContributionTest, ExpiredKeysFailWithoutRetryOrLeakedStaging) {
  keys();
  Runtime runtime;
  auto counted = countedServices();
  NativeRuntime native(runtime, counted);
  registerMatvecKernels(runtime, counted, FHN_NATIVE_TEST_LIBRARY);
  auto x = native.importCopy(input);
  auto prep = runtime.submit({"contrib.matvec.prepare", {x}, MatvecPrepareRequest{4}}).at(0);
  destroyClient(client);
  client = nullptr;
  int fallbacks = 0;
  runtime.addRecipe({"contrib.matvec.dot", "test.fallback", -1, [](const Request &, std::string &) { return true; },
                     [&](Runtime &, const Request &) {
                       ++fallbacks;
                       return Values{x};
                     }});
  const auto baseline = runtime.stats().live_bytes;
  expectError(runtime, {"contrib.matvec.dot", {x, x, prep}, {}}, ErrorCode::ExecutionFailed);
  EXPECT_EQ(runtime.stats().live_bytes, baseline);
  EXPECT_EQ(fallbacks, 0);
  EXPECT_EQ(allocations, 2u);
  EXPECT_EQ(frees, 1u);
}
TEST_F(MatvecContributionTest, ArtifactOutlivesSourceHostRuntimeAndLibraryWithoutRetainingClient) {
  keys();
  std::weak_ptr<FhnBackendCtx> context = services.context;
  Value survivor;
  {
    Runtime runtime;
    {
      NativeRuntime native(runtime, services);
      registerMatvecKernels(runtime, services, FHN_NATIVE_TEST_LIBRARY);
      auto x = native.importCopy(input);
      survivor = runtime.submit({"contrib.matvec.prepare", {x}, MatvecPrepareRequest{4}}).at(0);
      x = {};
      EXPECT_EQ(runtime.stats().live_bytes, survivor.bytes() + bytes());
    }
    releaseHost();
    services = {};
    dlclose(library);
    library = nullptr;
    EXPECT_FALSE(context.expired());
  }
  EXPECT_FALSE(context.expired());
  ASSERT_EQ(survivor.dependencies().size(), 1u);
  EXPECT_TRUE(survivor.dependencies()[0]);
  survivor = {};
  EXPECT_TRUE(context.expired());
}
} // namespace
