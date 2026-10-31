#include "FHN/NativeRuntime.h"
#include <array>
#include <dlfcn.h>
#include <gtest/gtest.h>

#ifndef FHN_NATIVE_TEST_LIBRARY
#error "FHN_NATIVE_TEST_LIBRARY must name the native backend built by CMake"
#endif

namespace {
using namespace fhenomenon::runtime;
class NativeRuntimeTest : public ::testing::Test {
  protected:
  void *library = nullptr;
  NativeServices services;
  FhnClientCtx *client = nullptr;
  FhnBuffer *input = nullptr;
  FhnClientDestroyFn destroyClient = nullptr;
  FhnClientPrepareFn prepare = nullptr;
  FhnClientEncryptVectorDoubleFn encrypt = nullptr;
  FhnClientDecryptVectorDoubleFn decrypt = nullptr;
  FhnClientPrepareExecutableFn prepareExecutable = nullptr;
  const std::array<double, 4> data{0.5, -1.25, 2, 0.125};
  template <class T> T resolve(const char *name) { return reinterpret_cast<T>(dlsym(library, name)); }
  void SetUp() override {
    library = dlopen(FHN_NATIVE_TEST_LIBRARY, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(library, nullptr) << dlerror();
    services = loadNativeServices(FHN_NATIVE_TEST_LIBRARY);
    auto create = resolve<FhnClientCreateFn>("ckks_fhn_client_create");
    destroyClient = resolve<FhnClientDestroyFn>("ckks_fhn_client_destroy");
    prepare = resolve<FhnClientPrepareFn>("ckks_fhn_client_prepare");
    encrypt = resolve<FhnClientEncryptVectorDoubleFn>("ckks_fhn_client_encrypt_vec_f64");
    decrypt = resolve<FhnClientDecryptVectorDoubleFn>("ckks_fhn_client_decrypt_vec_f64");
    prepareExecutable = resolve<FhnClientPrepareExecutableFn>("ckks_fhn_client_prepare_executable");
    ASSERT_TRUE(create && destroyClient && prepare && encrypt && decrypt && prepareExecutable);
    client = create(services.context.get());
    ASSERT_NE(client, nullptr);
    input = services.buffer_alloc(services.context.get());
    ASSERT_NE(input, nullptr);
    ASSERT_EQ(encrypt(client, input, data.data(), data.size()), 0);
  }
  void TearDown() override {
    if (input)
      services.buffer_free(services.context.get(), input);
    if (client)
      destroyClient(client);
    services = {};
    if (library)
      dlclose(library);
  }
  void keys(std::initializer_list<std::pair<FhnOpCode, std::int64_t>> operations) {
    std::vector<FhnInstruction> instructions;
    for (auto pair : operations) {
      FhnInstruction instruction{};
      instruction.opcode = pair.first;
      instruction.params[0] = pair.second;
      instructions.push_back(instruction);
    }
    FhnProgram p{FHN_ABI_VERSION, static_cast<std::uint32_t>(instructions.size()), instructions.data(), 0, nullptr, 0,
                 nullptr};
    ASSERT_EQ(prepare(client, &p), 0);
  }
  std::array<double, 4> values(const FhnBuffer *buffer) {
    std::array<double, 4> result{};
    EXPECT_EQ(decrypt(client, buffer, result.data(), result.size()), 0);
    return result;
  }
  void expectValues(NativeRuntime &native, const Value &v, const std::array<double, 4> &expected,
                    double tolerance = 1e-5) {
    auto actual = values(native.buffer(v));
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i], tolerance);
  }
};
TEST_F(NativeRuntimeTest, EagerPhysicalNeedsNoCompilerAndImportIsSnapshot) {
  auto computeOnly = services;
  computeOnly.compile = nullptr;
  computeOnly.run = nullptr;
  Runtime rt;
  NativeRuntime native(rt, computeOnly);
  auto x = native.importCopy(input);
  auto sum = native.op(FHN_ADD_CC, {x, x});
  auto tensor = native.op(FHN_MULT_CC, {x, x});
  FhnCkksBufferInfo metadata{};
  ASSERT_EQ(services.buffer_info(native.context(), native.buffer(tensor), &metadata), 0);
  EXPECT_EQ(metadata.components, 3u);
  EXPECT_EQ(metadata.level, services.fresh_level(native.context()));
  expectValues(native, sum, {1, -2.5, 4, 0.25});
  const std::array<double, 4> changed{3, 3, 3, 3};
  ASSERT_EQ(encrypt(client, input, changed.data(), changed.size()), 0);
  expectValues(native, x, data);
  EXPECT_EQ(x.bytes(), services.level_bytes(native.context(), services.fresh_level(native.context())));
  EXPECT_TRUE(x.dependencies().empty());
}
TEST_F(NativeRuntimeTest, MathematicalAdapterPreservesIntegerLevelAndNormalizesProducts) {
  keys({{FHN_HMULT, 0}});
  Runtime rt;
  NativeRuntime native(rt, services);
  auto x = native.importCopy(input);
  auto integer = native.semantic(FHN_CIRCUIT_SCALE, {x}, 0, -3);
  expectValues(native, integer, {-1.5, 3.75, -6, -0.375});
  FhnCkksBufferInfo metadata{};
  ASSERT_EQ(services.buffer_info(native.context(), native.buffer(integer), &metadata), 0);
  EXPECT_EQ(metadata.level, services.fresh_level(native.context()));
  auto product = native.semantic(FHN_CIRCUIT_MULTIPLY, {x, x});
  expectValues(native, product, {0.25, 1.5625, 4, 0.015625});
  // Product scales differ from fresh scales; never relabel them to force a join.
  EXPECT_THROW(native.semantic(FHN_CIRCUIT_ADD, {x, product}), Error);
  auto fraction = native.semantic(FHN_CIRCUIT_SCALE, {x}, 0, 0.5);
  expectValues(native, fraction, {0.25, -0.625, 1, 0.0625});
  auto mixed = native.semantic(FHN_CIRCUIT_ADD, {x, fraction});
  expectValues(native, mixed, {0.75, -1.875, 3, 0.1875});
}
TEST_F(NativeRuntimeTest, PreparedCommandsShareSourceAndRejectFreshIdentity) {
  ASSERT_TRUE(services.prepare_rotation && services.preparation_required_bytes);
  keys({{FHN_ROTATE, 1}, {FHN_ROTATE, -1}});
  Runtime rt;
  NativeRuntime native(rt, services);
  auto x = native.importCopy(input);
  auto preparation = native.prepareRotation(x);
  EXPECT_EQ(preparation.kind(), ResourceKind::Artifact);
  EXPECT_EQ(preparation.dependencies().at(0).identity(), x.identity());
  EXPECT_EQ(preparation.bytes(), services.preparation_required_bytes(native.context(), native.buffer(x)));
  auto left = native.rotate(x, 1, preparation), right = native.rotate(x, -1, preparation);
  expectValues(native, left, {-1.25, 2, 0.125, 0.5});
  expectValues(native, right, {0.125, 0.5, -1.25, 2});
  auto fresh = native.importCopy(input);
  EXPECT_THROW(native.rotate(fresh, 1, preparation), Error);
  const auto before = rt.stats().live_bytes;
  destroyClient(client);
  client = nullptr;
  EXPECT_THROW(native.rotate(x, 1, preparation), Error);
  EXPECT_EQ(rt.stats().live_bytes, before);
  EXPECT_NO_THROW(native.op(FHN_ADD_CC, {x, x}));
}
TEST_F(NativeRuntimeTest, RegisteredRecipesAndResourcesOutliveAdapterAndRuntime) {
  ASSERT_TRUE(services.prepare_rotation);
  keys({{FHN_ROTATE, 1}});
  Value survivor;
  const FhnBuffer *escaped = nullptr;
  {
    Runtime rt;
    Value source;
    {
      NativeRuntime native(rt, services);
      source = native.importCopy(input);
      survivor = native.op(FHN_ADD_CC, {source, source});
      escaped = native.buffer(survivor);
    }
    auto preparation = rt.submit({"native.prepare_rotation", {source}, {}}).at(0);
    EXPECT_EQ(preparation.dependencies().at(0).identity(), source.identity());
    EXPECT_NO_THROW(rt.submit({"native.rotate", {source, preparation}, std::int64_t{1}}));
  }
  const auto actual = values(escaped);
  for (std::size_t i = 0; i < data.size(); ++i)
    EXPECT_NEAR(actual[i], 2 * data[i], 1e-5);
  survivor = {};
}
TEST_F(NativeRuntimeTest, DifferentContextAndRuntimeHandlesAreRejected) {
  Runtime a, b;
  NativeRuntime na(a, services), nb(b, services);
  auto x = na.importCopy(input);
  EXPECT_THROW(nb.op(FHN_ADD_CC, {x, x}), Error);
  auto other = loadNativeServices(FHN_NATIVE_TEST_LIBRARY);
  NativeRuntime foreign(a, other);
  EXPECT_THROW(foreign.buffer(x), Error);
  EXPECT_THROW(foreign.op(FHN_ADD_CC, {x, x}), Error);
}
TEST_F(NativeRuntimeTest, ExplicitRegionRebindsEagerInputsAndProducesReusableMultipleOutputs) {
  Runtime rt;
  NativeRuntime native(rt, services);
  auto x = native.importCopy(input);
  auto eager = native.op(FHN_ADD_CC, {x, x});
  std::array<FhnCircuitNode, 2> nodes{};
  nodes[0].op = FHN_CIRCUIT_ADD;
  nodes[0].result_id = 2;
  nodes[0].operands[0] = 1;
  nodes[0].operands[1] = 1;
  nodes[1].op = FHN_CIRCUIT_NEGATE;
  nodes[1].result_id = 3;
  nodes[1].operands[0] = 1;
  std::uint32_t inputId = 1;
  std::array<std::uint32_t, 2> outputIds{2, 3};
  FhnCircuit circuit{FHN_CIRCUIT_VERSION, 2, nodes.data(), 1, &inputId, 2, outputIds.data()};
  const auto before = rt.stats().live_bytes;
  auto region = native.compileRegion(circuit, {eager});
  EXPECT_EQ(region.bytes(), RegionOptions{}.workspace_bytes);
  EXPECT_TRUE(region.dependencies().empty());
  EXPECT_EQ(rt.stats().live_bytes, before + region.bytes());
  ASSERT_EQ(prepareExecutable(client, native.executable(region)), 0);
  auto results = native.runRegion(region, {eager});
  ASSERT_EQ(results.size(), 2u);
  expectValues(native, results[0], {2, -5, 8, 0.5});
  expectValues(native, results[1], {-1, 2.5, -4, -0.25});
  expectValues(native, native.op(FHN_ADD_CC, {results[0], results[1]}), {1, -2.5, 4, 0.25});
  auto fresh = native.importCopy(input);
  expectValues(native, native.runRegion(region, {fresh}).at(0), {1, -2.5, 4, 0.25});
  EXPECT_GT(native.executableStats(region).storage_bytes, 0u);
}
TEST_F(NativeRuntimeTest, MultiOutputBudgetFailureReleasesUnpublishedBuffers) {
  const auto bytes = services.level_bytes(services.context.get(), services.fresh_level(services.context.get()));
  Runtime rt({static_cast<std::size_t>(2 * bytes) + RegionOptions{}.workspace_bytes, 256});
  NativeRuntime native(rt, services);
  auto x = native.importCopy(input);
  std::array<FhnCircuitNode, 2> nodes{};
  nodes[0].op = FHN_CIRCUIT_NEGATE;
  nodes[0].result_id = 2;
  nodes[0].operands[0] = 1;
  nodes[1].op = FHN_CIRCUIT_ADD;
  nodes[1].result_id = 3;
  nodes[1].operands[0] = 1;
  nodes[1].operands[1] = 1;
  std::uint32_t in = 1;
  std::array<std::uint32_t, 2> out{2, 3};
  FhnCircuit circuit{FHN_CIRCUIT_VERSION, 2, nodes.data(), 1, &in, 2, out.data()};
  auto region = native.compileRegion(circuit, {x});
  ASSERT_EQ(prepareExecutable(client, native.executable(region)), 0);
  auto before = rt.stats();
  EXPECT_THROW(native.runRegion(region, {x}), Error);
  EXPECT_EQ(rt.stats().live_bytes, before.live_bytes);
  EXPECT_EQ(rt.stats().live_resources, before.live_resources);
  expectValues(native, x, data);
}
TEST_F(NativeRuntimeTest, ResourceLeaseKeepsSoleContextOwnerAlive) {
  auto isolated = loadNativeServices(FHN_NATIVE_TEST_LIBRARY);
  auto create = resolve<FhnClientCreateFn>("ckks_fhn_client_create");
  auto isolatedClient = create(isolated.context.get());
  ASSERT_NE(isolatedClient, nullptr);
  auto source = isolated.buffer_alloc(isolated.context.get());
  ASSERT_NE(source, nullptr);
  ASSERT_EQ(encrypt(isolatedClient, source, data.data(), data.size()), 0);
  std::weak_ptr<FhnBackendCtx> weak = isolated.context;
  auto diagnostics = isolated.buffer_info;
  const auto context = isolated.context.get();
  Value survivor;
  const FhnBuffer *escaped = nullptr;
  {
    Runtime rt;
    NativeRuntime native(rt, isolated);
    survivor = native.importCopy(source);
    escaped = native.buffer(survivor);
  }
  isolated.buffer_free(context, source);
  destroyClient(isolatedClient);
  isolated = {};
  EXPECT_FALSE(weak.expired());
  FhnCkksBufferInfo metadata{};
  EXPECT_EQ(diagnostics(context, escaped, &metadata), 0);
  EXPECT_EQ(metadata.logical_slots, 4u);
  survivor = {};
  EXPECT_TRUE(weak.expired());
}
TEST_F(NativeRuntimeTest, FailedOrOversizedRegionsLeaveEagerStateUsable) {
  Runtime rt;
  NativeRuntime native(rt, services);
  auto x = native.importCopy(input);
  std::uint32_t inputId = 1, outputId = 2;
  FhnCircuitNode node{};
  node.op = FHN_CIRCUIT_NEGATE;
  node.result_id = 2;
  node.operands[0] = 1;
  FhnCircuit circuit{FHN_CIRCUIT_VERSION, 1, &node, 1, &inputId, 1, &outputId};
  const auto before = rt.stats().live_bytes;
  EXPECT_THROW(native.compileRegion(circuit, {x}, {0, 1024}), Error);
  EXPECT_THROW(native.compileRegion(circuit, {x}, {256, 1}), Error);
  EXPECT_EQ(rt.stats().live_bytes, before);
  expectValues(native, native.op(FHN_ADD_CC, {x, x}), {1, -2.5, 4, 0.25});
}
static FhnCompileCircuitFn actualCompile;
static FhnBufferAllocFn actualAlloc;
static FhnPrepareRotationSourceFn actualPrepare;
static int compiles, allocations, preparations;
FhnExecutable *countedCompile(FhnBackendCtx *c, const FhnCircuit *g, const FhnBuffer *const *p, uint32_t n,
                              const FhnCompileOptions *o, char *e, uint32_t s) {
  ++compiles;
  return actualCompile(c, g, p, n, o, e, s);
}
FhnBuffer *countedAlloc(FhnBackendCtx *c) {
  ++allocations;
  return actualAlloc(c);
}
FhnRotationPreparation *countedPrepare(FhnBackendCtx *c, const FhnBuffer *b) {
  ++preparations;
  return actualPrepare(c, b);
}
FhnKernelTable *noKernels(FhnBackendCtx *) { return nullptr; }
TEST_F(NativeRuntimeTest, UnavailableFirstNativeProviderDeclinesBeforeCompleteProviderExecutes) {
  auto unavailable = services;
  unavailable.get_kernels = noKernels;
  unavailable.compile = nullptr;
  unavailable.run = nullptr;
  unavailable.prepare_rotation = nullptr;
  unavailable.rotate_prepared = nullptr;
  Runtime rt;
  NativeRuntime first(rt, unavailable), complete(rt, services);
  auto x = complete.importCopy(input);
  expectValues(complete, complete.op(FHN_ADD_CC, {x, x}), {1, -2.5, 4, 0.25});
  keys({{FHN_ROTATE, 1}});
  auto preparation = complete.prepareRotation(x);
  expectValues(complete, complete.rotate(x, 1, preparation), {-1.25, 2, 0.125, 0.5});
  std::uint32_t in = 1, out = 2;
  FhnCircuitNode node{};
  node.op = FHN_CIRCUIT_NEGATE;
  node.result_id = 2;
  node.operands[0] = 1;
  FhnCircuit circuit{FHN_CIRCUIT_VERSION, 1, &node, 1, &in, 1, &out};
  auto region = complete.compileRegion(circuit, {x});
  ASSERT_EQ(prepareExecutable(client, complete.executable(region)), 0);
  expectValues(complete, complete.runRegion(region, {x}).at(0), {-0.5, 1.25, -2, -0.125});
  std::size_t declines = 0;
  for (const auto &event : rt.trace())
    if (event.kind == TraceKind::Declined)
      ++declines;
  EXPECT_EQ(declines, 5u); // op, prepare, rotate, compile and run each skipped unavailable provider
  const auto before = rt.stats().live_bytes;
  EXPECT_THROW(complete.compileRegion(circuit, {x}, {0, 1024}), Error);
  EXPECT_EQ(rt.stats().live_bytes, before);
  EXPECT_EQ(rt.trace().back().kind, TraceKind::Failed);
  auto trace = rt.trace();
  EXPECT_EQ(trace[trace.size() - 2].kind, TraceKind::Declined);
  expectValues(complete, complete.op(FHN_ADD_CC, {x, x}), {1, -2.5, 4, 0.25});
}
TEST_F(NativeRuntimeTest, ReservationsRejectBeforeNativeFactories) {
  auto counted = services;
  actualCompile = services.compile;
  actualAlloc = services.buffer_alloc;
  actualPrepare = services.prepare_rotation;
  counted.compile = countedCompile;
  counted.buffer_alloc = countedAlloc;
  counted.prepare_rotation = countedPrepare;
  compiles = allocations = preparations = 0;
  const auto bytes = services.level_bytes(services.context.get(), services.fresh_level(services.context.get()));
  Runtime rt({static_cast<std::size_t>(bytes), 256});
  NativeRuntime native(rt, counted);
  auto x = native.importCopy(input);
  EXPECT_EQ(allocations, 1);
  EXPECT_THROW(native.op(FHN_ADD_CC, {x, x}), Error);
  EXPECT_EQ(allocations, 1);
  if (services.prepare_rotation) {
    EXPECT_THROW(native.prepareRotation(x), Error);
    EXPECT_EQ(preparations, 0);
  }
  std::uint32_t in = 1, out = 2;
  FhnCircuitNode node{};
  node.op = FHN_CIRCUIT_NEGATE;
  node.result_id = 2;
  node.operands[0] = 1;
  FhnCircuit circuit{FHN_CIRCUIT_VERSION, 1, &node, 1, &in, 1, &out};
  EXPECT_THROW(native.compileRegion(circuit, {x}), Error);
  EXPECT_EQ(compiles, 0);
  EXPECT_EQ(rt.stats().live_bytes, bytes);
}
} // namespace
