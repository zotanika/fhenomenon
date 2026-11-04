#include "FHN/contrib/SpectralKernels.h"
#include <array>
#include <complex>
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <memory>

namespace {
using namespace fhenomenon::runtime;
using namespace fhenomenon::contrib;
class SpectralContributionTest : public ::testing::Test {
  protected:
  void *library = nullptr;
  NativeServices services;
  FhnClientCtx *client = nullptr;
  FhnClientDestroyFn destroy = nullptr;
  FhnClientEncryptVectorComplex64Fn encrypt = nullptr;
  FhnClientDecryptVectorComplex64Fn decrypt = nullptr;
  template <class T> T symbol(const char *name) { return reinterpret_cast<T>(dlsym(library, name)); }
  void SetUp() override {
    library = dlopen(FHN_NATIVE_TEST_LIBRARY, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(library, nullptr);
    services = loadNativeServices(FHN_NATIVE_TEST_LIBRARY);
    services.compile = nullptr;
    services.run = nullptr;
    services.executable_stats = nullptr;
    services.executable_free = nullptr;
    auto create = symbol<FhnClientCreateFn>("ckks_fhn_client_create");
    destroy = symbol<FhnClientDestroyFn>("ckks_fhn_client_destroy");
    encrypt = symbol<FhnClientEncryptVectorComplex64Fn>("ckks_fhn_client_encrypt_vec_c64");
    decrypt = symbol<FhnClientDecryptVectorComplex64Fn>("ckks_fhn_client_decrypt_vec_c64");
    ASSERT_TRUE(create && destroy && encrypt && decrypt);
    client = create(services.context.get());
    ASSERT_NE(client, nullptr);
    auto prepare = symbol<FhnClientPrepareFn>("ckks_fhn_client_prepare");
    const auto requirements = nativeSpectralKeyRequirements();
    ASSERT_TRUE(requirements.multiplication);
    ASSERT_TRUE(requirements.conjugation);
    std::array<FhnInstruction, 2> key_requests{};
    key_requests[0].opcode = FHN_HMULT;
    key_requests[1].opcode = FHN_CONJUGATE;
    FhnProgram evaluation_key_request{
      FHN_ABI_VERSION, static_cast<std::uint32_t>(key_requests.size()), key_requests.data(), 0, nullptr, 0, nullptr};
    ASSERT_EQ(prepare(client, &evaluation_key_request), 0);
  }
  void TearDown() override {
    if (client)
      destroy(client);
    services = {};
    if (library)
      dlclose(library);
  }
  Value value(NativeRuntime &native, const std::vector<FhnComplex64> &data) {
    auto release = [&](FhnBuffer *ptr) { services.buffer_free(services.context.get(), ptr); };
    std::unique_ptr<FhnBuffer, decltype(release)> buffer(services.buffer_alloc(services.context.get()), release);
    if (!buffer)
      throw std::runtime_error("test allocation failed");
    if (encrypt(client, buffer.get(), data.data(), data.size()))
      throw std::runtime_error("test encryption failed");
    return native.importCopy(buffer.get());
  }
  std::vector<FhnComplex64> decode(NativeRuntime &native, const Value &v) {
    std::vector<FhnComplex64> result(8);
    EXPECT_EQ(decrypt(client, native.buffer(v), result.data(), result.size()), 0);
    return result;
  }
  void rejects(Runtime &rt, const Request &request, ErrorCode code = ErrorCode::Unsupported) {
    try {
      rt.submit(request);
      FAIL() << "expected rejection";
    } catch (const Error &e) {
      EXPECT_EQ(e.code(), code) << e.what();
    }
  }
  bool selected(const Runtime &rt, const std::string &name) {
    for (const auto &e : rt.trace())
      if (e.kind == TraceKind::Selected && e.recipe == name)
        return true;
    return false;
  }
};

void expectProducts(const std::vector<FhnComplex64> &result, const std::vector<FhnComplex64> &sensor,
                    const std::vector<FhnComplex64> &ref, double factor = 1.0) {
  ASSERT_EQ(result.size(), sensor.size());
  ASSERT_EQ(result.size(), ref.size());
  for (std::size_t i = 0; i < result.size(); ++i) {
    const auto expected = factor * std::complex(sensor[i].real, sensor[i].imag) *
                          std::conj(std::complex(ref[i].real, ref[i].imag));
    EXPECT_NEAR(result[i].real, expected.real(), 1e-5) << i;
    EXPECT_NEAR(result[i].imag, expected.imag(), 1e-5) << i;
  }
}
const std::vector<FhnComplex64> reference{{1, 2}, {2, -1}, {0, -1}, {3, 0}, {-2, 3}, {0, 1}, {1, 0}, {1, -2}};
const std::vector<FhnComplex64> sensor{{3, -1}, {0, 2}, {-3, 1}, {2, -4}, {1, 1}, {1, -3}, {2, 1}, {0, -1}};

TEST_F(SpectralContributionTest, ReusesPreparedReferenceAcrossScopesAndExportsAllLiveouts) {
  Runtime rt;
  NativeRuntime native(rt, services);
  registerSpectralKernels(rt, services, FHN_NATIVE_TEST_LIBRARY);
  SpectralRuntime spectral(rt);
  auto r = value(native, reference);
  auto x = value(native, sensor);
  auto outputs = rt.runScope([&](Scope &scope) {
    auto ref = scope.input(r);
    auto [conjugate, preparation] = spectral.prepareReference(scope, ref);
    auto cross = spectral.pair(scope, scope.input(x), preparation);
    auto power = spectral.pair(scope, ref, preparation);
    auto sum = native.op(scope, FHN_ADD_CC, {cross, cross});
    return ScopeValues{power, conjugate, sum, cross, preparation};
  });
  ASSERT_EQ(outputs.size(), 5u);
  EXPECT_TRUE(selected(rt, "contrib.spectral.scope.pair-tile"));
  EXPECT_TRUE(selected(rt, "contrib.matvec.product.native"));
  expectProducts(decode(native, outputs[0]), reference, reference);
  expectProducts(decode(native, outputs[2]), sensor, reference, 2);
  expectProducts(decode(native, outputs[3]), sensor, reference);
  auto manual = native.op(FHN_HMULT, {x, outputs[1]});
  expectProducts(decode(native, manual), sensor, reference);
  auto reused = rt.runScope([&](Scope &scope) {
    return ScopeValues{spectral.pair(scope, scope.input(x), scope.input(outputs[4]))};
  });
  EXPECT_TRUE(selected(rt, "contrib.spectral.scope.pair"));
  expectProducts(decode(native, reused.at(0)), sensor, reference);
}

TEST_F(SpectralContributionTest, HonorsOverridesAndSupportsOrdinaryFallback) {
  Runtime rt;
  NativeRuntime native(rt, services);
  registerSpectralKernels(rt, services, "");
  SpectralRuntime spectral(rt);
  auto x = value(native, sensor);
  const auto prepared = spectral.prepareReference(value(native, reference));
  const auto &conjugate = prepared.first;
  const auto &preparation = prepared.second;
  expectProducts(decode(native, spectral.pair(x, preparation)), sensor, reference);
  EXPECT_TRUE(selected(rt, "contrib.spectral.pair.ordinary"));
  ScopeOptions baseline;
  baseline.use_kernels = false;
  const auto ordinary = rt.runScope([&](Scope &scope) {
    return ScopeValues{spectral.pair(scope, scope.input(x), scope.input(preparation))};
  }, baseline);
  expectProducts(decode(native, ordinary.at(0)), sensor, reference);
  rt.addRecipe({"contrib.spectral.pair", "test.pair.override", 100,
                [](const Request &, std::string &) { return true; },
                [](Runtime &, const Request &r) { return Values{r.inputs[0]}; }});
  for (bool useKernels : {true, false}) {
    ScopeOptions options;
    options.use_kernels = useKernels;
    const auto out = rt.runScope([&](Scope &scope) {
      auto ref = scope.input(preparation);
      auto first = spectral.pair(scope, scope.input(x), ref);
      auto second = spectral.pair(scope, scope.input(conjugate), ref);
      return ScopeValues{second, first};
    }, options);
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].identity(), conjugate.identity());
    EXPECT_EQ(out[1].identity(), x.identity());
  }
  EXPECT_TRUE(selected(rt, "contrib.spectral.scope.pair-tile"));
  EXPECT_TRUE(selected(rt, "test.pair.override"));
}

TEST_F(SpectralContributionTest, PropagatesSelectedMatvecPreparationFailure) {
  Runtime rt;
  NativeRuntime native(rt, services);
  registerSpectralKernels(rt, services, FHN_NATIVE_TEST_LIBRARY);
  rt.addRecipe({"contrib.matvec.prepare", "test.matvec.prepare.failure", 100,
                [](const Request &, std::string &) { return true; },
                [](Runtime &, const Request &) -> Values {
                  throw Error(ErrorCode::Unsupported, "selected Matvec preparation failed");
                }});
  const auto source = value(native, reference);
  const auto before = rt.stats();
  rejects(rt, {"contrib.spectral.prepare-reference", {source}, {}});
  EXPECT_TRUE(selected(rt, "test.matvec.prepare.failure"));
  EXPECT_EQ(rt.stats().live_resources, before.live_resources);
  EXPECT_EQ(rt.stats().live_bytes, before.live_bytes);
}

TEST_F(SpectralContributionTest, RejectsForeignAndForgedPreparation) {
  Runtime rt;
  NativeRuntime native(rt, services);
  registerSpectralKernels(rt, services, FHN_NATIVE_TEST_LIBRARY);
  SpectralRuntime spectral(rt);
  auto x = value(native, sensor);
  const auto prepared = spectral.prepareReference(x);
  const auto &conjugate = prepared.first;
  const auto &preparation = prepared.second;
  auto forged = rt.makeValue(preparation.format(), ResourceKind::Artifact, 0,
                             [&] { return preparation.payload(); }, preparation.dependencies());
  rejects(rt, {"contrib.spectral.pair", {x, forged}, {}});
  rejects(rt, {"contrib.spectral.pair", {x, conjugate}, {}});
  Runtime other;
  NativeRuntime otherNative(other, services);
  registerSpectralKernels(other, services, FHN_NATIVE_TEST_LIBRARY);
  auto otherX = value(otherNative, sensor);
  auto foreignContext = loadNativeServices(FHN_NATIVE_TEST_LIBRARY);
  NativeRuntime foreignNative(rt, foreignContext);
  auto release = [&](FhnBuffer *ptr) { foreignContext.buffer_free(foreignContext.context.get(), ptr); };
  std::unique_ptr<FhnBuffer, decltype(release)> foreignBuffer(foreignContext.buffer_alloc(foreignContext.context.get()),
                                                            release);
  ASSERT_TRUE(foreignBuffer);
  auto create = symbol<FhnClientCreateFn>("ckks_fhn_client_create");
  std::unique_ptr<FhnClientCtx, FhnClientDestroyFn> foreignClient(create(foreignContext.context.get()), destroy);
  ASSERT_TRUE(foreignClient);
  ASSERT_EQ(encrypt(foreignClient.get(), foreignBuffer.get(), sensor.data(), sensor.size()), 0);
  auto foreignValue = foreignNative.importCopy(foreignBuffer.get());
  rejects(rt, {"contrib.spectral.pair", {foreignValue, preparation}, {}});
  rejects(rt, {"contrib.spectral.prepare-reference", {foreignValue}, {}});
  rejects(other, {"contrib.spectral.pair", {otherX, preparation}, {}}, ErrorCode::InvalidRequest);
  EXPECT_THROW(other.runScope([&](Scope &scope) {
    return ScopeValues{spectral.prepareReference(scope, scope.input(otherX)).first};
  }), Error);
}
} // namespace
