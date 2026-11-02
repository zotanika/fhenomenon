#include "FHN/contrib/MatvecKernels.h"
#include "FHN/contrib/SearchKernels.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <limits>
#include <vector>
namespace {
using namespace fhenomenon::runtime;
using namespace fhenomenon::contrib;
TEST(SearchKeyRequirements, CoversNativeMultiplicationAndEveryReductionRotation) {
  auto single = nativeSearchKeyRequirements(1);
  EXPECT_TRUE(single.multiplication);
  EXPECT_TRUE(single.rotation_steps.empty());
  EXPECT_EQ(nativeSearchKeyRequirements(2).rotation_steps, (std::vector<std::int64_t>{1}));
  EXPECT_EQ(nativeSearchKeyRequirements(4).rotation_steps, (std::vector<std::int64_t>{1, 2}));
  auto search = nativeSearchKeyRequirements(512);
  EXPECT_TRUE(search.multiplication);
  EXPECT_EQ(search.rotation_steps, (std::vector<std::int64_t>{1, 2, 4, 8, 16, 32, 64, 128, 256}));
  auto largest = nativeSearchKeyRequirements(std::uint32_t{1} << 31);
  ASSERT_EQ(largest.rotation_steps.size(), 31u);
  EXPECT_EQ(largest.rotation_steps.back(), std::int64_t{1} << 30);
}
TEST(SearchKeyRequirements, RejectsInvalidReductionWidths) {
  for (auto width : {0u, 3u, 6u, std::numeric_limits<std::uint32_t>::max()}) {
    try {
      nativeSearchKeyRequirements(width);
      FAIL() << "invalid width accepted: " << width;
    } catch (const Error &error) {
      EXPECT_EQ(error.code(), ErrorCode::InvalidRequest);
    }
  }
}
class SearchContributionTest : public ::testing::Test {
  protected:
  void *library = nullptr;
  NativeServices services;
  FhnClientCtx *client = nullptr;
  FhnClientDestroyFn destroy = nullptr;
  FhnClientEncryptVectorDoubleFn encrypt = nullptr;
  FhnClientDecryptVectorDoubleFn decrypt = nullptr;
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
    encrypt = symbol<FhnClientEncryptVectorDoubleFn>("ckks_fhn_client_encrypt_vec_f64");
    decrypt = symbol<FhnClientDecryptVectorDoubleFn>("ckks_fhn_client_decrypt_vec_f64");
    ASSERT_TRUE(create && destroy && encrypt && decrypt);
    client = create(services.context.get());
    ASSERT_NE(client, nullptr);
    auto prepare = symbol<FhnClientPrepareFn>("ckks_fhn_client_prepare");
    const auto requirements = nativeSearchKeyRequirements(4);
    std::vector<FhnInstruction> key_requests;
    if (requirements.multiplication) {
      FhnInstruction multiply{};
      multiply.opcode = FHN_HMULT;
      key_requests.push_back(multiply);
    }
    for (auto step : requirements.rotation_steps) {
      FhnInstruction rotate{};
      rotate.opcode = FHN_ROTATE;
      rotate.params[0] = step;
      key_requests.push_back(rotate);
    }
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
  Value value(NativeRuntime &native, const std::vector<double> &data) {
    auto *buffer = services.buffer_alloc(services.context.get());
    if (!buffer)
      throw std::runtime_error("test allocation failed");
    if (encrypt(client, buffer, data.data(), data.size())) {
      services.buffer_free(services.context.get(), buffer);
      throw std::runtime_error("test encryption failed");
    }
    auto result = native.importCopy(buffer);
    services.buffer_free(services.context.get(), buffer);
    return result;
  }
  std::vector<double> decode(NativeRuntime &native, const Value &v) {
    std::vector<double> result(8);
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
TEST_F(SearchContributionTest, CollectionComposesOverrideableBlockRequestsAndOrdinaryInterop) {
  Runtime rt;
  NativeRuntime native(rt, services);
  registerSearchKernels(rt, services, FHN_NATIVE_TEST_LIBRARY);
  auto q = value(native, {1, 2, 1, 2, 1, 2, 1, 2});
  auto a = value(native, {1, 0, 0, 1, 1, 1, -1, 2});
  auto b = value(native, {2, 1, 0, 0, 0, 0, 0, 0});
  auto index = rt.submit({"contrib.search.index", {a, b}, SearchShape{2, 5, 4}}).at(0);
  auto scores = rt.submit({"contrib.search.score", {q, index}, {}});
  ASSERT_EQ(scores.size(), 2u);
  auto first = decode(native, scores[0]);
  auto last = decode(native, scores[1]);
  for (unsigned j = 0; j < 4; ++j)
    EXPECT_NEAR(first[j * 2], (std::array<double, 4>{1, 2, 3, 3})[j], 1e-5);
  EXPECT_NEAR(last[0], 4, 1e-5);
  auto twice = native.op(FHN_ADD_CC, {scores[0], scores[0]});
  EXPECT_NEAR(decode(native, twice)[4], 6, 1e-5);
  EXPECT_TRUE(selected(rt, "contrib.search.score-block.tile.ordinary"));
  EXPECT_TRUE(selected(rt, "contrib.search.score-block.row.ordinary"));
  unsigned calls = 0;
  rt.addRecipe({"contrib.search.score-block", "test.third-party", 100,
                [](const Request &r, std::string &) { return std::any_cast<SearchShape>(&r.attributes) != nullptr; },
                [&](Runtime &, const Request &r) {
                  ++calls;
                  auto product = native.op(FHN_HMULT, {r.inputs[0], r.inputs[1]});
                  const auto width = std::any_cast<SearchShape>(r.attributes).width;
                  for (auto shift = width / 2; shift; shift /= 2)
                    product = native.op(FHN_ADD_CC, {product, native.rotate(product, shift)});
                  return Values{product};
                }});
  auto overridden = rt.submit({"contrib.search.score", {q, index}, {}});
  EXPECT_EQ(calls, 2u);
  EXPECT_NEAR(decode(native, overridden[0])[4], 3, 1e-5);
  EXPECT_NEAR(decode(native, overridden[1])[0], 4, 1e-5);
}
} // namespace
namespace {
TEST_F(SearchContributionTest, PreparedAndOrdinaryLayoutsMatchAllValidDocumentScoresAndTails) {
  Runtime rt;
  NativeRuntime native(rt, services);
  registerSearchKernels(rt, services, FHN_NATIVE_TEST_LIBRARY);
  auto q = value(native, {1, 2, 1, 2, 1, 2, 1, 2});
  auto prep = rt.submit({"contrib.search.prepare-query", {q}, uint32_t{2}}).at(0);
  ASSERT_EQ(prep.dependencies().size(), 1u);
  EXPECT_EQ(prep.dependencies()[0].format(), "contrib.matvec.preparation");
  EXPECT_EQ(prep.dependencies()[0].dependencies()[0].identity(), q.identity());
  for (uint32_t count : {1u, 3u, 4u, 5u, 9u}) {
    Values rows, tiles;
    for (uint32_t d = 0; d < count; ++d) {
      const double x = double(d + 1) / 8, y = double(d % 3) - 1;
      rows.push_back(value(native, {x, y, x, y, x, y, x, y}));
    }
    for (uint32_t start = 0; start < count; start += 4) {
      std::vector<double> tile(8);
      for (uint32_t j = 0; j < 4 && start + j < count; ++j) {
        tile[j * 2] = double(start + j + 1) / 8;
        tile[j * 2 + 1] = double((start + j) % 3) - 1;
      }
      tiles.push_back(value(native, tile));
    }
    auto rowIndex = rt.submit({"contrib.search.index", rows, SearchShape{2, count, 1}}).at(0);
    auto tileIndex = rt.submit({"contrib.search.index", tiles, SearchShape{2, count, 4}}).at(0);
    for (bool prepared : {false, true}) {
      auto row = rt.submit({"contrib.search.score", prepared ? Values{q, rowIndex, prep} : Values{q, rowIndex}, {}});
      auto tile = rt.submit({"contrib.search.score", prepared ? Values{q, tileIndex, prep} : Values{q, tileIndex}, {}});
      std::vector<std::vector<double>> decoded;
      for (const auto &v : tile)
        decoded.push_back(decode(native, v));
      for (uint32_t d = 0; d < count; ++d) {
        const double expected = double(d + 1) / 8 + 2 * (double(d % 3) - 1);
        auto actual = decode(native, row[d]);
        EXPECT_NEAR(actual[0], expected, 1e-5) << "document " << d;
        EXPECT_NEAR(decoded[d / 4][(d % 4) * 2], expected, 1e-5) << "document " << d;
      }
    }
  }
  EXPECT_TRUE(selected(rt, "contrib.search.score-block.row.prepared"));
  EXPECT_TRUE(selected(rt, "contrib.search.score-block.tile.prepared"));
}
TEST_F(SearchContributionTest, MalformedShapesMetadataArtifactsAndContextsFailBeforeComputation) {
  Runtime rt;
  NativeRuntime native(rt, services);
  registerSearchKernels(rt, services, FHN_NATIVE_TEST_LIBRARY);
  auto q = value(native, {1, 2, 1, 2, 1, 2, 1, 2});
  auto block = value(native, {1, 0, 0, 1, 1, 1, -1, 2});
  for (auto shape : {SearchShape{0, 1, 1}, SearchShape{3, 1, 1}, SearchShape{2, 0, 1}, SearchShape{2, 1, 0},
                     SearchShape{2, 1, 5}, SearchShape{2, 5, 4}, SearchShape{0x80000000u, 1, 0xffffffffu}})
    rejects(rt, {"contrib.search.index", {block}, shape});
  rejects(rt, {"contrib.search.index", {}, SearchShape{2, 1, 4}});
  rejects(rt, {"contrib.search.index", {block, block}, SearchShape{2, 1, 4}});
  rejects(rt, {"contrib.search.score-block", {q, block}, SearchShape{2, 5, 4}});
  auto shortQuery = value(native, {1, 2, 1, 2});
  rejects(rt, {"contrib.search.score-block", {shortQuery, block}, SearchShape{2, 4, 4}});
  auto lower = native.op(FHN_HMULT, {q, q});
  rejects(rt, {"contrib.search.score-block", {q, lower}, SearchShape{2, 4, 4}});
  rejects(rt, {"contrib.search.index", {block, lower}, SearchShape{2, 5, 4}});
  auto index = rt.submit({"contrib.search.index", {block}, SearchShape{2, 4, 4}}).at(0);
  auto forged = rt.makeValue(index.format(), ResourceKind::Artifact, 0, [] { return std::any{17}; });
  rejects(rt, {"contrib.search.score", {q, forged}, {}});
  rejects(rt, {"contrib.search.score", {q, index}, uint32_t{2}});
  rejects(rt, {"contrib.search.score", {lower, index}, {}});
  auto other = loadNativeServices(FHN_NATIVE_TEST_LIBRARY);
  NativeRuntime foreign(rt, other);
  auto *foreignBuffer = other.buffer_alloc(other.context.get());
  auto create = symbol<FhnClientCreateFn>("ckks_fhn_client_create");
  auto *foreignClient = create(other.context.get());
  const std::array<double, 8> foreignData{1, 2, 1, 2, 1, 2, 1, 2};
  ASSERT_EQ(encrypt(foreignClient, foreignBuffer, foreignData.data(), foreignData.size()), 0);
  auto foreignValue = foreign.importCopy(foreignBuffer);
  other.buffer_free(other.context.get(), foreignBuffer);
  destroy(foreignClient);
  rejects(rt, {"contrib.search.index", {foreignValue}, SearchShape{2, 1, 1}});
  rejects(rt, {"contrib.search.score-block", {foreignValue, block}, SearchShape{2, 4, 4}});
}
TEST_F(SearchContributionTest, PreparationsBindProviderQueryWidthAndNeverFallBackAfterFailure) {
  Runtime rt;
  NativeRuntime native(rt, services);
  registerSearchKernels(rt, services, FHN_NATIVE_TEST_LIBRARY);
  auto q = value(native, {1, 2, 1, 2, 1, 2, 1, 2}), fresh = value(native, {1, 2, 1, 2, 1, 2, 1, 2});
  auto block = value(native, {1, 0, 0, 1, 1, 1, -1, 2});
  auto index = rt.submit({"contrib.search.index", {block}, SearchShape{2, 4, 4}}).at(0);
  for (uint32_t width : {0u, 3u, 16u})
    rejects(rt, {"contrib.search.prepare-query", {q}, width});
  auto prep = rt.submit({"contrib.search.prepare-query", {q}, uint32_t{2}}).at(0);
  auto wide = rt.submit({"contrib.search.prepare-query", {q}, uint32_t{4}}).at(0);
  auto narrow = rt.submit({"contrib.search.prepare-query", {q}, uint32_t{1}}).at(0);
  auto raw = rt.submit({"contrib.matvec.prepare", {q}, MatvecPrepareRequest{2}}).at(0);
  auto forged = rt.makeValue(prep.format(), ResourceKind::Artifact, 0, [] { return std::any{42}; });
  rejects(rt, {"contrib.search.score", {q, index, Value{}}, {}}, ErrorCode::InvalidRequest);
  rejects(rt, {"contrib.search.score-block", {q, block, Value{}}, SearchShape{2, 4, 4}}, ErrorCode::InvalidRequest);
  for (const auto &bad : Values{wide, narrow, raw, forged}) {
    rejects(rt, {"contrib.search.score", {q, index, bad}, {}});
    rejects(rt, {"contrib.search.score-block", {q, block, bad}, SearchShape{2, 4, 4}});
  }
  rejects(rt, {"contrib.search.score", {fresh, index, prep}, {}});
  Runtime other;
  NativeRuntime otherNative(other, services);
  registerSearchKernels(other, services, FHN_NATIVE_TEST_LIBRARY);
  auto otherQuery = value(otherNative, {1, 2, 1, 2, 1, 2, 1, 2});
  auto otherPrep = other.submit({"contrib.search.prepare-query", {otherQuery}, uint32_t{2}}).at(0);
  rejects(rt, {"contrib.search.score", {q, index, otherPrep}, {}}, ErrorCode::InvalidRequest);
  rejects(other, {"contrib.search.score", {otherQuery, index}, {}}, ErrorCode::InvalidRequest);
  auto stolenPrep =
    rt.makeValue(prep.format(), ResourceKind::Artifact, 0, [&] { return prep.payload(); }, wide.dependencies());
  rejects(rt, {"contrib.search.score", {q, index, stolenPrep}, {}});
  auto stolenIndex = rt.makeValue(index.format(), ResourceKind::Artifact, 0, [&] { return index.payload(); }, {q});
  rejects(rt, {"contrib.search.score", {q, stolenIndex}, {}});
  unsigned fallback = 0;
  rt.addRecipe({"contrib.search.score-block", "test.failure", 100, [](const Request &, std::string &) { return true; },
                [](Runtime &, const Request &) -> Values { throw Error(ErrorCode::ExecutionFailed, "deliberate"); }});
  rt.addRecipe({"contrib.search.score-block", "test.fallback", -100,
                [](const Request &, std::string &) { return true; },
                [&](Runtime &, const Request &) -> Values {
                  ++fallback;
                  return {block};
                }});
  rejects(rt, {"contrib.search.score", {q, index, prep}, {}}, ErrorCode::ExecutionFailed);
  EXPECT_EQ(fallback, 0u);
}
TEST_F(SearchContributionTest, IndexRetainsDistinctCiphertextDependenciesOnceAndBudgetRollback) {
  const auto bytes = services.level_bytes(services.context.get(), services.fresh_level(services.context.get()));
  {
    Runtime rt;
    NativeRuntime native(rt, services);
    registerSearchKernels(rt, services, FHN_NATIVE_TEST_LIBRARY);
    auto q = value(native, {1, 2, 1, 2, 1, 2, 1, 2});
    auto index = rt.submit({"contrib.search.index", {q, q}, SearchShape{2, 2, 1}}).at(0);
    ASSERT_EQ(index.dependencies().size(), 1u);
    const auto identity = q.identity();
    q = {};
    EXPECT_EQ(index.dependencies()[0].identity(), identity);
    EXPECT_EQ(rt.stats().live_bytes, bytes + index.bytes());
    index = {};
    EXPECT_EQ(rt.stats().live_bytes, 0u);
  }
  Runtime rt({2 * bytes, 256});
  NativeRuntime native(rt, services);
  registerSearchKernels(rt, services, FHN_NATIVE_TEST_LIBRARY);
  auto q = value(native, {1, 2, 1, 2, 1, 2, 1, 2});
  auto block = value(native, {1, 0, 0, 1, 1, 1, -1, 2});
  const auto baseline = rt.stats().live_bytes;
  rejects(rt, {"contrib.search.index", {block}, SearchShape{2, 4, 4}}, ErrorCode::BudgetExceeded);
  rejects(rt, {"contrib.search.prepare-query", {q}, uint32_t{2}}, ErrorCode::BudgetExceeded);
  rejects(rt, {"contrib.search.score-block", {q, block}, SearchShape{2, 4, 4}}, ErrorCode::BudgetExceeded);
  EXPECT_EQ(rt.stats().live_bytes, baseline);
  EXPECT_NEAR(decode(native, q)[1], 2, 1e-5);
}
} // namespace
namespace {
thread_local FhnCkksBufferInfoFn originalInfo = nullptr;
thread_local unsigned corruption = 0;
int corruptInfo(FhnBackendCtx *context, const FhnBuffer *buffer, FhnCkksBufferInfo *info) {
  const int status = originalInfo(context, buffer, info);
  if (status)
    return status;
  switch (corruption) {
  case 1:
    info->scale = std::numeric_limits<double>::quiet_NaN();
    break;
  case 2:
    info->scale = 0;
    break;
  case 3:
    info->level = -1;
    break;
  case 4:
    info->components = 3;
    break;
  case 5:
    info->logical_slots = 3;
    break;
  }
  return 0;
}
TEST_F(SearchContributionTest, CorruptedPublicMetadataDeclinesWithoutCreatingResources) {
  Runtime rt;
  auto checked = services;
  originalInfo = services.buffer_info;
  checked.buffer_info = corruptInfo;
  NativeRuntime native(rt, checked);
  registerSearchKernels(rt, checked, FHN_NATIVE_TEST_LIBRARY);
  auto q = value(native, {1, 2, 1, 2, 1, 2, 1, 2});
  const auto baseline = rt.stats().live_bytes;
  for (unsigned bad = 1; bad <= 5; ++bad) {
    corruption = bad;
    rejects(rt, {"contrib.search.index", {q}, SearchShape{2, 1, 1}});
    rejects(rt, {"contrib.search.prepare-query", {q}, uint32_t{2}});
    rejects(rt, {"contrib.search.score-block", {q, q}, SearchShape{2, 1, 1}});
    EXPECT_EQ(rt.stats().live_bytes, baseline);
  }
  corruption = 0;
}
TEST_F(SearchContributionTest, LaterCollectionBudgetFailureReleasesEarlierScoresAndTemporaryCiphertexts) {
  const auto bytes = services.level_bytes(services.context.get(), services.fresh_level(services.context.get()));
  Runtime rt({6 * bytes + 4096, 256});
  NativeRuntime native(rt, services);
  registerSearchKernels(rt, services, FHN_NATIVE_TEST_LIBRARY);
  auto q = value(native, {1, 2, 1, 2, 1, 2, 1, 2});
  auto a = value(native, {1, 0, 0, 1, 1, 1, -1, 2});
  auto b = value(native, {2, 1, 0, 0, 0, 0, 0, 0});
  auto index = rt.submit({"contrib.search.index", {a, b}, SearchShape{2, 5, 4}}).at(0);
  const auto baseline = rt.stats().live_bytes;
  rejects(rt, {"contrib.search.score", {q, index}, {}}, ErrorCode::BudgetExceeded);
  bool firstCompleted = false, tailFailed = false;
  for (const auto &event : rt.trace()) {
    firstCompleted |= event.recipe == "contrib.search.score-block.tile.ordinary" && event.kind == TraceKind::Completed;
    tailFailed |= event.recipe == "contrib.search.score-block.row.ordinary" && event.kind == TraceKind::Failed;
  }
  EXPECT_TRUE(firstCompleted);
  EXPECT_TRUE(tailFailed);
  EXPECT_EQ(rt.stats().live_bytes, baseline);
  EXPECT_NEAR(decode(native, a)[2], 0, 1e-5);
}
} // namespace
namespace {
TEST_F(SearchContributionTest, ReductionWidthOneAndFourPreserveOnlyBlockStartScoreContract) {
  Runtime rt;
  NativeRuntime native(rt, services);
  registerSearchKernels(rt, services, FHN_NATIVE_TEST_LIBRARY);
  for (uint32_t width : {1u, 4u}) {
    auto q = value(native, width == 1 ? std::vector<double>{1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5}
                                      : std::vector<double>{1, -2, .5, 3, 1, -2, .5, 3});
    auto block = value(native, width == 1 ? std::vector<double>{1, 2, 3, 4, 0, 0, 0, 0}
                                          : std::vector<double>{1, 1, 1, 1, 0, 2, 4, -1});
    const SearchShape shape{width, width == 1 ? 4u : 2u, width == 1 ? 4u : 2u};
    auto prep = rt.submit({"contrib.search.prepare-query", {q}, width}).at(0);
    for (bool prepared : {false, true}) {
      auto scores =
        rt.submit({"contrib.search.score-block", prepared ? Values{q, block, prep} : Values{q, block}, shape}).at(0);
      const auto decoded = decode(native, scores);
      if (width == 1) {
        for (unsigned j = 0; j < 4; ++j)
          EXPECT_NEAR(decoded[j], 1.5 * (j + 1), 1e-5);
      } else {
        EXPECT_NEAR(decoded[0], 2.5, 1e-5);
        EXPECT_NEAR(decoded[4], -5, 1e-5);
        EXPECT_GT(std::abs(decoded[1] - decoded[0]), .5);
      }
    }
  }
}
} // namespace
