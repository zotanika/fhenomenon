#include "../examples/private_search/ScopedSearch.h"
#include "Session/Session.h"
#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <gtest/gtest.h>
#include <memory>

namespace {
using namespace private_search;
using namespace fhenomenon::runtime;

class SearchScope : public ::testing::Test {
  public:
  static void SetUpTestSuite() {
    // Keep the plugin loaded across tests and repetitions. Apple ASan can
    // retain global registrations after unloading an instrumented dylib.
    static const std::unique_ptr<void, decltype(&dlclose)> library{
        dlopen(FHN_NATIVE_TEST_LIBRARY, RTLD_NOW | RTLD_LOCAL), dlclose};
    ASSERT_NE(library, nullptr) << dlerror();
  }
};

std::vector<Document> documents(unsigned count = 17) {
  std::vector<Document> result;
  for (unsigned i = 0; i < count; ++i) {
    std::vector<double> v(dimensions);
    v[0] = std::cos(.1 * i);
    v[1] = std::sin(.1 * i);
    result.push_back({std::to_string(i), "title", "text", "test", std::move(v)});
  }
  return result;
}
Query query() {
  std::vector<double> v(dimensions);
  v[0] = 1;
  return {"query", "question", {}, std::move(v)};
}
unsigned selected(const Runtime &runtime, const std::string &name) {
  unsigned count = 0;
  for (const auto &event : runtime.trace())
    count += event.kind == TraceKind::Selected && event.recipe == name;
  return count;
}
void scoresEqual(TrustedClient &client, const EncryptedScores &scores) {
  auto actual = client.decryptScores(scores);
  ASSERT_EQ(actual.size(), scores.shape.document_count);
  for (unsigned i = 0; i < actual.size(); ++i)
    EXPECT_NEAR(actual[i], std::cos(.1 * i), 1e-5) << "document " << i;
}

// Without scoped search contributions this still computes correct ciphertexts
// through ordinary fallback, but cannot select the required two-command span.
TEST_F(SearchScope, PreparedFullBlocksSelectCollectionAndTailSelectsBlock) {
  Runtime runtime({0, 4096});
  TrustedClient client(runtime, FHN_NATIVE_TEST_LIBRARY);
  EncryptedSearch evaluator(runtime, client.computeServices(), FHN_NATIVE_TEST_LIBRARY);
  const auto encrypted_query = client.encryptQuery(query());
  const auto encrypted_documents = client.encryptDocuments(documents());
  auto values = runtime.runScope([&](Scope &scope) {
    const auto q = scope.input(encrypted_query.ciphertext);
    const auto prep = scope.submit("contrib.search.prepare-query", {q}, std::uint32_t{width}).at(0);
    ScopeValues exports{prep};
    unsigned remaining = encrypted_documents.shape.document_count;
    for (const auto &block : encrypted_documents.blocks) {
      auto shape = encrypted_documents.shape;
      shape.document_count = std::min(remaining, shape.documents_per_ciphertext);
      exports.push_back(scope.submit("contrib.search.score-block", {q, scope.input(block), prep}, shape).at(0));
      remaining -= shape.document_count;
    }
    return exports;
  });
  ASSERT_EQ(values.size(), 4u);
  EncryptedScores scores{{values.begin() + 1, values.end()}, encrypted_documents.shape};
  scoresEqual(client, scores);
  EXPECT_EQ(selected(runtime, "contrib.search.scope.collection"), 1u);
  EXPECT_EQ(selected(runtime, "contrib.search.scope.block"), 1u);
  EXPECT_EQ(selected(runtime, "contrib.search.prepare-query.native"), 1u);
}
} // namespace

namespace {
TEST_F(SearchScope, DisabledKernelsAndUnpreparedRequestsUseOrdinaryDispatch) {
  Runtime runtime({0, 4096});
  TrustedClient client(runtime, FHN_NATIVE_TEST_LIBRARY);
  EncryptedSearch evaluator(runtime, client.computeServices(), FHN_NATIVE_TEST_LIBRARY);
  const auto q = client.encryptQuery(query());
  const auto docs = client.encryptDocuments(documents());
  for (bool prepared : {false, true}) {
    ScopeOptions options;
    options.use_kernels = !prepared;
    auto values = runtime.runScope([&](Scope &scope) {
      const auto qs = scope.input(q.ciphertext);
      ScopeValue prep;
      if (prepared)
        prep = scope.submit("contrib.search.prepare-query", {qs}, std::uint32_t{width}).at(0);
      ScopeValues exports;
      unsigned remaining = docs.shape.document_count;
      for (const auto &block : docs.blocks) {
        auto shape = docs.shape;
        shape.document_count = std::min(remaining, shape.documents_per_ciphertext);
        ScopeValues inputs{qs, scope.input(block)};
        if (prepared)
          inputs.push_back(prep);
        exports.push_back(scope.submit("contrib.search.score-block", inputs, shape).at(0));
        remaining -= shape.document_count;
      }
      return exports;
    }, options);
    scoresEqual(client, {values, docs.shape});
    EXPECT_EQ(selected(runtime, "contrib.search.scope.collection"), 0u);
    EXPECT_EQ(selected(runtime, "contrib.search.scope.block"), 0u);
  }
  EXPECT_EQ(selected(runtime, "contrib.search.score-block.tile.ordinary"), 2u);
  EXPECT_EQ(selected(runtime, "contrib.search.score-block.row.ordinary"), 1u);
  EXPECT_EQ(selected(runtime, "contrib.search.score-block.tile.prepared"), 2u);
  EXPECT_EQ(selected(runtime, "contrib.search.score-block.row.prepared"), 1u);
}
}

namespace {
TEST_F(SearchScope, TypedRecorderExportsPreparationAndReusesExactArtifactInLaterScope) {
  Runtime runtime({0, 4096});
  TrustedClient client(runtime, FHN_NATIVE_TEST_LIBRARY);
  EncryptedSearch evaluator(runtime, client.computeServices(), FHN_NATIVE_TEST_LIBRARY);
  const auto q = client.encryptQuery(query());
  const auto docs = client.encryptDocuments(documents());
  NativeRuntime native(runtime, client.computeServices());
  auto session = fhenomenon::Session::create(runtime);
  unsigned captures = 0;
  auto values = session->run([&](Scope &scope) {
    ++captures;
    ScopedSearch search(scope);
    auto prep = search.prepareQuery(q);
    auto scores = search.score(q, docs, prep);
    auto exports = search.exportResults(prep, scores);
    exports.push_back(native.op(scope, FHN_ADD_CC, {scores[0], scores[0]}));
    return exports;
  });
  ASSERT_EQ(values.size(), 5u);
  auto doubled = values.back();
  values.pop_back();
  auto result = ScopedSearch::results(std::move(values), docs.shape);
  scoresEqual(client, result.scores);
  const auto doubled_plaintext = client.decryptVector(native, doubled);
  for (unsigned i = 0; i < tile_capacity; ++i)
    EXPECT_NEAR(doubled_plaintext[i * width], 2 * std::cos(.1 * i), 1e-5);
  EXPECT_EQ(selected(runtime, "contrib.search.scope.collection"), 1u);
  EXPECT_EQ(selected(runtime, "contrib.search.scope.block"), 1u);
  EXPECT_EQ(selected(runtime, "native.physical"), 1u);
  const auto identity = result.preparation.artifact.identity();
  auto second = session->run([&](Scope &scope) {
    ++captures;
    ScopedSearch search(scope);
    auto prep = scope.input(result.preparation.artifact);
    return search.exportResults(prep, search.score(q, docs, prep));
  });
  auto reused = ScopedSearch::results(std::move(second), docs.shape);
  EXPECT_EQ(captures, 2u);
  EXPECT_EQ(reused.preparation.artifact.identity(), identity);
  EXPECT_EQ(selected(runtime, "contrib.search.prepare-query.native"), 1u);
  EXPECT_EQ(selected(runtime, "contrib.matvec.prepare.native"), 1u);
  scoresEqual(client, reused.scores);
}
}

namespace {
TEST_F(SearchScope, ExternalScopeKernelReplacesBuiltinsAndNestedBlockOverridesRemainVisible) {
  Runtime runtime({0, 4096});
  TrustedClient client(runtime, FHN_NATIVE_TEST_LIBRARY);
  EncryptedSearch evaluator(runtime, client.computeServices(), FHN_NATIVE_TEST_LIBRARY);
  NativeRuntime native(runtime, client.computeServices());
  const auto q = client.encryptQuery(query());
  const auto docs = client.encryptDocuments(documents());
  auto session = fhenomenon::Session::create(runtime);
  unsigned block_calls = 0, external_calls = 0;
  runtime.addRecipe({"contrib.search.score-block", "test.search.scope.block-replacement", 100,
    [](const Request &request, std::string &) {
      return request.inputs.size() == 3 && std::any_cast<SearchShape>(&request.attributes);
    },
    [&](Runtime &, const Request &request) {
      ++block_calls;
      auto product = native.op(FHN_HMULT, {request.inputs[0], request.inputs[1]});
      for (auto shift = std::any_cast<SearchShape>(request.attributes).width / 2; shift; shift /= 2)
        product = native.op(FHN_ADD_CC, {product, native.rotate(product, shift)});
      return Values{product};
    }});
  const auto run = [&] {
    return session->run([&](Scope &scope) {
      ScopedSearch search(scope);
      const auto prep = search.prepareQuery(q);
      return search.exportResults(prep, search.score(q, docs, prep));
    });
  };
  auto first = ScopedSearch::results(run(), docs.shape);
  scoresEqual(client, first.scores);
  EXPECT_EQ(block_calls, 3u);
  EXPECT_EQ(selected(runtime, "contrib.search.scope.collection"), 1u);
  EXPECT_EQ(selected(runtime, "contrib.search.scope.block"), 1u);
  runtime.addScopeKernel({"test.search.scope.replacement", 100, 8,
    [](const ScopeView &view, std::string &) {
      std::size_t count = 0;
      for (; count < view.size(); ++count) {
        const auto &command = view.command(count);
        if (command.operation != "contrib.search.score-block" || command.inputs.size() != 3 ||
            command.outputs.size() != 1 || !std::any_cast<SearchShape>(&command.attributes))
          break;
        for (auto symbol : command.inputs)
          if (!view.find(symbol))
            return count;
      }
      return count;
    },
    [&](Runtime &rt, const ScopeView &view) {
      ++external_calls;
      const auto liveouts = view.liveOuts(view.size());
      Values results(liveouts.size());
      for (std::size_t i = 0; i < view.size(); ++i) {
        const auto &command = view.command(i);
        Values inputs;
        for (auto symbol : command.inputs)
          inputs.push_back(view.value(symbol));
        auto block = rt.submit({command.operation, inputs, command.attributes}).at(0);
        const auto output = std::find(liveouts.begin(), liveouts.end(), command.outputs[0]);
        if (output != liveouts.end())
          results[static_cast<std::size_t>(output - liveouts.begin())] = std::move(block);
      }
      return results;
    }});
  auto replacement = ScopedSearch::results(run(), docs.shape);
  scoresEqual(client, replacement.scores);
  EXPECT_EQ(external_calls, 1u);
  EXPECT_EQ(block_calls, 6u);
  EXPECT_EQ(selected(runtime, "test.search.scope.replacement"), 1u);
  EXPECT_EQ(selected(runtime, "contrib.search.scope.collection"), 1u);
  EXPECT_EQ(selected(runtime, "contrib.search.scope.block"), 1u);
}

TEST_F(SearchScope, CopiedPreparationPayloadAndDifferentQueryCannotEnterSealedScopeKernels) {
  Runtime runtime({0, 4096});
  TrustedClient client(runtime, FHN_NATIVE_TEST_LIBRARY);
  EncryptedSearch evaluator(runtime, client.computeServices(), FHN_NATIVE_TEST_LIBRARY);
  const auto q = client.encryptQuery(query());
  const auto fresh_q = client.encryptQuery(query());
  const auto docs = client.encryptDocuments(documents(8));
  auto prep = evaluator.prepareQuery(q);
  auto copied = runtime.makeValue(prep.artifact.format(), ResourceKind::Artifact, 0,
    [&] { return prep.artifact.payload(); }, prep.artifact.dependencies());
  const auto baseline = runtime.stats().live_bytes;
  for (bool forged : {false, true}) {
    EXPECT_THROW(runtime.runScope([&](Scope &scope) {
      ScopedSearch search(scope);
      return search.score(forged ? q : fresh_q, docs, scope.input(forged ? copied : prep.artifact));
    }), Error);
    EXPECT_EQ(runtime.stats().live_bytes, baseline);
  }
  EXPECT_EQ(selected(runtime, "contrib.search.scope.collection"), 0u);
  EXPECT_EQ(selected(runtime, "contrib.search.scope.block"), 0u);
  auto actual = runtime.runScope([&](Scope &scope) {
    ScopedSearch search(scope);
    return search.score(q, docs, scope.input(prep.artifact));
  });
  scoresEqual(client, {actual, docs.shape});
}

TEST_F(SearchScope, CollectionStopsAtEightCommandsAndReturnsOnlyRequestedLiveouts) {
  Runtime runtime({0, 4096});
  TrustedClient client(runtime, FHN_NATIVE_TEST_LIBRARY);
  EncryptedSearch evaluator(runtime, client.computeServices(), FHN_NATIVE_TEST_LIBRARY);
  const auto q = client.encryptQuery(query());
  const auto docs = client.encryptDocuments(documents(8));
  const auto prep = evaluator.prepareQuery(q);
  auto values = runtime.runScope([&](Scope &scope) {
    const auto qs = scope.input(q.ciphertext), block = scope.input(docs.blocks[0]);
    const auto preparation = scope.input(prep.artifact);
    ScopeValues recorded;
    for (unsigned i = 0; i < 9; ++i)
      recorded.push_back(scope.submit("contrib.search.score-block", {qs, block, preparation}, docs.shape).at(0));
    return ScopeValues{recorded[0], recorded[7], recorded[8]};
  });
  ASSERT_EQ(values.size(), 3u);
  for (const auto &value : values)
    scoresEqual(client, {{value}, docs.shape});
  EXPECT_EQ(selected(runtime, "contrib.search.scope.collection"), 1u);
  EXPECT_EQ(selected(runtime, "contrib.search.scope.block"), 1u);
  EXPECT_EQ(selected(runtime, "contrib.search.score-block.tile.prepared"), 9u);
  bool collection_detail = false;
  for (const auto &event : runtime.trace())
    collection_detail |= event.kind == TraceKind::Selected && event.recipe == "contrib.search.scope.collection" &&
                         event.detail.find("count=8") != std::string::npos;
  EXPECT_TRUE(collection_detail);
}

TEST_F(SearchScope, CollectionSeparatesDifferentPreparationSymbolsAndBlockCapacities) {
  Runtime runtime({0, 4096});
  TrustedClient client(runtime, FHN_NATIVE_TEST_LIBRARY);
  EncryptedSearch evaluator(runtime, client.computeServices(), FHN_NATIVE_TEST_LIBRARY);
  const auto q = client.encryptQuery(query());
  const auto docs = client.encryptDocuments(documents(8));
  const SearchShape small{width, 4, 4};
  auto values = runtime.runScope([&](Scope &scope) {
    const auto qs = scope.input(q.ciphertext), block = scope.input(docs.blocks[0]);
    const auto first = scope.submit("contrib.search.prepare-query", {qs}, std::uint32_t{width}).at(0);
    const auto second = scope.submit("contrib.search.prepare-query", {qs}, std::uint32_t{width}).at(0);
    return ScopeValues{
      scope.submit("contrib.search.score-block", {qs, block, first}, docs.shape).at(0),
      scope.submit("contrib.search.score-block", {qs, block, second}, docs.shape).at(0),
      scope.submit("contrib.search.score-block", {qs, block, second}, small).at(0),
      scope.submit("contrib.search.score-block", {qs, block, second}, small).at(0)};
  });
  ASSERT_EQ(values.size(), 4u);
  for (unsigned i = 0; i < values.size(); ++i)
    scoresEqual(client, {{values[i]}, i < 2 ? docs.shape : small});
  EXPECT_EQ(selected(runtime, "contrib.search.scope.collection"), 1u);
  EXPECT_EQ(selected(runtime, "contrib.search.scope.block"), 2u);
}
} // namespace
