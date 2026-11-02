#include "../examples/private_search/SearchWorkflow.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace private_search;
std::vector<Document> documents() {
  std::vector<Document> result;
  for (unsigned i = 0; i < 9; ++i) {
    std::vector<double> v(dimensions);
    v[0] = std::cos(0.2 * i);
    v[1] = std::sin(0.2 * i);
    result.push_back({std::to_string(i), "title", "text", "test", v});
  }
  return result;
}
Query query() {
  std::vector<double> v(dimensions);
  v[0] = 1;
  return {"query", "question", {}, v};
}
TEST(PrivateSearchWorkflow, ClientEncryptsBothInputsEvaluatorReturnsOnlyEncryptedScoresIncludingTail) {
  fhenomenon::runtime::Runtime runtime;
  TrustedClient client(runtime, FHN_NATIVE_TEST_LIBRARY);
  EncryptedSearch evaluator(runtime, client.computeServices(), FHN_NATIVE_TEST_LIBRARY);
  auto q = client.encryptQuery(query());
  auto preparation = evaluator.prepareQuery(q);
  for (auto capacity : {1u, tile_capacity}) {
    auto encrypted = client.encryptDocuments(documents(), capacity);
    EXPECT_EQ(encrypted.blocks.size(), capacity == 1 ? 9u : 2u);
    auto index = evaluator.createIndex(encrypted);
    for (bool prepared : {false, true}) {
      auto encrypted_scores =
        evaluator.score(q, index, prepared ? std::optional<QueryPreparation>(preparation) : std::nullopt);
      EXPECT_EQ(encrypted_scores.blocks.size(), capacity == 1 ? 9u : 2u);
      auto actual = client.decryptScores(encrypted_scores);
      ASSERT_EQ(actual.size(), 9u);
      for (unsigned i = 0; i < 9; ++i)
        EXPECT_NEAR(actual[i], std::cos(0.2 * i), 1e-5);
    }
  }
  auto selected = evaluator.selectedRecipes();
  EXPECT_NE(std::find(selected.begin(), selected.end(), "contrib.search.score-block.tile.prepared"), selected.end());
  EXPECT_NE(std::find(selected.begin(), selected.end(), "contrib.search.score-block.row.prepared"), selected.end());
}
TEST(PrivateSearchWorkflow, ExplicitPreparationMustBelongToExactQueryAndRuntime) {
  fhenomenon::runtime::Runtime runtime;
  TrustedClient client(runtime, FHN_NATIVE_TEST_LIBRARY);
  EncryptedSearch evaluator(runtime, client.computeServices(), FHN_NATIVE_TEST_LIBRARY);
  auto index = evaluator.createIndex(client.encryptDocuments(documents()));
  auto q = client.encryptQuery(query());
  auto fresh_q = client.encryptQuery(query());
  auto preparation = evaluator.prepareQuery(q);
  EXPECT_THROW(evaluator.score(fresh_q, index, preparation), fhenomenon::runtime::Error);
  EXPECT_THROW(evaluator.score(q, index, QueryPreparation{}), fhenomenon::runtime::Error);
  fhenomenon::runtime::Runtime foreign_runtime;
  EncryptedSearch another(foreign_runtime, client.computeServices(), FHN_NATIVE_TEST_LIBRARY);
  EXPECT_THROW(another.createIndex(client.encryptDocuments(documents())), fhenomenon::runtime::Error);
  EXPECT_THROW(another.score(q, index, preparation), fhenomenon::runtime::Error);
}
TEST(PrivateSearchWorkflow, ClientRejectsBadPackingAndScoreMetadata) {
  fhenomenon::runtime::Runtime runtime;
  TrustedClient client(runtime, FHN_NATIVE_TEST_LIBRARY);
  EXPECT_THROW(client.encryptDocuments({}, tile_capacity), std::invalid_argument);
  EXPECT_THROW(client.encryptDocuments(documents(), 0), std::invalid_argument);
  EXPECT_THROW(client.encryptDocuments(documents(), tile_capacity + 1), std::invalid_argument);
  auto bad = query();
  bad.embedding.pop_back();
  EXPECT_THROW(client.encryptQuery(bad), std::invalid_argument);
  bad = query();
  bad.embedding[0] = 2;
  EXPECT_THROW(client.encryptQuery(bad), std::invalid_argument);
  bad = query();
  bad.embedding[0] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(client.encryptQuery(bad), std::invalid_argument);
  auto bad_documents = documents();
  bad_documents.back().embedding[0] = 3;
  EXPECT_THROW(client.encryptDocuments(bad_documents), std::invalid_argument);
  EXPECT_THROW(client.decryptScores({{}, SearchShape{width, 1, tile_capacity}}), std::invalid_argument);
}
} // namespace
