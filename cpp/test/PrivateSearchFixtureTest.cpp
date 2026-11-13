#include "../examples/private_search/Fixture.h"
#include <gtest/gtest.h>
#include <limits>
namespace {
using nlohmann::json;
json fixture() {
  std::vector<double> embedding(private_search::dimensions);
  embedding[0] = 1;
  return {
    {"schema_version", 1},
    {"model", {{"id", "local-model"}, {"revision", "pinned"}, {"dimensions", 384}}},
    {"documents",
     {{{"id", "d"}, {"title", "Title"}, {"text", "Document"}, {"source", "doc#L1"}, {"embedding", embedding}}}},
    {"queries", {{{"id", "q"}, {"text", "Question"}, {"relevant_document_ids", {"d"}}, {"embedding", embedding}}}}};
}
TEST(PrivateSearchFixture, PreservesClientTextVectorsAndLabels) {
  auto parsed = private_search::parseFixture(fixture());
  ASSERT_EQ(parsed.documents.size(), 1u);
  ASSERT_EQ(parsed.queries.size(), 1u);
  EXPECT_EQ(parsed.documents[0].text, "Document");
  EXPECT_EQ(parsed.queries[0].relevant[0], "d");
  EXPECT_EQ(parsed.documents[0].embedding.size(), 384u);
}
TEST(PrivateSearchFixture, RejectsMalformedVectorsAndSchemaBeforeCryptoSetup) {
  for (const auto &path : {json::json_pointer("/documents/0/embedding"), json::json_pointer("/queries/0/embedding")}) {
    for (int kind = 0; kind < 4; ++kind) {
      auto input = fixture();
      if (kind == 0)
        input[path].erase(input[path].begin());
      if (kind == 1)
        input[path][0] = 2;
      if (kind == 2)
        input[path][0] = std::numeric_limits<double>::infinity();
      if (kind == 3)
        input[path][0] = "1";
      EXPECT_THROW(private_search::parseFixture(input), std::exception);
    }
  }
  for (int kind = 0; kind < 8; ++kind) {
    auto input = fixture();
    if (kind == 0)
      input["schema_version"] = 2;
    if (kind == 1)
      input["documents"] = json::array();
    if (kind == 2)
      input["queries"] = json::array();
    if (kind == 3)
      input["model"]["dimensions"] = 512;
    if (kind == 4)
      input["documents"].push_back(input["documents"][0]);
    if (kind == 5)
      input["queries"][0]["relevant_document_ids"] = {"missing"};
    if (kind == 6)
      input["queries"].push_back(input["queries"][0]);
    if (kind == 7)
      input["queries"][0]["id"] = "d";
    EXPECT_THROW(private_search::parseFixture(input), std::exception) << kind;
  }
}
} // namespace
