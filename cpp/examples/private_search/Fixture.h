#pragma once
#include <cmath>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace private_search {
constexpr unsigned dimensions = 384, width = 512, slots = 4096, tile_capacity = slots / width;
struct Document {
  std::string id, title, text, source;
  std::vector<double> embedding;
};
struct Query {
  std::string id, text;
  std::vector<std::string> relevant;
  std::vector<double> embedding;
};
struct Fixture {
  nlohmann::json model;
  std::vector<Document> documents;
  std::vector<Query> queries;
};
inline Fixture parseFixture(const nlohmann::json &input) {
  auto check = [](bool ok, const char *message) {
    if (!ok)
      throw std::invalid_argument(message);
  };
  auto string = [&](const nlohmann::json &item, const char *field) {
    const auto &value = item.at(field);
    check(value.is_string() && !value.get_ref<const std::string &>().empty(), "expected nonempty string");
    return value.get<std::string>();
  };
  auto vector = [&](const nlohmann::json &item) {
    const auto &values = item.at("embedding");
    check(values.is_array() && values.size() == dimensions, "embedding must have 384 dimensions");
    std::vector<double> result;
    double squared_norm = 0;
    for (const auto &value : values) {
      check(value.is_number(), "embedding must contain numbers");
      const auto x = value.get<double>();
      check(std::isfinite(x), "embedding must be finite");
      result.push_back(x);
      squared_norm += x * x;
    }
    check(std::isfinite(squared_norm) && std::abs(squared_norm - 1) <= 1e-4, "embedding must be L2 normalized");
    return result;
  };
  check(input.at("schema_version").is_number_integer() && input.at("schema_version") == 1,
        "unsupported fixture schema");
  Fixture result;
  result.model = input.at("model");
  string(result.model, "id");
  string(result.model, "revision");
  check(result.model.at("dimensions").is_number_integer() && result.model.at("dimensions") == dimensions,
        "model dimensions must be 384");
  const auto &documents = input.at("documents");
  const auto &queries = input.at("queries");
  check(documents.is_array() && !documents.empty() && documents.size() <= 100000, "invalid document collection");
  check(queries.is_array() && !queries.empty(), "invalid query collection");
  std::set<std::string> document_ids, query_ids;
  for (const auto &item : documents) {
    auto id = string(item, "id");
    check(document_ids.insert(id).second, "duplicate document id");
    result.documents.push_back({id, string(item, "title"), string(item, "text"), string(item, "source"), vector(item)});
  }
  for (const auto &item : queries) {
    auto id = string(item, "id");
    check(!document_ids.count(id) && query_ids.insert(id).second, "duplicate query or document id");
    const auto &labels = item.at("relevant_document_ids");
    check(labels.is_array(), "relevance labels must be an array");
    std::vector<std::string> relevant;
    std::set<std::string> seen;
    for (const auto &label : labels) {
      check(label.is_string(), "relevance labels must be document ids");
      auto value = label.get<std::string>();
      check(document_ids.count(value) && seen.insert(value).second, "unknown or duplicate relevance label");
      relevant.push_back(value);
    }
    result.queries.push_back({id, string(item, "text"), relevant, vector(item)});
  }
  return result;
}
} // namespace private_search
