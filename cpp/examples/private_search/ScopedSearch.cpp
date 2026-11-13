#include "ScopedSearch.h"
#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace private_search {
namespace {
std::size_t blocks(const SearchShape &shape) {
  if (shape.width != width || !shape.document_count || !shape.documents_per_ciphertext ||
      shape.documents_per_ciphertext > tile_capacity)
    throw std::invalid_argument("invalid scoped example search layout");
  return (std::uint64_t(shape.document_count) + shape.documents_per_ciphertext - 1) / shape.documents_per_ciphertext;
}
} // namespace
ScopedSearch::ScopedSearch(fhenomenon::runtime::Scope &scope) : scope_(scope) {}
fhenomenon::runtime::ScopeValue ScopedSearch::prepareQuery(const EncryptedQuery &query) {
  return scope_.submit("contrib.search.prepare-query", {scope_.input(query.ciphertext)}, std::uint32_t{width}).at(0);
}
fhenomenon::runtime::ScopeValues ScopedSearch::score(const EncryptedQuery &query, const EncryptedDocuments &documents,
                                                     std::optional<fhenomenon::runtime::ScopeValue> preparation) {
  const auto count = blocks(documents.shape);
  if (documents.blocks.size() != count)
    throw std::invalid_argument("scoped search ciphertext count does not match document layout");
  const auto q = scope_.input(query.ciphertext);
  fhenomenon::runtime::ScopeValues results;
  results.reserve(count);
  auto remaining = documents.shape.document_count;
  for (const auto &block : documents.blocks) {
    auto shape = documents.shape;
    shape.document_count = std::min(remaining, shape.documents_per_ciphertext);
    fhenomenon::runtime::ScopeValues inputs{q, scope_.input(block)};
    if (preparation)
      inputs.push_back(*preparation);
    results.push_back(scope_.submit("contrib.search.score-block", std::move(inputs), shape).at(0));
    remaining -= shape.document_count;
  }
  return results;
}
fhenomenon::runtime::ScopeValues ScopedSearch::exportResults(fhenomenon::runtime::ScopeValue preparation,
                                                             const fhenomenon::runtime::ScopeValues &scores) {
  fhenomenon::runtime::ScopeValues exports{preparation};
  exports.insert(exports.end(), scores.begin(), scores.end());
  return exports;
}
ScopedSearchResult ScopedSearch::results(Values values, SearchShape shape) {
  if (values.size() != blocks(shape) + 1)
    throw std::invalid_argument("scoped search exports must contain preparation followed by every score block");
  QueryPreparation preparation{std::move(values.front())};
  Values scores(std::make_move_iterator(values.begin() + 1), std::make_move_iterator(values.end()));
  return {std::move(preparation), {std::move(scores), shape}};
}
} // namespace private_search
