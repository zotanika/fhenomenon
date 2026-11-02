#include "SearchWorkflow.h"
#include <set>

namespace private_search {
EncryptedSearch::EncryptedSearch(fhenomenon::runtime::Runtime &runtime, NativeServices services,
                                 const std::string &nativeLibraryPath)
  : runtime_(runtime), native_(runtime, services) {
  fhenomenon::contrib::registerSearchKernels(runtime_, std::move(services), nativeLibraryPath);
}
EncryptedIndex EncryptedSearch::createIndex(const EncryptedDocuments &documents) {
  // Public layout metadata and ciphertext handles; no document text is needed.
  auto index = runtime_.submit({"contrib.search.index", documents.blocks, documents.shape}).at(0);
  return {std::move(index), documents.shape};
}
QueryPreparation EncryptedSearch::prepareQuery(const EncryptedQuery &query) {
  // This is reusable encrypted-source state, bound to this exact query Value.
  return {runtime_.submit({"contrib.search.prepare-query", {query.ciphertext}, std::uint32_t{width}}).at(0)};
}
EncryptedScores EncryptedSearch::score(const EncryptedQuery &query, const EncryptedIndex &index,
                                       std::optional<QueryPreparation> preparation) {
  Values inputs{query.ciphertext, index.artifact};
  if (preparation)
    inputs.push_back(preparation->artifact);
  // The high-level encrypted computation starts here. The collection recipe
  // selects block kernels; contributions/search/ScoreBlock.h shows their math.
  // An explicit invalid preparation is passed through and rejected, not ignored.
  auto scores = runtime_.submit({"contrib.search.score", std::move(inputs), {}});
  return {std::move(scores), index.shape};
}
std::vector<std::string> EncryptedSearch::selectedRecipes() const {
  std::set<std::string> names;
  for (const auto &event : runtime_.trace())
    if (event.kind == fhenomenon::runtime::TraceKind::Selected && event.operation.rfind("contrib.search", 0) == 0)
      names.insert(event.recipe);
  return {names.begin(), names.end()}; // Names in bounded trace history, not invocation counts.
}
} // namespace private_search
