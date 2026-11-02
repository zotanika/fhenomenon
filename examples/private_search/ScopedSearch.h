#pragma once

#include "SearchWorkflow.h"
#include "FHN/Scope.h"

namespace private_search {
struct ScopedSearchResult {
  QueryPreparation preparation;
  EncryptedScores scores;
};

// Records separate preparation and ciphertext-block commands. The returned
// symbols remain local to this Scope; explicitly export artifacts for reuse.
class ScopedSearch {
  public:
  explicit ScopedSearch(fhenomenon::runtime::Scope &);
  fhenomenon::runtime::ScopeValue prepareQuery(const EncryptedQuery &);
  fhenomenon::runtime::ScopeValues score(
    const EncryptedQuery &, const EncryptedDocuments &,
    std::optional<fhenomenon::runtime::ScopeValue> preparation = std::nullopt);
  fhenomenon::runtime::ScopeValues exportResults(
    fhenomenon::runtime::ScopeValue preparation, const fhenomenon::runtime::ScopeValues &scores);
  static ScopedSearchResult results(Values, SearchShape);

  private:
  fhenomenon::runtime::Scope &scope_;
};
} // namespace private_search
