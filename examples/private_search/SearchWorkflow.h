#pragma once

#include "FHN/contrib/SearchKernels.h"
#include "Fixture.h"
#include <memory>
#include <optional>

namespace private_search {
using fhenomenon::contrib::SearchShape;
using fhenomenon::runtime::NativeRuntime;
using fhenomenon::runtime::NativeServices;
using fhenomenon::runtime::Value;
using fhenomenon::runtime::Values;

// Example-only role labels. They make the plaintext/ciphertext boundary visible;
// they are not wire formats or a security boundary against a malicious host.
// Preserve each shape with its ciphertext/artifact; do not edit it independently.
struct EncryptedDocuments {
  Values blocks;
  SearchShape shape;
};
struct EncryptedQuery {
  Value ciphertext;
};
struct EncryptedIndex {
  Value artifact;
  SearchShape shape;
};
struct QueryPreparation {
  Value artifact;
};
struct EncryptedScores {
  Values blocks;
  SearchShape shape;
};

// Owner-side capabilities: secret key, plaintext packing, encryption, decryption.
// Keep this client alive while evaluation uses its public evaluation-key leases.
// Both roles share one explicit Runtime for resource accounting in this process;
// that Runtime must outlive the client and evaluator. It does not own the secret.
class TrustedClient {
  public:
  TrustedClient(fhenomenon::runtime::Runtime &, const std::string &nativeLibraryPath);
  ~TrustedClient();
  TrustedClient(const TrustedClient &) = delete;
  TrustedClient &operator=(const TrustedClient &) = delete;
  NativeServices computeServices() const;
  double keySetupMs() const;
  EncryptedDocuments encryptDocuments(const std::vector<Document> &, std::uint32_t capacity = tile_capacity);
  EncryptedQuery encryptQuery(const Query &);
  std::vector<double> decryptScores(const EncryptedScores &);

  // Explicit low-level client operations used by the separate benchmark runner.
  Value encryptVector(NativeRuntime &, const std::vector<double> &);
  std::vector<double> decryptVector(NativeRuntime &, const Value &);

  private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Evaluator-side capabilities: ciphertext and public metadata only. These
// signatures never accept a TrustedClient, plaintext embedding or source text.
// This is a view of the shared runtime registry, not a separate provider identity.
// Inputs must belong to the same Runtime supplied to this evaluator.
class EncryptedSearch {
  public:
  EncryptedSearch(fhenomenon::runtime::Runtime &, NativeServices, const std::string &nativeLibraryPath);
  EncryptedIndex createIndex(const EncryptedDocuments &);
  QueryPreparation prepareQuery(const EncryptedQuery &);
  EncryptedScores score(const EncryptedQuery &, const EncryptedIndex &, std::optional<QueryPreparation> = std::nullopt);
  std::vector<std::string> selectedRecipes() const;

  private:
  fhenomenon::runtime::Runtime &runtime_;
  NativeRuntime native_;
};
} // namespace private_search
