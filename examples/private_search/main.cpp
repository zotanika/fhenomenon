#include "SearchWorkflow.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace {
using namespace private_search;

// Ordinary client-side search presentation: sort decrypted scores, then look up
// the owner's local document text. No FHE calculation occurs in this function.
std::vector<std::size_t> topK(const std::vector<Document> &documents, const std::vector<double> &scores,
                              std::size_t k) {
  std::vector<std::size_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](auto a, auto b) {
    return scores[a] != scores[b] ? scores[a] > scores[b] : documents[a].id < documents[b].id;
  });
  order.resize(std::min(k, order.size()));
  return order;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if ((argc != 3 && argc != 5) || (argc == 5 && std::string(argv[3]) != "--query"))
      throw std::invalid_argument(
        "usage: fhn-private-search NATIVE_LIBRARY FIXTURE_JSON [--query ID]\n"
        "For --compare, --verify-all, --reps, --limit-documents or --json, use fhn-bench-private-search.");
    std::ifstream input(argv[2]);
    if (!input)
      throw std::runtime_error("cannot open fixture");
    nlohmann::json source;
    input >> source;
    const Fixture fixture = parseFixture(source);
    const std::string query_id = argc == 5 ? argv[4] : fixture.queries.front().id;
    const auto question =
      std::find_if(fixture.queries.begin(), fixture.queries.end(), [&](const Query &q) { return q.id == query_id; });
    if (question == fixture.queries.end())
      throw std::invalid_argument("unknown query id: " + query_id);

    // This in-process demo shares resource accounting between the two roles.
    // Only TrustedClient owns a secret key; the evaluator receives compute services.
    fhenomenon::runtime::Runtime resources;
    TrustedClient client(resources, argv[1]);
    EncryptedSearch evaluator(resources, client.computeServices(), argv[1]);

    // CLIENT: plaintext embeddings cross the encryption boundary here.
    const EncryptedDocuments documents = client.encryptDocuments(fixture.documents);
    const EncryptedQuery query = client.encryptQuery(*question);

    // EVALUATOR: only ciphertext and public layout metadata enter this program.
    // Follow EncryptedSearch.cpp for runtime requests and ScoreBlock.h for kernels.
    const EncryptedIndex index = evaluator.createIndex(documents);
    const QueryPreparation prepared = evaluator.prepareQuery(query);
    const EncryptedScores encrypted_scores = evaluator.score(query, index, prepared);

    // CLIENT: only the owner decrypts. Ranking and document display are plaintext.
    const std::vector<double> scores = client.decryptScores(encrypted_scores);
    const auto matches = topK(fixture.documents, scores, 5);

    std::cout << "Question: " << question->text << "\n\nOwner's top matches:\n";
    for (auto i : matches) {
      const auto &document = fixture.documents[i];
      std::cout << std::fixed << std::setprecision(4) << scores[i] << "  " << document.title << '\n'
                << "  " << document.source << '\n'
                << "  " << document.text.substr(0, document.text.find('\n')) << '\n';
    }
    std::cout << "\nSelected search kernels (bounded runtime trace):\n";
    for (const auto &recipe : evaluator.selectedRecipes())
      std::cout << "  " << recipe << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "private-search: " << error.what() << '\n';
    return 1;
  }
}
