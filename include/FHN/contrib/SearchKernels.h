#pragma once
#include "FHN/NativeRuntime.h"
#include <cstdint>
#include <string>
#include <vector>

namespace fhenomenon::contrib {
struct SearchShape {
  std::uint32_t width;
  std::uint32_t document_count;
  std::uint32_t documents_per_ciphertext;
};
struct SearchKeyRequirements {
  bool multiplication;
  std::vector<std::int64_t> rotation_steps;
};
// Evaluation-key requirements for the built-in native ordinary/prepared search
// kernels, not a circuit or negotiation for arbitrary third-party recipes.
// Width must be a positive power of two. The caller also checks that it fits
// the selected CKKS profile and prepares these keys before encrypted execution.
SearchKeyRequirements nativeSearchKeyRequirements(std::uint32_t width);

// Registers contrib.search.index({ciphertext blocks}, SearchShape),
// contrib.search.prepare-query({query}, uint32_t width),
// contrib.search.score({query,index[,search preparation]}, {}), and
// contrib.search.score-block({query,block[,search preparation]}, SearchShape).
// Query repeats width slots across logical slots. A tile holds independent
// documents, one per width-slot block. Only slot j*width contains document j's
// score; other lanes contain mixed partial sums. Collection scoring composes
// block requests, allowing higher-priority contributed block recipes.
// Index and preparation artifacts bind this registration's provider and retain
// only ciphertext/artifact dependencies, never a client, plaintext or secret.
// Preparation wraps an existing contrib.matvec.prepare artifact, validating its
// width and exact query identity before dispatching contrib.matvec.dot.
void registerSearchKernels(runtime::Runtime &, runtime::NativeServices, const std::string &nativeLibraryPath);
} // namespace fhenomenon::contrib
