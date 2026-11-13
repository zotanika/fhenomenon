#pragma once
#include "FHN/NativeRuntime.h"
#include <vector>

namespace fhenomenon::contrib {
// Optional native packed-plaintext extension. Registration fails explicitly if
// unavailable; existing native providers are otherwise unaffected.
void registerPublicLinearKernels(runtime::Runtime &, runtime::NativeServices, const std::string &library);

// Correctness-oriented row-wise packed dot baseline. It consumes one level and
// returns a ciphertext with the row result repeated in every logical slot.
// Preparation is public and reusable for compatible inputs, not source-bound.
class PublicLinearRuntime {
  public:
  explicit PublicLinearRuntime(runtime::Runtime &runtime) : runtime_(runtime) {}
  runtime::Value prepareRow(runtime::Value prototype, std::vector<double> weights);
  runtime::Value dot(runtime::Value input, runtime::Value row, double bias = 0);
  runtime::ScopeValue dot(runtime::Scope &, runtime::ScopeValue input, runtime::ScopeValue row, double bias = 0);

  private:
  runtime::Runtime &runtime_;
};
} // namespace fhenomenon::contrib
