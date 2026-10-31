#pragma once
#include "FHN/fhn_circuit_api.h"
#include <optional>
#include <string>
#include <vector>

namespace fhenomenon::corpus {
// Owning semantic import of a corpus app. view() remains valid until mutation
// or destruction of this object; ask for a new view after moving it.
struct Circuit {
  std::vector<FhnCircuitNode> nodes;
  std::vector<uint32_t> inputs, outputs;
  FhnCircuit view() const;
};
// Explicitly imports the corpus value semantics. This is NOT an arbitrary
// FhnProgram optimizer: physical RELIN/RESCALE/LEVEL_DOWN and Boolean ops fail.
std::optional<Circuit> importCircuit(const FhnProgram &, std::string *error = nullptr);
} // namespace fhenomenon::corpus
