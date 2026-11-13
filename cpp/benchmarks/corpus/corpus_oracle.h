#pragma once

#include "corpus_types.h"

#include <optional>

namespace fhenomenon {
namespace corpus {

// Plaintext reference evaluator over slot vectors. Models the primitive
// compute opcodes only (fused composites are rejected — corpus generators
// never emit them). Returns id -> slots for every defined id, or nullopt
// on an unmodeled opcode, an undefined operand, or mismatched slot sizes.
std::optional<std::map<uint32_t, Slots>> evaluate(const FhnProgram &program, const std::map<uint32_t, Slots> &inputs);

} // namespace corpus
} // namespace fhenomenon
