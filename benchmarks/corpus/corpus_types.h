#pragma once

#include "FHN/fhn_program.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace fhenomenon {
namespace corpus {

using Slots = std::vector<int64_t>;
using FhnProgramPtr = std::unique_ptr<FhnProgram, decltype(&fhn_program_free)>;

// One corpus workload: a generated FhnProgram plus the plaintext world it
// computes over. `expected` values come from the oracle at run time, not
// from the generator.
struct Shape {
  std::string name;
  std::string axis;       // one line: which dataflow axis this stresses
  uint32_t slot_count;    // slots per ciphertext in the plaintext model
  uint32_t ct_mult_depth; // longest chain of ct*ct multiplies (execution
                          // eligibility vs a backend's exactness bound)
  FhnProgramPtr program;
  std::map<uint32_t, Slots> inputs; // input id -> slot values
  std::vector<uint32_t> output_ids;
};

} // namespace corpus
} // namespace fhenomenon
