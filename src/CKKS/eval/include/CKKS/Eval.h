#pragma once

#include "CKKS/Arena.h"
#include "CKKS/EvalTables.h"
#include "CKKS/Storage.h"

#include <cstddef>
#include <cstdint>

namespace fhenomenon::ckks::eval {

enum class EvalOperation {
  Add,
  Subtract,
  Negate,
  Tensor,
  Relinearize,
  Rescale,
  Rotate,
  HMultiply,
  AddScalar,
  MultiplyScalar,
  LevelDown
};

// Conservative operation-local bound, including alignment and worst-case
// supported polynomial count. This excludes caller-owned inputs/output, keys,
// and immutable tables. Returns zero for an unsupported operation/level.
// Kernels require this entire remaining capacity before touching output.
std::size_t scratchBytes(EvalOperation operation, const Params &params, uint32_t level) noexcept;

// Powers of 5 modulo 2N. Positive steps rotate canonical slots left. Invalid
// degree returns zero; signed steps, including INT64_MIN, are normalized.
uint64_t galoisElement(int64_t step, uint32_t log_degree);

// Inputs: reduced NTT residues, active Q prefix, no auxiliary limbs. Scales
// for add/subtract must differ by at most 1e-12 relative. Tensor multiplies
// scales. Exact aliases are supported; partial storage overlaps are rejected.
// All functions validate storage/keys/scratch before writing, allocate only
// from the supplied Arena, and publish output metadata only on success.
bool add(CiphertextView &out, ConstCiphertextView a, ConstCiphertextView b, const EvaluationTables &tables,
         Arena &arena);
bool subtract(CiphertextView &out, ConstCiphertextView a, ConstCiphertextView b, const EvaluationTables &tables,
              Arena &arena);
bool negate(CiphertextView &out, ConstCiphertextView a, const EvaluationTables &tables, Arena &arena);
bool tensor(CiphertextView &out, ConstCiphertextView a, ConstCiphertextView b, const EvaluationTables &tables,
            Arena &arena);
bool relinearize(CiphertextView &out, ConstCiphertextView a, EvaluationKeyView key, const EvaluationTables &tables,
                 Arena &arena);
bool rescale(CiphertextView &out, ConstCiphertextView a, const EvaluationTables &tables, Arena &arena);
bool rotate(CiphertextView &out, ConstCiphertextView a, uint64_t galois_element, EvaluationKeyView key,
            const EvaluationTables &tables, Arena &arena);
bool hmult(CiphertextView &out, ConstCiphertextView a, ConstCiphertextView b, EvaluationKeyView key,
           const EvaluationTables &tables, Arena &arena);
// Scalar coefficients are rounded with long-double arithmetic and must fit
// int64_t; nonfinite values or overflow return false without changing output.
bool addScalar(CiphertextView &out, ConstCiphertextView a, double scalar, const EvaluationTables &tables, Arena &arena);
// Scalar encoding uses the dropped prime as its scale. Every successful call
// consumes exactly one level, including multiplication by 0 or an integer.
bool multiplyScalar(CiphertextView &out, ConstCiphertextView a, double scalar, const EvaluationTables &tables,
                    Arena &arena);
bool levelDown(CiphertextView &out, ConstCiphertextView a, uint32_t target_level, const EvaluationTables &tables,
               Arena &arena);

} // namespace fhenomenon::ckks::eval
