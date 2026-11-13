#pragma once

#include "CKKS/Eval.h"

// Private kernel/compiler interface, not a stable external ABI. Callers certify
// canonical input/key residues at binding, retain immutable key lifetimes, and
// supply disjoint destination/input/key storage. Regions do not rescan residues.
// All temporary memory comes from Arena; public output publication is the
// caller's responsibility. Metadata/scratch failures return false.
namespace fhenomenon::ckks::region {
struct LinearTerm {
  ConstCiphertextView value;
  int64_t coefficient;
};
struct ProductTerm {
  ConstCiphertextView left;
  ConstCiphertextView right;
  int64_t coefficient;
};
// Trusted preparation owns these immutable arrays and certifies correspondence
// to the bound keys/source/galois element. Empty views retain ordinary arithmetic.
// Keys: [digit][component][original full Q followed by P][coefficient].
// Cipher: [component][active Q][coefficient], for multiplyCached's LEFT input.
struct KeyProducts {
  const uint64_t *quotients = nullptr;
  std::size_t count = 0;
};
struct CipherProducts {
  const uint64_t *quotients = nullptr;
  std::size_t count = 0;
};
struct RotationMap {
  const uint32_t *indices = nullptr;
  std::size_t count = 0;
};
bool validateInput(ConstCiphertextView, const EvaluationTables &);
bool validateKey(EvaluationKeyView, SwitchKeyKind, uint64_t galois, const EvaluationTables &);
std::size_t scratchBytes(const Params &) noexcept;
bool encodeConstant(double value, double scale, int64_t &encoded) noexcept;

// Integer coefficients preserve level/scale. A zero-term expression uses target
// for a constant-only two-component ciphertext. Constants use int64 rounding.
bool linear(CiphertextView &, const LinearTerm *, std::size_t count, double constant, CiphertextLayout target,
            const EvaluationTables &, Arena &);
// All operands have common basis/scale and two components. Accumulate weighted
// tensors and integer-scale-lifted linear terms, then one relin/rescale. Nonzero
// constants or linear terms require integral common scale fitting signed64.
// Constants are rounded at common scale first, then lifted modulo each prime.
bool productSum(CiphertextView &, const ProductTerm *, std::size_t product_count, const LinearTerm *,
                std::size_t linear_count, double constant, EvaluationKeyView, const EvaluationTables &, Arena &);
bool multiply(CiphertextView &, ConstCiphertextView, ConstCiphertextView, EvaluationKeyView, const EvaluationTables &,
              Arena &);
bool rotateAdd(CiphertextView &, ConstCiphertextView, uint64_t galois, EvaluationKeyView, const LinearTerm *,
               std::size_t count, const EvaluationTables &, Arena &);
// Canonical Shoup products preserve every residue and rounding decision. A
// nonempty KeyProducts also accelerates the fixed P-lift/mod-down multipliers.
// Arrays must be disjoint from destination; malformed views fail before writes.
bool multiplyCached(CiphertextView &, ConstCiphertextView, ConstCiphertextView, EvaluationKeyView, KeyProducts,
                    CipherProducts, const EvaluationTables &, Arena &);
bool rotateAddCached(CiphertextView &, ConstCiphertextView, uint64_t galois, EvaluationKeyView, KeyProducts,
                     RotationMap, const LinearTerm *, std::size_t count, const EvaluationTables &, Arena &);
// Source snapshot and its matching canonical coefficient-domain c1 are immutable.
// Same private disjoint-output contract as rotateAdd; only source inverse NTTs
// are shared. Basis conversion and key switching remain per rotation.
bool rotatePrepared(CiphertextView &, ConstCiphertextView, ConstRnsPolyView coefficients, uint64_t galois,
                    EvaluationKeyView, const EvaluationTables &, Arena &);
// Fractional scaling retains the checked multiplyScalar semantics: one level
// consumed, even for integers/zero. levelDown copies the requested Q prefix.
bool scaleFractional(CiphertextView &, ConstCiphertextView, double, const EvaluationTables &, Arena &);
bool levelDown(CiphertextView &, ConstCiphertextView, uint32_t, const EvaluationTables &, Arena &);
} // namespace fhenomenon::ckks::region
