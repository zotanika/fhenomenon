#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace fhenomenon::ckks {

// Precomputation for the negacyclic number-theoretic transform over one
// prime — layer L1.
//
// Keyed by (modulus, log_degree) and nothing else. In particular it does not
// take a Params: a transform over a single prime is arithmetic, and a CKKS
// parameter set is not one of its inputs. Taking one would make L1 depend on
// L0 for no arithmetic reason and would drag a parameter set into this
// layer's tests, which is exactly the erosion invariant I5 exists to stop.
//
// Immutable once built and free of interior pointers, so a single instance is
// shared as const across threads. The spec's L1 rule that a table cache is a
// memo table and never session state rests on that.
class NttTables {
  public:
  // The modulus bound is Harvey's, not a preference. Lazy butterflies carry
  // intermediates up to 4q without reducing, so 4q must fit a uint64 and
  // therefore q < 2^62. L0's Params::kMaxPrimeBits is the same number for
  // this reason; the two are stated independently because L1 does not see L0.
  static constexpr uint32_t kMaxModulusBits = 62;
  // Degrees below L0's cryptographic minimum are allowed here on purpose. A
  // transform is correct or not at N = 8 exactly as it is at N = 65536, and a
  // small ring is what makes an O(N^2) reference comparison affordable in a
  // test. Security is L0's bound to set, not this layer's.
  static constexpr uint32_t kMinLogDegree = 1;
  static constexpr uint32_t kMaxLogDegree = 17;

  // A multiplier and its Shoup companion quotient, as parallel arrays indexed
  // together. Parallel rather than interleaved because the vectorised kernels
  // this layer will grow want each stream contiguous.
  struct RootTable {
    const uint64_t *value;
    const uint64_t *quotient;
  };

  // Build the tables for X^N + 1 over Z_modulus. Returns nullopt when no
  // negacyclic transform exists for this pair; when `error` is non-null it
  // receives the first reason.
  static std::optional<NttTables> create(uint64_t modulus, uint32_t log_degree, std::string *error = nullptr);

  uint64_t modulus() const { return modulus_; }
  uint32_t logDegree() const { return log_degree_; }
  uint64_t degree() const { return uint64_t{1} << log_degree_; }
  // The primitive 2N-th root of unity the tables were built from.
  uint64_t root() const { return root_; }

  // Powers of the root in bit-reversed order, which is the order the
  // butterflies below consume them in. Index 0 is unused: the loops address
  // this table from 1, so the slot is kept rather than shifting every index.
  RootTable forwardRoots() const { return {forward_root_.data(), forward_quotient_.data()}; }
  RootTable inverseRoots() const { return {inverse_root_.data(), inverse_quotient_.data()}; }

  // N^-1 mod q and its Shoup companion, folded into the last pass of inverse().
  uint64_t degreeInverse() const { return degree_inverse_; }
  uint64_t degreeInverseQuotient() const { return degree_inverse_quotient_; }

  private:
  NttTables() = default;

  uint64_t modulus_ = 0;
  uint32_t log_degree_ = 0;
  uint64_t root_ = 0;
  std::vector<uint64_t> forward_root_;
  std::vector<uint64_t> forward_quotient_;
  std::vector<uint64_t> inverse_root_;
  std::vector<uint64_t> inverse_quotient_;
  uint64_t degree_inverse_ = 0;
  uint64_t degree_inverse_quotient_ = 0;
};

// The negacyclic transform itself. Free functions rather than members: the
// tables are data, and a transform is not a thing the data does.
//
// Both operate in place on exactly tables.degree() residues and allocate
// nothing — in-place Cooley-Tukey and Gentleman-Sande need no temporaries,
// which is why this layer needs no Arena and sits below L3.
//
// Contract: values are fully reduced to [0, modulus) on entry and on exit.
// The lazy, unreduced representation is an implementation detail that does
// not escape either function.
//
// forward() maps natural order to bit-reversed order and inverse() maps back,
// so neither performs a separate bit-reversal pass. The intermediate order is
// unspecified beyond the fact that the two agree, and it is irrelevant to the
// only thing done in that domain — pointwise multiplication commutes with any
// permutation of the slots.
namespace ntt {

void forward(uint64_t *values, const NttTables &tables);
void inverse(uint64_t *values, const NttTables &tables);

} // namespace ntt

} // namespace fhenomenon::ckks
