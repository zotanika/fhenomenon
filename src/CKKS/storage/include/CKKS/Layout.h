#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// L2 — what a polynomial and a ciphertext ARE, with no statement about where
// their residues live.
//
// Everything here is a trivially copyable value type over indices. Nothing
// holds a pointer, an allocator, a context or a table, which is what lets
// these types exist before any memory does. That is not tidiness: L4's
// scratch-byte estimate and L4's kernels are both written in terms of these
// types, so the figure a planner reports before execution and the figure the
// execution consumes come from the same expression rather than from two that
// have to be kept in step by hand. Requirement R1 is either true here or a
// coincidence everywhere else.
//
// A basis names primes by INDEX RANGE, never by value. Carrying the primes
// would make every ciphertext fat, make basis equality a vector compare, and
// stop every type in this file from being trivially copyable — and the values
// are recoverable from the parameter set the index range already identifies.

namespace fhenomenon::ckks {

class Params; // L0. Forward-declared: only ParamsId::of() needs the definition.

enum class NttForm : uint8_t { Coefficient = 0, Evaluation = 1 };

// A stable 64-bit fingerprint of a parameter set.
//
// Deliberately NOT Params::hash(). That one is built from std::hash, whose
// values are implementation-defined and carry no guarantee across builds or
// standard libraries, so a ciphertext written by one and read by another could
// be rejected — or worse, accepted under the wrong parameters. It is fine for
// what it is for, an in-process cache key, and unusable as a wire identity.
//
// Defined at L2 rather than at L0 because the obligation it creates belongs to
// whoever owns the wire format. The field order below is a compatibility
// surface: it may never change, and every field Params::operator== compares
// must appear in it, or two parameter sets that compare unequal would produce
// ciphertexts claiming to be interchangeable.
//
// A necessary condition, never a proof. Two parameter sets sharing a 64-bit
// fingerprint pass this check; at a trust boundary compare Params directly.
class ParamsId {
  public:
  static ParamsId of(const Params &params);

  constexpr ParamsId() = default;
  constexpr explicit ParamsId(uint64_t value) : value_(value) {}

  constexpr uint64_t value() const { return value_; }
  constexpr bool empty() const { return value_ == 0; }
  constexpr bool operator==(ParamsId other) const { return value_ == other.value_; }
  constexpr bool operator!=(ParamsId other) const { return !(*this == other); }

  private:
  uint64_t value_ = 0;
};

// Which primes the residues are carried modulo, as half-open index ranges into
// the parameter set's own prime lists.
//
// Two ranges rather than one because key switching raises to P*Q and comes
// back down, and the level accounting has to tell the two bases apart — the
// same reason Params keeps aux_primes separate from main_primes. Both ranges
// carry a begin as well as an end, which costs nothing here and is what lets a
// key-switching digit be a basis in its own right:
//
//   ciphertext at level L : {0, L+1, 0, 0}
//   ModUp target          : {0, L+1, 0, |P|}
//   hybrid digit i        : Params::digitRange(i) into main, no aux
struct RnsBasis {
  // A bound on limbs, stated here rather than on PolyLayout because limb count
  // is a property of the basis. It is what keeps wordCount() far inside a
  // size_t at every representable degree, and it is generous: 256 limbs is
  // four times the longest chain any published CKKS parameter set uses.
  static constexpr uint32_t kMaxLimbs = 256;

  ParamsId params_id{};
  uint32_t main_begin = 0;
  uint32_t main_end = 0;
  uint32_t aux_begin = 0;
  uint32_t aux_end = 0;

  uint32_t mainCount() const { return main_end - main_begin; }
  uint32_t auxCount() const { return aux_end - aux_begin; }
  uint32_t limbCount() const { return mainCount() + auxCount(); }
  bool isChainPrefix() const { return main_begin == 0; }
  bool hasAux() const { return aux_end != aux_begin; }

  bool wellFormed(std::string *error = nullptr) const;
  bool operator==(const RnsBasis &other) const;
  bool operator!=(const RnsBasis &other) const { return !(*this == other); }
};

// Everything about one RNS polynomial except where its residues are.
struct PolyLayout {
  // Lower than Params::kMinLogDegree (10) on purpose. L2 has no cryptographic
  // opinion — a degree too small to be secure is still a degree this layer can
  // describe, and refusing it here would only mean L2's own tests could not
  // use small rings. The bound that IS load-bearing is that a limb be a whole
  // number of 64-byte lines: 1 << 3 words of 8 bytes is exactly 64, so from
  // here up every limb stride is a multiple of the alignment kernels want.
  static constexpr uint32_t kMinLogDegree = 3;
  static constexpr uint32_t kMaxLogDegree = 17;

  RnsBasis basis{};
  uint32_t log_degree = 0;
  NttForm form = NttForm::Coefficient;

  uint64_t degree() const { return uint64_t{1} << log_degree; }
  std::size_t degreeWords() const { return std::size_t{1} << log_degree; }

  // wordCount() is size_t because it feeds indexing and allocation; byteCount()
  // is uint64_t because it feeds the planner and the wire, where the answer
  // must not depend on the host's word size. Returning one type from both
  // would put a cast at every call site that -Wuseless-cast then rejects on
  // LP64, which is how a build flag ends up dictating an interface.
  std::size_t wordCount() const { return std::size_t{basis.limbCount()} << log_degree; }
  uint64_t byteCount() const { return (uint64_t{basis.limbCount()} << log_degree) * 8U; }

  bool wellFormed(std::string *error = nullptr) const;
  // Same ring, which is a question about the parameter set and the degree
  // only. Two operands of a key switch differ in live range and in NTT form
  // and are still the same ring.
  bool sameRing(const PolyLayout &other) const {
    return basis.params_id == other.basis.params_id && log_degree == other.log_degree;
  }

  bool operator==(const PolyLayout &other) const;
  bool operator!=(const PolyLayout &other) const { return !(*this == other); }
};

// A ciphertext's shape: some number of polynomials that share one layout,
// plus the scale they carry.
//
// The layout is stored ONCE for all polynomials rather than per polynomial, so
// "poly 0 at level 5 and poly 1 at level 4" is unrepresentable rather than
// merely detectable. There is consequently no consistency check to remember to
// call, and basis and form transitions are ciphertext-level operations: a
// half-transformed ciphertext is not a valid operand at any function boundary.
struct CiphertextLayout {
  // 2 is a ciphertext, 3 is the product before relinearisation, and the fourth
  // slot admits exactly one deferred relinearisation. Raising this costs no
  // bytes in any handle, because a ciphertext is addressed by a base and a
  // uniform stride rather than by an array of pointers.
  static constexpr uint32_t kMaxPolys = 4;

  PolyLayout poly{};
  double scale = 0.0;
  uint32_t num_polys = 0;

  std::size_t wordCount() const { return std::size_t{num_polys} * poly.wordCount(); }
  uint64_t byteCount() const { return uint64_t{num_polys} * poly.byteCount(); }

  // Total rather than optional, because wellFormed() rejects a basis that is
  // not a chain prefix — so a key-switching digit is a PolyLayout and never
  // reaches this type. Matches Params: level L carries L+1 main primes.
  int64_t level() const { return static_cast<int64_t>(poly.basis.main_end) - 1; }

  // Words a slab must hold to back this ciphertext with `limb_capacity` limbs
  // reserved per polynomial. The reserve is per polynomial and sits at each
  // polynomial's own tail, which is what makes RESCALE a decrement of main_end
  // with no memmove and ModUp an append at each live tail.
  std::size_t slabWords(uint32_t limb_capacity) const {
    return std::size_t{num_polys} * (std::size_t{limb_capacity} << poly.log_degree);
  }

  bool wellFormed(std::string *error = nullptr) const;
  bool operator==(const CiphertextLayout &other) const;
  bool operator!=(const CiphertextLayout &other) const { return !(*this == other); }
};

} // namespace fhenomenon::ckks
