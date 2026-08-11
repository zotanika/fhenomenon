#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace fhenomenon::ckks {

// Distribution the secret key is drawn from. SparseTernary exists because
// bootstrapping's sparse-secret encapsulation needs it. It is representable
// here before any bootstrapping component exists, so adding one later does
// not reshape Params (spec: Non-goals v1).
enum class SecretDistribution { UniformTernary, SparseTernary };

// An immutable, hashable CKKS parameter set — layer L0.
//
// Params owns nothing but its own prime lists, holds no precomputation, and
// points at nothing. That is what lets two parameter sets coexist in one
// process without arbitration (requirement R4), and what lets a Params value
// be used as the identity key of a precomputation table at L1.
//
// Level convention: a ciphertext "at level L" lives modulo q_0 * ... * q_L,
// so it carries L+1 main primes. Fresh ciphertexts sit at freshLevel() and
// RESCALE drops one. The FHN level-model exports (fhn_fresh_level /
// fhn_level_bytes / fhn_opcode_level_effect) are defined against exactly this
// convention, which is why ciphertextBytes() below can serve them directly
// instead of a hand-maintained table (requirement R5).
class Params {
  public:
  // Primes are kept under 2^kMaxPrimeBits so that products of two residues
  // stay within a 128-bit intermediate with room for lazy reduction.
  static constexpr uint32_t kMaxPrimeBits = 62;
  static constexpr uint32_t kMinLogDegree = 10;
  static constexpr uint32_t kMaxLogDegree = 17;

  // Unvalidated input to create(). Nothing here is derived or defaulted
  // behind the caller's back — in particular `dnum` is always an explicit
  // choice. Key switching is hybrid parameterised by dnum, with dnum == 1
  // degenerating to GHS (one digit, so P must cover the whole chain) and
  // dnum == levelCount() to a BV-style decomposition. Hiding that number
  // behind a default would turn a measurable trade-off into a hidden one.
  struct Spec {
    uint32_t log_degree = 0;
    // The main modulus chain q_0 .. q_{L-1}, bottom prime first.
    std::vector<uint64_t> main_primes;
    // The special primes P of hybrid key switching. Kept separate from the
    // main chain because key switching raises to P*Q and comes back down, and
    // the level accounting has to be able to tell the two bases apart.
    std::vector<uint64_t> aux_primes;
    uint32_t dnum = 1;
    uint32_t log_default_scale = 0;
    SecretDistribution secret = SecretDistribution::UniformTernary;

    // Bootstrapping surface: representable, unused in v1.
    uint32_t sparse_hamming_weight = 0;
    std::vector<uint64_t> terminal_primes;
  };

  // Validate a spec into a parameter set. Returns nullopt when the spec is
  // not a usable CKKS parameter set; when `error` is non-null it receives the
  // first reason, because a bare nullopt is close to useless for something a
  // human authors by hand.
  static std::optional<Params> create(Spec spec, std::string *error = nullptr);

  uint32_t logDegree() const { return spec_.log_degree; }
  uint64_t degree() const { return uint64_t{1} << spec_.log_degree; }
  uint64_t numSlots() const { return degree() / 2; }

  const std::vector<uint64_t> &mainPrimes() const { return spec_.main_primes; }
  const std::vector<uint64_t> &auxPrimes() const { return spec_.aux_primes; }
  const std::vector<uint64_t> &terminalPrimes() const { return spec_.terminal_primes; }

  uint32_t dnum() const { return spec_.dnum; }
  // Main primes per key-switching digit, rounded up. The last digit may be
  // short when dnum does not divide levelCount().
  uint32_t alpha() const;
  // Half-open range [begin, end) of main-prime indices belonging to digit i.
  std::pair<uint32_t, uint32_t> digitRange(uint32_t digit) const;

  uint32_t logDefaultScale() const { return spec_.log_default_scale; }
  SecretDistribution secret() const { return spec_.secret; }
  uint32_t sparseHammingWeight() const { return spec_.sparse_hamming_weight; }

  // Number of main primes, i.e. the number of usable levels.
  uint32_t levelCount() const { return static_cast<uint32_t>(spec_.main_primes.size()); }
  // Level of a freshly encrypted ciphertext: levelCount() - 1.
  int64_t freshLevel() const { return static_cast<int64_t>(levelCount()) - 1; }

  // Bytes a ciphertext of `num_polys` polynomials occupies at `level`.
  // Returns 0 for a level outside [0, freshLevel()], matching the
  // fhn_level_bytes contract where 0 means "invalid level".
  uint64_t ciphertextBytes(int64_t level, uint32_t num_polys = 2) const;

  bool operator==(const Params &other) const;
  bool operator!=(const Params &other) const { return !(*this == other); }

  // Identity of this parameter set, for use as a precomputation cache key.
  std::size_t hash() const;

  private:
  explicit Params(Spec spec) : spec_(std::move(spec)) {}

  Spec spec_;
};

} // namespace fhenomenon::ckks

namespace std {

template <> struct hash<fhenomenon::ckks::Params> {
  std::size_t operator()(const fhenomenon::ckks::Params &p) const { return p.hash(); }
};

} // namespace std
