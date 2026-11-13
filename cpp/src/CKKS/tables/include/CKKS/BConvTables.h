#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// L1 — fast RNS base conversion.
//
// A residue-number-system value carried modulo a source basis
// Q = q_0 * ... * q_{k-1} has to be re-expressed modulo a target basis
// B = p_0 * ... * p_{l-1} at every ModUp and ModDown, which is to say at every
// key switch and every rescale. Doing that exactly would require reconstructing
// the integer, so CKKS uses the "fast" conversion of Bajard et al.: with
//
//     Q_i = Q / q_i,   y_i = x_i * Q_i^{-1} mod q_i   (so 0 <= y_i < q_i)
//
// the sum S = sum_i y_i * Q_i satisfies S = x mod Q by construction, and this
// class computes S mod p_j for each target prime.
//
// S is NOT reduced modulo Q first, and cannot be without the reconstruction
// the scheme is avoiding. So the result is
//
//     x + alpha * Q   (mod p_j),   alpha = floor(sum_i y_i / q_i) < k
//
// The alpha * Q overflow term is inherent to the method, not an approximation
// introduced here: every value is exact, it is the representative that is off
// by a bounded multiple of Q. Callers that need it gone remove it downstream
// (ModDown's subtraction, or an extra prime carrying alpha); callers in the
// key-switching path tolerate it, which is why the method is used at all.
//
// Two consequences worth stating because they are easy to misread as bugs:
// a single-prime source basis converts exactly (y_0 = x_0 < q_0 forces
// alpha = 0), and converting Q -> B -> Q does not round-trip.

namespace fhenomenon::ckks {

class BConvTables {
  public:
  // The same 62-bit ceiling as NttTables, and for the same reason: the
  // accumulator below carries partial sums unreduced under 4p, which must fit
  // a uint64. Stated independently rather than shared because these are two
  // different lazy windows that happen to close at the same width, and because
  // L1 cannot see L0's identical constant either.
  static constexpr uint32_t kMaxModulusBits = 62;

  // A basis and its target, both fully validated. `from` and `to` must each be
  // non-empty, and all k + l primes must be pairwise distinct: the CRT is only
  // a bijection over coprime moduli, and distinct primes is what coprime means
  // here.
  static std::optional<BConvTables> create(const std::vector<uint64_t> &from, const std::vector<uint64_t> &to,
                                           std::string *error = nullptr);

  const std::vector<uint64_t> &fromBasis() const { return from_; }
  const std::vector<uint64_t> &toBasis() const { return to_; }

  // A multiplier and its Shoup companion, laid out as parallel arrays so the
  // conversion loop reads two contiguous streams. Mirrors NttTables::RootTable.
  struct Multipliers {
    const uint64_t *value;
    const uint64_t *quotient;
  };

  // hatInverses().value[i] = (Q / q_i)^{-1} mod q_i, k entries.
  Multipliers hatInverses() const { return {hat_inverse_.data(), hat_inverse_quotient_.data()}; }

  // hatResidues().value[i * toBasis().size() + j] = (Q / q_i) mod p_j, k*l entries,
  // row-major over the source index because the inner loop runs over j.
  Multipliers hatResidues() const { return {hat_residue_.data(), hat_residue_quotient_.data()}; }

  private:
  BConvTables() = default;

  std::vector<uint64_t> from_;
  std::vector<uint64_t> to_;
  std::vector<uint64_t> hat_inverse_;
  std::vector<uint64_t> hat_inverse_quotient_;
  std::vector<uint64_t> hat_residue_;
  std::vector<uint64_t> hat_residue_quotient_;
};

namespace bconv {

// in[i] is the residue modulo fromBasis()[i] and must be fully reduced to
// [0, q_i); out[j] receives the residue modulo toBasis()[j], fully reduced to
// [0, p_j). The lazy representation used internally never escapes. `in` and
// `out` must not overlap: the bases are independent, so the caller has two
// buffers anyway.
void convert(const uint64_t *in, uint64_t *out, const BConvTables &tables);

} // namespace bconv

} // namespace fhenomenon::ckks
