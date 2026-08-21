#include "CKKS/BConvTables.h"

#include "CKKS/ModArith.h"

#include <cstddef>

namespace fhenomenon::ckks {

namespace {

bool fail(std::string *error, std::string reason) {
  if (error != nullptr) {
    *error = std::move(reason);
  }
  return false;
}

// Width and primality, checked for one basis. Width first: it is the cheaper
// test, and a 63-bit value reaching isPrime() would be answered correctly and
// then rejected anyway.
bool validateBasis(const std::vector<uint64_t> &basis, const char *name, std::string *error) {
  if (basis.empty()) {
    return fail(error, std::string{name} + " basis is empty");
  }
  for (uint64_t modulus : basis) {
    if (modulus >> BConvTables::kMaxModulusBits != 0) {
      return fail(error, "modulus " + std::to_string(modulus) + " in the " + name + " basis exceeds " +
                           std::to_string(BConvTables::kMaxModulusBits) +
                           " bits: the conversion accumulates partial sums up to 4p, which must fit a uint64");
    }
    // The inverse below is Fermat's little theorem and returns silent garbage
    // on a composite, exactly as in NttTables::create. Load-bearing, not
    // defensive.
    if (!modarith::isPrime(modulus)) {
      return fail(error, "modulus " + std::to_string(modulus) + " in the " + name + " basis is not prime");
    }
  }
  return true;
}

} // namespace

std::optional<BConvTables> BConvTables::create(const std::vector<uint64_t> &from, const std::vector<uint64_t> &to,
                                               std::string *error) {
  if (!validateBasis(from, "source", error) || !validateBasis(to, "target", error)) {
    return std::nullopt;
  }

  // Distinctness is coprimality here, and it is needed in three places at
  // once: Q must be squarefree for Q/q_i to be invertible mod q_i, the CRT
  // over the target basis must be a bijection, and a target prime dividing Q
  // would make its whole column the conversion of zero.
  for (std::size_t i = 0; i < from.size(); ++i) {
    for (std::size_t l = i + 1; l < from.size(); ++l) {
      if (from[i] == from[l]) {
        fail(error, "modulus " + std::to_string(from[i]) + " is repeated in the source basis");
        return std::nullopt;
      }
    }
    for (uint64_t target : to) {
      if (from[i] == target) {
        fail(error, "modulus " + std::to_string(target) + " appears in both bases");
        return std::nullopt;
      }
    }
  }
  for (std::size_t j = 0; j < to.size(); ++j) {
    for (std::size_t l = j + 1; l < to.size(); ++l) {
      if (to[j] == to[l]) {
        fail(error, "modulus " + std::to_string(to[j]) + " is repeated in the target basis");
        return std::nullopt;
      }
    }
  }

  BConvTables tables;
  tables.from_ = from;
  tables.to_ = to;
  tables.hat_inverse_.resize(from.size());
  tables.hat_inverse_quotient_.resize(from.size());
  tables.hat_residue_.resize(from.size() * to.size());
  tables.hat_residue_quotient_.resize(from.size() * to.size());

  // Q_i = Q / q_i is never formed as an integer — it does not fit. Only its
  // residues are needed, one modulo q_i for the inverse and one modulo each
  // p_j for the conversion, and each of those is a product of the other source
  // primes reduced as it goes.
  for (std::size_t i = 0; i < from.size(); ++i) {
    uint64_t hat_mod_q = 1 % from[i];
    for (std::size_t l = 0; l < from.size(); ++l) {
      if (l != i) {
        hat_mod_q = modarith::mulMod(hat_mod_q, from[l] % from[i], from[i]);
      }
    }
    tables.hat_inverse_[i] = modarith::invMod(hat_mod_q, from[i]);
    tables.hat_inverse_quotient_[i] = modarith::shoupFactor(tables.hat_inverse_[i], from[i]);

    for (std::size_t j = 0; j < to.size(); ++j) {
      uint64_t hat_mod_p = 1 % to[j];
      for (std::size_t l = 0; l < from.size(); ++l) {
        if (l != i) {
          hat_mod_p = modarith::mulMod(hat_mod_p, from[l] % to[j], to[j]);
        }
      }
      const std::size_t slot = i * to.size() + j;
      tables.hat_residue_[slot] = hat_mod_p;
      tables.hat_residue_quotient_[slot] = modarith::shoupFactor(hat_mod_p, to[j]);
    }
  }

  if (error != nullptr) {
    error->clear();
  }
  return tables;
}

namespace bconv {

void convert(const uint64_t *in, uint64_t *out, const BConvTables &tables) {
  const std::vector<uint64_t> &from = tables.fromBasis();
  const std::vector<uint64_t> &to = tables.toBasis();
  const BConvTables::Multipliers hat_inverses = tables.hatInverses();
  const BConvTables::Multipliers hat_residues = tables.hatResidues();

  for (std::size_t j = 0; j < to.size(); ++j) {
    out[j] = 0;
  }

  for (std::size_t i = 0; i < from.size(); ++i) {
    // y_i must be fully reduced, not merely under 2q_i. The lazy form differs
    // from the reduced one by q_i, and y_i is weighted by Q_i, so leaving it
    // unreduced shifts the sum by exactly Q — which is invisible modulo Q and
    // very visible modulo p_j, and would push alpha past its stated bound.
    const uint64_t lazy = modarith::mulModShoupLazy(in[i], hat_inverses.value[i], hat_inverses.quotient[i], from[i]);
    const uint64_t y = lazy >= from[i] ? lazy - from[i] : lazy;

    for (std::size_t j = 0; j < to.size(); ++j) {
      const uint64_t two_p = 2 * to[j];
      const std::size_t slot = i * to.size() + j;
      // The Shoup product lands in [0, 2p) for any 64-bit y, and out[j] is
      // already under 2p, so the sum is under 4p and one subtraction restores
      // the window. This is the 62-bit ceiling being spent.
      const uint64_t term = modarith::mulModShoupLazy(y, hat_residues.value[slot], hat_residues.quotient[slot], to[j]);
      const uint64_t sum = out[j] + term;
      out[j] = sum >= two_p ? sum - two_p : sum;
    }
  }

  for (std::size_t j = 0; j < to.size(); ++j) {
    out[j] = out[j] >= to[j] ? out[j] - to[j] : out[j];
  }
}

} // namespace bconv

} // namespace fhenomenon::ckks
