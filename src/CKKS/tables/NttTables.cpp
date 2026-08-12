#include "CKKS/NttTables.h"

#include "CKKS/ModArith.h"

namespace fhenomenon::ckks {

namespace {

bool fail(std::string *error, std::string reason) {
  if (error != nullptr) {
    *error = std::move(reason);
  }
  return false;
}

uint64_t bitReverse(uint64_t index, uint32_t bits) {
  uint64_t result = 0;
  for (uint32_t i = 0; i < bits; ++i) {
    result = (result << 1U) | ((index >> i) & 1U);
  }
  return result;
}

// A primitive 2N-th root of unity mod q, or 0 if the search somehow exhausts.
//
// For a candidate a, psi = a^((q-1)/2N) satisfies psi^N = a^((q-1)/2), which
// is -1 exactly when a is a quadratic non-residue. Half of Z_q^* qualifies, so
// the loop below finds one almost immediately. And psi^N = -1 pins the order
// of psi to exactly 2N rather than a proper divisor: the order divides 2N and
// does not divide N, and since N is a power of two those are all the divisors
// there are.
//
// The scan is ordered rather than random on purpose. Tables keyed by
// (modulus, log_degree) must be reproducible — two processes that build the
// same tables have to agree on the root, or a ciphertext serialised by one is
// noise to the other.
uint64_t findPrimitiveRoot(uint64_t modulus, uint64_t degree) {
  const uint64_t exponent = (modulus - 1) / (2 * degree);
  for (uint64_t candidate = 2; candidate < modulus; ++candidate) {
    const uint64_t root = modarith::powMod(candidate, exponent, modulus);
    if (modarith::powMod(root, degree, modulus) == modulus - 1) {
      return root;
    }
  }
  return 0;
}

// Successive powers of `base`, permuted into bit-reversed index order, with
// the Shoup companion of each. Built by walking the powers once and scattering
// them, so this is O(N) multiplications rather than O(N log N).
void buildRootTable(uint64_t base, uint64_t modulus, uint32_t log_degree, std::vector<uint64_t> &value,
                    std::vector<uint64_t> &quotient) {
  const uint64_t degree = uint64_t{1} << log_degree;
  value.resize(degree);
  quotient.resize(degree);
  uint64_t power = 1;
  for (uint64_t i = 0; i < degree; ++i) {
    const uint64_t slot = bitReverse(i, log_degree);
    value[slot] = power;
    quotient[slot] = modarith::shoupFactor(power, modulus);
    power = modarith::mulMod(power, base, modulus);
  }
}

} // namespace

std::optional<NttTables> NttTables::create(uint64_t modulus, uint32_t log_degree, std::string *error) {
  if (log_degree < kMinLogDegree || log_degree > kMaxLogDegree) {
    fail(error, "log_degree " + std::to_string(log_degree) + " outside [" + std::to_string(kMinLogDegree) + ", " +
                  std::to_string(kMaxLogDegree) + "]");
    return std::nullopt;
  }
  if (modulus >> kMaxModulusBits != 0) {
    fail(error, "modulus " + std::to_string(modulus) + " exceeds " + std::to_string(kMaxModulusBits) +
                  " bits: lazy butterflies carry intermediates up to 4q, which must fit a uint64");
    return std::nullopt;
  }

  const uint64_t degree = uint64_t{1} << log_degree;
  const uint64_t two_n = 2 * degree;
  if (modulus % two_n != 1) {
    fail(error, "modulus " + std::to_string(modulus) + " is not 1 mod 2N (" + std::to_string(two_n) +
                  "), so it admits no primitive 2N-th root of unity");
    return std::nullopt;
  }
  // Checked after the congruence because the congruence is the cheaper test
  // and the likelier typo, and because the root-finding argument above needs
  // Z_q^* to be cyclic of order q-1 — which needs q prime, not merely odd.
  // invMod() below is Fermat's little theorem and silently returns garbage on
  // a composite, so this check is load-bearing rather than defensive.
  if (!modarith::isPrime(modulus)) {
    fail(error, "modulus " + std::to_string(modulus) + " is not prime");
    return std::nullopt;
  }

  const uint64_t root = findPrimitiveRoot(modulus, degree);
  if (root == 0) {
    fail(error,
         "modulus " + std::to_string(modulus) + " admits no primitive " + std::to_string(two_n) + "-th root of unity");
    return std::nullopt;
  }

  NttTables tables;
  tables.modulus_ = modulus;
  tables.log_degree_ = log_degree;
  tables.root_ = root;
  buildRootTable(root, modulus, log_degree, tables.forward_root_, tables.forward_quotient_);
  buildRootTable(modarith::invMod(root, modulus), modulus, log_degree, tables.inverse_root_, tables.inverse_quotient_);
  tables.degree_inverse_ = modarith::invMod(degree % modulus, modulus);
  tables.degree_inverse_quotient_ = modarith::shoupFactor(tables.degree_inverse_, modulus);

  if (error != nullptr) {
    error->clear();
  }
  return tables;
}

namespace ntt {

// Both transforms are the Longa-Naehrig merged form: the powers of psi that
// make the transform negacyclic are folded into the butterfly twiddles, so
// there is no separate pre-scaling pass over the input and no bit-reversal
// pass at either end.
//
// Reduction is lazy in Harvey's sense. Residues are carried unreduced — under
// 4q in forward(), under 2q in inverse() — and brought back to [0, q) only in
// the final pass, which is why the modulus must stay under 2^62. Each
// conditional subtraction below is what keeps a value inside its stated
// window; none of them is a defensive check.

void forward(uint64_t *values, const NttTables &tables) {
  const uint64_t modulus = tables.modulus();
  const uint64_t two_q = 2 * modulus;
  const uint64_t degree = tables.degree();
  const NttTables::RootTable roots = tables.forwardRoots();

  uint64_t gap = degree;
  for (uint64_t m = 1; m < degree; m <<= 1U) {
    gap >>= 1U;
    for (uint64_t i = 0; i < m; ++i) {
      const uint64_t root = roots.value[m + i];
      const uint64_t quotient = roots.quotient[m + i];
      const uint64_t begin = 2 * i * gap;
      for (uint64_t j = begin; j < begin + gap; ++j) {
        // x < 4q on entry; one subtraction puts it under 2q so that the sum
        // and the difference below both stay under 4q.
        uint64_t x = values[j];
        if (x >= two_q) {
          x -= two_q;
        }
        const uint64_t t = modarith::mulModShoupLazy(values[j + gap], root, quotient, modulus);
        values[j] = x + t;
        values[j + gap] = x + two_q - t;
      }
    }
  }

  for (uint64_t i = 0; i < degree; ++i) {
    uint64_t value = values[i];
    if (value >= two_q) {
      value -= two_q;
    }
    values[i] = value >= modulus ? value - modulus : value;
  }
}

void inverse(uint64_t *values, const NttTables &tables) {
  const uint64_t modulus = tables.modulus();
  const uint64_t two_q = 2 * modulus;
  const uint64_t degree = tables.degree();
  const NttTables::RootTable roots = tables.inverseRoots();

  uint64_t gap = 1;
  for (uint64_t m = degree; m > 1; m >>= 1U) {
    const uint64_t half = m >> 1U;
    uint64_t begin = 0;
    for (uint64_t i = 0; i < half; ++i) {
      const uint64_t root = roots.value[half + i];
      const uint64_t quotient = roots.quotient[half + i];
      for (uint64_t j = begin; j < begin + gap; ++j) {
        // Gentleman-Sande: the sum is reduced back under 2q, while the
        // difference is left under 4q because the lazy multiply that consumes
        // it accepts any 64-bit input and returns under 2q regardless.
        const uint64_t x = values[j];
        const uint64_t y = values[j + gap];
        const uint64_t sum = x + y;
        values[j] = sum >= two_q ? sum - two_q : sum;
        values[j + gap] = modarith::mulModShoupLazy(x + two_q - y, root, quotient, modulus);
      }
      begin += 2 * gap;
    }
    gap <<= 1U;
  }

  const uint64_t scale = tables.degreeInverse();
  const uint64_t scale_quotient = tables.degreeInverseQuotient();
  for (uint64_t i = 0; i < degree; ++i) {
    const uint64_t value = modarith::mulModShoupLazy(values[i], scale, scale_quotient, modulus);
    values[i] = value >= modulus ? value - modulus : value;
  }
}

} // namespace ntt

} // namespace fhenomenon::ckks
