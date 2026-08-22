// Tests for L1 — the NTT tables and the negacyclic transform.
//
// This executable links fhn_ckks_tables and nothing else: not L0, not L3, not
// ${PROJECT_LIB_NAME}. That is the check invariant I5 actually buys here. A
// negacyclic NTT over a single prime is arithmetic over a span of residues —
// it has no ciphertext, no key, and no scratch — so if this test ever needs
// something from another layer in order to build, the layering has already
// failed and the link fails here rather than a reviewer having to notice.

#include "CKKS/ModArith.h"
#include "CKKS/NttTables.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using fhenomenon::ckks::NttTables;

// 12289 = 6 * 2048 + 1: the classic small NTT-friendly prime. It is 1 mod 4096
// as well (12288 = 2^12 * 3), so it serves every degree up to N = 2048.
constexpr uint64_t kSmallPrime = 12289;

TEST(NttTables, RejectsAModulusWithNoPrimitiveRootOfUnity) {
  // 12289 mod 8192 == 4097, so at log_degree 12 there is no primitive 2N-th
  // root of unity and the negacyclic transform does not exist.
  std::string error;
  EXPECT_FALSE(NttTables::create(kSmallPrime, 12, &error).has_value());
  EXPECT_NE(error.find("1 mod 2N"), std::string::npos) << "actual: " << error;
}

TEST(NttTables, RejectsACompositeModulus) {
  // 2049 = 3 * 683 is 1 mod 16, so it passes the congruence check and would
  // still produce silent garbage: the root-finding argument below needs the
  // multiplicative group to be cyclic of order q-1, which needs q prime.
  std::string error;
  EXPECT_FALSE(NttTables::create(2049, 3, &error).has_value());
  EXPECT_NE(error.find("not prime"), std::string::npos) << "actual: " << error;
}

TEST(NttTables, RejectsAModulusWiderThanTheLazyReductionBound) {
  // 9223372036854775073 is a 63-bit prime and 1 mod 16, so only the width
  // check can reject it. Harvey's lazy butterflies keep intermediates below
  // 4q, which fits a uint64 only while q < 2^62.
  std::string error;
  EXPECT_FALSE(NttTables::create(9223372036854775073ULL, 3, &error).has_value());
  EXPECT_NE(error.find("62"), std::string::npos) << "actual: " << error;
}

TEST(NttTables, AcceptsAnNttFriendlyPrimeAndReportsItsRing) {
  std::string error = "not cleared";
  const auto tables = NttTables::create(kSmallPrime, 10, &error);

  ASSERT_TRUE(tables.has_value()) << error;
  EXPECT_EQ(tables->modulus(), kSmallPrime);
  EXPECT_EQ(tables->logDegree(), 10U);
  EXPECT_EQ(tables->degree(), 1024U);
  EXPECT_TRUE(error.empty()) << "create() left a stale error on success: " << error;
}

TEST(NttTables, RootHasOrderExactlyTwoN) {
  const auto tables = NttTables::create(kSmallPrime, 10, nullptr);
  ASSERT_TRUE(tables.has_value());

  // psi^N == -1 is the whole negacyclic property: it is what makes X^N + 1
  // split, and for a power-of-two N it also pins the order of psi to exactly
  // 2N rather than merely dividing it.
  uint64_t power = 1;
  for (uint64_t i = 0; i < tables->degree(); ++i) {
    power = power * tables->root() % kSmallPrime;
  }
  EXPECT_EQ(power, kSmallPrime - 1);
}

// Schoolbook multiplication in Z_q[X]/(X^N + 1), used as the oracle for the
// transform. Deliberately written without any helper from the code under
// test: with q = 12289 every product stays under 2^28, so this needs no
// modular-multiplication routine at all and shares no code with ModArith.h.
// The sign flip on wrap is the negacyclic property being asserted.
std::vector<uint64_t> schoolbookNegacyclicMultiply(const std::vector<uint64_t> &a, const std::vector<uint64_t> &b,
                                                   uint64_t q) {
  const std::size_t n = a.size();
  std::vector<uint64_t> out(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const uint64_t term = a[i] * b[j] % q;
      const std::size_t k = (i + j) % n;
      out[k] = (i + j < n) ? (out[k] + term) % q : (out[k] + q - term) % q;
    }
  }
  return out;
}

// A deterministic filler. std::rand would make a failure unreproducible, and
// the point of a counterexample is that it can be replayed.
//
// The LCG output is passed through the SplitMix64 finaliser rather than being
// shifted down, because the residues must reach the full width of the modulus.
// An earlier version took `state >> 33`, which is 31 bits: against a 62-bit
// prime it meant the "at the modulus bound" tests below never presented a
// residue above 2^31, and the wide lazy windows they exist to exercise were
// never driven anywhere near their limits. Taking the raw LCG state instead
// would fix the width and break the quality, since the low bits of a power-of-
// two LCG have a short period; the finaliser gives full width at every modulus.
uint64_t mix(uint64_t z) {
  z += 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31U);
}

std::vector<uint64_t> pseudoRandomResidues(std::size_t count, uint64_t q, uint64_t seed) {
  std::vector<uint64_t> out(count);
  uint64_t state = seed;
  for (uint64_t &value : out) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    value = mix(state) % q;
  }
  return out;
}

TEST(Ntt, PointwiseProductInTheTransformDomainIsNegacyclicConvolution) {
  constexpr uint32_t kLogDegree = 3;
  const auto tables = NttTables::create(kSmallPrime, kLogDegree, nullptr);
  ASSERT_TRUE(tables.has_value());
  const std::size_t n = tables->degree();

  for (uint64_t seed = 1; seed <= 64; ++seed) {
    const std::vector<uint64_t> a = pseudoRandomResidues(n, kSmallPrime, seed);
    const std::vector<uint64_t> b = pseudoRandomResidues(n, kSmallPrime, seed + 1000);

    std::vector<uint64_t> product = a;
    std::vector<uint64_t> other = b;
    fhenomenon::ckks::ntt::forward(product.data(), *tables);
    fhenomenon::ckks::ntt::forward(other.data(), *tables);
    for (std::size_t i = 0; i < n; ++i) {
      product[i] = product[i] * other[i] % kSmallPrime;
    }
    fhenomenon::ckks::ntt::inverse(product.data(), *tables);

    EXPECT_EQ(product, schoolbookNegacyclicMultiply(a, b, kSmallPrime)) << "seed " << seed;
  }
}

TEST(Ntt, RoundTripRecoversTheInput) {
  const auto tables = NttTables::create(kSmallPrime, 10, nullptr);
  ASSERT_TRUE(tables.has_value());

  for (uint64_t seed = 1; seed <= 32; ++seed) {
    const std::vector<uint64_t> original = pseudoRandomResidues(tables->degree(), kSmallPrime, seed);
    std::vector<uint64_t> values = original;
    fhenomenon::ckks::ntt::forward(values.data(), *tables);
    fhenomenon::ckks::ntt::inverse(values.data(), *tables);

    EXPECT_EQ(values, original) << "seed " << seed;
  }
}

// The transform carries residues unreduced between passes, so those windows
// fitting a uint64 is the entire justification for kMaxModulusBits. At 62 bits
// there are two spare bits and no more; an off-by-one in any of the lazy
// windows wraps here and nowhere else, which is why this case is separate from
// the 14-bit one above rather than folded into it.
//
// Swept over seeds, not drawn once. A single draw is not enough and that is
// not hypothetical: narrowing inverse()'s Gentleman-Sande sum window from 2q
// to q leaves the one-draw version of this test passing, because the failure
// appears for only about one input in six. Seed 4 — what this test used to
// pin — is one of the passing ones.
TEST(Ntt, RoundTripRecoversTheInputAtTheModulusBound) {
  constexpr uint64_t kWidePrime = 4611686018427365377ULL; // 62 bits, 1 mod 2048
  const auto tables = NttTables::create(kWidePrime, 10, nullptr);
  ASSERT_TRUE(tables.has_value());

  for (uint64_t seed = 1; seed <= 32; ++seed) {
    const std::vector<uint64_t> original = pseudoRandomResidues(tables->degree(), kWidePrime, seed);
    std::vector<uint64_t> values = original;

    fhenomenon::ckks::ntt::forward(values.data(), *tables);
    // forward() promises fully reduced output too, and nothing else in this
    // file checks it. Without this loop the final 2q subtraction in forward()
    // can be deleted outright and every test still passes, because every other
    // test launders the result — two feed it back into inverse(), which
    // tolerates unreduced input, and the convolution tests reduce modulo a
    // 14-bit prime where even 4q squared still fits a uint64. The first real
    // consumer to multiply two 62-bit forward outputs would overflow instead.
    for (uint64_t value : values) {
      ASSERT_LT(value, kWidePrime) << "forward() leaked an unreduced residue, seed " << seed;
    }

    fhenomenon::ckks::ntt::inverse(values.data(), *tables);

    EXPECT_EQ(values, original) << "seed " << seed;
    for (uint64_t value : values) {
      ASSERT_LT(value, kWidePrime) << "inverse() leaked an unreduced residue";
    }
  }
}

// X^(N-1) * X = X^N = -1 in Z_q[X]/(X^N + 1). The expected answer is written
// down rather than computed, so this pins the negacyclic sign without relying
// on any multiplication outside the code path being tested.
TEST(Ntt, MultiplyingPastTheRingDegreeWrapsWithANegativeSign) {
  constexpr uint64_t kWidePrime = 4611686018427365377ULL;
  const auto tables = NttTables::create(kWidePrime, 10, nullptr);
  ASSERT_TRUE(tables.has_value());
  const std::size_t n = tables->degree();

  std::vector<uint64_t> top(n, 0);
  top[n - 1] = 1;
  std::vector<uint64_t> shift(n, 0);
  shift[1] = 1;

  fhenomenon::ckks::ntt::forward(top.data(), *tables);
  fhenomenon::ckks::ntt::forward(shift.data(), *tables);
  for (std::size_t i = 0; i < n; ++i) {
    top[i] = fhenomenon::ckks::modarith::mulMod(top[i], shift[i], kWidePrime);
  }
  fhenomenon::ckks::ntt::inverse(top.data(), *tables);

  std::vector<uint64_t> expected(n, 0);
  expected[0] = kWidePrime - 1;
  EXPECT_EQ(top, expected);
}

TEST(NttTables, RejectsALogDegreeOutsideTheSupportedRange) {
  std::string error;
  EXPECT_FALSE(NttTables::create(kSmallPrime, 0, &error).has_value());
  EXPECT_NE(error.find("log_degree"), std::string::npos) << "actual: " << error;
  EXPECT_FALSE(NttTables::create(kSmallPrime, NttTables::kMaxLogDegree + 1, &error).has_value());
}

} // namespace
