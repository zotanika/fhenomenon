// Tests for L1 — fast RNS base conversion.
//
// Like NttTest, this executable links fhn_ckks_tables and nothing else. A
// base conversion is a function of two prime bases; it has no ciphertext, no
// key and no arena, so if this test ever needs another layer to build, the
// layering has already failed.
//
// The oracle below is a small arbitrary-precision integer, deliberately
// sharing no code with the implementation: the conversion is defined as
// S = sum_i y_i * (Q / q_i) reduced modulo each target prime, and that sum is
// computed here exactly, in bignum, with modular inverses from the extended
// Euclidean algorithm rather than the Fermat exponentiation ModArith.h uses.
// Two different algorithms agreeing is evidence; one algorithm agreeing with
// itself is not.

#include "CKKS/BConvTables.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

using fhenomenon::ckks::BConvTables;

// --- the oracle -----------------------------------------------------------
//
// Little-endian base 2^32. Only three operations are needed: multiply by a
// 64-bit scalar, add, and reduce modulo a 64-bit scalar. The reduction is
// bitwise long division rather than the usual limbwise form because the
// moduli here reach 62 bits: a limbwise remainder would need a 94-bit
// intermediate, and this file must not name a 128-bit type.

using Big = std::vector<uint32_t>;

Big bigFrom(uint64_t value) {
  Big out{static_cast<uint32_t>(value & 0xFFFFFFFFULL), static_cast<uint32_t>(value >> 32U)};
  while (out.size() > 1 && out.back() == 0) {
    out.pop_back();
  }
  return out;
}

void bigAddInPlace(Big &a, const Big &b) {
  a.resize(std::max(a.size(), b.size()) + 1, 0);
  uint64_t carry = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const uint64_t sum = uint64_t{a[i]} + (i < b.size() ? uint64_t{b[i]} : 0) + carry;
    a[i] = static_cast<uint32_t>(sum & 0xFFFFFFFFULL);
    carry = sum >> 32U;
  }
  while (a.size() > 1 && a.back() == 0) {
    a.pop_back();
  }
}

// Multiplication by a 32-bit half, shifted up by `limb_shift` limbs. The
// 64-bit scalar case is two of these, which is why the scalar is split: a
// full 32x64 limb product would overflow.
Big bigMulHalf(const Big &a, uint32_t half, std::size_t limb_shift) {
  if (half == 0) {
    return Big{0};
  }
  Big out(a.size() + limb_shift + 1, 0);
  uint64_t carry = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const uint64_t product = uint64_t{a[i]} * uint64_t{half} + carry;
    out[i + limb_shift] = static_cast<uint32_t>(product & 0xFFFFFFFFULL);
    carry = product >> 32U;
  }
  out[a.size() + limb_shift] = static_cast<uint32_t>(carry);
  while (out.size() > 1 && out.back() == 0) {
    out.pop_back();
  }
  return out;
}

Big bigMul(const Big &a, uint64_t scalar) {
  Big out = bigMulHalf(a, static_cast<uint32_t>(scalar & 0xFFFFFFFFULL), 0);
  bigAddInPlace(out, bigMulHalf(a, static_cast<uint32_t>(scalar >> 32U), 1));
  return out;
}

uint64_t bigMod(const Big &a, uint64_t modulus) {
  uint64_t remainder = 0;
  for (std::size_t limb = a.size(); limb-- > 0;) {
    for (uint32_t bit = 32; bit-- > 0;) {
      remainder = (remainder << 1U) | ((a[limb] >> bit) & 1U);
      if (remainder >= modulus) {
        remainder -= modulus;
      }
    }
  }
  return remainder;
}

// Extended Euclid, so that the oracle's inverse and ModArith.h's Fermat
// inverse are independent computations of the same quantity.
uint64_t inverseByEuclid(uint64_t a, uint64_t modulus) {
  int64_t previous_coefficient = 1;
  int64_t coefficient = 0;
  uint64_t previous_remainder = a % modulus;
  uint64_t remainder = modulus;
  while (remainder != 0) {
    const uint64_t quotient = previous_remainder / remainder;
    const uint64_t next_remainder = previous_remainder - quotient * remainder;
    previous_remainder = remainder;
    remainder = next_remainder;
    // The Bezout coefficients stay bounded by the modulus, so this fits an
    // int64 for every modulus this class accepts.
    const int64_t next_coefficient = previous_coefficient - static_cast<int64_t>(quotient) * coefficient;
    previous_coefficient = coefficient;
    coefficient = next_coefficient;
  }
  return previous_coefficient < 0 ? static_cast<uint64_t>(previous_coefficient + static_cast<int64_t>(modulus))
                                  : static_cast<uint64_t>(previous_coefficient);
}

// The defining sum, computed exactly. Returns S mod p_j for each p_j in `to`.
std::vector<uint64_t> exactConversion(const std::vector<uint64_t> &in, const std::vector<uint64_t> &from,
                                      const std::vector<uint64_t> &to) {
  Big sum{0};
  for (std::size_t i = 0; i < from.size(); ++i) {
    Big hat{1};
    for (std::size_t l = 0; l < from.size(); ++l) {
      if (l != i) {
        hat = bigMul(hat, from[l]);
      }
    }
    const uint64_t hat_inverse = inverseByEuclid(bigMod(hat, from[i]), from[i]);
    const uint64_t y = bigMod(bigMul(bigFrom(in[i]), hat_inverse), from[i]);
    bigAddInPlace(sum, bigMul(hat, y));
  }

  std::vector<uint64_t> out(to.size());
  for (std::size_t j = 0; j < to.size(); ++j) {
    out[j] = bigMod(sum, to[j]);
  }
  return out;
}

// --- fixtures -------------------------------------------------------------

// Small NTT-friendly primes: the arithmetic is easy to follow by hand and a
// failure prints legibly.
const std::vector<uint64_t> kSmallFrom{12289, 40961, 65537};
const std::vector<uint64_t> kSmallTo{114689, 147457};

// 62-bit primes, all 1 mod 2048. The width is the point: the accumulator
// carries partial sums under 4p, so at 62 bits there are exactly two spare
// bits and an off-by-one in the lazy window wraps here and nowhere else.
constexpr uint64_t kWideA = 4611686018427365377ULL;
constexpr uint64_t kWideB = 4611686018427322369ULL;
constexpr uint64_t kWideC = 4611686018427289601ULL;
constexpr uint64_t kWideD = 4611686018427277313ULL;
constexpr uint64_t kWideE = 4611686018427246593ULL;

// Sixteen more of the same family, for the long-basis case below. A real
// CKKS chain is this length, and the accumulator's behaviour depends on how
// many terms are summed before the final reduction, so basis length is a
// dimension the short fixtures do not cover.
const std::vector<uint64_t> kLongWideBasis{4611686018427228161ULL, 4611686018427215873ULL, 4611686018427199489ULL,
                                           4611686018427185153ULL, 4611686018427156481ULL, 4611686018427136001ULL,
                                           4611686018427045889ULL, 4611686018427013121ULL, 4611686018426963969ULL,
                                           4611686018426953729ULL, 4611686018426933249ULL, 4611686018426884097ULL,
                                           4611686018426877953ULL};

std::vector<uint64_t> pseudoRandomResidues(const std::vector<uint64_t> &basis, uint64_t seed) {
  std::vector<uint64_t> out(basis.size());
  uint64_t state = seed;
  for (std::size_t i = 0; i < basis.size(); ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    out[i] = state % basis[i];
  }
  return out;
}

// One thing these tests do NOT cover, recorded so that green is not mistaken
// for complete. The conversion carries its accumulator unreduced under 2p and
// subtracts 2p when it exceeds that; mutating the threshold to p leaves every
// test here passing. That is not a missing test case, it is a property of the
// mutant: the loop becomes out <- out + term - p, and the lazy Shoup product
// exceeds p only when its input y satisfies a condition of probability at most
// y / 2^64 < 1/4, so E[term] < 0.6p and the walk drifts downward instead of
// overflowing. Driving it upward would need that probability above 1/2, which
// no modulus under 2^62 can produce. The stated window is therefore justified
// by the bound in ModArith.h rather than by anything below, and an off-by-one
// in it is a review question, not a test question.

// --- validation -----------------------------------------------------------

TEST(BConvTables, RejectsAnEmptyBasis) {
  std::string error;
  EXPECT_FALSE(BConvTables::create({}, kSmallTo, &error).has_value());
  EXPECT_NE(error.find("empty"), std::string::npos) << "actual: " << error;
  EXPECT_FALSE(BConvTables::create(kSmallFrom, {}, &error).has_value());
  EXPECT_NE(error.find("empty"), std::string::npos) << "actual: " << error;
}

TEST(BConvTables, RejectsACompositeModulus) {
  // The inverse of Q/q_i is taken modulo q_i by Fermat, which returns silent
  // garbage on a composite. 2047 = 23 * 89 is odd and coprime to everything
  // else here, so only a primality test rejects it.
  std::string error;
  EXPECT_FALSE(BConvTables::create({12289, 2047}, kSmallTo, &error).has_value());
  EXPECT_NE(error.find("not prime"), std::string::npos) << "actual: " << error;
  EXPECT_FALSE(BConvTables::create(kSmallFrom, {2047}, &error).has_value());
  EXPECT_NE(error.find("not prime"), std::string::npos) << "actual: " << error;
}

TEST(BConvTables, RejectsAModulusWiderThanTheLazyAccumulationBound) {
  // A 63-bit prime. Nothing else rejects it: it is prime and coprime to the
  // rest, and only 4p overflowing a uint64 makes it unusable.
  constexpr uint64_t kTooWide = 9223372036854775073ULL;
  std::string error;
  EXPECT_FALSE(BConvTables::create({kTooWide}, kSmallTo, &error).has_value());
  EXPECT_NE(error.find("62"), std::string::npos) << "actual: " << error;
  EXPECT_FALSE(BConvTables::create(kSmallFrom, {kTooWide}, &error).has_value());
  EXPECT_NE(error.find("62"), std::string::npos) << "actual: " << error;
}

TEST(BConvTables, RejectsAPrimeRepeatedWithinABasis) {
  // Q would then not be squarefree and Q/q_i would still be divisible by q_i,
  // so the inverse that the whole method rests on does not exist.
  std::string error;
  EXPECT_FALSE(BConvTables::create({12289, 40961, 12289}, kSmallTo, &error).has_value());
  EXPECT_NE(error.find("repeated"), std::string::npos) << "actual: " << error;
}

TEST(BConvTables, RejectsAPrimeSharedBetweenTheTwoBases) {
  // Not a coprimality failure inside either basis, but the pair still has to
  // be coprime: a target prime dividing Q makes its output identically the
  // conversion of zero, which is silently wrong rather than loudly wrong.
  std::string error;
  EXPECT_FALSE(BConvTables::create(kSmallFrom, {114689, 40961}, &error).has_value());
  EXPECT_NE(error.find("both bases"), std::string::npos) << "actual: " << error;
}

TEST(BConvTables, AcceptsTwoDisjointPrimeBasesAndReportsThem) {
  std::string error = "not cleared";
  const auto tables = BConvTables::create(kSmallFrom, kSmallTo, &error);

  ASSERT_TRUE(tables.has_value()) << error;
  EXPECT_EQ(tables->fromBasis(), kSmallFrom);
  EXPECT_EQ(tables->toBasis(), kSmallTo);
  EXPECT_TRUE(error.empty()) << "create() left a stale error on success: " << error;
}

// --- conversion -----------------------------------------------------------

// Swept over many inputs rather than one. Several branches inside the
// conversion are taken only for some residues — the Shoup product exceeds its
// modulus roughly half the time, and the reduction that follows is invisible
// on the draws where it does not. A single draw was in fact not enough: it
// left that reduction uncovered, and only sweeping found it.
TEST(BConv, MatchesTheExactCrtSum) {
  const auto tables = BConvTables::create(kSmallFrom, kSmallTo, nullptr);
  ASSERT_TRUE(tables.has_value());

  for (uint64_t seed = 1; seed <= 64; ++seed) {
    const std::vector<uint64_t> in = pseudoRandomResidues(kSmallFrom, seed);
    std::vector<uint64_t> out(kSmallTo.size(), 0);
    fhenomenon::ckks::bconv::convert(in.data(), out.data(), *tables);

    EXPECT_EQ(out, exactConversion(in, kSmallFrom, kSmallTo)) << "seed " << seed;
  }
}

TEST(BConv, MatchesTheExactCrtSumAtTheModulusBound) {
  const std::vector<uint64_t> from{kWideA, kWideB, kWideC};
  const std::vector<uint64_t> to{kWideD, kWideE};
  const auto tables = BConvTables::create(from, to, nullptr);
  ASSERT_TRUE(tables.has_value());

  for (uint64_t seed = 1; seed <= 64; ++seed) {
    const std::vector<uint64_t> in = pseudoRandomResidues(from, seed);
    std::vector<uint64_t> out(to.size(), 0);
    fhenomenon::ckks::bconv::convert(in.data(), out.data(), *tables);

    EXPECT_EQ(out, exactConversion(in, from, to)) << "seed " << seed;
    for (std::size_t j = 0; j < to.size(); ++j) {
      EXPECT_LT(out[j], to[j]) << "convert() leaked an unreduced residue";
    }
  }
}

// Every source residue at its maximum drives the lazy accumulator as hard as
// the representation allows, and a wider source basis means more additions
// before the final reduction. If the unreduced window is off by one, this is
// where it wraps.
TEST(BConv, MatchesTheExactCrtSumWithEveryResidueAtItsMaximum) {
  const std::vector<uint64_t> from{kWideA, kWideB, kWideC, kWideD};
  const std::vector<uint64_t> to{12289, 40961, kWideE};
  const auto tables = BConvTables::create(from, to, nullptr);
  ASSERT_TRUE(tables.has_value());

  std::vector<uint64_t> in(from.size());
  for (std::size_t i = 0; i < from.size(); ++i) {
    in[i] = from[i] - 1;
  }
  std::vector<uint64_t> out(to.size(), 0);
  fhenomenon::ckks::bconv::convert(in.data(), out.data(), *tables);

  EXPECT_EQ(out, exactConversion(in, from, to));
}

// With one source prime, y_0 = x_0 < q_0 forces alpha = floor(y_0/q_0) = 0, so
// the conversion is exact and the answer can be written down without an
// oracle at all. This is the case ModDown relies on, and it is worth pinning
// separately: a sign or an off-by-one in the alpha term is invisible here and
// visible above, and vice versa.
// A full-length source basis at the modulus bound. Every additional source
// prime is another term added before the single final reduction, so if the
// unreduced window is stated wrongly the error accumulates with basis length
// rather than appearing at once — a three-prime basis can carry a wrong window
// for a long time without ever leaving the correct range by chance. This is
// also the realistic shape: ModUp converts a whole modulus chain, not three
// primes.
TEST(BConv, MatchesTheExactCrtSumOverALongSourceBasis) {
  const std::vector<uint64_t> &from = kLongWideBasis;
  const std::vector<uint64_t> to{kWideA, kWideB};
  const auto tables = BConvTables::create(from, to, nullptr);
  ASSERT_TRUE(tables.has_value());

  for (uint64_t seed = 1; seed <= 16; ++seed) {
    const std::vector<uint64_t> in = pseudoRandomResidues(from, seed);
    std::vector<uint64_t> out(to.size(), 0);
    fhenomenon::ckks::bconv::convert(in.data(), out.data(), *tables);

    EXPECT_EQ(out, exactConversion(in, from, to)) << "seed " << seed;
  }
}

TEST(BConv, ConvertsExactlyFromASinglePrimeBasis) {
  const std::vector<uint64_t> from{kWideA};
  const std::vector<uint64_t> to{kWideB, 12289};
  const auto tables = BConvTables::create(from, to, nullptr);
  ASSERT_TRUE(tables.has_value());

  const std::vector<uint64_t> in{kWideA - 12345};
  std::vector<uint64_t> out(to.size(), 0);
  fhenomenon::ckks::bconv::convert(in.data(), out.data(), *tables);

  EXPECT_EQ(out[0], in[0] % kWideB);
  EXPECT_EQ(out[1], in[0] % 12289);
}

// Zero converts to zero regardless of alpha, so this catches a table that is
// wrong by an additive constant — which the oracle comparisons above would
// also catch, but which would otherwise only ever be exercised at random
// inputs.
TEST(BConv, ConvertsZeroToZero) {
  const auto tables = BConvTables::create(kSmallFrom, kSmallTo, nullptr);
  ASSERT_TRUE(tables.has_value());

  const std::vector<uint64_t> in(kSmallFrom.size(), 0);
  std::vector<uint64_t> out(kSmallTo.size(), 12345);
  fhenomenon::ckks::bconv::convert(in.data(), out.data(), *tables);

  EXPECT_EQ(out, std::vector<uint64_t>(kSmallTo.size(), 0));
}

} // namespace
