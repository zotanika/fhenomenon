#include "CKKS/Params.h"

#include <gtest/gtest.h>

#include <string>
#include <unordered_map>

// This test links fhn_ckks_params and nothing else. If it ever needs another
// layer to build, L0 has stopped being a leaf and the layering has failed —
// that is the point of the target boundary, not an accident of it.

using fhenomenon::ckks::Params;
using fhenomenon::ckks::SecretDistribution;

namespace {

// N = 1024, so every prime must be 1 mod 2048.
//   12289 = 3*2^12+1   40961 = 5*2^13+1   61441 = 15*2^12+1   65537 = 2^16+1
// Special primes are wider so that P can exceed a two-prime digit.
//   786433 = 3*2^18+1  1179649 = 9*2^17+1
Params::Spec goodSpec() {
  Params::Spec spec;
  spec.log_degree = 10;
  spec.main_primes = {12289, 40961, 61441, 65537};
  spec.aux_primes = {786433, 1179649};
  spec.dnum = 2;
  spec.log_default_scale = 13;
  return spec;
}

Params make(const Params::Spec &spec) {
  std::string error;
  auto params = Params::create(spec, &error);
  EXPECT_TRUE(params.has_value()) << error;
  return params.value();
}

} // namespace

TEST(CkksParamsTest, AcceptsAWellFormedSpec) {
  std::string error;
  auto params = Params::create(goodSpec(), &error);
  ASSERT_TRUE(params.has_value()) << error;
  EXPECT_TRUE(error.empty());

  EXPECT_EQ(params->logDegree(), 10u);
  EXPECT_EQ(params->degree(), 1024u);
  EXPECT_EQ(params->numSlots(), 512u);
  EXPECT_EQ(params->levelCount(), 4u);
  EXPECT_EQ(params->freshLevel(), 3);
}

TEST(CkksParamsTest, RejectsAPrimeThatIsNotNttFriendly) {
  Params::Spec spec = goodSpec();
  spec.main_primes[1] = 40960; // 0 mod 2048, not 1
  std::string error;
  EXPECT_FALSE(Params::create(spec, &error).has_value());
  EXPECT_NE(error.find("not 1 mod 2N"), std::string::npos) << error;
}

TEST(CkksParamsTest, RejectsARepeatedPrimeAcrossLists) {
  Params::Spec spec = goodSpec();
  spec.aux_primes[0] = spec.main_primes[0];
  std::string error;
  EXPECT_FALSE(Params::create(spec, &error).has_value());
  EXPECT_NE(error.find("more than once"), std::string::npos) << error;
}

TEST(CkksParamsTest, RejectsAnOversizedPrime) {
  Params::Spec spec = goodSpec();
  // 1 mod 2048 but past the 62-bit ceiling that keeps products inside a
  // 128-bit intermediate with room for lazy reduction.
  spec.main_primes[0] = (uint64_t{1} << 62) + 1;
  std::string error;
  EXPECT_FALSE(Params::create(spec, &error).has_value());
  EXPECT_NE(error.find("exceeds 62 bits"), std::string::npos) << error;
}

TEST(CkksParamsTest, RejectsDnumOutsideItsRange) {
  Params::Spec spec = goodSpec();
  spec.dnum = 0;
  EXPECT_FALSE(Params::create(spec).has_value());
  spec.dnum = 5; // levelCount() is 4
  EXPECT_FALSE(Params::create(spec).has_value());
}

// dnum is the whole point of the key-switching decision: hybrid is a
// continuum whose endpoints are GHS (dnum == 1) and a BV-style decomposition
// (dnum == levelCount()). The validator has to enforce what each endpoint
// actually costs rather than accept them all silently.
TEST(CkksParamsTest, DnumSweepsFromGhsToBv) {
  Params::Spec spec = goodSpec();

  // dnum == 1 makes the single digit the entire chain, so P would have to
  // cover all of Q. These special primes do not, so it is rejected — with a
  // reason that names the fix.
  spec.dnum = 1;
  std::string error;
  EXPECT_FALSE(Params::create(spec, &error).has_value());
  EXPECT_NE(error.find("widest key-switching digit"), std::string::npos) << error;

  // Splitting the same chain into narrower digits makes the same P sufficient.
  spec.dnum = 2;
  EXPECT_TRUE(Params::create(spec).has_value());
  spec.dnum = 4; // one prime per digit: the BV endpoint
  EXPECT_TRUE(Params::create(spec).has_value());
}

TEST(CkksParamsTest, DigitsPartitionTheMainChain) {
  Params::Spec spec = goodSpec();
  spec.dnum = 3; // 4 primes over 3 digits: alpha 2, so the last digit is short
  const Params params = make(spec);

  EXPECT_EQ(params.alpha(), 2u);
  EXPECT_EQ(params.digitRange(0), std::make_pair(0u, 2u));
  EXPECT_EQ(params.digitRange(1), std::make_pair(2u, 4u));
  EXPECT_EQ(params.digitRange(2), std::make_pair(4u, 4u)); // empty tail

  // Whatever the split, the digits must cover every main prime exactly once.
  uint32_t covered = 0;
  for (uint32_t digit = 0; digit < params.dnum(); ++digit) {
    const auto range = params.digitRange(digit);
    EXPECT_LE(range.first, range.second);
    covered += range.second - range.first;
  }
  EXPECT_EQ(covered, params.levelCount());
}

// This is the R5 export trio reduced to a pure function of Params, which is
// the whole reason the level convention is pinned in the header.
TEST(CkksParamsTest, CiphertextBytesFollowTheLevel) {
  const Params params = make(goodSpec());
  const uint64_t poly_bytes = params.degree() * sizeof(uint64_t);

  EXPECT_EQ(params.ciphertextBytes(0), 2 * 1 * poly_bytes);
  EXPECT_EQ(params.ciphertextBytes(3), 2 * 4 * poly_bytes);
  EXPECT_EQ(params.ciphertextBytes(1, 3), 3 * 2 * poly_bytes);

  // 0 means "invalid level", matching the fhn_level_bytes contract.
  EXPECT_EQ(params.ciphertextBytes(-1), 0u);
  EXPECT_EQ(params.ciphertextBytes(params.freshLevel() + 1), 0u);
  EXPECT_EQ(params.ciphertextBytes(0, 0), 0u);
}

// Requirement R4: nothing about a parameter set is process-global, so two of
// them are simply two values. A backend that stored degree or the modulus
// chain in a static could not pass this.
TEST(CkksParamsTest, TwoParameterSetsCoexist) {
  Params::Spec small = goodSpec();
  Params::Spec large = goodSpec();
  large.log_degree = 11;
  // 1 mod 4096: 12289 = 3*2^12+1, 40961 = 10*4096+1, 61441 = 15*4096+1.
  large.main_primes = {12289, 40961, 61441};
  large.aux_primes = {786433, 1179649};
  large.dnum = 3;

  const Params a = make(small);
  const Params b = make(large);

  EXPECT_EQ(a.degree(), 1024u);
  EXPECT_EQ(b.degree(), 2048u);
  EXPECT_NE(a, b);
  EXPECT_EQ(a, make(small));
}

TEST(CkksParamsTest, HashesByIdentitySoItCanKeyAPrecomputationCache) {
  const Params a = make(goodSpec());
  const Params b = make(goodSpec());
  EXPECT_EQ(a.hash(), b.hash());

  Params::Spec other = goodSpec();
  other.log_default_scale = 14;
  EXPECT_NE(a.hash(), make(other).hash());

  std::unordered_map<Params, int> cache;
  cache[a] = 1;
  EXPECT_EQ(cache.count(b), 1u);
}

TEST(CkksParamsTest, SparseSecretNeedsAHammingWeight) {
  Params::Spec spec = goodSpec();
  spec.secret = SecretDistribution::SparseTernary;
  EXPECT_FALSE(Params::create(spec).has_value());

  spec.sparse_hamming_weight = 32;
  EXPECT_TRUE(Params::create(spec).has_value());

  // The bootstrapping surface is representable but inert in v1: carrying
  // terminal primes must not change how a leveled parameter set validates.
  spec.terminal_primes = {114689};
  EXPECT_TRUE(Params::create(spec).has_value());
}
