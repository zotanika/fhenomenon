// Tests for L2 — the storage layouts.
//
// This executable links fhn_ckks_storage, which brings fhn_ckks_params with it
// because L2 legitimately depends on L0. It must never need L1 or L3: a layout
// is value arithmetic over indices, it holds no residues, and it is available
// before any memory exists. That last property is the point. eval::scratchBytes
// at L4 will be written in terms of these types, so the number a planner
// reports before execution and the number a kernel consumes during it come
// from the same expression rather than from two that have to be kept in step.

#include "CKKS/Layout.h"
#include "CKKS/Params.h"

#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <vector>

namespace {

using fhenomenon::ckks::CiphertextLayout;
using fhenomenon::ckks::NttForm;
using fhenomenon::ckks::Params;
using fhenomenon::ckks::ParamsId;
using fhenomenon::ckks::PolyLayout;
using fhenomenon::ckks::RnsBasis;
using fhenomenon::ckks::SecretDistribution;

// The same shape ParamsTest uses: N = 1024, four main primes, two special.
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

// A basis that does not need a Params to exist. L2's minimum log_degree is
// below L0's on purpose, so these tests can use tiny rings.
RnsBasis basisOf(uint32_t main_begin, uint32_t main_end, uint32_t aux_begin = 0, uint32_t aux_end = 0) {
  RnsBasis basis;
  basis.params_id = ParamsId{0x1234};
  basis.main_begin = main_begin;
  basis.main_end = main_end;
  basis.aux_begin = aux_begin;
  basis.aux_end = aux_end;
  return basis;
}

PolyLayout polyOf(const RnsBasis &basis, uint32_t log_degree = 3, NttForm form = NttForm::Coefficient) {
  PolyLayout layout;
  layout.basis = basis;
  layout.log_degree = log_degree;
  layout.form = form;
  return layout;
}

// --- ParamsId -------------------------------------------------------------

TEST(ParamsId, IsEmptyWhenDefaultConstructed) {
  EXPECT_TRUE(ParamsId{}.empty());
  EXPECT_EQ(ParamsId{}.value(), 0U);
  EXPECT_FALSE(ParamsId::of(make(goodSpec())).empty());
}

TEST(ParamsId, AgreesWithParameterSetEquality) {
  const Params a = make(goodSpec());
  const Params b = make(goodSpec());
  ASSERT_EQ(a, b);
  EXPECT_EQ(ParamsId::of(a), ParamsId::of(b));
}

// Every field Params::operator== compares must reach the fingerprint. A field
// that does not is a silent wire-compatibility hole: two parameter sets that
// compare unequal would produce ciphertexts that claim to be interchangeable.
// This is the test that pins the documented field order as a permanent
// obligation rather than an implementation detail.
TEST(ParamsId, SeparatesParameterSetsDifferingInAnyComparedField) {
  std::vector<Params::Spec> variants;

  Params::Spec degree = goodSpec();
  degree.log_degree = 11;
  variants.push_back(degree);

  Params::Spec main = goodSpec();
  main.main_primes = {12289, 40961, 61441};
  variants.push_back(main);

  Params::Spec aux = goodSpec();
  aux.aux_primes = {786433, 1179649, 1312769}; // 641*2048+1
  variants.push_back(aux);

  Params::Spec dnum = goodSpec();
  dnum.dnum = 4;
  variants.push_back(dnum);

  Params::Spec scale = goodSpec();
  scale.log_default_scale = 14;
  variants.push_back(scale);

  Params::Spec secret = goodSpec();
  secret.secret = SecretDistribution::SparseTernary;
  secret.sparse_hamming_weight = 64;
  variants.push_back(secret);

  Params::Spec terminal = goodSpec();
  terminal.terminal_primes = {1318913};
  variants.push_back(terminal);

  const Params base = make(goodSpec());
  const ParamsId base_id = ParamsId::of(base);
  for (const Params::Spec &spec : variants) {
    std::string error;
    const auto varied = Params::create(spec, &error);
    ASSERT_TRUE(varied.has_value()) << error;
    ASSERT_NE(base, *varied) << "the variant is not actually different, so this proves nothing";
    EXPECT_NE(base_id, ParamsId::of(*varied));
  }
}

// The variants above change `secret` and `sparse_hamming_weight` together,
// because a sparse secret needs a weight — so dropping the weight from the
// fingerprint still passes that test. Two sparse sets differing ONLY in weight
// is what actually pins it.
TEST(ParamsId, SeparatesSecretsOfDifferentHammingWeight) {
  Params::Spec light = goodSpec();
  light.secret = SecretDistribution::SparseTernary;
  light.sparse_hamming_weight = 64;

  Params::Spec heavy = light;
  heavy.sparse_hamming_weight = 128;

  const Params a = make(light);
  const Params b = make(heavy);
  ASSERT_NE(a, b);
  EXPECT_NE(ParamsId::of(a), ParamsId::of(b));
}

// The fingerprint walks three prime lists in sequence. Without a length prefix
// on each, the boundary between them is invisible: moving a prime from the aux
// list to the terminal list leaves the byte sequence identical while
// Params::operator== reports the two sets as different. That is the exact
// shape of a wire-compatibility hole — two parameter sets claiming one
// identity — so it gets its own test rather than being left to the
// field-by-field sweep, which cannot construct it.
TEST(ParamsId, SeparatesParameterSetsThatOnlyRepartitionTheSamePrimes) {
  Params::Spec split = goodSpec();
  split.dnum = 4; // one prime per digit, so a single special prime suffices
  Params::Spec moved = split;
  moved.aux_primes = {786433};
  moved.terminal_primes = {1179649};

  const Params a = make(split);
  const Params b = make(moved);
  ASSERT_NE(a, b);
  EXPECT_NE(ParamsId::of(a), ParamsId::of(b));
}

// --- RnsBasis -------------------------------------------------------------

TEST(RnsBasis, CountsMainAndAuxLimbsSeparatelyAndTogether) {
  const RnsBasis basis = basisOf(0, 5, 0, 2);
  EXPECT_EQ(basis.mainCount(), 5U);
  EXPECT_EQ(basis.auxCount(), 2U);
  EXPECT_EQ(basis.limbCount(), 7U);
  EXPECT_TRUE(basis.hasAux());
  EXPECT_TRUE(basis.isChainPrefix());

  const RnsBasis digit = basisOf(2, 4);
  EXPECT_EQ(digit.limbCount(), 2U);
  EXPECT_FALSE(digit.hasAux());
  EXPECT_FALSE(digit.isChainPrefix());
}

TEST(RnsBasis, RejectsAnInvertedRange) {
  std::string error;
  RnsBasis basis = basisOf(4, 2);
  EXPECT_FALSE(basis.wellFormed(&error));
  EXPECT_NE(error.find("main"), std::string::npos) << error;

  basis = basisOf(0, 2, 3, 1);
  EXPECT_FALSE(basis.wellFormed(&error));
  EXPECT_NE(error.find("aux"), std::string::npos) << error;
}

TEST(RnsBasis, RejectsAnEmptyBasis) {
  std::string error;
  const RnsBasis basis = basisOf(2, 2);
  EXPECT_FALSE(basis.wellFormed(&error));
  EXPECT_NE(error.find("no limbs"), std::string::npos) << error;
}

// An unidentified basis is the dangerous one: it compares equal to any other
// unidentified basis, so it would let residues under one parameter set be
// accepted as residues under another.
TEST(RnsBasis, RejectsAnUnidentifiedParameterSet) {
  std::string error;
  RnsBasis basis = basisOf(0, 3);
  basis.params_id = ParamsId{};
  EXPECT_FALSE(basis.wellFormed(&error));
  EXPECT_NE(error.find("params"), std::string::npos) << error;
}

TEST(RnsBasis, RejectsMoreLimbsThanTheLayoutBound) {
  std::string error;
  const RnsBasis basis = basisOf(0, RnsBasis::kMaxLimbs + 1);
  EXPECT_FALSE(basis.wellFormed(&error));
  EXPECT_NE(error.find("limbs"), std::string::npos) << error;
}

TEST(RnsBasis, EqualityComparesEveryField) {
  const RnsBasis basis = basisOf(1, 4, 0, 2);
  EXPECT_EQ(basis, basisOf(1, 4, 0, 2));
  EXPECT_NE(basis, basisOf(0, 4, 0, 2));
  EXPECT_NE(basis, basisOf(1, 5, 0, 2));
  EXPECT_NE(basis, basisOf(1, 4, 1, 2));
  EXPECT_NE(basis, basisOf(1, 4, 0, 3));

  RnsBasis other = basisOf(1, 4, 0, 2);
  other.params_id = ParamsId{0x9999};
  EXPECT_NE(basis, other);
}

// --- PolyLayout -----------------------------------------------------------

TEST(PolyLayout, SizesItselfFromLimbCountAndDegree) {
  const PolyLayout layout = polyOf(basisOf(0, 5, 0, 2), 10);
  EXPECT_EQ(layout.degree(), 1024U);
  EXPECT_EQ(layout.wordCount(), 7U * 1024U);
  EXPECT_EQ(layout.byteCount(), 7ULL * 1024ULL * 8ULL);
}

// R1 in one line: the planner reads byteCount() and the kernel indexes
// wordCount(), so they must be the same quantity in two units. If these ever
// diverge, a budget is reported that the execution does not honour.
TEST(PolyLayout, ByteCountIsWordCountInBytesAtEveryRepresentableSize) {
  for (uint32_t log_degree = PolyLayout::kMinLogDegree; log_degree <= PolyLayout::kMaxLogDegree; ++log_degree) {
    for (uint32_t limbs : {1U, 2U, 17U, RnsBasis::kMaxLimbs}) {
      const PolyLayout layout = polyOf(basisOf(0, limbs), log_degree);
      ASSERT_TRUE(layout.wellFormed()) << "log_degree " << log_degree << " limbs " << limbs;
      EXPECT_EQ(layout.byteCount(), static_cast<uint64_t>(layout.wordCount()) * 8ULL)
        << "log_degree " << log_degree << " limbs " << limbs;
    }
  }
}

TEST(PolyLayout, RejectsALogDegreeOutsideTheSupportedRange) {
  std::string error;
  EXPECT_FALSE(polyOf(basisOf(0, 2), PolyLayout::kMinLogDegree - 1).wellFormed(&error));
  EXPECT_NE(error.find("log_degree"), std::string::npos) << error;
  EXPECT_FALSE(polyOf(basisOf(0, 2), PolyLayout::kMaxLogDegree + 1).wellFormed(&error));
}

TEST(PolyLayout, RejectsAMalformedBasisRatherThanRepeatingItsChecks) {
  std::string error;
  EXPECT_FALSE(polyOf(basisOf(2, 2)).wellFormed(&error));
  EXPECT_NE(error.find("no limbs"), std::string::npos) << error;
}

// sameRing asks whether two polynomials live in the same ring, which is a
// question about the parameter set and the degree only. Two operands of a
// key switch differ in live range and in NTT form and are still the same ring;
// that is exactly the case this predicate exists to accept.
TEST(PolyLayout, SameRingIgnoresTheLiveRangeAndTheNttForm) {
  const PolyLayout a = polyOf(basisOf(0, 5), 10, NttForm::Coefficient);
  EXPECT_TRUE(a.sameRing(polyOf(basisOf(2, 4), 10, NttForm::Evaluation)));
  EXPECT_FALSE(a.sameRing(polyOf(basisOf(0, 5), 11)));

  PolyLayout other = polyOf(basisOf(0, 5), 10);
  other.basis.params_id = ParamsId{0x9999};
  EXPECT_FALSE(a.sameRing(other));
}

TEST(PolyLayout, EqualityDistinguishesTheNttForm) {
  const PolyLayout a = polyOf(basisOf(0, 5), 10, NttForm::Coefficient);
  EXPECT_EQ(a, polyOf(basisOf(0, 5), 10, NttForm::Coefficient));
  EXPECT_NE(a, polyOf(basisOf(0, 5), 10, NttForm::Evaluation));
}

// --- CiphertextLayout -----------------------------------------------------

CiphertextLayout ciphertextOf(uint32_t main_end, uint32_t num_polys = 2, double scale = 8192.0) {
  CiphertextLayout layout;
  layout.poly = polyOf(basisOf(0, main_end), 10);
  layout.num_polys = num_polys;
  layout.scale = scale;
  return layout;
}

TEST(CiphertextLayout, LevelFollowsTheMainRangeByTheProjectConvention) {
  // A ciphertext "at level L" carries L+1 main primes, matching Params.
  EXPECT_EQ(ciphertextOf(4).level(), 3);
  EXPECT_EQ(ciphertextOf(1).level(), 0);
}

TEST(CiphertextLayout, SizesItselfFromItsPolynomials) {
  const CiphertextLayout layout = ciphertextOf(4, 3);
  EXPECT_EQ(layout.wordCount(), 3U * 4U * 1024U);
  EXPECT_EQ(layout.byteCount(), 3ULL * 4ULL * 1024ULL * 8ULL);
}

// The reserve is per polynomial, not a single tail on the whole ciphertext.
// That is what makes RESCALE a decrement of main_end with no memmove and
// ModUp an append at each polynomial's own live tail.
TEST(CiphertextLayout, SlabWordsReservesCapacityPerPolynomial) {
  const CiphertextLayout layout = ciphertextOf(4, 2);
  EXPECT_EQ(layout.slabWords(4), 2U * 4U * 1024U);
  EXPECT_EQ(layout.slabWords(6), 2U * 6U * 1024U);
  EXPECT_GT(layout.slabWords(6), layout.wordCount());
}

TEST(CiphertextLayout, RejectsAPolynomialCountOutsideTheSupportedRange) {
  std::string error;
  EXPECT_FALSE(ciphertextOf(4, 0).wellFormed(&error));
  EXPECT_NE(error.find("num_polys"), std::string::npos) << error;
  EXPECT_FALSE(ciphertextOf(4, CiphertextLayout::kMaxPolys + 1).wellFormed(&error));
}

TEST(CiphertextLayout, RejectsAScaleThatCannotDivide) {
  std::string error;
  EXPECT_FALSE(ciphertextOf(4, 2, 0.0).wellFormed(&error));
  EXPECT_NE(error.find("scale"), std::string::npos) << error;
  EXPECT_FALSE(ciphertextOf(4, 2, -1.0).wellFormed(&error));
}

// A key-switching digit covers a slice of the chain and is a PolyLayout. A
// ciphertext is always a prefix, because its level is defined as main_end - 1
// and that number is meaningless for a basis starting anywhere else.
TEST(CiphertextLayout, RejectsABasisThatIsNotAChainPrefix) {
  std::string error;
  CiphertextLayout layout = ciphertextOf(4);
  layout.poly.basis = basisOf(2, 4);
  EXPECT_FALSE(layout.wellFormed(&error));
  EXPECT_NE(error.find("prefix"), std::string::npos) << error;
}

TEST(CiphertextLayout, AcceptsAnExtendedBasisBecauseModUpProducesOne) {
  CiphertextLayout layout = ciphertextOf(4);
  layout.poly.basis = basisOf(0, 4, 0, 2);
  std::string error;
  EXPECT_TRUE(layout.wellFormed(&error)) << error;
}

// Every one of these crosses the C ABI eventually, so none of them may acquire
// a destructor, a vtable or a heap member without that being a deliberate
// decision. Asserting it here makes the decision visible if it is ever made.
TEST(Layout, TypesAreTriviallyCopyableValues) {
  static_assert(std::is_trivially_copyable_v<ParamsId>);
  static_assert(std::is_trivially_copyable_v<RnsBasis>);
  static_assert(std::is_trivially_copyable_v<PolyLayout>);
  static_assert(std::is_trivially_copyable_v<CiphertextLayout>);
  static_assert(std::is_trivially_copyable_v<NttForm>);
  SUCCEED();
}

} // namespace
