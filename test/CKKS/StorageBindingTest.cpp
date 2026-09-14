// Tests for L2 — the memory-binding half of storage.
//
// This executable links fhn_ckks_storage and nothing else, exactly as
// LayoutTest does. Everything here is about where residues live and how a
// handle addresses them; nothing here transforms a residue or needs a table,
// an arena or a key. If a test in this file ever needs to link L1 or L3, the
// binding half has started to know things that are not its business.

#include "CKKS/Ciphertext.h"
#include "CKKS/Layout.h"
#include "CKKS/PolySlab.h"
#include "CKKS/RnsPoly.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using fhenomenon::ckks::BasicCiphertext;
using fhenomenon::ckks::BasicRnsPoly;
using fhenomenon::ckks::Ciphertext;
using fhenomenon::ckks::CiphertextLayout;
using fhenomenon::ckks::CiphertextRef;
using fhenomenon::ckks::CiphertextView;
using fhenomenon::ckks::NttForm;
using fhenomenon::ckks::ParamsId;
using fhenomenon::ckks::PolyLayout;
using fhenomenon::ckks::PolySlab;
using fhenomenon::ckks::RnsBasis;
using fhenomenon::ckks::RnsPolyRef;
using fhenomenon::ckks::RnsPolyView;

constexpr uint32_t kLogDegree = 3; // 8 words per limb: one cache line, the tightest alignment case
constexpr std::size_t kDegree = std::size_t{1} << kLogDegree;

RnsBasis basisOf(uint32_t main_begin, uint32_t main_end, uint32_t aux_begin = 0, uint32_t aux_end = 0) {
  RnsBasis basis;
  basis.params_id = ParamsId{0x1234};
  basis.main_begin = main_begin;
  basis.main_end = main_end;
  basis.aux_begin = aux_begin;
  basis.aux_end = aux_end;
  return basis;
}

PolyLayout polyOf(const RnsBasis &basis, uint32_t log_degree = kLogDegree, NttForm form = NttForm::Coefficient) {
  PolyLayout layout;
  layout.basis = basis;
  layout.log_degree = log_degree;
  layout.form = form;
  return layout;
}

CiphertextLayout ciphertextOf(uint32_t main_end, uint32_t num_polys = 2, uint32_t aux_end = 0, double scale = 8192.0) {
  CiphertextLayout layout;
  layout.poly = polyOf(basisOf(0, main_end, 0, aux_end));
  layout.num_polys = num_polys;
  layout.scale = scale;
  return layout;
}

// A value that identifies (polynomial, limb, coefficient) uniquely, so a test
// can tell WHICH word ended up somewhere and not merely that some word did.
uint64_t tag(uint32_t poly, uint32_t limb, std::size_t coeff) {
  return (uint64_t{poly} << 32U) | (uint64_t{limb} << 16U) | static_cast<uint64_t>(coeff);
}

// Write the tag pattern into every live word of a ciphertext handle.
void fillTagged(CiphertextRef ct) {
  for (uint32_t p = 0; p < ct.numPolys(); ++p) {
    RnsPolyRef poly = ct.poly(p);
    for (uint32_t i = 0; i < poly.limbCount(); ++i) {
      for (std::size_t n = 0; n < poly.degree(); ++n) {
        poly.limb(i)[n] = tag(p, i, n);
      }
    }
  }
}

bool liveWordsEqual(CiphertextView a, CiphertextView b) {
  if (a.layout() != b.layout()) {
    return false;
  }
  for (uint32_t p = 0; p < a.numPolys(); ++p) {
    if (std::memcmp(a.poly(p).data(), b.poly(p).data(), a.poly(p).words() * sizeof(uint64_t)) != 0) {
      return false;
    }
  }
  return true;
}

bool isAligned(const void *pointer) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return reinterpret_cast<std::uintptr_t>(pointer) % PolySlab::kAlignment == 0;
}

// --- PolySlab -------------------------------------------------------------

TEST(PolySlab, OwnsNothingByDefault) {
  const PolySlab slab;
  EXPECT_TRUE(slab.empty());
  EXPECT_EQ(slab.words(), 0U);
  EXPECT_EQ(slab.bytes(), 0U);
  EXPECT_EQ(slab.data(), nullptr);
}

TEST(PolySlab, AZeroWordRequestOwnsNothing) {
  const PolySlab slab(0);
  EXPECT_TRUE(slab.empty());
  EXPECT_EQ(slab.data(), nullptr);
}

TEST(PolySlab, OwnsExactlyTheRequestedWordsAndTheyAreWritable) {
  PolySlab slab(3 * kDegree);
  ASSERT_FALSE(slab.empty());
  EXPECT_EQ(slab.words(), 3 * kDegree);
  EXPECT_EQ(slab.bytes(), 3 * kDegree * sizeof(uint64_t));
  for (std::size_t i = 0; i < slab.words(); ++i) {
    slab.data()[i] = i;
  }
  for (std::size_t i = 0; i < slab.words(); ++i) {
    EXPECT_EQ(slab.data()[i], i);
  }
}

// The alignment is the one promise the slab makes to kernels. Sizes that are
// not a multiple of the line are included so that the guarantee is seen to
// come from the allocation and not from a coincidence of the request.
TEST(PolySlab, IsCacheLineAlignedAtEverySize) {
  for (std::size_t words : {1U, 7U, 8U, 9U, 64U, 1000U}) {
    const PolySlab slab(words);
    EXPECT_TRUE(isAligned(slab.data())) << words << " words";
  }
}

// SIZE_MAX / 8 + 1 words is 2^64 + 8 bytes, which wraps to 8: without a
// guard this would be an eight-byte allocation claiming to hold 2^61 words.
TEST(PolySlab, RefusesAWordCountWhoseByteSizeWraps) {
  EXPECT_THROW(PolySlab(SIZE_MAX / sizeof(uint64_t) + 1), std::bad_alloc);
}

TEST(PolySlab, MoveTransfersTheWordsAndEmptiesTheSource) {
  PolySlab source(kDegree);
  uint64_t *const original = source.data();
  source.data()[0] = 42;

  PolySlab moved(std::move(source));
  EXPECT_EQ(moved.data(), original) << "a move must not reallocate";
  EXPECT_EQ(moved.words(), kDegree);
  EXPECT_EQ(moved.data()[0], 42U);
  EXPECT_TRUE(source.empty());       // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.data(), nullptr); // NOLINT(bugprone-use-after-move)

  PolySlab assigned(2 * kDegree);
  assigned = std::move(moved);
  EXPECT_EQ(assigned.data(), original);
  EXPECT_EQ(assigned.words(), kDegree);
  EXPECT_TRUE(moved.empty()); // NOLINT(bugprone-use-after-move)
}

TEST(PolySlab, MoveAssigningOntoItselfChangesNothing) {
  PolySlab slab(kDegree);
  uint64_t *const original = slab.data();
  slab.data()[3] = 7;
  PolySlab &alias = slab;
  slab = std::move(alias);
  EXPECT_EQ(slab.data(), original);
  EXPECT_EQ(slab.words(), kDegree);
  EXPECT_EQ(slab.data()[3], 7U);
}

TEST(PolySlab, CloneIsAnIndependentCopyOfEveryWord) {
  PolySlab slab(2 * kDegree);
  for (std::size_t i = 0; i < slab.words(); ++i) {
    slab.data()[i] = 1000 + i;
  }
  PolySlab clone = slab.clone();
  ASSERT_EQ(clone.words(), slab.words());
  ASSERT_NE(clone.data(), slab.data());
  EXPECT_EQ(std::memcmp(clone.data(), slab.data(), slab.bytes()), 0);

  clone.data()[5] = 0;
  EXPECT_EQ(slab.data()[5], 1005U) << "writing the clone must not touch the original";
  slab.data()[6] = 0;
  EXPECT_EQ(clone.data()[6], 1006U) << "writing the original must not touch the clone";
}

TEST(PolySlab, CloneOfNothingIsNothing) {
  const PolySlab empty;
  const PolySlab clone = empty.clone();
  EXPECT_TRUE(clone.empty());
}

// --- BasicRnsPoly ---------------------------------------------------------

TEST(RnsPoly, BindsNothingByDefault) {
  const RnsPolyView view;
  EXPECT_TRUE(view.empty());
  EXPECT_EQ(view.data(), nullptr);
  EXPECT_EQ(view.words(), 0U);
}

TEST(RnsPoly, AddressesLimbsContiguouslyInStorageOrder) {
  std::vector<uint64_t> words(5 * kDegree);
  const RnsPolyRef poly(polyOf(basisOf(0, 3, 0, 2)), words.data());
  EXPECT_EQ(poly.words(), 5 * kDegree);
  EXPECT_EQ(poly.limbCount(), 5U);
  EXPECT_EQ(poly.degree(), kDegree);
  for (uint32_t i = 0; i < 5; ++i) {
    EXPECT_EQ(poly.limb(i), words.data() + i * kDegree) << "limb " << i;
  }
}

// The basis names primes by absolute index. A digit at main [2, 4) has its
// first limb at mainLimb(2), and aux limbs come after ALL main limbs in
// storage, wherever the aux range begins in the parameter set's list.
TEST(RnsPoly, AddressesLimbsByThePrimeIndexTheBasisNames) {
  std::vector<uint64_t> words(5 * kDegree);
  const RnsPolyRef poly(polyOf(basisOf(2, 5, 1, 3)), words.data());
  EXPECT_EQ(poly.mainLimb(2), poly.limb(0));
  EXPECT_EQ(poly.mainLimb(3), poly.limb(1));
  EXPECT_EQ(poly.mainLimb(4), poly.limb(2));
  EXPECT_EQ(poly.auxLimb(1), poly.limb(3));
  EXPECT_EQ(poly.auxLimb(2), poly.limb(4));
}

TEST(RnsPoly, AWriteThroughARefIsVisibleThroughAViewOfTheSameWords) {
  std::vector<uint64_t> words(2 * kDegree, 0);
  const RnsPolyRef ref(polyOf(basisOf(0, 2)), words.data());
  const RnsPolyView view = ref; // implicit Ref -> View
  EXPECT_EQ(view.data(), ref.data());
  EXPECT_EQ(view.layout(), ref.layout());
  ref.limb(1)[3] = 99;
  EXPECT_EQ(view.limb(1)[3], 99U);
  static_assert(!std::is_convertible_v<RnsPolyView, RnsPolyRef>, "a View must never become a Ref");
}

TEST(RnsPoly, MainSliceAddressesTheSameWordsUnderADigitBasis) {
  std::vector<uint64_t> words(6 * kDegree);
  const RnsPolyRef poly(polyOf(basisOf(0, 4, 0, 2), kLogDegree, NttForm::Evaluation), words.data());
  const auto digit = poly.mainSlice(1, 3);
  ASSERT_TRUE(digit.has_value());
  EXPECT_EQ(digit->data(), poly.mainLimb(1));
  EXPECT_EQ(digit->limbCount(), 2U);
  EXPECT_EQ(digit->basis(), basisOf(1, 3));
  EXPECT_EQ(digit->form(), NttForm::Evaluation) << "a slice does not change the NTT form";
  EXPECT_EQ(digit->layout().log_degree, kLogDegree);
  EXPECT_FALSE(digit->basis().hasAux()) << "a digit is main-only";
  // The slice is a view: a write through it lands in the original.
  digit->limb(1)[0] = 77;
  EXPECT_EQ(poly.mainLimb(2)[0], 77U);
}

// A slice of a slice must be addressed in the outer basis's terms, which is
// what makes mainLimb() the right primitive: mainSlice(2, 3) of a digit at
// [1, 3) is main limb 2, and the offset is relative to the digit's own begin.
TEST(RnsPoly, MainSliceOfASliceStaysInAbsoluteTerms) {
  std::vector<uint64_t> words(4 * kDegree);
  const RnsPolyRef poly(polyOf(basisOf(0, 4)), words.data());
  const auto digit = poly.mainSlice(1, 3);
  ASSERT_TRUE(digit.has_value());
  const auto inner = digit->mainSlice(2, 3);
  ASSERT_TRUE(inner.has_value());
  EXPECT_EQ(inner->data(), poly.mainLimb(2));
  EXPECT_EQ(inner->basis(), basisOf(2, 3));
}

TEST(RnsPoly, MainSliceRefusesARangeOutsideItsOwnMainRange) {
  std::vector<uint64_t> words(3 * kDegree);
  const RnsPolyView poly(polyOf(basisOf(1, 4)), words.data());
  EXPECT_FALSE(poly.mainSlice(0, 2).has_value()) << "starts before main_begin";
  EXPECT_FALSE(poly.mainSlice(2, 5).has_value()) << "ends after main_end";
  EXPECT_FALSE(poly.mainSlice(2, 2).has_value()) << "empty";
  EXPECT_FALSE(poly.mainSlice(3, 2).has_value()) << "inverted";
  EXPECT_TRUE(poly.mainSlice(1, 4).has_value()) << "the whole main range is a valid slice";
}

TEST(RnsPoly, CopyMovesEveryLiveWordBetweenIdenticalLayouts) {
  std::vector<uint64_t> src_words(3 * kDegree);
  std::vector<uint64_t> dst_words(3 * kDegree, 0);
  for (std::size_t i = 0; i < src_words.size(); ++i) {
    src_words[i] = 500 + i;
  }
  const PolyLayout layout = polyOf(basisOf(0, 2, 0, 1));
  ASSERT_TRUE(copy(RnsPolyRef(layout, dst_words.data()), RnsPolyView(layout, src_words.data())));
  EXPECT_EQ(src_words, dst_words);
}

TEST(RnsPoly, CopyRefusesADifferentLayoutAndWritesNothing) {
  std::vector<uint64_t> src_words(3 * kDegree, 1);
  std::vector<uint64_t> dst_words(3 * kDegree, 0);
  const std::vector<uint64_t> untouched = dst_words;

  PolyLayout src = polyOf(basisOf(0, 3));
  PolyLayout dst = src;
  dst.basis.main_end = 2;
  EXPECT_FALSE(copy(RnsPolyRef(dst, dst_words.data()), RnsPolyView(src, src_words.data())));
  EXPECT_EQ(dst_words, untouched);

  dst = src;
  dst.form = NttForm::Evaluation;
  EXPECT_FALSE(copy(RnsPolyRef(dst, dst_words.data()), RnsPolyView(src, src_words.data())));
  EXPECT_EQ(dst_words, untouched);

  dst = src;
  dst.basis.params_id = ParamsId{0x9999};
  EXPECT_FALSE(copy(RnsPolyRef(dst, dst_words.data()), RnsPolyView(src, src_words.data())));
  EXPECT_EQ(dst_words, untouched);
}

TEST(RnsPoly, CopyOntoItselfIsANoOp) {
  std::vector<uint64_t> words(2 * kDegree);
  for (std::size_t i = 0; i < words.size(); ++i) {
    words[i] = i;
  }
  const std::vector<uint64_t> before = words;
  const PolyLayout layout = polyOf(basisOf(0, 2));
  EXPECT_TRUE(copy(RnsPolyRef(layout, words.data()), RnsPolyView(layout, words.data())));
  EXPECT_EQ(words, before);
}

// --- BasicCiphertext ------------------------------------------------------

TEST(CiphertextHandle, AddressesPolynomialsAtTheGivenStride) {
  const CiphertextLayout layout = ciphertextOf(3, 2);
  const std::size_t stride = 5 * kDegree; // two limbs of reserve per polynomial
  std::vector<uint64_t> words(2 * stride);
  const CiphertextRef ct(layout, words.data(), stride);
  EXPECT_EQ(ct.numPolys(), 2U);
  EXPECT_EQ(ct.level(), 2);
  EXPECT_EQ(ct.scale(), 8192.0);
  EXPECT_EQ(ct.polyStride(), stride);
  EXPECT_EQ(ct.poly(0).data(), words.data());
  EXPECT_EQ(ct.poly(1).data(), words.data() + stride);
  EXPECT_EQ(ct.poly(1).layout(), layout.poly);
  EXPECT_EQ(ct.poly(1).limb(2), words.data() + stride + 2 * kDegree);
}

TEST(CiphertextHandle, ATightStrideIsTheArenaShape) {
  const CiphertextLayout layout = ciphertextOf(3, 3);
  std::vector<uint64_t> words(layout.wordCount());
  const CiphertextView ct(layout, words.data(), layout.poly.wordCount());
  EXPECT_EQ(ct.poly(2).data() + ct.poly(2).words(), words.data() + words.size())
    << "three polynomials at the live stride exactly fill wordCount()";
}

TEST(CiphertextHandle, RefConvertsToViewOverTheSameWords) {
  const CiphertextLayout layout = ciphertextOf(2, 2);
  std::vector<uint64_t> words(layout.wordCount(), 0);
  const CiphertextRef ref(layout, words.data(), layout.poly.wordCount());
  const CiphertextView view = ref;
  EXPECT_EQ(view.data(), ref.data());
  EXPECT_EQ(view.polyStride(), ref.polyStride());
  EXPECT_EQ(view.layout(), ref.layout());
  ref.poly(1).limb(1)[2] = 5;
  EXPECT_EQ(view.poly(1).limb(1)[2], 5U);
  static_assert(!std::is_convertible_v<CiphertextView, CiphertextRef>, "a View must never become a Ref");
}

// --- Ciphertext (owner) ---------------------------------------------------

TEST(Ciphertext, OwnsNothingByDefaultAndCannotBeGivenAShape) {
  Ciphertext ct;
  EXPECT_TRUE(ct.empty());
  EXPECT_EQ(ct.limbCapacity(), 0U);
  EXPECT_EQ(ct.polyCapacity(), 0U);
  EXPECT_TRUE(ct.view().empty());
  std::string error;
  EXPECT_FALSE(ct.reshape(ciphertextOf(2), &error));
  EXPECT_NE(error.find("ring"), std::string::npos) << error;
}

TEST(Ciphertext, CreateSizesTheSlabFromBothCapacities) {
  std::string error;
  const auto ct = Ciphertext::create(ciphertextOf(3, 2), 5, 3, &error);
  ASSERT_TRUE(ct.has_value()) << error;
  EXPECT_EQ(ct->limbCapacity(), 5U);
  EXPECT_EQ(ct->polyCapacity(), 3U);
  EXPECT_EQ(ct->polyStride(), 5 * kDegree);
  EXPECT_EQ(ct->slab().words(), 3 * 5 * kDegree);
  EXPECT_EQ(ct->slab().words(), Ciphertext::slabWords(kLogDegree, 5, 3));
  EXPECT_EQ(ct->level(), 2);
  EXPECT_EQ(ct->layout(), ciphertextOf(3, 2));
}

// R1 in one line, for the owner: when the capacities equal the live shape,
// the slab is exactly what the layout said it would be before any memory
// existed, and every word of it is addressed.
TEST(Ciphertext, ATightCiphertextIsExactlyItsLayoutsWordCount) {
  const CiphertextLayout layout = ciphertextOf(4, 2, 1);
  const auto ct = Ciphertext::create(layout, layout.poly.basis.limbCount(), layout.num_polys);
  ASSERT_TRUE(ct.has_value());
  EXPECT_EQ(ct->slab().words(), layout.wordCount());
  EXPECT_EQ(ct->slab().words(), layout.slabWords(layout.poly.basis.limbCount()));
  EXPECT_EQ(ct->poly(1).data() + ct->poly(1).words(), ct->slab().data() + ct->slab().words());
}

TEST(Ciphertext, CreateRefusesAMalformedLayout) {
  std::string error;
  CiphertextLayout layout = ciphertextOf(3, 2);
  layout.poly.basis = basisOf(1, 3);
  EXPECT_FALSE(Ciphertext::create(layout, 3, 2, &error).has_value());
  EXPECT_NE(error.find("prefix"), std::string::npos) << error;
}

TEST(Ciphertext, CreateRefusesACapacityBelowTheLiveShape) {
  std::string error;
  EXPECT_FALSE(Ciphertext::create(ciphertextOf(3, 2), 2, 2, &error).has_value());
  EXPECT_NE(error.find("limb capacity 2"), std::string::npos) << error;
  EXPECT_FALSE(Ciphertext::create(ciphertextOf(3, 2, 1), 3, 2, &error).has_value())
    << "aux limbs are live limbs and count against the capacity";
  EXPECT_FALSE(Ciphertext::create(ciphertextOf(3, 3), 3, 2, &error).has_value());
  EXPECT_NE(error.find("polynomial capacity 2"), std::string::npos) << error;
}

TEST(Ciphertext, CreateRefusesACapacityOverTheLayoutBounds) {
  std::string error;
  EXPECT_FALSE(Ciphertext::create(ciphertextOf(3, 2), RnsBasis::kMaxLimbs + 1, 2, &error).has_value());
  EXPECT_NE(error.find("limb capacity"), std::string::npos) << error;
  EXPECT_FALSE(Ciphertext::create(ciphertextOf(3, 2), 3, CiphertextLayout::kMaxPolys + 1, &error).has_value());
  EXPECT_NE(error.find("polynomial capacity"), std::string::npos) << error;
  EXPECT_TRUE(Ciphertext::create(ciphertextOf(3, 2), RnsBasis::kMaxLimbs, CiphertextLayout::kMaxPolys).has_value())
    << "the bounds themselves are allowed";
}

// Every limb of every polynomial must start on a cache line, at the smallest
// degree — where a limb is exactly one line and any off-by-one in the stride
// arithmetic would break the alignment for polynomial 1 onwards.
TEST(Ciphertext, EveryLimbOfEveryPolynomialIsCacheLineAligned) {
  for (uint32_t limb_capacity : {3U, 4U, 7U}) {
    auto ct = Ciphertext::create(ciphertextOf(3, 2), limb_capacity, 4);
    ASSERT_TRUE(ct.has_value());
    CiphertextLayout grown = ct->layout();
    grown.num_polys = 4;
    ASSERT_TRUE(ct->reshape(grown));
    for (uint32_t p = 0; p < 4; ++p) {
      for (uint32_t i = 0; i < 3; ++i) {
        EXPECT_TRUE(isAligned(ct->poly(p).limb(i))) << "capacity " << limb_capacity << " poly " << p << " limb " << i;
      }
    }
  }
}

TEST(Ciphertext, RefAndViewAndPolyAddressTheSameWords) {
  auto ct = Ciphertext::create(ciphertextOf(3, 2), 4, 2);
  ASSERT_TRUE(ct.has_value());
  fillTagged(ct->ref());
  EXPECT_EQ(ct->view().data(), ct->slab().data());
  EXPECT_EQ(ct->ref().data(), ct->slab().data());
  EXPECT_EQ(ct->view().polyStride(), ct->polyStride());
  for (uint32_t p = 0; p < 2; ++p) {
    EXPECT_EQ(ct->poly(p).data(), ct->view().poly(p).data());
    EXPECT_EQ(ct->poly(p).data(), ct->slab().data() + p * ct->polyStride());
    EXPECT_EQ(ct->poly(p).limb(2)[5], tag(p, 2, 5));
  }
}

// The claim the per-polynomial reserve was designed for: dropping a limb is a
// layout change and nothing else. Every word that stays live stays where it
// was, verified by pointer AND by content.
TEST(Ciphertext, RescaleShapedReshapeMovesNoWord) {
  auto ct = Ciphertext::create(ciphertextOf(3, 2), 4, 2);
  ASSERT_TRUE(ct.has_value());
  fillTagged(ct->ref());
  const uint64_t *const poly1_before = ct->poly(1).data();

  CiphertextLayout rescaled = ct->layout();
  rescaled.poly.basis.main_end = 2;
  rescaled.scale = 4096.0;
  std::string error;
  ASSERT_TRUE(ct->reshape(rescaled, &error)) << error;

  EXPECT_EQ(ct->level(), 1);
  EXPECT_EQ(ct->scale(), 4096.0);
  EXPECT_EQ(ct->poly(1).data(), poly1_before);
  EXPECT_EQ(ct->poly(1).limbCount(), 2U);
  for (uint32_t p = 0; p < 2; ++p) {
    for (uint32_t i = 0; i < 2; ++i) {
      EXPECT_EQ(ct->poly(p).limb(i)[7], tag(p, i, 7)) << "poly " << p << " limb " << i;
    }
  }
}

// ModUp appends aux limbs at EACH polynomial's own tail, not after all the
// polynomials. If the reserve were a single tail on the whole ciphertext,
// polynomial 0's new aux limb would land on top of polynomial 1's first main
// limb; this test would see the tag of the wrong polynomial there.
TEST(Ciphertext, ModUpShapedReshapeAppendsAtEachPolynomialsOwnTail) {
  auto ct = Ciphertext::create(ciphertextOf(3, 2), 5, 2);
  ASSERT_TRUE(ct.has_value());
  fillTagged(ct->ref());

  CiphertextLayout extended = ct->layout();
  extended.poly.basis.aux_end = 2;
  std::string error;
  ASSERT_TRUE(ct->reshape(extended, &error)) << error;

  for (uint32_t p = 0; p < 2; ++p) {
    const RnsPolyRef poly = ct->poly(p);
    EXPECT_EQ(poly.limbCount(), 5U);
    EXPECT_EQ(poly.auxLimb(0), poly.mainLimb(2) + kDegree) << "aux follows this polynomial's last main limb";
    EXPECT_EQ(poly.auxLimb(1), ct->slab().data() + p * ct->polyStride() + 4 * kDegree);
    for (uint32_t i = 0; i < 3; ++i) {
      EXPECT_EQ(poly.limb(i)[1], tag(p, i, 1)) << "main limbs untouched, poly " << p << " limb " << i;
    }
  }
  // Writing the new aux limbs of polynomial 0 must not disturb polynomial 1.
  for (std::size_t n = 0; n < kDegree; ++n) {
    ct->poly(0).auxLimb(0)[n] = ~uint64_t{0};
    ct->poly(0).auxLimb(1)[n] = ~uint64_t{0};
  }
  EXPECT_EQ(ct->poly(1).limb(0)[0], tag(1, 0, 0));
}

TEST(Ciphertext, TensorShapedReshapeExposesTheThirdPolynomialAtTheStride) {
  auto ct = Ciphertext::create(ciphertextOf(3, 2), 3, 3);
  ASSERT_TRUE(ct.has_value());
  fillTagged(ct->ref());

  CiphertextLayout product = ct->layout();
  product.num_polys = 3;
  product.scale = 8192.0 * 8192.0;
  std::string error;
  ASSERT_TRUE(ct->reshape(product, &error)) << error;
  EXPECT_EQ(ct->poly(2).data(), ct->slab().data() + 2 * ct->polyStride());
  EXPECT_EQ(ct->poly(1).limb(0)[0], tag(1, 0, 0));

  // And back to two after relinearisation, again without moving anything.
  CiphertextLayout relinearised = product;
  relinearised.num_polys = 2;
  ASSERT_TRUE(ct->reshape(relinearised, &error)) << error;
  EXPECT_EQ(ct->poly(1).limb(0)[0], tag(1, 0, 0));
}

// A refused reshape must not partially apply. The pending layout differs
// from the current one in a field that would pass on its own (scale) and a
// field that fails (limbs), and the check is that the scale did not change.
TEST(Ciphertext, ReshapeRefusesMoreLimbsThanTheCapacityAndLeavesTheLayoutAlone) {
  auto ct = Ciphertext::create(ciphertextOf(3, 2), 3, 2);
  ASSERT_TRUE(ct.has_value());
  const CiphertextLayout before = ct->layout();

  CiphertextLayout extended = before;
  extended.poly.basis.aux_end = 1;
  extended.scale = 1.0;
  std::string error;
  EXPECT_FALSE(ct->reshape(extended, &error));
  EXPECT_NE(error.find("4 limbs"), std::string::npos) << error;
  EXPECT_EQ(ct->layout(), before);
}

TEST(Ciphertext, ReshapeRefusesMorePolynomialsThanTheCapacityAndLeavesTheLayoutAlone) {
  auto ct = Ciphertext::create(ciphertextOf(3, 2), 3, 2);
  ASSERT_TRUE(ct.has_value());
  const CiphertextLayout before = ct->layout();

  CiphertextLayout product = before;
  product.num_polys = 3;
  product.scale = 1.0;
  std::string error;
  EXPECT_FALSE(ct->reshape(product, &error));
  EXPECT_NE(error.find("3 polynomials"), std::string::npos) << error;
  EXPECT_EQ(ct->layout(), before);
}

TEST(Ciphertext, ReshapeRefusesAChangeOfRing) {
  auto ct = Ciphertext::create(ciphertextOf(3, 2), 3, 2);
  ASSERT_TRUE(ct.has_value());
  const CiphertextLayout before = ct->layout();
  std::string error;

  CiphertextLayout other_params = before;
  other_params.poly.basis.params_id = ParamsId{0x9999};
  EXPECT_FALSE(ct->reshape(other_params, &error));
  EXPECT_NE(error.find("ring"), std::string::npos) << error;
  EXPECT_EQ(ct->layout(), before);

  // Refused even though this would fit the slab at a lower limb capacity:
  // the stride is in words and was computed from the degree the slab was
  // bound with, so a different degree would misaddress every polynomial.
  CiphertextLayout other_degree = before;
  other_degree.poly.log_degree = kLogDegree + 1;
  ASSERT_TRUE(other_degree.wellFormed());
  EXPECT_FALSE(ct->reshape(other_degree, &error));
  EXPECT_NE(error.find("ring"), std::string::npos) << error;
  EXPECT_EQ(ct->layout(), before);
}

TEST(Ciphertext, ReshapeRefusesAMalformedLayout) {
  auto ct = Ciphertext::create(ciphertextOf(3, 2), 3, 2);
  ASSERT_TRUE(ct.has_value());
  const CiphertextLayout before = ct->layout();
  CiphertextLayout bad = before;
  bad.scale = 0.0;
  std::string error;
  EXPECT_FALSE(ct->reshape(bad, &error));
  EXPECT_NE(error.find("scale"), std::string::npos) << error;
  EXPECT_EQ(ct->layout(), before);
}

TEST(Ciphertext, DuplicateIsAnIndependentOwnerWithTheSameLiveWordsAndCapacities) {
  auto ct = Ciphertext::create(ciphertextOf(3, 2), 5, 3);
  ASSERT_TRUE(ct.has_value());
  fillTagged(ct->ref());

  Ciphertext dup = ct->duplicate();
  EXPECT_EQ(dup.layout(), ct->layout());
  EXPECT_EQ(dup.limbCapacity(), ct->limbCapacity());
  EXPECT_EQ(dup.polyCapacity(), ct->polyCapacity());
  EXPECT_EQ(dup.polyStride(), ct->polyStride());
  EXPECT_NE(dup.slab().data(), ct->slab().data());
  EXPECT_TRUE(liveWordsEqual(dup.view(), ct->view()));

  dup.poly(1).limb(2)[4] = 0;
  EXPECT_EQ(ct->poly(1).limb(2)[4], tag(1, 2, 4)) << "writing the duplicate must not touch the original";

  // The duplicate has the original's headroom, not merely its live shape.
  CiphertextLayout grown = dup.layout();
  grown.num_polys = 3;
  grown.poly.basis.aux_end = 2;
  std::string error;
  EXPECT_TRUE(dup.reshape(grown, &error)) << error;
}

TEST(Ciphertext, DuplicateOfNothingIsNothing) {
  const Ciphertext empty;
  EXPECT_TRUE(empty.duplicate().empty());
}

TEST(Ciphertext, MoveTransfersTheSlabAndLeavesTheSourceShapeless) {
  auto created = Ciphertext::create(ciphertextOf(3, 2), 4, 2);
  ASSERT_TRUE(created.has_value());
  Ciphertext source = std::move(*created);
  fillTagged(source.ref());
  const uint64_t *const slab = source.slab().data();

  Ciphertext moved(std::move(source));
  EXPECT_EQ(moved.slab().data(), slab) << "a move must not reallocate";
  EXPECT_EQ(moved.layout(), ciphertextOf(3, 2));
  EXPECT_EQ(moved.limbCapacity(), 4U);
  EXPECT_EQ(moved.poly(1).limb(1)[1], tag(1, 1, 1));
  EXPECT_TRUE(source.empty());                    // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.layout(), CiphertextLayout{}); // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.limbCapacity(), 0U);           // NOLINT(bugprone-use-after-move)
  EXPECT_EQ(source.polyCapacity(), 0U);           // NOLINT(bugprone-use-after-move)

  Ciphertext assigned;
  assigned = std::move(moved);
  EXPECT_EQ(assigned.slab().data(), slab);
  EXPECT_EQ(assigned.layout(), ciphertextOf(3, 2));
  EXPECT_TRUE(moved.empty()); // NOLINT(bugprone-use-after-move)
}

TEST(Ciphertext, AdoptBindsACallerOwnedSlab) {
  PolySlab slab(Ciphertext::slabWords(kLogDegree, 4, 2) + 3); // excess is allowed and never addressed
  const uint64_t *const data = slab.data();
  std::string error;
  auto ct = Ciphertext::adopt(std::move(slab), ciphertextOf(3, 2), 4, 2, &error);
  ASSERT_TRUE(ct.has_value()) << error;
  EXPECT_EQ(ct->slab().data(), data) << "adopting must not copy";
  EXPECT_EQ(ct->limbCapacity(), 4U);
  EXPECT_EQ(ct->polyCapacity(), 2U);
}

TEST(Ciphertext, AdoptAcceptsASlabOfExactlyTheRequiredSize) {
  std::string error;
  const auto ct =
    Ciphertext::adopt(PolySlab(Ciphertext::slabWords(kLogDegree, 4, 2)), ciphertextOf(3, 2), 4, 2, &error);
  ASSERT_TRUE(ct.has_value()) << error;
  EXPECT_EQ(ct->slab().words(), Ciphertext::slabWords(kLogDegree, 4, 2));
}

TEST(Ciphertext, AdoptRefusesASlabTooSmallForTheCapacities) {
  std::string error;
  EXPECT_FALSE(
    Ciphertext::adopt(PolySlab(Ciphertext::slabWords(kLogDegree, 4, 2) - 1), ciphertextOf(3, 2), 4, 2, &error)
      .has_value());
  EXPECT_NE(error.find("slab holds"), std::string::npos) << error;
  EXPECT_FALSE(Ciphertext::adopt(PolySlab{}, ciphertextOf(3, 2), 3, 2, &error).has_value());
}

TEST(Ciphertext, AdoptAppliesTheSameCapacityChecksAsCreate) {
  std::string error;
  EXPECT_FALSE(Ciphertext::adopt(PolySlab(1000), ciphertextOf(3, 2), 2, 2, &error).has_value());
  EXPECT_NE(error.find("limb capacity 2"), std::string::npos) << error;
}

TEST(Ciphertext, CopyLandsAnArenaShapedTemporaryInAReservedBuffer) {
  // Source: tight stride, as an Arena temporary would be bound.
  const CiphertextLayout layout = ciphertextOf(3, 3);
  std::vector<uint64_t> scratch(layout.wordCount());
  const CiphertextRef temp(layout, scratch.data(), layout.poly.wordCount());
  fillTagged(temp);

  // Destination: the same layout over a slab with headroom in both dimensions.
  auto ct = Ciphertext::create(layout, 6, 4);
  ASSERT_TRUE(ct.has_value());
  ASSERT_NE(ct->polyStride(), temp.polyStride());
  ASSERT_TRUE(copy(ct->ref(), temp));
  EXPECT_TRUE(liveWordsEqual(ct->view(), temp));
  EXPECT_EQ(ct->poly(2).limb(2)[7], tag(2, 2, 7));
}

TEST(Ciphertext, CopyRefusesADifferentLayoutAndWritesNothing) {
  auto src = Ciphertext::create(ciphertextOf(3, 2), 3, 2);
  auto dst = Ciphertext::create(ciphertextOf(3, 2), 3, 2);
  ASSERT_TRUE(src.has_value() && dst.has_value());
  fillTagged(src->ref());
  for (uint32_t p = 0; p < 2; ++p) {
    for (std::size_t i = 0; i < dst->poly(p).words(); ++i) {
      dst->poly(p).data()[i] = 0xDEAD;
    }
  }

  CiphertextLayout rescaled = dst->layout();
  rescaled.poly.basis.main_end = 2;
  ASSERT_TRUE(dst->reshape(rescaled));
  EXPECT_FALSE(copy(dst->ref(), src->view()));
  for (uint32_t p = 0; p < 2; ++p) {
    for (std::size_t i = 0; i < dst->poly(p).words(); ++i) {
      ASSERT_EQ(dst->poly(p).data()[i], 0xDEADU) << "poly " << p << " word " << i;
    }
  }
}

TEST(Ciphertext, CopyOntoItselfIsANoOp) {
  auto ct = Ciphertext::create(ciphertextOf(3, 2), 4, 2);
  ASSERT_TRUE(ct.has_value());
  fillTagged(ct->ref());
  EXPECT_TRUE(copy(ct->ref(), ct->view()));
  EXPECT_EQ(ct->poly(1).limb(2)[3], tag(1, 2, 3));
}

// The handles cross kernel boundaries by value and the owners must not be
// copyable by accident (invariant I4: duplication is a named operation).
TEST(StorageBinding, HandlesAreTrivialValuesAndOwnersAreMoveOnly) {
  static_assert(std::is_trivially_copyable_v<RnsPolyView>);
  static_assert(std::is_trivially_copyable_v<RnsPolyRef>);
  static_assert(std::is_trivially_copyable_v<CiphertextView>);
  static_assert(std::is_trivially_copyable_v<CiphertextRef>);
  static_assert(!std::is_copy_constructible_v<PolySlab>);
  static_assert(!std::is_copy_assignable_v<PolySlab>);
  static_assert(std::is_nothrow_move_constructible_v<PolySlab>);
  static_assert(!std::is_copy_constructible_v<Ciphertext>);
  static_assert(!std::is_copy_assignable_v<Ciphertext>);
  static_assert(std::is_nothrow_move_constructible_v<Ciphertext>);
  SUCCEED();
}

} // namespace
