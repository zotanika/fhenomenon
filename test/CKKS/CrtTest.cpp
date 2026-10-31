#include "CKKS/CrtTables.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <string>

namespace {
using fhenomenon::ckks::CrtTables;

TEST(Crt, ValidatesTheEntireBasis) {
  std::string error;
  EXPECT_FALSE(CrtTables::create({}, &error));
  EXPECT_FALSE(error.empty());
  EXPECT_FALSE(CrtTables::create({3, 3}));
  EXPECT_FALSE(CrtTables::create({3, 9}));
  EXPECT_FALSE(CrtTables::create({1}));
  EXPECT_FALSE(CrtTables::create({uint64_t{1} << 62}));
  ASSERT_TRUE(CrtTables::create({3, 5, 7}, &error));
  EXPECT_TRUE(error.empty());
}

TEST(Crt, ExhaustivelyMatchesIndependentSmallIntegerArithmetic) {
  const auto tables = CrtTables::create({3, 5, 7});
  ASSERT_TRUE(tables);
  for (uint64_t value = 0; value < 105; ++value) {
    const std::array<uint64_t, 3> residues{value % 3, value % 5, value % 7};
    std::array<uint64_t, 3> digits{};
    ASSERT_TRUE(tables->toMixedRadix(residues.data(), digits.data()));
    EXPECT_EQ(digits[0] + 3 * digits[1] + 15 * digits[2], value);
    const int64_t centered = value > 52 ? static_cast<int64_t>(value) - 105 : static_cast<int64_t>(value);
    EXPECT_EQ(tables->isNegative(digits.data()), centered < 0);
    EXPECT_EQ(tables->centeredValue(digits.data()), static_cast<long double>(centered));
    EXPECT_EQ(tables->centeredMod(digits.data(), 11), static_cast<uint64_t>((centered % 11 + 11) % 11));
    EXPECT_EQ(tables->centeredMod(digits.data(), 4), static_cast<uint64_t>((centered % 4 + 4) % 4));
    EXPECT_EQ(tables->centeredMod(digits.data(), 1), 0U);
  }
}

TEST(Crt, ConvertsInPlaceAndRejectsInvalidInputsBeforeWriting) {
  const auto tables = CrtTables::create({3, 5, 7});
  ASSERT_TRUE(tables);
  std::array<uint64_t, 3> values{1, 3, 3}; // 73, centered -32.
  ASSERT_TRUE(tables->toMixedRadix(values.data(), values.data()));
  EXPECT_EQ(values, (std::array<uint64_t, 3>{1, 4, 4}));
  const std::array<uint64_t, 3> invalid{1, 5, 0};
  const auto before = values;
  EXPECT_FALSE(tables->toMixedRadix(invalid.data(), values.data()));
  EXPECT_EQ(values, before);
  EXPECT_FALSE(tables->toMixedRadix(nullptr, values.data()));
  EXPECT_FALSE(tables->toMixedRadix(values.data(), nullptr));
  std::array<uint64_t, 4> overlap{1, 2, 3, 999};
  const auto untouched = overlap;
  EXPECT_FALSE(tables->toMixedRadix(overlap.data(), overlap.data() + 1));
  EXPECT_EQ(overlap, untouched);
}

TEST(Crt, ExactHalfBoundaryIncludesEvenBases) {
  const auto tables = CrtTables::create({2, 3, 5});
  ASSERT_TRUE(tables);
  std::array<uint64_t, 3> half{1, 0, 0}; // 15 = Q/2, convention is positive half.
  std::array<uint64_t, 3> digits{};
  ASSERT_TRUE(tables->toMixedRadix(half.data(), digits.data()));
  EXPECT_FALSE(tables->isNegative(digits.data()));
  EXPECT_EQ(tables->centeredValue(digits.data()), 15.0L);
  half = {0, 1, 1}; // 16 maps to -14.
  ASSERT_TRUE(tables->toMixedRadix(half.data(), digits.data()));
  EXPECT_TRUE(tables->isNegative(digits.data()));
  EXPECT_EQ(tables->centeredValue(digits.data()), -14.0L);
}

TEST(Crt, NegativeValuesAcross118BitsAvoidFloatingCancellation) {
  const std::array<uint64_t, 4> moduli{2305843009213693951ULL, 2147483647, 524287, 127};
  const auto tables = CrtTables::create({moduli.begin(), moduli.end()});
  ASSERT_TRUE(tables);
  for (uint64_t magnitude : {uint64_t{1}, uint64_t{17}, uint64_t{12345}}) {
    std::array<uint64_t, 4> residues{};
    for (size_t i = 0; i < residues.size(); ++i)
      residues[i] = (moduli[i] - magnitude % moduli[i]) % moduli[i];
    std::array<uint64_t, 4> digits{};
    ASSERT_TRUE(tables->toMixedRadix(residues.data(), digits.data()));
    EXPECT_TRUE(tables->isNegative(digits.data()));
    EXPECT_EQ(tables->centeredValue(digits.data()), -static_cast<long double>(magnitude));
    EXPECT_EQ(tables->centeredMod(digits.data(), 65537), 65537U - magnitude);
    EXPECT_EQ(tables->centeredMod(digits.data(), std::numeric_limits<uint64_t>::max()),
              std::numeric_limits<uint64_t>::max() - magnitude);
  }
}

TEST(Crt, LargePositiveValueHasCorrectSignAndResidue) {
  // 2^80 + 12345, independently reduced with integer arithmetic.
  const auto tables = CrtTables::create({2305843009213693951ULL, 2147483647, 524287, 127});
  ASSERT_TRUE(tables);
  const std::array<uint64_t, 4> residues{536633, 274489, 12361, 34};
  std::array<uint64_t, 4> digits{};
  ASSERT_TRUE(tables->toMixedRadix(residues.data(), digits.data()));
  EXPECT_FALSE(tables->isNegative(digits.data()));
  EXPECT_EQ(tables->centeredMod(digits.data(), 65537), 12344U);
  EXPECT_EQ(tables->centeredValue(digits.data()), 1208925819614629174718521.0L);
}

TEST(Crt, ExactSignAcross118BitMidpointDoesNotUseFloatingPoint) {
  // Q = 329710221489888311046522117729812353. These residues and
  // centered mod-65537 answers were calculated independently as integers.
  const auto tables = CrtTables::create({2305843009213693951ULL, 2147483647, 524287, 127});
  ASSERT_TRUE(tables);
  const std::array<uint64_t, 4> below{1152921504606846974ULL, 1073741822, 262142, 62};
  const std::array<uint64_t, 4> middle{1152921504606846975ULL, 1073741823, 262143, 63};
  const std::array<uint64_t, 4> above{1152921504606846976ULL, 1073741824, 262144, 64};
  std::array<uint64_t, 4> digits{};
  ASSERT_TRUE(tables->toMixedRadix(below.data(), digits.data()));
  EXPECT_FALSE(tables->isNegative(digits.data()));
  EXPECT_EQ(tables->centeredMod(digits.data(), 65537), 34565U);
  ASSERT_TRUE(tables->toMixedRadix(middle.data(), digits.data()));
  EXPECT_FALSE(tables->isNegative(digits.data()));
  EXPECT_EQ(tables->centeredMod(digits.data(), 65537), 34566U);
  ASSERT_TRUE(tables->toMixedRadix(above.data(), digits.data()));
  EXPECT_TRUE(tables->isNegative(digits.data()));
  EXPECT_EQ(tables->centeredMod(digits.data(), 65537), 30971U);
}

TEST(Crt, SinglePrimeAndUnsortedBasisAreSupported) {
  const auto single = CrtTables::create({65537});
  ASSERT_TRUE(single);
  uint64_t residue = 65536;
  uint64_t digit = 0;
  ASSERT_TRUE(single->toMixedRadix(&residue, &digit));
  EXPECT_EQ(single->centeredValue(&digit), -1.0L);
  EXPECT_EQ(single->centeredMod(&digit, 0), 0U);
  const auto unsorted = CrtTables::create({7, 3, 5});
  ASSERT_TRUE(unsorted);
  const std::array<uint64_t, 3> residues{5, 1, 1}; // 61 modulo 105.
  std::array<uint64_t, 3> digits{};
  ASSERT_TRUE(unsorted->toMixedRadix(residues.data(), digits.data()));
  EXPECT_EQ(digits, (std::array<uint64_t, 3>{5, 2, 2}));
  EXPECT_EQ(unsorted->centeredValue(digits.data()), -44.0L);
}
} // namespace
