#include "CKKS/CrtTables.h"

#include "CKKS/ModArith.h"

#include <cstddef>
#include <limits>
#include <utility>

namespace fhenomenon::ckks {
namespace {
// Both arguments are reduced, including when modulus uses all 64 bits.
uint64_t addMod(uint64_t a, uint64_t b, uint64_t modulus) { return a >= modulus - b ? a - (modulus - b) : a + b; }
} // namespace

std::optional<CrtTables> CrtTables::create(std::vector<uint64_t> moduli, std::string *error) {
  const auto fail = [error](const char *message) -> std::optional<CrtTables> {
    if (error != nullptr)
      *error = message;
    return std::nullopt;
  };
  const std::size_t count = moduli.size();
  if (count == 0)
    return fail("CRT basis is empty");
  if (count > std::numeric_limits<std::size_t>::max() / count / sizeof(uint64_t))
    return fail("CRT inverse table size overflows");
  for (std::size_t i = 0; i < count; ++i) {
    if (moduli[i] >= (uint64_t{1} << 62) || !modarith::isPrime(moduli[i]))
      return fail("CRT moduli must be primes below 2^62");
    for (std::size_t j = 0; j < i; ++j)
      if (moduli[i] == moduli[j])
        return fail("CRT moduli must be pairwise distinct");
  }
  CrtTables result;
  result.moduli_ = std::move(moduli);
  result.inverses_.resize(count * count);
  result.half_.resize(count);
  for (std::size_t i = 0; i < count; ++i)
    for (std::size_t j = 0; j < i; ++j)
      result.inverses_[i * count + j] = modarith::invMod(result.moduli_[j] % result.moduli_[i], result.moduli_[i]);

  // Divide Q, represented by a leading 1 and count zero digits, by 2.
  // Carry is only 0 or 1; this also handles a basis containing prime 2.
  uint64_t carry = 1;
  for (std::size_t i = count; i != 0; --i) {
    const uint64_t value = carry * result.moduli_[i - 1];
    result.half_[i - 1] = value / 2;
    carry = value % 2;
  }
  if (error != nullptr)
    error->clear();
  return result;
}

bool CrtTables::toMixedRadix(const uint64_t *residues, uint64_t *digits) const {
  if (residues == nullptr || digits == nullptr)
    return false;
  const std::size_t count = moduli_.size();
  const std::size_t bytes = count * sizeof(uint64_t);
  const auto source = reinterpret_cast<uintptr_t>(residues);
  const auto destination = reinterpret_cast<uintptr_t>(digits);
  if (source != destination && (source < destination ? destination - source < bytes : source - destination < bytes))
    return false;
  for (std::size_t i = 0; i < count; ++i)
    if (residues[i] >= moduli_[i])
      return false;

  // Garner elimination. Earlier output digits are the only scratch needed,
  // and the complete residue validation above precedes any writes.
  for (std::size_t i = 0; i < count; ++i) {
    const uint64_t modulus = moduli_[i];
    uint64_t value = residues[i];
    for (std::size_t j = 0; j < i; ++j) {
      const uint64_t earlier = digits[j] % modulus;
      value = value >= earlier ? value - earlier : modulus - (earlier - value);
      value = modarith::mulMod(value, inverses_[i * count + j], modulus);
    }
    digits[i] = value;
  }
  return true;
}

bool CrtTables::isNegative(const uint64_t *digits) const {
  for (std::size_t i = moduli_.size(); i != 0; --i)
    if (digits[i - 1] != half_[i - 1])
      return digits[i - 1] > half_[i - 1];
  return false;
}

uint64_t CrtTables::centeredMod(const uint64_t *digits, uint64_t modulus) const {
  if (modulus <= 1)
    return 0;
  uint64_t value = 0;
  uint64_t product = 1;
  for (std::size_t i = moduli_.size(); i != 0; --i) {
    value = addMod(modarith::mulMod(value, moduli_[i - 1], modulus), digits[i - 1] % modulus, modulus);
    product = modarith::mulMod(product, moduli_[i - 1], modulus);
  }
  if (isNegative(digits))
    value = value >= product ? value - product : modulus - (product - value);
  return value;
}

long double CrtTables::centeredValue(const uint64_t *digits) const {
  const bool negative = isNegative(digits);
  long double value = 0.0L;
  for (std::size_t i = moduli_.size(); i != 0; --i) {
    const uint64_t digit = negative ? moduli_[i - 1] - 1 - digits[i - 1] : digits[i - 1];
    value = value * static_cast<long double>(moduli_[i - 1]) + static_cast<long double>(digit);
  }
  // Q - x = (Q - 1 - x) + 1. Every complement digit is nonnegative;
  // no subtraction of two huge, already-rounded floating values occurs.
  return negative ? -(value + 1.0L) : value;
}
} // namespace fhenomenon::ckks
