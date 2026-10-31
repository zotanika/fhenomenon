#include "CKKS/Encoding.h"
#include "CKKS/Arena.h"
#include "CKKS/EvalTables.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <new>

namespace fhenomenon::ckks::encoding {
namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;
using Complex = std::complex<double>;

bool validPoly(ConstRnsPolyView p, const EvaluationTables &tables) {
  const auto &b = p.layout.basis;
  return p.wellFormed() && b.params_id == tables.paramsId() && p.layout.log_degree == tables.params().logDegree() &&
         p.layout.form == NttForm::Evaluation && b.main_begin == 0 && b.main_end != 0 &&
         b.main_end <= tables.params().levelCount() && b.aux_begin == 0 && b.aux_end == 0;
}

// A positive transform evaluates at roots exp(+2*pi*i*k/N). The negative
// transform interpolates when divided by N. The half-root twist supplies
// the odd 2N-th roots of X^N+1 used by the canonical embedding.
void transform(Complex *a, std::size_t n, bool positive) {
  for (std::size_t i = 1, j = 0; i < n; ++i) {
    std::size_t bit = n >> 1;
    for (; (j & bit) != 0; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j)
      std::swap(a[i], a[j]);
  }
  for (std::size_t length = 2; length <= n; length <<= 1) {
    const double angle = (positive ? 2.0 : -2.0) * kPi / static_cast<double>(length);
    const Complex step(std::cos(angle), std::sin(angle));
    for (std::size_t start = 0; start < n; start += length) {
      Complex root(1.0, 0.0);
      for (std::size_t j = 0; j < length / 2; ++j) {
        const Complex even = a[start + j];
        const Complex odd = a[start + j + length / 2] * root;
        a[start + j] = even + odd;
        a[start + j + length / 2] = even - odd;
        root *= step;
      }
    }
  }
}
uint64_t signedResidue(int64_t value, uint64_t q) {
  if (value >= 0)
    return static_cast<uint64_t>(value) % q;
  const uint64_t magnitude = static_cast<uint64_t>(-(value + 1)) + 1;
  const uint64_t remainder = magnitude % q;
  return remainder == 0 ? 0 : q - remainder;
}
} // namespace

std::size_t scratchBytes(const Params &params) noexcept {
  // Decode dominates: one Q polynomial, one complex FFT, and CRT vectors.
  // Include coefficient rounding storage and alignment as a conservative bound.
  const std::size_t n = params.degree();
  return n * (sizeof(Complex) + sizeof(int64_t) + params.levelCount() * sizeof(uint64_t)) +
         2 * params.levelCount() * sizeof(uint64_t) + 512;
}

// Rounded coefficients must fit both the int64 encoding bound and the centered
// interval [-floor(Qactive/2), floor(Qactive/2)]. Reject before any output write.
bool encode(RnsPolyView &out, const double *values, uint32_t count, double scale, const EvaluationTables &tables,
            Arena &arena) {
  const auto &params = tables.params();
  const std::size_t n = params.degree();
  if (!validPoly(asConst(out), tables) || values == nullptr || count == 0 || count > n / 2 ||
      (count & (count - 1)) != 0 || !std::isfinite(scale) || scale <= 0 || arena.remaining() < scratchBytes(params))
    return false;
  for (uint32_t i = 0; i < count; ++i)
    if (!std::isfinite(values[i]))
      return false;
  uint64_t modulus_product = 1;
  bool product_fits = true;
  for (uint32_t limb = 0; limb < out.layout.basis.main_end; ++limb) {
    const uint64_t q = params.mainPrimes()[limb];
    if (modulus_product > std::numeric_limits<uint64_t>::max() / q) {
      product_fits = false;
      break;
    }
    modulus_product *= q;
  }
  // Once Q exceeds uint64, floor(Q/2) is larger than every permitted int64
  // magnitude. Otherwise the exact integer half avoids double rounding at
  // large boundaries and detects wraparound even on the final active limb.
  const uint64_t coefficient_limit = product_fits ? modulus_product / 2 : std::numeric_limits<uint64_t>::max();
  Arena::Scope scope(arena);
  auto *frequency = arena.allocate<Complex>(n);
  auto *rounded = arena.allocate<int64_t>(n);
  if (frequency == nullptr || rounded == nullptr)
    return false;
  for (std::size_t i = 0; i < n; ++i)
    new (frequency + i) Complex(0.0, 0.0);
  uint64_t exponent = 1;
  const uint64_t two_n = 2 * n;
  for (std::size_t j = 0; j < n / 2; ++j) {
    const double value = values[j % count];
    frequency[(exponent - 1) / 2] = value;
    frequency[(two_n - exponent - 1) / 2] = value;
    exponent = (exponent * 5) % two_n;
  }
  transform(frequency, n, false);
  const double divisor = static_cast<double>(n);
  for (std::size_t j = 0; j < n; ++j) {
    const double angle = -kPi * static_cast<double>(j) / divisor;
    const double coefficient = (frequency[j] * Complex(std::cos(angle), std::sin(angle))).real() / divisor;
    const double value = std::round(coefficient * scale);
    // 2^63 is exactly representable in double, unlike INT64_MAX. Test before
    // the cast; excluding both endpoints also keeps magnitude conversion safe.
    if (!std::isfinite(value) || value <= -0x1p63 || value >= 0x1p63)
      return false;
    rounded[j] = static_cast<int64_t>(value);
    const uint64_t magnitude = rounded[j] < 0 ? static_cast<uint64_t>(-rounded[j]) : static_cast<uint64_t>(rounded[j]);
    if (magnitude > coefficient_limit)
      return false;
  }
  for (uint32_t limb = 0; limb < out.layout.basis.main_end; ++limb) {
    const uint64_t q = params.mainPrimes()[limb];
    uint64_t *destination = out.limb(limb);
    for (std::size_t j = 0; j < n; ++j)
      destination[j] = signedResidue(rounded[j], q);
    ntt::forward(destination, tables.mainNtt(limb));
  }
  return true;
}

bool decode(double *values, uint32_t count, ConstRnsPolyView input, double scale, const EvaluationTables &tables,
            Arena &arena) {
  const auto &params = tables.params();
  const std::size_t n = params.degree();
  if (!validPoly(input, tables) || values == nullptr || count == 0 || count > n / 2 || !std::isfinite(scale) ||
      scale <= 0 || arena.remaining() < scratchBytes(params))
    return false;
  const uint32_t limbs = input.layout.basis.main_end;
  for (uint32_t limb = 0; limb < limbs; ++limb)
    for (std::size_t j = 0; j < n; ++j)
      if (input.limb(limb)[j] >= params.mainPrimes()[limb])
        return false;
  Arena::Scope scope(arena);
  auto *coefficients = arena.allocate<uint64_t>(n * limbs);
  auto *frequency = arena.allocate<Complex>(n);
  auto *residues = arena.allocate<uint64_t>(limbs);
  auto *digits = arena.allocate<uint64_t>(limbs);
  if (coefficients == nullptr || frequency == nullptr || residues == nullptr || digits == nullptr)
    return false;
  for (uint32_t limb = 0; limb < limbs; ++limb) {
    std::copy_n(input.limb(limb), n, coefficients + limb * n);
    ntt::inverse(coefficients + limb * n, tables.mainNtt(limb));
  }
  const auto &crt = tables.mainCrt(limbs - 1);
  for (std::size_t j = 0; j < n; ++j) {
    for (uint32_t limb = 0; limb < limbs; ++limb)
      residues[limb] = coefficients[limb * n + j];
    if (!crt.toMixedRadix(residues, digits))
      return false;
    const double coefficient = static_cast<double>(crt.centeredValue(digits) / static_cast<long double>(scale));
    if (!std::isfinite(coefficient))
      return false;
    const double angle = kPi * static_cast<double>(j) / static_cast<double>(n);
    new (frequency + j) Complex(coefficient * std::cos(angle), coefficient * std::sin(angle));
  }
  transform(frequency, n, true);
  uint64_t exponent = 1;
  for (uint32_t j = 0; j < count; ++j) {
    if (!std::isfinite(frequency[(exponent - 1) / 2].real()))
      return false;
    exponent = (exponent * 5) % (2 * n);
  }
  exponent = 1;
  for (uint32_t j = 0; j < count; ++j) {
    values[j] = frequency[(exponent - 1) / 2].real();
    exponent = (exponent * 5) % (2 * n);
  }
  return true;
}
} // namespace fhenomenon::ckks::encoding
