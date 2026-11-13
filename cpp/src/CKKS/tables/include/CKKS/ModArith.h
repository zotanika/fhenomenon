#pragma once

#include <cstdint>

// Modular arithmetic over a single 64-bit modulus.
//
// This header exists to be the ONLY place in the backend that spells a
// 128-bit integer type. `unsigned __int128` is a compiler extension, so the
// build's -Wpedantic -Werror rejects it on sight; suppressing that warning
// per use site would spread the exemption across every file that multiplies.
// Confining it here keeps -Wpedantic at full strength everywhere else, and
// gives the eventual aarch64/SVE2 or x86-64 specialisation a single seam to
// replace. Nothing else in the CKKS tree may name __int128 directly.
//
// Every function here is a pure function of its arguments. There is no state,
// no table, and no allocation, so this header is usable from any layer.

namespace fhenomenon::ckks::modarith {

#if !defined(__SIZEOF_INT128__)
#error "The CKKS backend needs a 128-bit integer type for 64-bit modular multiplication."
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
using uint128 = unsigned __int128;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

// High 64 bits of a 64x64 product — `umulh` on aarch64, `mulx` on x86-64.
inline uint64_t mulHigh(uint64_t a, uint64_t b) { return static_cast<uint64_t>((uint128{a} * uint128{b}) >> 64); }

// a * b mod q. The general form, used to build tables; the hot path uses the
// Shoup form below, which trades a precomputation for the division.
inline uint64_t mulMod(uint64_t a, uint64_t b, uint64_t q) {
  return static_cast<uint64_t>(uint128{a} * uint128{b} % uint128{q});
}

inline uint64_t powMod(uint64_t base, uint64_t exponent, uint64_t q) {
  uint64_t result = 1 % q;
  uint64_t acc = base % q;
  while (exponent != 0) {
    if ((exponent & 1U) != 0) {
      result = mulMod(result, acc, q);
    }
    acc = mulMod(acc, acc, q);
    exponent >>= 1U;
  }
  return result;
}

// Multiplicative inverse by Fermat's little theorem. Valid only for prime q,
// which is why NttTables::create() proves primality before building anything.
inline uint64_t invMod(uint64_t a, uint64_t q) { return powMod(a, q - 2, q); }

// Shoup's precomputed quotient floor(w * 2^64 / q), the companion of a fixed
// multiplier w. Requires w < q.
inline uint64_t shoupFactor(uint64_t w, uint64_t q) { return static_cast<uint64_t>((uint128{w} << 64) / uint128{q}); }

// Shoup multiplication by a precomputed multiplier, WITHOUT the final
// conditional subtraction: the result is congruent to y * w and lies in
// [0, 2q), not [0, q).
//
// The bound holds for any 64-bit y, not just y < q. Writing w' = floor(w*B/q)
// with B = 2^64 and t = floor(y*w'/B), the returned y*w - t*q is at least 0
// because t <= y*w/q, and is under 2q because t > y*w/q - y/B - 1 > y*w/q - 2.
// That is what lets the butterflies below feed unreduced values straight back
// in, which is the whole point of the lazy form.
//
// PRECONDITION, and the only one in this header that is not range-free:
// q <= 2^63. The bound above is about the mathematical value; the return type
// is a uint64, so a window of width 2q is only representable while 2q fits.
// Above 2^63 the result is not merely out of window, it stops being congruent
// to y*w at all. Every other function here is correct across the whole uint64
// range. Both current callers cap their moduli at 62 bits and say so in their
// own errors, but those caps live in the callers, and this file advertises
// itself as usable from any layer.
//
// The sharper form of the bound is worth having, because callers that sum many
// of these need it: with w' = floor(w*B/q), s = (w*B) mod q and r = (y*w') mod
// B, the returned value is exactly (y*s + r*q)/B. That is continuous on
// [0, q*(1 + y_max/B)) rather than concentrated at 0 or q — so for y under
// 2^62 the true supremum is 1.25q, not 2q, and it is reachable by a caller who
// chooses y rather than only by an unlucky one.
inline uint64_t mulModShoupLazy(uint64_t y, uint64_t w, uint64_t w_shoup, uint64_t q) {
  return y * w - mulHigh(y, w_shoup) * q;
}

// Deterministic Miller-Rabin. These twelve bases are proven correct for every
// n < 3.3 * 10^24, so there is no probabilistic gap anywhere in the uint64
// range and no RNG to inject.
inline bool isPrime(uint64_t n) {
  constexpr uint64_t kBases[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
  if (n < 2) {
    return false;
  }
  for (uint64_t base : kBases) {
    if (n % base == 0) {
      return n == base;
    }
  }

  uint64_t d = n - 1;
  uint32_t twos = 0;
  while ((d & 1U) == 0) {
    d >>= 1U;
    ++twos;
  }

  for (uint64_t base : kBases) {
    uint64_t x = powMod(base, d, n);
    if (x == 1 || x == n - 1) {
      continue;
    }
    bool witnessed = false;
    for (uint32_t i = 1; i < twos; ++i) {
      x = mulMod(x, x, n);
      if (x == n - 1) {
        witnessed = true;
        break;
      }
    }
    if (!witnessed) {
      return false;
    }
  }
  return true;
}

} // namespace fhenomenon::ckks::modarith
