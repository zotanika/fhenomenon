// Copyright 2026 Scalable Computer Architecture Laboratory, Seoul National University.
// MIT; see LICENSE.cheddar. Extracted unchanged butterfly functions from
// Cheddar c428a134bd871c57f73b286c695eb55a13f4f2e6/include/core/NTTUtils.cuh.
#pragma once
#include "common/Basic.cuh"

namespace cheddar {
namespace kernel {

/**
 * @brief Compute a = a + bw, b = a - bw
 *
 * @tparam word
 * @param a in range [-q, q)
 * @param b in range [-q, q)
 * @param w in range [0, q)
 * @param q (prime number)
 * @param q_inv (montgomery const)
 */
template <typename word>
__device__ __inline__ void ButterflyNTT(make_signed_t<word> &a,
                                        make_signed_t<word> &b, const word w,
                                        const word q,
                                        const make_signed_t<word> q_inv) {
  using signed_word = make_signed_t<word>;

  if (a < 0) a += q;  // [0, q) range

  signed_word mult = basic::detail::__mult_montgomery_lazy<word>(
      b, static_cast<signed_word>(w), q, q_inv);  // (-q, q)
  if (mult < 0) mult += q;                        // [0, q) range

  b = a - mult;        // (-q, q)
  a = (a - q) + mult;  // [-q, q)
}

/**
 * @brief Compute a = (a + b), b = (a - b)w
 *
 * @tparam word
 * @param a in range [-q, q)
 * @param b in range [-q, q)
 * @param w in range [0, q)
 * @param q (prime number)
 * @param q_inv (montgomery const)
 */
template <typename word>
__device__ __inline__ void ButterflyINTT(make_signed_t<word> &a,
                                         make_signed_t<word> &b, const word w,
                                         const word q,
                                         const make_signed_t<word> q_inv) {
  using signed_word = make_signed_t<word>;

  if (a < 0) a += q;         // [0, q)
  if (b < 0) b += q;         // [0, q)
  signed_word diff = a - b;  // (-q, q)
  a = (a - q) + b;           // [-q, q)
  b = basic::detail::__mult_montgomery_lazy<word>(
      diff, static_cast<signed_word>(w), q,
      q_inv);  // (-q, q)
}

} // namespace kernel
} // namespace cheddar
