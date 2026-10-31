#include "CKKS/Eval.h"
#include "CKKS/ModArith.h"
#include "CKKS/detail/Region.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace fhenomenon::ckks::eval {
namespace {
std::size_t padded(std::size_t words) {
  return (words * sizeof(uint64_t) + Arena::kAlignment - 1) & ~(Arena::kAlignment - 1);
}
uint64_t plusMod(uint64_t a, uint64_t b, uint64_t q) {
  const uint64_t sum = a + b;
  return sum >= q ? sum - q : sum;
}
uint64_t canonicalProduct(uint64_t y, uint64_t w, uint64_t quotient, uint64_t q) {
  const auto value = modarith::mulModShoupLazy(y, w, quotient, q);
  return value >= q ? value - q : value;
}
uint64_t minusMod(uint64_t a, uint64_t b, uint64_t q) { return a >= b ? a - b : a + q - b; }
uint64_t signedMod(int64_t value, uint64_t q) {
  const bool negative = value < 0;
  const uint64_t magnitude = negative ? uint64_t{0} - static_cast<uint64_t>(value) : static_cast<uint64_t>(value);
  const uint64_t reduced = magnitude % q;
  return negative && reduced != 0 ? q - reduced : reduced;
}
bool quantize(double scalar, double scale, int64_t &result) {
  const long double rounded = std::round(static_cast<long double>(scalar) * static_cast<long double>(scale));
  constexpr long double bound = 9223372036854775808.0L;
  if (!std::isfinite(rounded) || rounded < -bound || rounded >= bound)
    return false;
  result = static_cast<int64_t>(rounded);
  return true;
}
std::size_t spanWords(ConstCiphertextView value) {
  return (value.layout.num_polys - 1U) * value.poly_stride_words + value.layout.poly.wordCount();
}
bool overlap(ConstCiphertextView a, ConstCiphertextView b) {
  const auto left = reinterpret_cast<std::uintptr_t>(a.data), right = reinterpret_cast<std::uintptr_t>(b.data);
  const auto left_bytes = spanWords(a) * sizeof(uint64_t), right_bytes = spanWords(b) * sizeof(uint64_t);
  return left < right + right_bytes && right < left + left_bytes;
}
bool aliasAllowed(ConstCiphertextView out, ConstCiphertextView input) {
  return !overlap(out, input) || (out.data == input.data && out.poly_stride_words == input.poly_stride_words);
}
bool canonical(ConstCiphertextView value, const EvaluationTables &tables, bool extended) {
  const auto k = value.layout.poly.basis.main_end;
  const auto p = extended ? value.layout.poly.basis.aux_end : 0U;
  for (uint32_t poly = 0; poly < value.layout.num_polys; ++poly) {
    for (uint32_t limb = 0; limb < k + p; ++limb) {
      const auto q = limb < k ? tables.params().mainPrimes()[limb] : tables.params().auxPrimes()[limb - k];
      const auto *data = value.poly(poly).limb(limb);
      for (std::size_t i = 0; i < tables.params().degree(); ++i)
        if (data[i] >= q)
          return false;
    }
  }
  return true;
}
bool inputMetadata(ConstCiphertextView value, const EvaluationTables &tables) {
  if (!value.wellFormed())
    return false;
  const auto &poly = value.layout.poly;
  const auto &basis = poly.basis;
  if (basis.params_id != tables.paramsId() || poly.log_degree != tables.params().logDegree() ||
      poly.form != NttForm::Evaluation || basis.main_begin != 0 || basis.main_end > tables.params().levelCount() ||
      basis.aux_begin != 0 || basis.aux_end != 0)
    return false;
  return true;
}
bool validInput(ConstCiphertextView value, const EvaluationTables &tables) {
  return inputMetadata(value, tables) && canonical(value, tables, false);
}
bool sameBasis(ConstCiphertextView a, ConstCiphertextView b) { return a.layout.poly == b.layout.poly; }
bool sameScale(double a, double b) { return std::abs(a - b) <= 1e-12 * std::max(a, b); }
bool outputReady(CiphertextView out, CiphertextLayout layout, ConstCiphertextView a,
                 const ConstCiphertextView *b = nullptr) {
  out.layout = layout;
  if (!out.wellFormed() || !aliasAllowed(asConst(out), a))
    return false;
  return b == nullptr || aliasAllowed(asConst(out), *b);
}
bool scratchReady(EvalOperation op, ConstCiphertextView input, const EvaluationTables &tables, Arena &arena) {
  const auto bytes = scratchBytes(op, tables.params(), input.layout.poly.basis.main_end - 1);
  return bytes != 0 && arena.remaining() >= bytes;
}
CiphertextView temporary(CiphertextLayout layout, Arena &arena) {
  const auto words = layout.wordCount();
  return {arena.allocate<uint64_t>(words), words, layout.poly.wordCount(), layout};
}
void commit(CiphertextView &out, ConstCiphertextView value) {
  for (uint32_t poly = 0; poly < value.layout.num_polys; ++poly)
    std::copy_n(value.data + poly * value.poly_stride_words, value.layout.poly.wordCount(),
                out.data + poly * out.poly_stride_words);
  out.layout = value.layout;
}
void copyInto(CiphertextView out, ConstCiphertextView input) {
  for (uint32_t poly = 0; poly < out.layout.num_polys; ++poly)
    std::copy_n(input.data + poly * input.poly_stride_words, out.layout.poly.wordCount(),
                out.data + poly * out.poly_stride_words);
}
bool keyMetadata(EvaluationKeyView key, SwitchKeyKind kind, uint64_t galois, const EvaluationTables &tables) {
  const auto &params = tables.params();
  const auto k = params.levelCount();
  const auto p = static_cast<uint32_t>(params.auxPrimes().size());
  if (key.digits == nullptr || key.digit_count != params.dnum() || key.params_id != tables.paramsId() ||
      key.log_degree != params.logDegree() || key.main_count != k || key.aux_count != p || key.kind != kind ||
      key.galois_element != galois)
    return false;
  for (uint32_t d = 0; d < key.digit_count; ++d) {
    const auto digit = key.digits[d];
    if (!digit.wellFormed() || digit.layout.num_polys != 2 || digit.layout.scale != 1.0 ||
        digit.layout.poly != PolyLayout{{tables.paramsId(), 0, k, 0, p}, params.logDegree(), NttForm::Evaluation})
      return false;
  }
  return true;
}
bool keyDisjoint(EvaluationKeyView key, ConstCiphertextView output) {
  for (uint32_t d = 0; d < key.digit_count; ++d)
    if (overlap(output, key.digits[d]))
      return false;
  return true;
}
bool validKey(EvaluationKeyView key, SwitchKeyKind kind, uint64_t galois, const EvaluationTables &tables,
              CiphertextView output, CiphertextLayout target) {
  if (!keyMetadata(key, kind, galois, tables))
    return false;
  output.layout = target;
  if (!keyDisjoint(key, asConst(output)))
    return false;
  for (uint32_t d = 0; d < key.digit_count; ++d)
    if (!canonical(key.digits[d], tables, true))
      return false;
  return true;
}
bool linear(CiphertextView &out, ConstCiphertextView a, ConstCiphertextView b, bool subtract_values,
            const EvaluationTables &tables, Arena &arena) {
  const auto operation = subtract_values ? EvalOperation::Subtract : EvalOperation::Add;
  if (!validInput(a, tables) || !validInput(b, tables) || !sameBasis(a, b) ||
      a.layout.num_polys != b.layout.num_polys || !sameScale(a.layout.scale, b.layout.scale) ||
      !outputReady(out, a.layout, a, &b) || !scratchReady(operation, a, tables, arena))
    return false;
  Arena::Scope scope(arena);
  auto tmp = temporary(a.layout, arena);
  if (tmp.data == nullptr)
    return false;
  for (uint32_t poly = 0; poly < a.layout.num_polys; ++poly) {
    for (uint32_t limb = 0; limb < a.layout.poly.basis.main_end; ++limb) {
      const auto q = tables.params().mainPrimes()[limb];
      const auto *left = a.poly(poly).limb(limb), *right = b.poly(poly).limb(limb);
      auto *dst = tmp.poly(poly).limb(limb);
      for (std::size_t i = 0; i < tables.params().degree(); ++i)
        dst[i] = subtract_values ? minusMod(left[i], right[i], q) : plusMod(left[i], right[i], q);
    }
  }
  commit(out, asConst(tmp));
  return true;
}
uint64_t bitReverse(uint64_t index, uint32_t bits) {
  uint64_t result = 0;
  for (uint32_t i = 0; i < bits; ++i) {
    result = (result << 1U) | (index & 1U);
    index >>= 1U;
  }
  return result;
}

// Decompose one public polynomial and accumulate the two switch components in
// active Q followed by P. Source limbs already have their exact NTT values;
// only converted complement limbs need a forward transform.
template <bool Cached = false>
bool switchAccumulate(uint64_t *accum, ConstRnsPolyView input, EvaluationKeyView key, const EvaluationTables &tables,
                      Arena &arena, const uint64_t *coefficients = nullptr, const uint64_t *quotients = nullptr) {
  Arena::Scope scope(arena);
  const auto k = input.layout.basis.main_end;
  const auto p = static_cast<uint32_t>(tables.params().auxPrimes().size());
  const auto ext = k + p;
  const auto n = tables.params().degree();
  const auto stride = ext * n;
  auto *digit = arena.allocate<uint64_t>(stride);
  auto *gather = arena.allocate<uint64_t>(ext);
  auto *converted = arena.allocate<uint64_t>(ext);
  if (!digit || !gather || !converted)
    return false;
  std::fill_n(accum, 2 * stride, uint64_t{0});
  for (uint32_t d = 0; d < key.digit_count; ++d) {
    const auto &plan = tables.digit(k - 1, d);
    if (plan.begin == plan.end)
      continue;
    for (uint32_t limb = plan.begin; limb < plan.end; ++limb) {
      if (coefficients)
        std::copy_n(coefficients + limb * n, n, digit + limb * n);
      else {
        std::copy_n(input.limb(limb), n, digit + limb * n);
        ntt::inverse(digit + limb * n, tables.mainNtt(limb));
      }
    }
    for (std::size_t i = 0; i < n; ++i) {
      for (uint32_t limb = plan.begin; limb < plan.end; ++limb)
        gather[limb - plan.begin] = digit[limb * n + i];
      bconv::convert(gather, converted, *plan.conversion);
      for (std::size_t j = 0; j < plan.destinations.size(); ++j)
        digit[plan.destinations[j] * n + i] = converted[j];
    }
    for (uint32_t limb = 0; limb < ext; ++limb) {
      const auto &table = limb < k ? tables.mainNtt(limb) : tables.auxNtt(limb - k);
      const auto q = table.modulus();
      const bool source = limb >= plan.begin && limb < plan.end;
      if (!source)
        ntt::forward(digit + limb * n, table);
      const auto *values = source ? input.limb(limb) : digit + limb * n;
      // P follows the ORIGINAL full Q in the immutable evaluation key.
      const uint32_t key_limb = limb < k ? limb : key.main_count + limb - k;
      for (uint32_t poly = 0; poly < 2; ++poly) {
        const auto *hint = key.digits[d].poly(poly).limb(key_limb);
        auto *dst = accum + poly * stride + limb * n;
        if constexpr (Cached) {
          const auto *companion = quotients + ((d * 2U + poly) * (key.main_count + key.aux_count) + key_limb) * n;
          for (std::size_t i = 0; i < n; ++i)
            dst[i] = plusMod(dst[i], canonicalProduct(values[i], hint[i], companion[i], q), q);
        } else {
          for (std::size_t i = 0; i < n; ++i)
            dst[i] = plusMod(dst[i], modarith::mulMod(values[i], hint[i], q), q);
        }
      }
    }
  }
  return true;
}

// Standalone exact P ModDown remains the primitive differential reference.
template <bool Cached = false>
bool switchCore(CiphertextView out, ConstRnsPolyView input, EvaluationKeyView key, const EvaluationTables &tables,
                Arena &arena, const uint64_t *coefficients = nullptr, const uint64_t *quotients = nullptr) {
  Arena::Scope scope(arena);
  const auto k = input.layout.basis.main_end;
  const auto p = static_cast<uint32_t>(tables.params().auxPrimes().size());
  const auto n = tables.params().degree();
  const auto stride = (k + p) * n;
  auto *accum = arena.allocate<uint64_t>(2 * stride);
  if (!accum || !switchAccumulate<Cached>(accum, input, key, tables, arena, coefficients, quotients))
    return false;
  auto *corrections = arena.allocate<uint64_t>(k * n);
  auto *gather = arena.allocate<uint64_t>(p);
  auto *mixed = arena.allocate<uint64_t>(p);
  if (!corrections || !gather || !mixed)
    return false;
  for (uint32_t poly = 0; poly < 2; ++poly) {
    auto *component = accum + poly * stride;
    for (uint32_t limb = 0; limb < p; ++limb)
      ntt::inverse(component + (k + limb) * n, tables.auxNtt(limb));
    for (std::size_t i = 0; i < n; ++i) {
      for (uint32_t limb = 0; limb < p; ++limb)
        gather[limb] = component[(k + limb) * n + i];
      if (!tables.auxCrt().toMixedRadix(gather, mixed))
        return false;
      for (uint32_t limb = 0; limb < k; ++limb)
        corrections[limb * n + i] = tables.auxCrt().centeredMod(mixed, tables.params().mainPrimes()[limb]);
    }
    for (uint32_t limb = 0; limb < k; ++limb) {
      const auto q = tables.params().mainPrimes()[limb];
      auto *correction = corrections + limb * n;
      ntt::forward(correction, tables.mainNtt(limb));
      auto *dst = out.poly(poly).limb(limb);
      if constexpr (Cached) {
        const auto factor = tables.auxProductInverse(limb);
        const auto companion = modarith::shoupFactor(factor, q);
        for (std::size_t i = 0; i < n; ++i)
          dst[i] = canonicalProduct(minusMod(component[limb * n + i], correction[i], q), factor, companion, q);
      } else {
        for (std::size_t i = 0; i < n; ++i)
          dst[i] =
            modarith::mulMod(minusMod(component[limb * n + i], correction[i], q), tables.auxProductInverse(limb), q);
      }
    }
  }
  return true;
}

// D_q(D_P(A)+t) = D_(Pq)(A+P*t): for odd P,q the centered remainder
// r_P + P*r_q is already in the centered Pq interval. Thus one exact CRT
// removes both bases without changing a residue or a rounding decision.
template <bool Cached = false, typename TensorValue>
bool fusedProduct(CiphertextView out, PolyLayout input_layout, TensorValue value, EvaluationKeyView key,
                  const EvaluationTables &tables, Arena &arena, const uint64_t *quotients = nullptr) {
  Arena::Scope scope(arena);
  const auto k = input_layout.basis.main_end;
  const auto level = k - 1;
  const auto p = static_cast<uint32_t>(tables.params().auxPrimes().size());
  const auto n = tables.params().degree();
  const auto stride = (k + p) * n;
  auto *source = arena.allocate<uint64_t>(k * n);
  auto *accum = arena.allocate<uint64_t>(2 * stride);
  if (!source || !accum)
    return false;
  for (uint32_t limb = 0; limb < k; ++limb)
    for (std::size_t i = 0; i < n; ++i)
      source[limb * n + i] = value(2U, limb, i);
  if (!switchAccumulate<Cached>(accum, {source, k * n, input_layout}, key, tables, arena, nullptr, quotients))
    return false;
  auto *gather = arena.allocate<uint64_t>(p + 1U);
  auto *mixed = arena.allocate<uint64_t>(p + 1U);
  if (!gather || !mixed)
    return false;
  const auto &crt = tables.switchRescaleCrt(level);
  for (uint32_t poly = 0; poly < 2; ++poly) {
    auto *component = accum + poly * stride;
    for (uint32_t limb = 0; limb < k; ++limb) {
      const auto q = tables.params().mainPrimes()[limb];
      const auto factor = tables.auxProductMod(limb);
      if constexpr (Cached) {
        const auto companion = modarith::shoupFactor(factor, q);
        for (std::size_t i = 0; i < n; ++i)
          component[limb * n + i] =
            plusMod(component[limb * n + i], canonicalProduct(value(poly, limb, i), factor, companion, q), q);
      } else {
        for (std::size_t i = 0; i < n; ++i)
          component[limb * n + i] =
            plusMod(component[limb * n + i], modarith::mulMod(factor, value(poly, limb, i), q), q);
      }
    }
    for (uint32_t limb = 0; limb < p; ++limb)
      ntt::inverse(component + (k + limb) * n, tables.auxNtt(limb));
    ntt::inverse(component + level * n, tables.mainNtt(level));
    // t2 is dead: use its storage for the surviving-Q centered corrections.
    for (std::size_t i = 0; i < n; ++i) {
      for (uint32_t limb = 0; limb < p; ++limb)
        gather[limb] = component[(k + limb) * n + i];
      gather[p] = component[level * n + i];
      if (!crt.toMixedRadix(gather, mixed))
        return false;
      for (uint32_t limb = 0; limb < level; ++limb)
        source[limb * n + i] = crt.centeredMod(mixed, tables.params().mainPrimes()[limb]);
    }
    for (uint32_t limb = 0; limb < level; ++limb) {
      const auto q = tables.params().mainPrimes()[limb];
      const auto inverse = modarith::mulMod(tables.auxProductInverse(limb), tables.dropInverse(level, limb), q);
      auto *correction = source + limb * n;
      ntt::forward(correction, tables.mainNtt(limb));
      auto *dst = out.poly(poly).limb(limb);
      if constexpr (Cached) {
        const auto companion = modarith::shoupFactor(inverse, q);
        for (std::size_t i = 0; i < n; ++i)
          dst[i] = canonicalProduct(minusMod(component[limb * n + i], correction[i], q), inverse, companion, q);
      } else {
        for (std::size_t i = 0; i < n; ++i)
          dst[i] = modarith::mulMod(minusMod(component[limb * n + i], correction[i], q), inverse, q);
      }
    }
  }
  return true;
}
auto tensorValue(ConstCiphertextView a, ConstCiphertextView b, const EvaluationTables &tables) {
  return [a, b, &tables](uint32_t poly, uint32_t limb, std::size_t i) {
    const auto q = tables.params().mainPrimes()[limb];
    if (poly == 0)
      return modarith::mulMod(a.poly(0).limb(limb)[i], b.poly(0).limb(limb)[i], q);
    if (poly == 2)
      return modarith::mulMod(a.poly(1).limb(limb)[i], b.poly(1).limb(limb)[i], q);
    return plusMod(modarith::mulMod(a.poly(0).limb(limb)[i], b.poly(1).limb(limb)[i], q),
                   modarith::mulMod(a.poly(1).limb(limb)[i], b.poly(0).limb(limb)[i], q), q);
  };
}
auto tensorValueCached(ConstCiphertextView a, ConstCiphertextView b, const uint64_t *quotients,
                       const EvaluationTables &tables) {
  return [a, b, quotients, &tables](uint32_t poly, uint32_t limb, std::size_t i) {
    const auto q = tables.params().mainPrimes()[limb];
    const auto n = tables.params().degree();
    const auto stride = a.layout.poly.wordCount();
    const auto product = [&](uint32_t left, uint32_t right) {
      return canonicalProduct(b.poly(right).limb(limb)[i], a.poly(left).limb(limb)[i],
                              quotients[left * stride + limb * n + i], q);
    };
    if (poly == 0)
      return product(0, 0);
    if (poly == 2)
      return product(1, 1);
    return plusMod(product(0, 1), product(1, 0), q);
  };
}

} // namespace

std::size_t scratchBytes(EvalOperation operation, const Params &params, uint32_t level) noexcept {
  if (level >= params.levelCount() || uint64_t{params.levelCount()} + params.auxPrimes().size() > RnsBasis::kMaxLimbs)
    return 0;
  const std::size_t n = params.degree(), k = std::size_t{level} + 1, p = params.auxPrimes().size(), ext = k + p;
  constexpr auto margin = Arena::kAlignment - 1;
  const auto switch_bytes = padded(3 * ext * n) + padded(p * n) + padded(n) + 2 * padded(ext) + padded(p);
  switch (operation) {
  case EvalOperation::Add:
  case EvalOperation::Subtract:
  case EvalOperation::Negate:
  case EvalOperation::AddScalar:
  case EvalOperation::LevelDown:
    return padded(4 * k * n) + margin;
  case EvalOperation::Tensor:
    return padded(3 * k * n) + margin;
  case EvalOperation::Relinearize:
    return padded(2 * k * n) + switch_bytes + margin;
  case EvalOperation::Rotate:
    return padded(4 * k * n) + switch_bytes + margin;
  case EvalOperation::Rescale:
    return level == 0 ? 0 : padded(4 * (k - 1) * n) + 2 * padded(n) + margin;
  case EvalOperation::MultiplyScalar:
    return level == 0 ? 0 : padded(4 * k * n) + scratchBytes(EvalOperation::Rescale, params, level) + margin;
  case EvalOperation::HMultiply:
    return level == 0 ? 0
                      : padded(3 * k * n) + margin +
                          std::max({scratchBytes(EvalOperation::Tensor, params, level),
                                    scratchBytes(EvalOperation::Relinearize, params, level),
                                    scratchBytes(EvalOperation::Rescale, params, level)});
  }
  return 0;
}
uint64_t galoisElement(int64_t step, uint32_t log_degree) {
  if (log_degree < NttTables::kMinLogDegree || log_degree > NttTables::kMaxLogDegree)
    return 0;
  const int64_t slots = int64_t{1} << (log_degree - 1);
  const int64_t remainder = step % slots;
  return modarith::powMod(5, static_cast<uint64_t>(remainder < 0 ? remainder + slots : remainder),
                          uint64_t{1} << (log_degree + 1));
}
bool add(CiphertextView &out, ConstCiphertextView a, ConstCiphertextView b, const EvaluationTables &tables,
         Arena &arena) {
  return linear(out, a, b, false, tables, arena);
}
bool subtract(CiphertextView &out, ConstCiphertextView a, ConstCiphertextView b, const EvaluationTables &tables,
              Arena &arena) {
  return linear(out, a, b, true, tables, arena);
}
bool negate(CiphertextView &out, ConstCiphertextView a, const EvaluationTables &tables, Arena &arena) {
  if (!validInput(a, tables) || !outputReady(out, a.layout, a) ||
      !scratchReady(EvalOperation::Negate, a, tables, arena))
    return false;
  Arena::Scope scope(arena);
  auto tmp = temporary(a.layout, arena);
  if (tmp.data == nullptr)
    return false;
  for (uint32_t poly = 0; poly < a.layout.num_polys; ++poly) {
    for (uint32_t limb = 0; limb < a.layout.poly.basis.main_end; ++limb) {
      const auto q = tables.params().mainPrimes()[limb];
      const auto *src = a.poly(poly).limb(limb);
      auto *dst = tmp.poly(poly).limb(limb);
      for (std::size_t i = 0; i < tables.params().degree(); ++i)
        dst[i] = src[i] == 0 ? 0 : q - src[i];
    }
  }
  commit(out, asConst(tmp));
  return true;
}
bool tensor(CiphertextView &out, ConstCiphertextView a, ConstCiphertextView b, const EvaluationTables &tables,
            Arena &arena) {
  if (!validInput(a, tables) || !validInput(b, tables) || !sameBasis(a, b) || a.layout.num_polys != 2 ||
      b.layout.num_polys != 2)
    return false;
  auto layout = a.layout;
  layout.num_polys = 3;
  layout.scale *= b.layout.scale;
  if (!outputReady(out, layout, a, &b) || !scratchReady(EvalOperation::Tensor, a, tables, arena))
    return false;
  Arena::Scope scope(arena);
  auto tmp = temporary(layout, arena);
  if (tmp.data == nullptr)
    return false;
  for (uint32_t limb = 0; limb < layout.poly.basis.main_end; ++limb) {
    const auto q = tables.params().mainPrimes()[limb];
    const auto *a0 = a.poly(0).limb(limb), *a1 = a.poly(1).limb(limb), *b0 = b.poly(0).limb(limb),
               *b1 = b.poly(1).limb(limb);
    auto *c0 = tmp.poly(0).limb(limb), *c1 = tmp.poly(1).limb(limb), *c2 = tmp.poly(2).limb(limb);
    for (std::size_t i = 0; i < tables.params().degree(); ++i) {
      c0[i] = modarith::mulMod(a0[i], b0[i], q);
      c1[i] = plusMod(modarith::mulMod(a0[i], b1[i], q), modarith::mulMod(a1[i], b0[i], q), q);
      c2[i] = modarith::mulMod(a1[i], b1[i], q);
    }
  }
  commit(out, asConst(tmp));
  return true;
}

bool relinearize(CiphertextView &out, ConstCiphertextView a, EvaluationKeyView key, const EvaluationTables &tables,
                 Arena &arena) {
  if (!validInput(a, tables) || a.layout.num_polys != 3)
    return false;
  auto layout = a.layout;
  layout.num_polys = 2;
  if (!outputReady(out, layout, a) || !validKey(key, SwitchKeyKind::Relinearization, 1, tables, out, layout) ||
      !scratchReady(EvalOperation::Relinearize, a, tables, arena))
    return false;
  Arena::Scope scope(arena);
  auto tmp = temporary(layout, arena);
  if (tmp.data == nullptr || !switchCore(tmp, a.poly(2), key, tables, arena))
    return false;
  for (uint32_t poly = 0; poly < 2; ++poly) {
    for (uint32_t limb = 0; limb < layout.poly.basis.main_end; ++limb) {
      const auto q = tables.params().mainPrimes()[limb];
      auto *dst = tmp.poly(poly).limb(limb);
      const auto *src = a.poly(poly).limb(limb);
      for (std::size_t i = 0; i < tables.params().degree(); ++i)
        dst[i] = plusMod(dst[i], src[i], q);
    }
  }
  commit(out, asConst(tmp));
  return true;
}

bool rescale(CiphertextView &out, ConstCiphertextView a, const EvaluationTables &tables, Arena &arena) {
  if (!validInput(a, tables) || a.layout.poly.basis.main_end < 2)
    return false;
  const auto level = a.layout.poly.basis.main_end - 1;
  const auto divisor = tables.params().mainPrimes()[level];
  auto layout = a.layout;
  --layout.poly.basis.main_end;
  layout.scale /= static_cast<double>(divisor);
  if (!outputReady(out, layout, a) || !scratchReady(EvalOperation::Rescale, a, tables, arena))
    return false;
  Arena::Scope scope(arena);
  auto tmp = temporary(layout, arena);
  const auto n = tables.params().degree();
  auto *removed = arena.allocate<uint64_t>(n), *correction = arena.allocate<uint64_t>(n);
  if (tmp.data == nullptr || removed == nullptr || correction == nullptr)
    return false;
  for (uint32_t poly = 0; poly < layout.num_polys; ++poly) {
    std::copy_n(a.poly(poly).limb(level), n, removed);
    ntt::inverse(removed, tables.mainNtt(level));
    for (uint32_t limb = 0; limb < level; ++limb) {
      const auto q = tables.params().mainPrimes()[limb];
      for (std::size_t i = 0; i < n; ++i) {
        if (removed[i] <= divisor / 2)
          correction[i] = removed[i] % q;
        else {
          const auto magnitude = (divisor - removed[i]) % q;
          correction[i] = magnitude == 0 ? 0 : q - magnitude;
        }
      }
      ntt::forward(correction, tables.mainNtt(limb));
      auto *dst = tmp.poly(poly).limb(limb);
      const auto *src = a.poly(poly).limb(limb);
      for (std::size_t i = 0; i < n; ++i)
        dst[i] = modarith::mulMod(minusMod(src[i], correction[i], q), tables.dropInverse(level, limb), q);
    }
  }
  commit(out, asConst(tmp));
  return true;
}

bool rotate(CiphertextView &out, ConstCiphertextView a, uint64_t galois_element, EvaluationKeyView key,
            const EvaluationTables &tables, Arena &arena) {
  if (!validInput(a, tables) || a.layout.num_polys != 2)
    return false;
  const auto n = tables.params().degree();
  if ((galois_element & 1U) == 0 || galois_element >= 2 * n || !outputReady(out, a.layout, a) ||
      !scratchReady(EvalOperation::Rotate, a, tables, arena))
    return false;
  if (galois_element != 1 && !validKey(key, SwitchKeyKind::Galois, galois_element, tables, out, a.layout))
    return false;
  Arena::Scope scope(arena);
  auto transformed = temporary(a.layout, arena);
  if (transformed.data == nullptr)
    return false;
  if (galois_element == 1) {
    copyInto(transformed, a);
    commit(out, asConst(transformed));
    return true;
  }
  auto tmp = temporary(a.layout, arena);
  if (tmp.data == nullptr)
    return false;
  const auto bits = tables.params().logDegree();
  for (std::size_t i = 0; i < n; ++i) {
    const auto odd = (galois_element * (2 * bitReverse(i, bits) + 1)) % (2 * n);
    const auto source = bitReverse((odd - 1) / 2, bits);
    for (uint32_t poly = 0; poly < 2; ++poly)
      for (uint32_t limb = 0; limb < a.layout.poly.basis.main_end; ++limb)
        transformed.poly(poly).limb(limb)[i] = a.poly(poly).limb(limb)[source];
  }
  if (!switchCore(tmp, asConst(transformed.poly(1)), key, tables, arena))
    return false;
  for (uint32_t limb = 0; limb < a.layout.poly.basis.main_end; ++limb) {
    const auto q = tables.params().mainPrimes()[limb];
    auto *dst = tmp.poly(0).limb(limb);
    const auto *src = transformed.poly(0).limb(limb);
    for (std::size_t i = 0; i < n; ++i)
      dst[i] = plusMod(dst[i], src[i], q);
  }
  commit(out, asConst(tmp));
  return true;
}

bool hmult(CiphertextView &out, ConstCiphertextView a, ConstCiphertextView b, EvaluationKeyView key,
           const EvaluationTables &tables, Arena &arena) {
  if (!validInput(a, tables) || !validInput(b, tables) || !sameBasis(a, b) || a.layout.num_polys != 2 ||
      b.layout.num_polys != 2 || a.layout.poly.basis.main_end < 2)
    return false;
  auto product_layout = a.layout;
  product_layout.num_polys = 3;
  product_layout.scale *= b.layout.scale;
  if (!std::isfinite(product_layout.scale) || product_layout.scale <= 0)
    return false;
  auto layout = product_layout;
  layout.num_polys = 2;
  --layout.poly.basis.main_end;
  layout.scale /= static_cast<double>(tables.params().mainPrimes()[layout.poly.basis.main_end]);
  if (!outputReady(out, layout, a, &b) || !validKey(key, SwitchKeyKind::Relinearization, 1, tables, out, layout) ||
      !scratchReady(EvalOperation::HMultiply, a, tables, arena))
    return false;
  Arena::Scope scope(arena);
  auto tmp = temporary(layout, arena);
  if (!tmp.data || !fusedProduct(tmp, a.layout.poly, tensorValue(a, b, tables), key, tables, arena))
    return false;
  commit(out, asConst(tmp));
  return true;
}

bool addScalar(CiphertextView &out, ConstCiphertextView a, double scalar, const EvaluationTables &tables,
               Arena &arena) {
  int64_t encoded = 0;
  if (!validInput(a, tables) || !quantize(scalar, a.layout.scale, encoded) || !outputReady(out, a.layout, a) ||
      !scratchReady(EvalOperation::AddScalar, a, tables, arena))
    return false;
  Arena::Scope scope(arena);
  auto tmp = temporary(a.layout, arena);
  if (tmp.data == nullptr)
    return false;
  copyInto(tmp, a);
  for (uint32_t limb = 0; limb < a.layout.poly.basis.main_end; ++limb) {
    const auto q = tables.params().mainPrimes()[limb], value = signedMod(encoded, q);
    auto *dst = tmp.poly(0).limb(limb);
    for (std::size_t i = 0; i < tables.params().degree(); ++i)
      dst[i] = plusMod(dst[i], value, q);
  }
  commit(out, asConst(tmp));
  return true;
}

bool multiplyScalar(CiphertextView &out, ConstCiphertextView a, double scalar, const EvaluationTables &tables,
                    Arena &arena) {
  if (!validInput(a, tables) || a.layout.poly.basis.main_end < 2)
    return false;
  const auto level = a.layout.poly.basis.main_end - 1;
  const double divisor = static_cast<double>(tables.params().mainPrimes()[level]);
  int64_t encoded = 0;
  auto layout = a.layout;
  --layout.poly.basis.main_end;
  auto product_layout = a.layout;
  product_layout.scale *= divisor;
  if (!quantize(scalar, divisor, encoded) || !std::isfinite(product_layout.scale) || !outputReady(out, layout, a) ||
      !scratchReady(EvalOperation::MultiplyScalar, a, tables, arena))
    return false;
  Arena::Scope scope(arena);
  auto product = temporary(product_layout, arena);
  if (product.data == nullptr)
    return false;
  for (uint32_t poly = 0; poly < a.layout.num_polys; ++poly) {
    for (uint32_t limb = 0; limb <= level; ++limb) {
      const auto q = tables.params().mainPrimes()[limb], value = signedMod(encoded, q);
      const auto *src = a.poly(poly).limb(limb);
      auto *dst = product.poly(poly).limb(limb);
      for (std::size_t i = 0; i < tables.params().degree(); ++i)
        dst[i] = modarith::mulMod(src[i], value, q);
    }
  }
  if (!rescale(out, asConst(product), tables, arena))
    return false;
  // Compensate only floating multiplication/division roundoff in metadata.
  out.layout.scale = a.layout.scale;
  return true;
}

bool levelDown(CiphertextView &out, ConstCiphertextView a, uint32_t target_level, const EvaluationTables &tables,
               Arena &arena) {
  if (!validInput(a, tables) || target_level >= a.layout.poly.basis.main_end)
    return false;
  auto layout = a.layout;
  layout.poly.basis.main_end = target_level + 1;
  if (!outputReady(out, layout, a) || !scratchReady(EvalOperation::LevelDown, a, tables, arena))
    return false;
  Arena::Scope scope(arena);
  auto tmp = temporary(layout, arena);
  if (tmp.data == nullptr)
    return false;
  copyInto(tmp, a);
  commit(out, asConst(tmp));
  return true;
}
} // namespace fhenomenon::ckks::eval

namespace fhenomenon::ckks::region {
namespace {
// Most affine edges have coefficients 0, +1 or -1. Avoid a wide modular
// multiplication for these exact additions/subtractions, including rotations.
void accumulateScaled(uint64_t *dst, const uint64_t *src, std::size_t n, uint64_t factor, uint64_t q) {
  if (factor == 0)
    return;
  if (factor == 1) {
    for (std::size_t i = 0; i < n; ++i)
      dst[i] = eval::plusMod(dst[i], src[i], q);
  } else if (factor == q - 1) {
    for (std::size_t i = 0; i < n; ++i)
      dst[i] = eval::minusMod(dst[i], src[i], q);
  } else {
    for (std::size_t i = 0; i < n; ++i)
      dst[i] = eval::plusMod(dst[i], modarith::mulMod(factor, src[i], q), q);
  }
}
bool metadata(ConstCiphertextView value, const EvaluationTables &tables) {
  return eval::inputMetadata(value, tables) && value.layout.num_polys == 2;
}
bool destination(CiphertextView out, CiphertextLayout target, const EvaluationTables &tables) {
  out.layout = target;
  return metadata(asConst(out), tables);
}
bool disjoint(CiphertextView out, CiphertextLayout target, ConstCiphertextView source) {
  out.layout = target;
  return !eval::overlap(asConst(out), source);
}
bool compatible(ConstCiphertextView value, CiphertextLayout target, const EvaluationTables &tables) {
  return metadata(value, tables) && value.layout.poly == target.poly &&
         eval::sameScale(value.layout.scale, target.scale);
}
bool workspace(const EvaluationTables &tables, Arena &arena) {
  const auto bound = region::scratchBytes(tables.params());
  return bound != 0 && arena.remaining() >= bound;
}
bool boundKey(EvaluationKeyView key, SwitchKeyKind kind, uint64_t g, CiphertextView out, CiphertextLayout target,
              const EvaluationTables &tables) {
  out.layout = target;
  return eval::keyMetadata(key, kind, g, tables) && eval::keyDisjoint(key, asConst(out));
}
bool productLayout(ConstCiphertextView a, ConstCiphertextView b, CiphertextLayout &target,
                   const EvaluationTables &tables) {
  if (!metadata(a, tables) || !metadata(b, tables) || a.layout.poly != b.layout.poly ||
      a.layout.poly.basis.main_end < 2)
    return false;
  target = a.layout;
  target.scale *= b.layout.scale;
  if (!std::isfinite(target.scale) || target.scale <= 0)
    return false;
  --target.poly.basis.main_end;
  target.scale /= static_cast<double>(tables.params().mainPrimes()[target.poly.basis.main_end]);
  return std::isfinite(target.scale) && target.scale > 0;
}
// Raw checked rescale remains independent; this private copy has no residue
// scan and writes only compiler-private, disjoint storage after preflight.
bool rescaleInto(CiphertextView out, ConstCiphertextView a, const EvaluationTables &tables, Arena &arena) {
  Arena::Scope scope(arena);
  const auto level = a.layout.poly.basis.main_end - 1;
  const auto divisor = tables.params().mainPrimes()[level];
  const auto n = tables.params().degree();
  auto *removed = arena.allocate<uint64_t>(n);
  auto *correction = arena.allocate<uint64_t>(n);
  if (!removed || !correction)
    return false;
  for (uint32_t poly = 0; poly < 2; ++poly) {
    std::copy_n(a.poly(poly).limb(level), n, removed);
    ntt::inverse(removed, tables.mainNtt(level));
    for (uint32_t limb = 0; limb < level; ++limb) {
      const auto q = tables.params().mainPrimes()[limb];
      for (std::size_t i = 0; i < n; ++i) {
        const auto magnitude = removed[i] <= divisor / 2 ? removed[i] % q : (divisor - removed[i]) % q;
        correction[i] = removed[i] <= divisor / 2 || magnitude == 0 ? magnitude : q - magnitude;
      }
      ntt::forward(correction, tables.mainNtt(limb));
      const auto *src = a.poly(poly).limb(limb);
      auto *dst = out.poly(poly).limb(limb);
      for (std::size_t i = 0; i < n; ++i)
        dst[i] = modarith::mulMod(eval::minusMod(src[i], correction[i], q), tables.dropInverse(level, limb), q);
    }
  }
  return true;
}
} // namespace

bool validateInput(ConstCiphertextView value, const EvaluationTables &tables) {
  return metadata(value, tables) && eval::canonical(value, tables, false);
}
bool validateKey(EvaluationKeyView key, SwitchKeyKind kind, uint64_t g, const EvaluationTables &tables) {
  if (!eval::keyMetadata(key, kind, g, tables))
    return false;
  for (uint32_t d = 0; d < key.digit_count; ++d)
    if (!eval::canonical(key.digits[d], tables, true))
      return false;
  return true;
}
std::size_t scratchBytes(const Params &params) noexcept {
  if (uint64_t{params.levelCount()} + params.auxPrimes().size() > RnsBasis::kMaxLimbs)
    return 0;
  std::size_t maximum = 0;
  for (uint32_t level = 0; level < params.levelCount(); ++level)
    for (auto op : {eval::EvalOperation::HMultiply, eval::EvalOperation::Rotate, eval::EvalOperation::MultiplyScalar,
                    eval::EvalOperation::Add})
      maximum = std::max(maximum, eval::scratchBytes(op, params, level));
  // One weighted tensor is live across fused switching. Keep the conservative
  // public bound too, so a caller can interleave reference kernels in this Arena.
  return maximum + eval::padded(3 * params.levelCount() * params.degree()) + 8 * Arena::kAlignment;
}
bool encodeConstant(double value, double scale, int64_t &encoded) noexcept {
  return std::isfinite(value) && std::isfinite(scale) && scale > 0 && eval::quantize(value, scale, encoded);
}
bool linear(CiphertextView &out, const LinearTerm *terms, std::size_t count, double constant, CiphertextLayout target,
            const EvaluationTables &tables, Arena &arena) {
  int64_t encoded = 0;
  if ((count && !terms) || !destination(out, target, tables) || !encodeConstant(constant, target.scale, encoded) ||
      !workspace(tables, arena))
    return false;
  for (std::size_t t = 0; t < count; ++t)
    if (!compatible(terms[t].value, target, tables) || !disjoint(out, target, terms[t].value))
      return false;
  auto result = out;
  result.layout = target;
  const auto n = tables.params().degree();
  for (uint32_t poly = 0; poly < 2; ++poly)
    for (uint32_t limb = 0; limb < target.poly.basis.main_end; ++limb) {
      const auto q = tables.params().mainPrimes()[limb];
      auto *dst = result.poly(poly).limb(limb);
      std::fill_n(dst, n, poly == 0 ? eval::signedMod(encoded, q) : uint64_t{0});
      for (std::size_t t = 0; t < count; ++t) {
        const auto factor = eval::signedMod(terms[t].coefficient, q);
        const auto *src = terms[t].value.poly(poly).limb(limb);
        accumulateScaled(dst, src, n, factor, q);
      }
    }
  out.layout = target;
  return true;
}
bool multiply(CiphertextView &out, ConstCiphertextView a, ConstCiphertextView b, EvaluationKeyView key,
              const EvaluationTables &tables, Arena &arena) {
  CiphertextLayout target;
  if (!productLayout(a, b, target, tables) || !destination(out, target, tables) || !disjoint(out, target, a) ||
      !disjoint(out, target, b) || !boundKey(key, SwitchKeyKind::Relinearization, 1, out, target, tables) ||
      !workspace(tables, arena))
    return false;
  auto result = out;
  result.layout = target;
  if (!eval::fusedProduct(result, a.layout.poly, eval::tensorValue(a, b, tables), key, tables, arena))
    return false;
  out.layout = target;
  return true;
}
namespace {
// These views cross only the private trusted-preparation boundary. Check their
// shape and destination aliasing here; correspondence was certified at binding.
template <typename T>
bool cachedSpan(const T *data, std::size_t count, std::size_t expected, CiphertextView out, CiphertextLayout target) {
  if (!data)
    return count == 0;
  if (count != expected || count > std::numeric_limits<std::size_t>::max() / sizeof(T))
    return false;
  const auto start = reinterpret_cast<std::uintptr_t>(data);
  const auto bytes = count * sizeof(T);
  if (start % alignof(T) || start > std::numeric_limits<std::uintptr_t>::max() - bytes)
    return false;
  out.layout = target;
  const auto out_start = reinterpret_cast<std::uintptr_t>(out.data);
  const auto out_bytes = eval::spanWords(asConst(out)) * sizeof(uint64_t);
  if (out_start > std::numeric_limits<std::uintptr_t>::max() - out_bytes)
    return false;
  return start >= out_start + out_bytes || out_start >= start + bytes;
}
bool cachedKey(KeyProducts products, EvaluationKeyView key, CiphertextView out, CiphertextLayout target) {
  const std::size_t words =
    std::size_t{key.digit_count} * 2 * (key.main_count + key.aux_count) * (std::size_t{1} << key.log_degree);
  return cachedSpan(products.quotients, products.count, words, out, target);
}
template <bool CachedKey, typename Tensor>
bool cachedMultiplyInto(CiphertextView &out, CiphertextLayout target, PolyLayout input, Tensor value,
                        EvaluationKeyView key, KeyProducts products, const EvaluationTables &tables, Arena &arena) {
  auto result = out;
  result.layout = target;
  if (!eval::fusedProduct<CachedKey>(result, input, value, key, tables, arena, products.quotients))
    return false;
  out.layout = target;
  return true;
}
} // namespace
bool multiplyCached(CiphertextView &out, ConstCiphertextView a, ConstCiphertextView b, EvaluationKeyView key,
                    KeyProducts products, CipherProducts source, const EvaluationTables &tables, Arena &arena) {
  CiphertextLayout target;
  if (!productLayout(a, b, target, tables) || !destination(out, target, tables) || !disjoint(out, target, a) ||
      !disjoint(out, target, b) || !boundKey(key, SwitchKeyKind::Relinearization, 1, out, target, tables) ||
      !workspace(tables, arena) || !cachedKey(products, key, out, target) ||
      !cachedSpan(source.quotients, source.count, 2 * a.layout.poly.wordCount(), out, target))
    return false;
  if (source.quotients) {
    const auto value = eval::tensorValueCached(a, b, source.quotients, tables);
    return products.quotients
             ? cachedMultiplyInto<true>(out, target, a.layout.poly, value, key, products, tables, arena)
             : cachedMultiplyInto<false>(out, target, a.layout.poly, value, key, products, tables, arena);
  }
  const auto value = eval::tensorValue(a, b, tables);
  return products.quotients
           ? cachedMultiplyInto<true>(out, target, a.layout.poly, value, key, products, tables, arena)
           : cachedMultiplyInto<false>(out, target, a.layout.poly, value, key, products, tables, arena);
}
bool productSum(CiphertextView &out, const ProductTerm *products, std::size_t product_count,
                const LinearTerm *linear_terms, std::size_t linear_count, double constant, EvaluationKeyView key,
                const EvaluationTables &tables, Arena &arena) {
  if (!products || product_count == 0 || (linear_count && !linear_terms) || !std::isfinite(constant))
    return false;
  const auto input = products[0].left;
  const auto common = input.layout;
  CiphertextLayout target;
  if (!productLayout(input, input, target, tables) || !destination(out, target, tables) ||
      !boundKey(key, SwitchKeyKind::Relinearization, 1, out, target, tables) || !workspace(tables, arena))
    return false;
  int64_t scale = 0, encoded = 0;
  if (linear_count || constant != 0) {
    if (std::trunc(common.scale) != common.scale || !encodeConstant(common.scale, 1, scale) ||
        !encodeConstant(constant, common.scale, encoded))
      return false;
  }
  for (std::size_t t = 0; t < product_count; ++t)
    if (!compatible(products[t].left, common, tables) || !compatible(products[t].right, common, tables) ||
        !disjoint(out, target, products[t].left) || !disjoint(out, target, products[t].right))
      return false;
  for (std::size_t t = 0; t < linear_count; ++t)
    if (!compatible(linear_terms[t].value, common, tables) || !disjoint(out, target, linear_terms[t].value))
      return false;
  Arena::Scope scope(arena);
  auto tensor_layout = common;
  tensor_layout.num_polys = 3;
  tensor_layout.scale *= common.scale;
  auto tensor = eval::temporary(tensor_layout, arena);
  if (!tensor.data)
    return false;
  const auto n = tables.params().degree();
  for (uint32_t limb = 0; limb < common.poly.basis.main_end; ++limb) {
    const auto q = tables.params().mainPrimes()[limb];
    for (uint32_t poly = 0; poly < 3; ++poly)
      std::fill_n(tensor.poly(poly).limb(limb), n, uint64_t{0});
    for (std::size_t t = 0; t < product_count; ++t) {
      const auto factor = eval::signedMod(products[t].coefficient, q);
      const auto value = eval::tensorValue(products[t].left, products[t].right, tables);
      for (uint32_t poly = 0; poly < 3; ++poly) {
        auto *dst = tensor.poly(poly).limb(limb);
        for (std::size_t i = 0; i < n; ++i)
          dst[i] = eval::plusMod(dst[i], modarith::mulMod(factor, value(poly, limb, i), q), q);
      }
    }
    const auto lift = eval::signedMod(scale, q);
    for (std::size_t t = 0; t < linear_count; ++t) {
      const auto factor = modarith::mulMod(eval::signedMod(linear_terms[t].coefficient, q), lift, q);
      for (uint32_t poly = 0; poly < 2; ++poly) {
        auto *dst = tensor.poly(poly).limb(limb);
        const auto *src = linear_terms[t].value.poly(poly).limb(limb);
        accumulateScaled(dst, src, n, factor, q);
      }
    }
    // Quantize at the input scale first, then lift modulo each prime; do not
    // require the product encoded*scale to fit int64 or round at scale squared.
    const auto lifted_constant = modarith::mulMod(eval::signedMod(encoded, q), lift, q);
    auto *c0 = tensor.poly(0).limb(limb);
    for (std::size_t i = 0; i < n; ++i)
      c0[i] = eval::plusMod(c0[i], lifted_constant, q);
  }
  auto result = out;
  result.layout = target;
  const auto value = [tensor](uint32_t poly, uint32_t limb, std::size_t i) { return tensor.poly(poly).limb(limb)[i]; };
  if (!eval::fusedProduct(result, common.poly, value, key, tables, arena))
    return false;
  out.layout = target;
  return true;
}
template <bool Cached = false>
static bool rotateAddImpl(CiphertextView &out, ConstCiphertextView a, uint64_t g, EvaluationKeyView key,
                          const LinearTerm *terms, std::size_t count, const EvaluationTables &tables, Arena &arena,
                          const uint64_t *quotients = nullptr, const uint32_t *indices = nullptr) {
  const auto n = tables.params().degree();
  if (!metadata(a, tables) || (count && !terms) || (g & 1U) == 0 || g >= 2 * n || !destination(out, a.layout, tables) ||
      !disjoint(out, a.layout, a) || !workspace(tables, arena) ||
      (g != 1 && !boundKey(key, SwitchKeyKind::Galois, g, out, a.layout, tables)))
    return false;
  for (std::size_t t = 0; t < count; ++t)
    if (!compatible(terms[t].value, a.layout, tables) || !disjoint(out, a.layout, terms[t].value))
      return false;
  Arena::Scope scope(arena);
  auto result = out;
  result.layout = a.layout;
  const auto k = a.layout.poly.basis.main_end;
  if (g == 1) {
    eval::copyInto(result, a);
  } else {
    // Each source index is independent of component and modulus. Compute it
    // once, retaining contiguous per-limb loops without repeating bit reversal.
    // N <= 2^17, so uint32_t covers every index. This N/2-word map plus the
    // kN-word transformed polynomial fits within the checked rotation bound's
    // 4kN-word staging allowance already included in region::scratchBytes.
    const uint32_t *permutation = indices;
    if (!permutation) {
      auto *generated = arena.allocate<uint32_t>(n);
      if (!generated)
        return false;
      const auto bits = tables.params().logDegree();
      for (std::size_t i = 0; i < n; ++i) {
        const auto odd = (g * (2 * eval::bitReverse(i, bits) + 1)) % (2 * n);
        generated[i] = static_cast<uint32_t>(eval::bitReverse((odd - 1) / 2, bits));
      }
      permutation = generated;
    }
    auto *data = arena.allocate<uint64_t>(k * n);
    if (!data)
      return false;
    RnsPolyView transformed{data, k * n, a.layout.poly};
    for (uint32_t limb = 0; limb < k; ++limb) {
      const auto *src = a.poly(1).limb(limb);
      auto *dst = transformed.limb(limb);
      for (std::size_t i = 0; i < n; ++i)
        dst[i] = src[permutation[i]];
    }
    if (!eval::switchCore<Cached>(result, asConst(transformed), key, tables, arena, nullptr, quotients))
      return false;
    for (uint32_t limb = 0; limb < k; ++limb) {
      const auto q = tables.params().mainPrimes()[limb];
      const auto *src = a.poly(0).limb(limb);
      auto *dst = result.poly(0).limb(limb);
      for (std::size_t i = 0; i < n; ++i)
        dst[i] = eval::plusMod(dst[i], src[permutation[i]], q);
    }
  }
  for (uint32_t poly = 0; poly < 2; ++poly)
    for (uint32_t limb = 0; limb < k; ++limb) {
      const auto q = tables.params().mainPrimes()[limb];
      auto *dst = result.poly(poly).limb(limb);
      for (std::size_t t = 0; t < count; ++t) {
        const auto factor = eval::signedMod(terms[t].coefficient, q);
        const auto *src = terms[t].value.poly(poly).limb(limb);
        accumulateScaled(dst, src, n, factor, q);
      }
    }
  out.layout = a.layout;
  return true;
}
bool rotateAdd(CiphertextView &out, ConstCiphertextView a, uint64_t g, EvaluationKeyView key, const LinearTerm *terms,
               std::size_t count, const EvaluationTables &tables, Arena &arena) {
  return rotateAddImpl(out, a, g, key, terms, count, tables, arena);
}
bool rotateAddCached(CiphertextView &out, ConstCiphertextView a, uint64_t g, EvaluationKeyView key,
                     KeyProducts products, RotationMap map, const LinearTerm *terms, std::size_t count,
                     const EvaluationTables &tables, Arena &arena) {
  if (!metadata(a, tables) || !destination(out, a.layout, tables) ||
      !cachedSpan(map.indices, map.count, tables.params().degree(), out, a.layout))
    return false;
  // Identity may omit its key entirely, but a supplied companion still has to
  // describe an actual bound key before computing sizes from its metadata.
  if (products.quotients || products.count) {
    if (!boundKey(key, SwitchKeyKind::Galois, g, out, a.layout, tables) || !cachedKey(products, key, out, a.layout))
      return false;
  }
  if (map.indices)
    for (std::size_t i = 0; i < map.count; ++i)
      if (map.indices[i] >= tables.params().degree())
        return false;
  return products.quotients
           ? rotateAddImpl<true>(out, a, g, key, terms, count, tables, arena, products.quotients, map.indices)
           : rotateAddImpl<false>(out, a, g, key, terms, count, tables, arena, nullptr, map.indices);
}
// Reuse only inverse transforms, never converted digits: approximate basis
// conversion is not assumed to commute with the signed automorphism. Apply
// X^i -> X^(g*i) in Q coefficients first, then use the unchanged switch core.
bool rotatePrepared(CiphertextView &out, ConstCiphertextView a, ConstRnsPolyView coefficients, uint64_t g,
                    EvaluationKeyView key, const EvaluationTables &tables, Arena &arena) {
  const auto n = tables.params().degree();
  auto coefficient_layout = a.layout.poly;
  coefficient_layout.form = NttForm::Coefficient;
  const ConstCiphertextView coefficient_view{
    coefficients.data, coefficients.capacity_words, coefficients.capacity_words, {coefficients.layout, 1.0, 1}};
  if (!metadata(a, tables) || !coefficients.wellFormed() || coefficients.layout != coefficient_layout ||
      (g & 1U) == 0 || g >= 2 * n || !destination(out, a.layout, tables) || !disjoint(out, a.layout, a) ||
      !disjoint(out, a.layout, coefficient_view) || !workspace(tables, arena) ||
      (g != 1 && !boundKey(key, SwitchKeyKind::Galois, g, out, a.layout, tables)))
    return false;
  Arena::Scope scope(arena);
  auto result = out;
  result.layout = a.layout;
  if (g == 1) {
    eval::copyInto(result, a);
    out.layout = a.layout;
    return true;
  }
  const auto k = a.layout.poly.basis.main_end;
  auto *permutation = arena.allocate<uint32_t>(n);
  auto *data = arena.allocate<uint64_t>(k * n);
  auto *rotated_coefficients = arena.allocate<uint64_t>(k * n);
  if (!permutation || !data || !rotated_coefficients)
    return false;
  const auto bits = tables.params().logDegree();
  for (std::size_t i = 0; i < n; ++i) {
    const auto odd = (g * (2 * eval::bitReverse(i, bits) + 1)) % (2 * n);
    permutation[i] = static_cast<uint32_t>(eval::bitReverse((odd - 1) / 2, bits));
  }
  RnsPolyView transformed{data, k * n, a.layout.poly};
  for (uint32_t limb = 0; limb < k; ++limb) {
    const auto q = tables.params().mainPrimes()[limb];
    const auto *src = a.poly(1).limb(limb);
    const auto *coeff = coefficients.limb(limb);
    auto *dst = transformed.limb(limb);
    auto *rotated = rotated_coefficients + limb * n;
    for (std::size_t i = 0; i < n; ++i) {
      dst[i] = src[permutation[i]];
      const auto exponent = (g * i) % (2 * n);
      rotated[exponent % n] = exponent >= n && coeff[i] != 0 ? q - coeff[i] : coeff[i];
    }
  }
  if (!eval::switchCore(result, asConst(transformed), key, tables, arena, rotated_coefficients))
    return false;
  for (uint32_t limb = 0; limb < k; ++limb) {
    const auto q = tables.params().mainPrimes()[limb];
    const auto *src = a.poly(0).limb(limb);
    auto *dst = result.poly(0).limb(limb);
    for (std::size_t i = 0; i < n; ++i)
      dst[i] = eval::plusMod(dst[i], src[permutation[i]], q);
  }
  out.layout = a.layout;
  return true;
}
bool scaleFractional(CiphertextView &out, ConstCiphertextView a, double scalar, const EvaluationTables &tables,
                     Arena &arena) {
  if (!metadata(a, tables) || a.layout.poly.basis.main_end < 2 || !workspace(tables, arena))
    return false;
  const auto level = a.layout.poly.basis.main_end - 1;
  const auto divisor = static_cast<double>(tables.params().mainPrimes()[level]);
  int64_t encoded = 0;
  auto target = a.layout;
  --target.poly.basis.main_end;
  auto product_layout = a.layout;
  product_layout.scale *= divisor;
  if (!encodeConstant(scalar, divisor, encoded) || !std::isfinite(product_layout.scale) ||
      !destination(out, target, tables) || !disjoint(out, target, a))
    return false;
  Arena::Scope scope(arena);
  auto product = eval::temporary(product_layout, arena);
  if (!product.data)
    return false;
  const auto n = tables.params().degree();
  for (uint32_t poly = 0; poly < 2; ++poly)
    for (uint32_t limb = 0; limb <= level; ++limb) {
      const auto q = tables.params().mainPrimes()[limb];
      const auto factor = eval::signedMod(encoded, q);
      const auto *src = a.poly(poly).limb(limb);
      auto *dst = product.poly(poly).limb(limb);
      for (std::size_t i = 0; i < n; ++i)
        dst[i] = modarith::mulMod(factor, src[i], q);
    }
  auto result = out;
  result.layout = target;
  if (!rescaleInto(result, asConst(product), tables, arena))
    return false;
  out.layout = target;
  return true;
}
bool levelDown(CiphertextView &out, ConstCiphertextView a, uint32_t level, const EvaluationTables &tables,
               Arena &arena) {
  if (!metadata(a, tables) || level >= a.layout.poly.basis.main_end || !workspace(tables, arena))
    return false;
  auto target = a.layout;
  target.poly.basis.main_end = level + 1;
  if (!destination(out, target, tables) || !disjoint(out, target, a))
    return false;
  auto result = out;
  result.layout = target;
  eval::copyInto(result, a);
  out.layout = target;
  return true;
}
} // namespace fhenomenon::ckks::region
