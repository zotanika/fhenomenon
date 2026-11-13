#include "CKKS/detail/Region.h"
#include "CKKS/ModArith.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <vector>
using namespace fhenomenon::ckks;
namespace {
using eval::EvalOperation;
Params parameters() {
  Params::Spec spec;
  spec.log_degree = 10;
  spec.main_primes = {12289, 40961, 61441, 65537};
  spec.aux_primes = {786433, 1179649};
  spec.dnum = 2;
  spec.log_default_scale = 13;
  return Params::create(spec).value();
}
uint64_t residue(int64_t value, uint64_t prime) {
  const auto p = static_cast<int64_t>(prime);
  const auto r = value % p;
  return static_cast<uint64_t>(r < 0 ? r + p : r);
}
struct Cipher {
  PolySlab slab;
  CiphertextView view;
  Cipher(const EvaluationTables &t, uint32_t level, uint32_t count, double scale = 8192)
    : slab(4U * t.params().degree() * t.params().levelCount()),
      view{slab.data(),
           slab.capacityWords(),
           t.params().degree() * t.params().levelCount(),
           {{{t.paramsId(), 0, level + 1, 0, 0}, t.params().logDegree(), NttForm::Evaluation}, scale, count}} {
    std::fill_n(slab.data(), slab.capacityWords(), uint64_t{0});
  }
  void set(uint32_t poly, const std::vector<int64_t> &v, const EvaluationTables &t) {
    for (uint32_t limb = 0; limb < view.layout.poly.basis.main_end; ++limb) {
      auto *dst = view.poly(poly).limb(limb);
      const auto q = t.params().mainPrimes()[limb];
      for (std::size_t i = 0; i < t.params().degree(); ++i)
        dst[i] = i < v.size() ? residue(v[i], q) : 0;
      ntt::forward(dst, t.mainNtt(limb));
    }
  }
};
std::vector<uint64_t> coefficients(ConstCiphertextView v, uint32_t poly, uint32_t limb, const EvaluationTables &t) {
  const auto *src = v.poly(poly).limb(limb);
  std::vector<uint64_t> out(src, src + t.params().degree());
  ntt::inverse(out.data(), t.mainNtt(limb));
  return out;
}
void expectCoefficients(ConstCiphertextView v, uint32_t poly, const std::vector<int64_t> &want,
                        const EvaluationTables &t) {
  for (uint32_t limb = 0; limb < v.layout.poly.basis.main_end; ++limb) {
    auto got = coefficients(v, poly, limb, t);
    const auto q = t.params().mainPrimes()[limb];
    for (std::size_t i = 0; i < got.size(); ++i)
      EXPECT_EQ(got[i], i < want.size() ? residue(want[i], q) : 0) << "limb " << limb << " coefficient " << i;
  }
}
// Mathematical hints for old=new=1 with zero noise. Arbitrary RNS a forces
// modular division, while b+a=P*projector gives an independent sum oracle.
struct SyntheticKey {
  std::vector<PolySlab> slabs;
  std::vector<ConstCiphertextView> digits;
  EvaluationKeyView view;
  SyntheticKey(const EvaluationTables &t, SwitchKeyKind kind = SwitchKeyKind::Relinearization, uint64_t g = 1) {
    const auto &p = t.params();
    const auto k = p.levelCount();
    const auto aux = static_cast<uint32_t>(p.auxPrimes().size());
    const auto n = p.degree();
    const auto stride = (k + aux) * n;
    slabs.reserve(p.dnum());
    digits.reserve(p.dnum());
    for (uint32_t d = 0; d < p.dnum(); ++d) {
      slabs.emplace_back(2 * stride);
      const auto range = p.digitRange(d);
      for (uint32_t limb = 0; limb < k + aux; ++limb) {
        const auto q = limb < k ? p.mainPrimes()[limb] : p.auxPrimes()[limb - k];
        uint64_t m = 0;
        if (limb >= range.first && limb < range.second) {
          m = 1;
          for (auto factor : p.auxPrimes())
            m = m * (factor % q) % q;
        }
        for (std::size_t i = 0; i < n; ++i) {
          const auto a = ((d + 3U) * (i + 7U) * 11U) % q;
          slabs.back().data()[stride + limb * n + i] = a;
          slabs.back().data()[limb * n + i] = (m + q - a) % q;
        }
      }
      digits.push_back({slabs.back().data(),
                        2 * stride,
                        stride,
                        {{{t.paramsId(), 0, k, 0, aux}, p.logDegree(), NttForm::Evaluation}, 1.0, 2}});
    }
    view = {digits.data(), p.dnum(), t.paramsId(), p.logDegree(), k, aux, kind, g};
  }
};

class Region : public ::testing::Test {
  protected:
  Params params = parameters();
  std::unique_ptr<EvaluationTables> tables = EvaluationTables::create(params);
  void equal(ConstCiphertextView a, ConstCiphertextView b) {
    ASSERT_EQ(a.layout, b.layout);
    for (uint32_t p = 0; p < a.layout.num_polys; ++p)
      for (uint32_t l = 0; l < a.layout.poly.basis.main_end; ++l)
        EXPECT_TRUE(std::equal(a.poly(p).limb(l), a.poly(p).limb(l) + params.degree(), b.poly(p).limb(l)));
  }
};
TEST_F(Region, BoundaryValidationRejectsBadResiduesAndKeyKinds) {
  Cipher a(*tables, 3, 2);
  SyntheticKey key(*tables);
  EXPECT_TRUE(region::validateInput(asConst(a.view), *tables));
  EXPECT_TRUE(region::validateKey(key.view, SwitchKeyKind::Relinearization, 1, *tables));
  EXPECT_FALSE(region::validateKey(key.view, SwitchKeyKind::Galois, 5, *tables));
  a.view.poly(1).limb(2)[19] = params.mainPrimes()[2];
  EXPECT_FALSE(region::validateInput(asConst(a.view), *tables));
  key.slabs[0].data()[17] = params.mainPrimes()[0];
  EXPECT_FALSE(region::validateKey(key.view, SwitchKeyKind::Relinearization, 1, *tables));
}
TEST_F(Region, ConstantEncodingRoundsSignedValuesAndGuardsInt64Endpoints) {
  int64_t e = 91;
  ASSERT_TRUE(region::encodeConstant(.5, 3, e));
  EXPECT_EQ(e, 2);
  ASSERT_TRUE(region::encodeConstant(-.5, 3, e));
  EXPECT_EQ(e, -2);
  ASSERT_TRUE(region::encodeConstant(-std::ldexp(1., 63), 1, e));
  EXPECT_EQ(e, INT64_MIN);
  EXPECT_FALSE(region::encodeConstant(std::ldexp(1., 63), 1, e));
  EXPECT_EQ(e, INT64_MIN);
  EXPECT_FALSE(region::encodeConstant(1, 0, e));
  EXPECT_FALSE(region::encodeConstant(1, -1, e));
  EXPECT_FALSE(region::encodeConstant(std::numeric_limits<double>::infinity(), 1, e));
}
TEST_F(Region, AffineArithmeticAndConstantOnlyPreserveLevelAndScale) {
  Cipher a(*tables, 3, 2, 8), b(*tables, 3, 2, 8), out(*tables, 3, 2, 8);
  a.set(0, {2, -3}, *tables);
  a.set(1, {7, 1}, *tables);
  b.set(0, {-4, 5}, *tables);
  b.set(1, {3, -2}, *tables);
  const region::LinearTerm terms[] = {{asConst(a.view), -3}, {asConst(b.view), 2}};
  Arena arena(region::scratchBytes(params));
  ASSERT_TRUE(region::linear(out.view, terms, 2, .25, a.view.layout, *tables, arena));
  expectCoefficients(asConst(out.view), 0, {-12, 19}, *tables);
  expectCoefficients(asConst(out.view), 1, {-15, -7}, *tables);
  EXPECT_EQ(out.view.layout, a.view.layout);
  ASSERT_TRUE(region::linear(out.view, nullptr, 0, -.5, a.view.layout, *tables, arena));
  expectCoefficients(asConst(out.view), 0, {-4}, *tables);
  expectCoefficients(asConst(out.view), 1, {}, *tables);
}
TEST_F(Region, IntegerScalingHandlesZeroAndBothInt64Endpoints) {
  Cipher a(*tables, 2, 2, 8), out(*tables, 2, 2, 8);
  a.set(0, {1, -1}, *tables);
  Arena arena(region::scratchBytes(params));
  for (auto c : {INT64_MIN, INT64_MAX, int64_t{0}}) {
    const region::LinearTerm term{asConst(a.view), c};
    ASSERT_TRUE(region::linear(out.view, &term, 1, 0, a.view.layout, *tables, arena));
    for (uint32_t l = 0; l < 3; ++l) {
      auto v = coefficients(asConst(out.view), 0, l, *tables);
      auto q = params.mainPrimes()[l];
      EXPECT_EQ(v[0], residue(c, q));
      EXPECT_EQ(v[1], v[0] ? q - v[0] : 0);
    }
    EXPECT_EQ(out.view.layout, a.view.layout);
  }
}
TEST_F(Region, FusedMultiplicationMatchesPrimitiveResiduesAtEveryDroppingLevel) {
  SyntheticKey key(*tables);
  Arena arena(std::max(region::scratchBytes(params), eval::scratchBytes(EvalOperation::HMultiply, params, 3)));
  for (uint32_t level = 1; level < 4; ++level) {
    Cipher a(*tables, level, 2, 17), b(*tables, level, 2, 31), out(*tables, level, 2), reference(*tables, level, 2),
      checked(*tables, level, 2);
    for (uint32_t p = 0; p < 2; ++p)
      for (uint32_t l = 0; l <= level; ++l)
        for (size_t i = 0; i < params.degree(); ++i) {
          auto q = params.mainPrimes()[l];
          a.view.poly(p).limb(l)[i] = (i * 7937 + (p + 1) * 17413 + l * 93) % q;
          b.view.poly(p).limb(l)[i] = (i * 12287 + (p + 3) * 8153 + l * 97) % q;
        }
    ASSERT_TRUE(eval::tensor(reference.view, asConst(a.view), asConst(b.view), *tables, arena));
    ASSERT_TRUE(eval::relinearize(reference.view, asConst(reference.view), key.view, *tables, arena));
    ASSERT_TRUE(eval::rescale(reference.view, asConst(reference.view), *tables, arena));
    ASSERT_TRUE(region::multiply(out.view, asConst(a.view), asConst(b.view), key.view, *tables, arena));
    ASSERT_TRUE(eval::hmult(checked.view, asConst(a.view), asConst(b.view), key.view, *tables, arena));
    equal(asConst(out.view), asConst(reference.view));
    equal(asConst(checked.view), asConst(reference.view));
  }
}
TEST_F(Region, WeightedProductSumLiftsLinearTermsBeforeOneRounding) {
  Cipher a(*tables, 3, 2, 8), b(*tables, 3, 2, 8), out(*tables, 3, 2), ref(*tables, 3, 3), term(*tables, 3, 3);
  a.set(0, {19, -23, 7}, *tables);
  a.set(1, {-5, 31}, *tables);
  b.set(0, {-11, 13}, *tables);
  b.set(1, {37, -17}, *tables);
  SyntheticKey key(*tables);
  Arena arena(region::scratchBytes(params));
  const region::ProductTerm products[] = {{asConst(a.view), asConst(b.view), 2},
                                          {asConst(a.view), asConst(a.view), -3}};
  const region::LinearTerm linear_terms[] = {{asConst(a.view), -2}};
  ASSERT_TRUE(eval::tensor(ref.view, asConst(a.view), asConst(b.view), *tables, arena));
  ASSERT_TRUE(eval::tensor(term.view, asConst(a.view), asConst(a.view), *tables, arena));
  for (uint32_t p = 0; p < 3; ++p)
    for (uint32_t l = 0; l < 4; ++l)
      for (size_t i = 0; i < params.degree(); ++i) {
        const auto q = params.mainPrimes()[l];
        __int128 value = 2 * static_cast<__int128>(ref.view.poly(p).limb(l)[i]) -
                         3 * static_cast<__int128>(term.view.poly(p).limb(l)[i]);
        if (p < 2)
          value -= 16 * static_cast<__int128>(a.view.poly(p).limb(l)[i]);
        if (p == 0)
          value += 16;
        value %= q;
        if (value < 0)
          value += q;
        ref.view.poly(p).limb(l)[i] = static_cast<uint64_t>(value);
      }
  ASSERT_TRUE(eval::relinearize(ref.view, asConst(ref.view), key.view, *tables, arena));
  ASSERT_TRUE(eval::rescale(ref.view, asConst(ref.view), *tables, arena));
  ASSERT_TRUE(region::productSum(out.view, products, 2, linear_terms, 1, .25, key.view, *tables, arena));
  equal(asConst(out.view), asConst(ref.view));
}
TEST_F(Region, RotationWithSignedAffineAddendsMatchesCheckedPrimitives) {
  Cipher a(*tables, 2, 2, 8), b(*tables, 2, 2, 8), out(*tables, 2, 2, 8), ref(*tables, 2, 2, 8);
  a.set(0, {1, -2, 3}, *tables);
  a.set(1, {-4, 5}, *tables);
  b.set(0, {8, -7}, *tables);
  b.set(1, {6, -9}, *tables);
  const auto g = eval::galoisElement(-1, params.logDegree());
  SyntheticKey key(*tables, SwitchKeyKind::Galois, g);
  Arena arena(region::scratchBytes(params));
  ASSERT_TRUE(eval::rotate(ref.view, asConst(a.view), g, key.view, *tables, arena));
  ASSERT_TRUE(eval::subtract(ref.view, asConst(ref.view), asConst(b.view), *tables, arena));
  ASSERT_TRUE(eval::add(ref.view, asConst(ref.view), asConst(a.view), *tables, arena));
  region::LinearTerm terms[] = {{asConst(b.view), -1}, {asConst(a.view), 1}};
  ASSERT_TRUE(region::rotateAdd(out.view, asConst(a.view), g, key.view, terms, 2, *tables, arena));
  equal(asConst(out.view), asConst(ref.view));
  ASSERT_TRUE(region::rotateAdd(out.view, asConst(a.view), 1, {}, nullptr, 0, *tables, arena));
  equal(asConst(out.view), asConst(a.view));
}
TEST_F(Region, FractionalScaleAndLevelDownKeepRawSemantics) {
  Cipher a(*tables, 3, 2, 8), out(*tables, 3, 2), ref(*tables, 3, 2);
  a.set(0, {65537, -65537, 2, -2}, *tables);
  a.set(1, {-5, 9}, *tables);
  Arena arena(region::scratchBytes(params));
  for (double c : {.5, -.25, 0., 2.}) {
    ASSERT_TRUE(eval::multiplyScalar(ref.view, asConst(a.view), c, *tables, arena));
    ASSERT_TRUE(region::scaleFractional(out.view, asConst(a.view), c, *tables, arena));
    equal(asConst(out.view), asConst(ref.view));
  }
  ASSERT_TRUE(eval::levelDown(ref.view, asConst(a.view), 1, *tables, arena));
  ASSERT_TRUE(region::levelDown(out.view, asConst(a.view), 1, *tables, arena));
  equal(asConst(out.view), asConst(ref.view));
}
TEST_F(Region, CheapGuardsRejectBeforeChangingDestination) {
  Cipher a(*tables, 3, 2, 8), out(*tables, 3, 2, 8);
  out.set(0, {73}, *tables);
  region::LinearTerm term{asConst(a.view), 1};
  SyntheticKey key(*tables);
  Arena arena(region::scratchBytes(params));
  const auto layout = out.view.layout;
  const std::vector<uint64_t> before(out.slab.data(), out.slab.data() + out.slab.capacityWords());
  EXPECT_FALSE(region::linear(out.view, nullptr, 1, 0, layout, *tables, arena));
  EXPECT_FALSE(region::linear(out.view, &term, 1, std::ldexp(1., 63), layout, *tables, arena));
  EXPECT_FALSE(region::linear(a.view, &term, 1, 0, layout, *tables, arena));
  auto wrong = layout;
  wrong.scale = 7;
  EXPECT_FALSE(region::linear(out.view, &term, 1, 0, wrong, *tables, arena));
  Arena tiny(64);
  EXPECT_FALSE(region::linear(out.view, &term, 1, 0, layout, *tables, tiny));
  auto fraction = asConst(a.view);
  fraction.layout.scale = 8.5;
  region::ProductTerm product{fraction, fraction, 1};
  region::LinearTerm lift{fraction, 1};
  EXPECT_FALSE(region::productSum(out.view, &product, 1, &lift, 1, 0, key.view, *tables, arena));
  EXPECT_EQ(out.view.layout, layout);
  EXPECT_TRUE(std::equal(before.begin(), before.end(), out.slab.data()));
  EXPECT_EQ(arena.used(), 0U);
}
} // namespace
#include <cstdlib>
#include <new>
namespace region_memory_probe {
thread_local bool enabled = false;
thread_local std::size_t allocations = 0;
} // namespace region_memory_probe
void *operator new(std::size_t bytes) {
  if (region_memory_probe::enabled)
    ++region_memory_probe::allocations;
  if (void *p = std::malloc(bytes == 0 ? 1 : bytes))
    return p;
  throw std::bad_alloc();
}
void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { ::operator delete(p); }
void operator delete(void *p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void *p, std::size_t) noexcept { ::operator delete(p); }
void *operator new(std::size_t bytes, std::align_val_t alignment) {
  if (region_memory_probe::enabled)
    ++region_memory_probe::allocations;
  void *p = nullptr;
  if (posix_memalign(&p, static_cast<std::size_t>(alignment), bytes == 0 ? 1 : bytes) != 0)
    throw std::bad_alloc();
  return p;
}
void *operator new[](std::size_t bytes, std::align_val_t alignment) { return ::operator new(bytes, alignment); }
void operator delete(void *p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void *p, std::align_val_t alignment) noexcept { ::operator delete(p, alignment); }
void operator delete(void *p, std::size_t, std::align_val_t alignment) noexcept { ::operator delete(p, alignment); }
void operator delete[](void *p, std::size_t, std::align_val_t alignment) noexcept { ::operator delete(p, alignment); }

TEST_F(Region, AllRegionPreflightFailuresPreserveOutputBytes) {
  Cipher a(*tables, 3, 2, 8), b(*tables, 3, 2, 8), out(*tables, 3, 2, 8), bottom(*tables, 0, 2, 8);
  out.set(0, {19, -23}, *tables);
  SyntheticKey key(*tables);
  Arena arena(region::scratchBytes(params));
  const auto layout = out.view.layout;
  const std::vector<uint64_t> before(out.slab.data(), out.slab.data() + out.slab.capacityWords());
  const region::LinearTerm term{asConst(a.view), 1};
  const region::ProductTerm product{asConst(a.view), asConst(b.view), 1};
  EXPECT_FALSE(region::multiply(out.view, asConst(a.view), asConst(b.view), {}, *tables, arena));
  EXPECT_FALSE(region::multiply(out.view, asConst(bottom.view), asConst(bottom.view), key.view, *tables, arena));
  EXPECT_FALSE(region::productSum(out.view, nullptr, 0, nullptr, 0, 0, key.view, *tables, arena));
  EXPECT_FALSE(region::productSum(out.view, &product, 1, nullptr, 1, 0, key.view, *tables, arena));
  EXPECT_FALSE(region::productSum(out.view, &product, 1, &term, 1, std::numeric_limits<double>::infinity(), key.view,
                                  *tables, arena));
  EXPECT_FALSE(region::rotateAdd(out.view, asConst(a.view), 2, key.view, nullptr, 0, *tables, arena));
  EXPECT_FALSE(region::rotateAdd(out.view, asConst(a.view), 5, key.view, nullptr, 0, *tables, arena));
  EXPECT_FALSE(region::rotateAdd(out.view, asConst(a.view), 1, {}, nullptr, 1, *tables, arena));
  EXPECT_FALSE(
    region::scaleFractional(out.view, asConst(a.view), std::numeric_limits<double>::infinity(), *tables, arena));
  EXPECT_FALSE(region::scaleFractional(out.view, asConst(bottom.view), .5, *tables, arena));
  EXPECT_FALSE(region::levelDown(out.view, asConst(a.view), 4, *tables, arena));
  auto small = out.view;
  small.capacity_words = 1;
  EXPECT_FALSE(region::linear(small, &term, 1, 0, layout, *tables, arena));
  auto wrong = asConst(a.view);
  wrong.layout.poly.form = NttForm::Coefficient;
  EXPECT_FALSE(region::multiply(out.view, wrong, asConst(b.view), key.view, *tables, arena));
  EXPECT_EQ(out.view.layout, layout);
  EXPECT_TRUE(std::equal(before.begin(), before.end(), out.slab.data()));
  EXPECT_EQ(arena.used(), 0U);
}
TEST_F(Region, CombinedRoundingMatchesIndependentCenteredIntegerBoundaryOracle) {
  Params::Spec spec;
  spec.log_degree = params.logDegree();
  spec.main_primes = params.mainPrimes();
  spec.aux_primes = params.auxPrimes();
  spec.log_default_scale = params.logDefaultScale();
  spec.dnum = 4;
  auto local = Params::create(spec);
  ASSERT_TRUE(local);
  auto t = EvaluationTables::create(*local);
  ASSERT_TRUE(t);
  Cipher a(*t, 3, 2, 8), b(*t, 3, 2, 8), out(*t, 3, 2), ref(*t, 3, 2);
  a.set(1, {1}, *t);
  b.set(1, {1}, *t);
  a.set(0, {3, -3, 1, -1}, *t);
  SyntheticKey key(*t);
  const int64_t P = static_cast<int64_t>(spec.aux_primes[0] * spec.aux_primes[1]);
  const int64_t q = static_cast<int64_t>(spec.main_primes.back());
  const int64_t divisor = P * q;
  const std::vector<int64_t> values = {divisor / 2, divisor / 2 + 1, -divisor / 2, -divisor / 2 - 1,
                                       P / 2,       P / 2 + 1,       -P / 2,       -P / 2 - 1};
  for (auto &slab : key.slabs)
    std::fill_n(slab.data(), slab.capacityWords(), uint64_t{0});
  const auto n = local->degree();
  const auto k = local->levelCount();
  const auto stride = (k + spec.aux_primes.size()) * n;
  for (uint32_t poly = 0; poly < 2; ++poly)
    for (uint32_t l = 0; l < k + spec.aux_primes.size(); ++l) {
      auto modulus = l < k ? spec.main_primes[l] : spec.aux_primes[l - k];
      auto *dst = key.slabs[0].data() + poly * stride + l * n;
      for (size_t i = 0; i < values.size(); ++i)
        dst[i] = residue(values[i], modulus);
      ntt::forward(dst, l < k ? t->mainNtt(l) : t->auxNtt(l - k));
    }
  Arena arena(region::scratchBytes(*local));
  ASSERT_TRUE(region::multiply(out.view, asConst(a.view), asConst(b.view), key.view, *t, arena));
  for (uint32_t poly = 0; poly < 2; ++poly) {
    std::vector<int64_t> expected(values.size());
    const int64_t add[] = {3, -3, 1, -1};
    for (size_t i = 0; i < values.size(); ++i) {
      const int64_t numerator = values[i] + (poly == 1 && i < 4 ? P * add[i] : 0);
      expected[i] = numerator >= 0 ? (numerator + divisor / 2) / divisor : -((-numerator + divisor / 2) / divisor);
    }
    expectCoefficients(asConst(out.view), poly, expected, *t);
  }
  ASSERT_TRUE(eval::tensor(ref.view, asConst(a.view), asConst(b.view), *t, arena));
  ASSERT_TRUE(eval::relinearize(ref.view, asConst(ref.view), key.view, *t, arena));
  ASSERT_TRUE(eval::rescale(ref.view, asConst(ref.view), *t, arena));
  equal(asConst(out.view), asConst(ref.view));
}
TEST_F(Region, ProductConstantsRoundAtInputScaleAndLiftBeyondInt64) {
  SyntheticKey key(*tables);
  Arena arena(region::scratchBytes(params));
  for (double scale : {8., std::ldexp(1., 40)}) {
    Cipher a(*tables, 3, 2, scale), out(*tables, 3, 2), ref(*tables, 3, 3);
    a.set(0, {1, -1}, *tables);
    a.set(1, {2, -2}, *tables);
    const region::ProductTerm product{asConst(a.view), asConst(a.view), INT64_MIN};
    ASSERT_TRUE(eval::tensor(ref.view, asConst(a.view), asConst(a.view), *tables, arena));
    const auto encoded = static_cast<int64_t>(std::round(.1 * scale));
    for (uint32_t poly = 0; poly < 3; ++poly)
      for (uint32_t l = 0; l < 4; ++l) {
        const auto q = params.mainPrimes()[l];
        auto *dst = ref.view.poly(poly).limb(l);
        for (size_t i = 0; i < params.degree(); ++i) {
          __int128 value = static_cast<__int128>(INT64_MIN) * dst[i];
          if (poly == 0)
            value += static_cast<__int128>(encoded) * static_cast<int64_t>(scale);
          value %= q;
          if (value < 0)
            value += q;
          dst[i] = static_cast<uint64_t>(value);
        }
      }
    ASSERT_TRUE(eval::relinearize(ref.view, asConst(ref.view), key.view, *tables, arena));
    ASSERT_TRUE(eval::rescale(ref.view, asConst(ref.view), *tables, arena));
    ASSERT_TRUE(region::productSum(out.view, &product, 1, nullptr, 0, .1, key.view, *tables, arena));
    equal(asConst(out.view), asConst(ref.view));
  }
  Cipher a(*tables, 3, 2, 8.5), out(*tables, 3, 2), ref(*tables, 3, 2);
  a.set(0, {3, -7}, *tables);
  const region::ProductTerm pure{asConst(a.view), asConst(a.view), 1};
  ASSERT_TRUE(region::productSum(out.view, &pure, 1, nullptr, 0, 0, key.view, *tables, arena));
  ASSERT_TRUE(eval::tensor(ref.view, asConst(a.view), asConst(a.view), *tables, arena));
  ASSERT_TRUE(eval::relinearize(ref.view, asConst(ref.view), key.view, *tables, arena));
  ASSERT_TRUE(eval::rescale(ref.view, asConst(ref.view), *tables, arena));
  equal(asConst(out.view), asConst(ref.view));
}
TEST_F(Region, MultilimbRotationsReuseBoundedScratchForEverySignedPermutation) {
  const auto bound = region::scratchBytes(params);
  Arena arena(bound + Arena::kAlignment);
  auto *guard = arena.allocate<unsigned char>(1);
  ASSERT_NE(guard, nullptr);
  *guard = 37;
  const auto entry = arena.used();
  Arena short_arena(bound - Arena::kAlignment);
  ASSERT_LT(short_arena.remaining(), bound);
  const uint64_t elements[] = {5, eval::galoisElement(-1, params.logDegree()), 2 * params.degree() - 1};
  for (auto g : elements) {
    SyntheticKey key(*tables, SwitchKeyKind::Galois, g);
    for (uint32_t level = 0; level < params.levelCount(); ++level) {
      Cipher a(*tables, level, 2, 8), out(*tables, level, 2, 8), reference(*tables, level, 2, 8);
      for (uint32_t poly = 0; poly < 2; ++poly)
        for (uint32_t limb = 0; limb <= level; ++limb)
          for (std::size_t i = 0; i < params.degree(); ++i)
            a.view.poly(poly).limb(limb)[i] = (i * 7937 + (poly + 1) * 17413 + limb * 93) % params.mainPrimes()[limb];
      const region::LinearTerm addend{asConst(a.view), 1};
      ASSERT_TRUE(eval::rotate(reference.view, asConst(a.view), g, key.view, *tables, arena));
      ASSERT_TRUE(eval::add(reference.view, asConst(reference.view), asConst(a.view), *tables, arena));
      const std::vector<uint64_t> before(out.slab.data(), out.slab.data() + out.slab.capacityWords());
      const auto layout = out.view.layout;
      EXPECT_FALSE(region::rotateAdd(out.view, asConst(a.view), g, key.view, &addend, 1, *tables, short_arena));
      EXPECT_EQ(out.view.layout, layout);
      EXPECT_TRUE(std::equal(before.begin(), before.end(), out.slab.data()));
      region_memory_probe::allocations = 0;
      region_memory_probe::enabled = true;
      const bool success = region::rotateAdd(out.view, asConst(a.view), g, key.view, &addend, 1, *tables, arena);
      region_memory_probe::enabled = false;
      ASSERT_TRUE(success);
      EXPECT_EQ(region_memory_probe::allocations, 0U);
      equal(asConst(out.view), asConst(reference.view));
      EXPECT_EQ(arena.used(), entry);
      EXPECT_EQ(*guard, 37);
      EXPECT_LE(arena.highWater(), arena.capacity());
    }
  }
}
TEST_F(Region, PreparedRegionsAllocateOnlyCallerWorkspace) {
  Cipher a(*tables, 3, 2, 8), b(*tables, 3, 2, 8), out(*tables, 3, 2, 8);
  a.set(0, {1, 2}, *tables);
  a.set(1, {-3, 4}, *tables);
  b.set(0, {5, -6}, *tables);
  SyntheticKey key(*tables), rotation(*tables, SwitchKeyKind::Galois, 5);
  const auto bound = region::scratchBytes(params);
  Arena arena(bound + Arena::kAlignment);
  auto *guard = arena.allocate<unsigned char>(1);
  ASSERT_NE(guard, nullptr);
  *guard = 91;
  const auto used = arena.used();
  region::LinearTerm term{asConst(a.view), -3};
  region::ProductTerm product{asConst(a.view), asConst(b.view), 1};
  bool success[6]{};
  region_memory_probe::allocations = 0;
  region_memory_probe::enabled = true;
  success[0] = region::linear(out.view, &term, 1, .25, a.view.layout, *tables, arena);
  success[1] = region::multiply(out.view, asConst(a.view), asConst(b.view), key.view, *tables, arena);
  success[2] = region::productSum(out.view, &product, 1, &term, 1, .25, key.view, *tables, arena);
  success[3] = region::rotateAdd(out.view, asConst(a.view), 5, rotation.view, &term, 1, *tables, arena);
  success[4] = region::scaleFractional(out.view, asConst(a.view), .25, *tables, arena);
  success[5] = region::levelDown(out.view, asConst(a.view), 1, *tables, arena);
  region_memory_probe::enabled = false;
  for (bool ok : success)
    EXPECT_TRUE(ok);
  EXPECT_EQ(region_memory_probe::allocations, 0U);
  EXPECT_EQ(arena.used(), used);
  EXPECT_EQ(*guard, 91);
  EXPECT_LE(arena.highWater(), arena.capacity());
}
