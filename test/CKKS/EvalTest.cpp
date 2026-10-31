#include "CKKS/Eval.h"
#include <gtest/gtest.h>
using namespace fhenomenon::ckks;
TEST(CkksEvaluationSetup, CreatesAllLevelsAndDigitConversions) {
  Params::Spec spec;
  spec.log_degree = 10;
  spec.main_primes = {12289, 40961, 61441, 65537};
  spec.aux_primes = {786433, 1179649};
  spec.dnum = 2;
  spec.log_default_scale = 13;
  auto params = Params::create(spec);
  ASSERT_TRUE(params.has_value());
  auto tables = EvaluationTables::create(*params);
  ASSERT_NE(tables, nullptr);
  EXPECT_EQ(tables->mainCrt(2).moduli().size(), 3U);
  EXPECT_EQ(tables->digit(2, 1).end, 3U);
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

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
class Evaluation : public ::testing::Test {
  protected:
  Params params = parameters();
  std::unique_ptr<EvaluationTables> tables;
  void SetUp() override {
    std::string error;
    tables = EvaluationTables::create(params, &error);
    ASSERT_NE(tables, nullptr) << error;
  }
};
TEST_F(Evaluation, AdditionSubtractionAndNegationPreserveRingCoefficientsInPlace) {
  Cipher a(*tables, 3, 2), b(*tables, 3, 2);
  a.set(0, {8, -7, 3}, *tables);
  a.set(1, {2, 4}, *tables);
  b.set(0, {-2, 9, 5}, *tables);
  b.set(1, {5, -1}, *tables);
  Arena arena(eval::scratchBytes(EvalOperation::Add, params, 3));
  ASSERT_TRUE(eval::add(a.view, asConst(a.view), asConst(b.view), *tables, arena));
  expectCoefficients(asConst(a.view), 0, {6, 2, 8}, *tables);
  expectCoefficients(asConst(a.view), 1, {7, 3}, *tables);
  ASSERT_TRUE(eval::subtract(a.view, asConst(a.view), asConst(b.view), *tables, arena));
  ASSERT_TRUE(eval::negate(a.view, asConst(a.view), *tables, arena));
  expectCoefficients(asConst(a.view), 0, {-8, 7, -3}, *tables);
  EXPECT_EQ(arena.used(), 0U);
}
TEST_F(Evaluation, TensorUsesNegacyclicRingProductAndRetainsThreeComponents) {
  Cipher a(*tables, 3, 2, 8), b(*tables, 3, 2, 16);
  std::vector<int64_t> high(params.degree(), 0);
  high.back() = 2;
  a.set(0, high, *tables);
  a.set(1, {3}, *tables);
  b.set(0, {0, 5}, *tables);
  b.set(1, {7}, *tables);
  Arena arena(eval::scratchBytes(EvalOperation::Tensor, params, 3));
  ASSERT_TRUE(eval::tensor(a.view, asConst(a.view), asConst(b.view), *tables, arena));
  EXPECT_EQ(a.view.layout.num_polys, 3U);
  EXPECT_EQ(a.view.layout.scale, 128);
  expectCoefficients(asConst(a.view), 0, {-10}, *tables);
  std::vector<int64_t> mid(params.degree(), 0);
  mid[1] = 15;
  mid.back() = 14;
  expectCoefficients(asConst(a.view), 1, mid, *tables);
  expectCoefficients(asConst(a.view), 2, {21}, *tables);
}
TEST_F(Evaluation, RescaleRoundsCenteredCoefficientsExactlyAcrossBothSigns) {
  Cipher a(*tables, 3, 2, 65537.0 * 8192);
  a.set(0, {32768, 32769, -32768, -32769, 65540, -65540}, *tables);
  a.set(1, {196610, -196610}, *tables);
  Arena arena(eval::scratchBytes(EvalOperation::Rescale, params, 3));
  ASSERT_TRUE(eval::rescale(a.view, asConst(a.view), *tables, arena));
  EXPECT_EQ(a.view.layout.level(), 2);
  EXPECT_DOUBLE_EQ(a.view.layout.scale, 8192);
  expectCoefficients(asConst(a.view), 0, {0, 1, 0, -1, 1, -1}, *tables);
  expectCoefficients(asConst(a.view), 1, {3, -3}, *tables);
}
TEST_F(Evaluation, HybridRelinearizationProjectsFullKeysAtEveryLevel) {
  SyntheticKey key(*tables);
  for (uint32_t level = 0; level < params.levelCount(); ++level) {
    Cipher a(*tables, level, 3);
    a.set(0, {1, -7, 8}, *tables);
    a.set(1, {2, 10, -3}, *tables);
    a.set(2, {-9, 4, 11}, *tables);
    Arena arena(eval::scratchBytes(EvalOperation::Relinearize, params, level));
    ASSERT_TRUE(eval::relinearize(a.view, asConst(a.view), key.view, *tables, arena));
    ASSERT_EQ(a.view.layout.num_polys, 2U);
    for (uint32_t limb = 0; limb <= level; ++limb) {
      auto a0 = coefficients(asConst(a.view), 0, limb, *tables), a1 = coefficients(asConst(a.view), 1, limb, *tables);
      auto q = params.mainPrimes()[limb];
      const int64_t want[] = {-6, 7, 16};
      for (std::size_t i = 0; i < a0.size(); ++i)
        EXPECT_EQ((a0[i] + a1[i]) % q, i < 3 ? residue(want[i], q) : 0) << "level " << level;
    }
    EXPECT_LE(arena.highWater(), arena.capacity());
  }
}
TEST_F(Evaluation, GaloisPermutationAndKeySwitchAgreeWithCoefficientSubstitution) {
  SyntheticKey key(*tables, SwitchKeyKind::Galois, 5);
  Cipher a(*tables, 2, 2);
  a.set(0, {1, 2, 3}, *tables);
  a.set(1, {4, 5, 6}, *tables);
  Arena arena(eval::scratchBytes(EvalOperation::Rotate, params, 2));
  ASSERT_TRUE(eval::rotate(a.view, asConst(a.view), 5, key.view, *tables, arena));
  for (uint32_t limb = 0; limb < 3; ++limb) {
    auto a0 = coefficients(asConst(a.view), 0, limb, *tables), a1 = coefficients(asConst(a.view), 1, limb, *tables);
    auto q = params.mainPrimes()[limb];
    for (std::size_t i = 0; i < a0.size(); ++i)
      EXPECT_EQ((a0[i] + a1[i]) % q, i == 0 ? 5U : (i == 5 ? 7U : (i == 10 ? 9U : 0U)));
  }
  EXPECT_EQ(eval::galoisElement(1, 10), 5U);
  EXPECT_EQ(eval::galoisElement(-1, 10), 1229U);
  EXPECT_EQ(eval::galoisElement(std::numeric_limits<int64_t>::min(), 10), 1U);
}
TEST_F(Evaluation, ScalarsUseQuantizedConstantAndMultiplicationConsumesLevel) {
  Cipher a(*tables, 3, 2, 4);
  a.set(0, {8, 65537, -65537}, *tables);
  Arena arena(std::max(eval::scratchBytes(EvalOperation::MultiplyScalar, params, 3),
                       eval::scratchBytes(EvalOperation::AddScalar, params, 3)));
  ASSERT_TRUE(eval::addScalar(a.view, asConst(a.view), -1.25, *tables, arena));
  expectCoefficients(asConst(a.view), 0, {3, 65537, -65537}, *tables);
  ASSERT_TRUE(eval::multiplyScalar(a.view, asConst(a.view), 2.0, *tables, arena));
  EXPECT_EQ(a.view.layout.level(), 2);
  EXPECT_DOUBLE_EQ(a.view.layout.scale, 4);
  expectCoefficients(asConst(a.view), 0, {6, 131074, -131074}, *tables);
}
TEST_F(Evaluation, LevelDownPreservesScaleAndCoefficients) {
  Cipher a(*tables, 3, 2);
  a.set(0, {-100, 20, 3}, *tables);
  Arena arena(eval::scratchBytes(EvalOperation::LevelDown, params, 3));
  ASSERT_TRUE(eval::levelDown(a.view, asConst(a.view), 1, *tables, arena));
  EXPECT_EQ(a.view.layout.level(), 1);
  EXPECT_EQ(a.view.layout.scale, 8192);
  expectCoefficients(asConst(a.view), 0, {-100, 20, 3}, *tables);
}
TEST_F(Evaluation, InvalidScaleKeyScratchCapacityAndPartialAliasLeaveOutputUnchanged) {
  Cipher a(*tables, 3, 2), b(*tables, 3, 2), out(*tables, 3, 2);
  a.set(0, {1, 2}, *tables);
  b.set(0, {3, 4}, *tables);
  out.set(0, {777}, *tables);
  const std::vector<uint64_t> before(out.slab.data(), out.slab.data() + out.slab.capacityWords());
  const auto layout = out.view.layout;
  Arena arena(eval::scratchBytes(EvalOperation::Rotate, params, 3));
  b.view.layout.scale *= 2;
  EXPECT_FALSE(eval::add(out.view, asConst(a.view), asConst(b.view), *tables, arena));
  b.view.layout.scale /= 2;
  Arena short_arena(eval::scratchBytes(EvalOperation::Add, params, 3) - Arena::kAlignment);
  ASSERT_LT(short_arena.capacity(), eval::scratchBytes(EvalOperation::Add, params, 3));
  EXPECT_FALSE(eval::add(out.view, asConst(a.view), asConst(b.view), *tables, short_arena));
  auto narrow = out.view;
  narrow.capacity_words = 1;
  EXPECT_FALSE(eval::add(narrow, asConst(a.view), asConst(b.view), *tables, arena));
  auto partial = a.view;
  partial.data += 8;
  partial.capacity_words -= 8;
  EXPECT_FALSE(eval::negate(partial, asConst(a.view), *tables, arena));
  SyntheticKey key(*tables);
  EXPECT_FALSE(eval::rotate(out.view, asConst(a.view), 5, key.view, *tables, arena));
  EXPECT_FALSE(eval::addScalar(out.view, asConst(a.view), std::numeric_limits<double>::infinity(), *tables, arena));
  EXPECT_EQ(out.view.layout, layout);
  EXPECT_TRUE(std::equal(before.begin(), before.end(), out.slab.data()));
}
} // namespace

TEST(CkksEvaluationSetup, FusedMultiplyMatchesExplicitTensorRelinearizeRescale) {
  auto params = parameters();
  auto tables = EvaluationTables::create(params);
  ASSERT_NE(tables, nullptr);
  Cipher a(*tables, 3, 2, 65537), b(*tables, 3, 2, 65537), fused(*tables, 3, 2), split(*tables, 3, 2);
  a.set(0, {65537, 2}, *tables);
  a.set(1, {3}, *tables);
  b.set(0, {131074, -5}, *tables);
  b.set(1, {7}, *tables);
  SyntheticKey key(*tables);
  Arena arena(eval::scratchBytes(EvalOperation::HMultiply, params, 3));
  ASSERT_TRUE(eval::hmult(fused.view, asConst(a.view), asConst(b.view), key.view, *tables, arena));
  ASSERT_TRUE(eval::tensor(split.view, asConst(a.view), asConst(b.view), *tables, arena));
  ASSERT_TRUE(eval::relinearize(split.view, asConst(split.view), key.view, *tables, arena));
  ASSERT_TRUE(eval::rescale(split.view, asConst(split.view), *tables, arena));
  EXPECT_EQ(fused.view.layout, split.view.layout);
  for (uint32_t poly = 0; poly < 2; ++poly)
    for (uint32_t limb = 0; limb < 3; ++limb) {
      auto f = coefficients(asConst(fused.view), poly, limb, *tables),
           s = coefficients(asConst(split.view), poly, limb, *tables);
      EXPECT_EQ(f, s);
    }
}

TEST_F(Evaluation, EveryFailureIsTransactionalForHybridKeyAndInputMetadata) {
  Cipher input(*tables, 3, 3), out(*tables, 3, 2);
  input.set(2, {1, 2, 3}, *tables);
  out.set(0, {999}, *tables);
  SyntheticKey hint(*tables);
  Arena arena(eval::scratchBytes(EvalOperation::Relinearize, params, 3));
  const std::vector<uint64_t> before(out.slab.data(), out.slab.data() + out.slab.capacityWords());
  const auto shape = out.view.layout;
  auto bad = hint.view;
  bad.galois_element = 5;
  EXPECT_FALSE(eval::relinearize(out.view, asConst(input.view), bad, *tables, arena));
  bad = hint.view;
  --bad.aux_count;
  EXPECT_FALSE(eval::relinearize(out.view, asConst(input.view), bad, *tables, arena));
  bad = hint.view;
  --bad.digit_count;
  EXPECT_FALSE(eval::relinearize(out.view, asConst(input.view), bad, *tables, arena));
  bad = hint.view;
  bad.params_id = ParamsId{42};
  EXPECT_FALSE(eval::relinearize(out.view, asConst(input.view), bad, *tables, arena));
  const auto saved = hint.digits[0];
  hint.digits[0].layout.poly.form = NttForm::Coefficient;
  EXPECT_FALSE(eval::relinearize(out.view, asConst(input.view), hint.view, *tables, arena));
  hint.digits[0] = saved;
  const auto input_shape = input.view.layout;
  input.view.layout.poly.form = NttForm::Coefficient;
  EXPECT_FALSE(eval::relinearize(out.view, asConst(input.view), hint.view, *tables, arena));
  input.view.layout = input_shape;
  input.view.poly(0).limb(0)[0] = params.mainPrimes()[0];
  EXPECT_FALSE(eval::relinearize(out.view, asConst(input.view), hint.view, *tables, arena));
  EXPECT_EQ(out.view.layout, shape);
  EXPECT_TRUE(std::equal(before.begin(), before.end(), out.slab.data()));
  EXPECT_EQ(arena.used(), 0U);
}

TEST_F(Evaluation, OppositeRotationsAndConjugationRespectNegacyclicSigns) {
  const auto n = params.degree();
  const std::vector<uint64_t> elements = {5, 1229, 2 * n - 1};
  for (auto g : elements) {
    Cipher input(*tables, 3, 2), out(*tables, 3, 2);
    std::vector<int64_t> message(n, 0);
    message[0] = 2;
    message[1] = -3;
    message[n / 2] = 5;
    message[n - 1] = -7;
    input.set(0, message, *tables);
    SyntheticKey hint(*tables, SwitchKeyKind::Galois, g);
    Arena arena(eval::scratchBytes(EvalOperation::Rotate, params, 3));
    ASSERT_TRUE(eval::rotate(out.view, asConst(input.view), g, hint.view, *tables, arena));
    std::vector<int64_t> expected(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
      auto exponent = i * g % (2 * n);
      expected[exponent % n] = exponent < n ? message[i] : -message[i];
    }
    for (uint32_t limb = 0; limb < 4; ++limb) {
      auto a = coefficients(asConst(out.view), 0, limb, *tables), b = coefficients(asConst(out.view), 1, limb, *tables);
      auto q = params.mainPrimes()[limb];
      for (std::size_t i = 0; i < n; ++i)
        EXPECT_EQ((a[i] + b[i]) % q, residue(expected[i], q));
    }
  }
}

TEST_F(Evaluation, ScratchBoundWorksAfterUnalignedAllocationAndIdentityNeedsNoKey) {
  Cipher input(*tables, 3, 2), out(*tables, 3, 2);
  input.set(0, {4, -3, 2}, *tables);
  const auto bound = eval::scratchBytes(EvalOperation::Rotate, params, 3);
  Arena arena(bound + Arena::kAlignment);
  auto *guard = arena.allocate<unsigned char>(1);
  ASSERT_NE(guard, nullptr);
  *guard = 73;
  const auto before = arena.used();
  out.view.layout = {};
  ASSERT_TRUE(eval::rotate(out.view, asConst(input.view), 1, {}, *tables, arena));
  expectCoefficients(asConst(out.view), 0, {4, -3, 2}, *tables);
  EXPECT_EQ(arena.used(), before);
  EXPECT_EQ(*guard, 73);
}

TEST_F(Evaluation, FractionalAndZeroScalarStillPerformRealRescale) {
  Cipher input(*tables, 3, 2, 4);
  input.set(0, {65537, -65537, 2, -2}, *tables);
  Arena arena(eval::scratchBytes(EvalOperation::MultiplyScalar, params, 3));
  ASSERT_TRUE(eval::multiplyScalar(input.view, asConst(input.view), 0.5, *tables, arena));
  EXPECT_EQ(input.view.layout.level(), 2);
  expectCoefficients(asConst(input.view), 0, {32769, -32769, 1, -1}, *tables);
  ASSERT_TRUE(eval::multiplyScalar(input.view, asConst(input.view), 0, *tables, arena));
  EXPECT_EQ(input.view.layout.level(), 1);
  expectCoefficients(asConst(input.view), 0, {}, *tables);
  auto bad = input.view;
  bad.layout.scale = std::numeric_limits<double>::max();
  EXPECT_FALSE(eval::multiplyScalar(input.view, asConst(bad), 2, *tables, arena));
}

#include <cstdlib>
#include <new>
namespace eval_memory_probe {
thread_local bool enabled = false;
thread_local std::size_t allocations = 0;
} // namespace eval_memory_probe
void *operator new(std::size_t bytes) {
  if (eval_memory_probe::enabled)
    ++eval_memory_probe::allocations;
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
  if (eval_memory_probe::enabled)
    ++eval_memory_probe::allocations;
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

TEST_F(Evaluation, EverySuccessfulKernelUsesOnlyCallerScratch) {
  Cipher a(*tables, 3, 2), b(*tables, 3, 2), three(*tables, 3, 3), out(*tables, 3, 2);
  a.set(0, {1, 2, 3}, *tables);
  b.set(0, {3, 4, 5}, *tables);
  three.set(2, {7, 8, 9}, *tables);
  SyntheticKey key(*tables), rotation(*tables, SwitchKeyKind::Galois, 5);
  const auto bound = std::max({eval::scratchBytes(EvalOperation::HMultiply, params, 3),
                               eval::scratchBytes(EvalOperation::Rotate, params, 3),
                               eval::scratchBytes(EvalOperation::MultiplyScalar, params, 3)});
  Arena arena(bound);
  bool success[11]{};
  eval_memory_probe::allocations = 0;
  eval_memory_probe::enabled = true;
  success[0] = eval::add(out.view, asConst(a.view), asConst(b.view), *tables, arena);
  success[1] = eval::subtract(out.view, asConst(a.view), asConst(b.view), *tables, arena);
  success[2] = eval::negate(out.view, asConst(a.view), *tables, arena);
  success[3] = eval::tensor(out.view, asConst(a.view), asConst(b.view), *tables, arena);
  success[4] = eval::relinearize(out.view, asConst(three.view), key.view, *tables, arena);
  success[5] = eval::rescale(out.view, asConst(a.view), *tables, arena);
  success[6] = eval::rotate(out.view, asConst(a.view), 5, rotation.view, *tables, arena);
  success[7] = eval::hmult(out.view, asConst(a.view), asConst(b.view), key.view, *tables, arena);
  success[8] = eval::addScalar(out.view, asConst(a.view), 0.5, *tables, arena);
  success[9] = eval::multiplyScalar(out.view, asConst(a.view), 0.5, *tables, arena);
  success[10] = eval::levelDown(out.view, asConst(a.view), 1, *tables, arena);
  eval_memory_probe::enabled = false;
  for (bool result : success)
    EXPECT_TRUE(result);
  EXPECT_EQ(eval_memory_probe::allocations, 0U);
  EXPECT_EQ(arena.used(), 0U);
  EXPECT_LE(arena.highWater(), bound);
}
