#include "CKKS/ModArith.h"
#include "CKKS/detail/Region.h"
#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <vector>
using namespace fhenomenon::ckks;
namespace {
uint64_t residue(int64_t value, uint64_t q) {
  const auto r = value % static_cast<int64_t>(q);
  return static_cast<uint64_t>(r < 0 ? r + static_cast<int64_t>(q) : r);
}
Params parameters(uint32_t digits, bool wide = false) {
  Params::Spec s;
  s.log_degree = 10;
  s.main_primes = {12289, 40961, 61441, 65537};
  // P must exceed even the single digit containing all four Q primes.
  // Use two distinct 62-bit NTT primes to cover the widest fixture.
  uint64_t candidate = 4611686018427365377ULL - 2048;
  while (s.aux_primes.size() < 2) {
    if (modarith::isPrime(candidate))
      s.aux_primes.push_back(candidate);
    candidate -= 2048;
  }
  if (wide)
    s.main_primes[0] = 4611686018427365377ULL;
  s.dnum = digits;
  s.log_default_scale = 13;
  return Params::create(s).value();
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
            m = modarith::mulMod(m, factor % q, q);
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

void fill(Cipher &c, const EvaluationTables &t, uint64_t seed) {
  for (uint32_t p = 0; p < 2; ++p)
    for (uint32_t l = 0; l < c.view.layout.poly.basis.main_end; ++l) {
      auto q = t.params().mainPrimes()[l];
      for (size_t i = 0; i < t.params().degree(); ++i) {
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        c.view.poly(p).limb(l)[i] = i % 7 == 0 ? 0 : i % 7 == 1 ? q - 1 : (seed * 2685821657736338717ULL) % q;
      }
    }
}
std::vector<uint64_t> keyProducts(EvaluationKeyView key, const EvaluationTables &t) {
  std::vector<uint64_t> v;
  for (uint32_t d = 0; d < key.digit_count; ++d)
    for (uint32_t p = 0; p < 2; ++p)
      for (uint32_t l = 0; l < key.main_count + key.aux_count; ++l) {
        auto q = l < key.main_count ? t.params().mainPrimes()[l] : t.params().auxPrimes()[l - key.main_count];
        for (size_t i = 0; i < t.params().degree(); ++i)
          v.push_back(modarith::shoupFactor(key.digits[d].poly(p).limb(l)[i], q));
      }
  return v;
}
std::vector<uint64_t> cipherProducts(ConstCiphertextView c, const EvaluationTables &t) {
  std::vector<uint64_t> v;
  for (uint32_t p = 0; p < 2; ++p)
    for (uint32_t l = 0; l < c.layout.poly.basis.main_end; ++l)
      for (size_t i = 0; i < t.params().degree(); ++i)
        v.push_back(modarith::shoupFactor(c.poly(p).limb(l)[i], t.params().mainPrimes()[l]));
  return v;
}
uint32_t reverse(uint32_t i, uint32_t bits) {
  uint32_t out = 0;
  for (uint32_t b = 0; b < bits; ++b)
    out |= ((i >> b) & 1U) << (bits - b - 1);
  return out;
}
std::vector<uint32_t> rotationMap(uint64_t g, const EvaluationTables &t) {
  std::vector<uint32_t> v(t.params().degree());
  auto bits = t.params().logDegree();
  for (size_t i = 0; i < v.size(); ++i)
    v[i] = reverse(
      static_cast<uint32_t>(((g * (2 * reverse(static_cast<uint32_t>(i), bits) + 1)) % (2 * v.size()) - 1) / 2), bits);
  return v;
}
void equal(ConstCiphertextView a, ConstCiphertextView b, const EvaluationTables &t) {
  ASSERT_EQ(a.layout, b.layout);
  for (uint32_t p = 0; p < 2; ++p)
    for (uint32_t l = 0; l < a.layout.poly.basis.main_end; ++l)
      EXPECT_TRUE(std::equal(a.poly(p).limb(l), a.poly(p).limb(l) + t.params().degree(), b.poly(p).limb(l)))
        << p << ":" << l;
}
// Catches incorrect quotient indexing, missing canonical subtraction, or any
// changed fused rounding. The reference uses the original generic arithmetic.
TEST(PreparedArithmetic, ProductsMatchEveryResidueAtAllLevelsAndDigitCounts) {
  for (auto d : {1U, 2U, 4U})
    for (bool wide : {false, true}) {
      auto t = EvaluationTables::create(parameters(d, wide));
      ASSERT_TRUE(t);
      SyntheticKey key(*t);
      auto kp = keyProducts(key.view, *t);
      Arena arena(region::scratchBytes(t->params()));
      for (uint32_t level = 1; level < t->params().levelCount(); ++level) {
        Cipher a(*t, level, 2, 17), b(*t, level, 2, 31), out(*t, level, 2), ref(*t, level, 2);
        fill(a, *t, 73);
        fill(b, *t, 119);
        auto cp = cipherProducts(asConst(a.view), *t);
        ASSERT_TRUE(region::multiply(ref.view, asConst(a.view), asConst(b.view), key.view, *t, arena));
        for (uint32_t flags = 0; flags < 4; ++flags) {
          ASSERT_TRUE(region::multiplyCached(
            out.view, asConst(a.view), asConst(b.view), key.view,
            flags & 1 ? region::KeyProducts{kp.data(), kp.size()} : region::KeyProducts{},
            flags & 2 ? region::CipherProducts{cp.data(), cp.size()} : region::CipherProducts{}, *t, arena));
          equal(asConst(out.view), asConst(ref.view), *t);
          EXPECT_EQ(arena.used(), 0U);
        }
      }
    }
}
TEST(PreparedArithmetic, SignedRotationsAndIdentityMatchAllResidues) {
  for (auto d : {1U, 2U, 4U}) {
    auto t = EvaluationTables::create(parameters(d, true));
    ASSERT_TRUE(t);
    Arena arena(region::scratchBytes(t->params()));
    const uint64_t elements[]{eval::galoisElement(-1, t->params().logDegree()), 1,
                              eval::galoisElement(4, t->params().logDegree()), 2 * t->params().degree() - 1};
    for (auto g : elements) {
      SyntheticKey key(*t, SwitchKeyKind::Galois, g);
      auto kp = keyProducts(key.view, *t);
      auto map = rotationMap(g, *t);
      for (uint32_t level = 0; level < t->params().levelCount(); ++level) {
        Cipher a(*t, level, 2), b(*t, level, 2), out(*t, level, 2), ref(*t, level, 2);
        fill(a, *t, 93);
        fill(b, *t, 113);
        region::LinearTerm terms[] = {{asConst(a.view), 3}, {asConst(b.view), -7}};
        ASSERT_TRUE(region::rotateAdd(ref.view, asConst(a.view), g, key.view, terms, 2, *t, arena));
        for (uint32_t flags = 0; flags < 4; ++flags) {
          ASSERT_TRUE(region::rotateAddCached(
            out.view, asConst(a.view), g, key.view,
            flags & 1 ? region::KeyProducts{kp.data(), kp.size()} : region::KeyProducts{},
            flags & 2 ? region::RotationMap{map.data(), map.size()} : region::RotationMap{}, terms, 2, *t, arena));
          equal(asConst(out.view), asConst(ref.view), *t);
          EXPECT_EQ(arena.used(), 0U);
        }
      }
    }
  }
}
TEST(PreparedArithmetic, MalformedViewsAndScratchFailBeforeDestinationWrites) {
  auto t = EvaluationTables::create(parameters(2));
  ASSERT_TRUE(t);
  Cipher a(*t, 3, 2), b(*t, 3, 2), out(*t, 3, 2);
  fill(a, *t, 17);
  fill(b, *t, 91);
  fill(out, *t, 49);
  SyntheticKey key(*t);
  auto kp = keyProducts(key.view, *t);
  auto cp = cipherProducts(asConst(a.view), *t);
  Arena arena(region::scratchBytes(t->params()));
  Arena tiny(64);
  auto before = std::vector<uint64_t>(out.slab.data(), out.slab.data() + out.slab.capacityWords());
  auto layout = out.view.layout;
  for (auto bad : {region::KeyProducts{nullptr, kp.size()}, region::KeyProducts{kp.data(), 0},
                   region::KeyProducts{kp.data(), kp.size() - 1}, region::KeyProducts{out.slab.data(), kp.size()},
                   region::KeyProducts{kp.data(), std::numeric_limits<size_t>::max()}})
    EXPECT_FALSE(region::multiplyCached(out.view, asConst(a.view), asConst(b.view), key.view, bad, {}, *t, arena));
  for (auto bad :
       {region::CipherProducts{nullptr, cp.size()}, region::CipherProducts{cp.data(), 0},
        region::CipherProducts{cp.data(), cp.size() - 1}, region::CipherProducts{out.slab.data(), cp.size()}})
    EXPECT_FALSE(region::multiplyCached(out.view, asConst(a.view), asConst(b.view), key.view, {}, bad, *t, arena));
  EXPECT_FALSE(region::multiplyCached(out.view, asConst(a.view), asConst(b.view), key.view, {kp.data(), kp.size()},
                                      {cp.data(), cp.size()}, *t, tiny));
  EXPECT_FALSE(region::multiplyCached(out.view, asConst(a.view), asConst(b.view), {}, {}, {}, *t, arena));
  auto g = eval::galoisElement(-1, t->params().logDegree());
  SyntheticKey rk(*t, SwitchKeyKind::Galois, g);
  auto map = rotationMap(g, *t);
  for (auto bad : {region::RotationMap{nullptr, map.size()}, region::RotationMap{map.data(), 0},
                   region::RotationMap{map.data(), map.size() - 1},
                   region::RotationMap{reinterpret_cast<uint32_t *>(out.slab.data()), map.size()}})
    EXPECT_FALSE(region::rotateAddCached(out.view, asConst(a.view), g, rk.view, {}, bad, nullptr, 0, *t, arena));
  map.back() = static_cast<uint32_t>(map.size());
  EXPECT_FALSE(region::rotateAddCached(out.view, asConst(a.view), g, rk.view, {}, {map.data(), map.size()}, nullptr, 0,
                                       *t, arena));
  EXPECT_EQ(out.view.layout, layout);
  EXPECT_TRUE(std::equal(before.begin(), before.end(), out.slab.data()));
  EXPECT_EQ(arena.used(), 0U);
}
} // namespace

#include <cstdlib>
#include <new>
namespace prepared_memory_probe {
thread_local bool enabled = false;
thread_local std::size_t allocations = 0;
} // namespace prepared_memory_probe
void *operator new(std::size_t bytes) {
  if (prepared_memory_probe::enabled)
    ++prepared_memory_probe::allocations;
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
  if (prepared_memory_probe::enabled)
    ++prepared_memory_probe::allocations;
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

TEST(PreparedArithmetic, CachedKernelsAllocateOnlyCallerWorkspace) {
  auto t = EvaluationTables::create(parameters(2));
  ASSERT_TRUE(t);
  Cipher a(*t, 3, 2), b(*t, 3, 2), out(*t, 3, 2);
  fill(a, *t, 17);
  fill(b, *t, 91);
  SyntheticKey key(*t), rk(*t, SwitchKeyKind::Galois, 5);
  auto kp = keyProducts(key.view, *t), rp = keyProducts(rk.view, *t), cp = cipherProducts(asConst(a.view), *t);
  auto map = rotationMap(5, *t);
  auto bound = region::scratchBytes(t->params());
  Arena arena(bound + Arena::kAlignment);
  auto *guard = arena.allocate<unsigned char>(1);
  ASSERT_NE(guard, nullptr);
  *guard = 57;
  auto entry = arena.used();
  region::LinearTerm term{asConst(a.view), 1};
  prepared_memory_probe::allocations = 0;
  prepared_memory_probe::enabled = true;
  bool product = region::multiplyCached(out.view, asConst(a.view), asConst(b.view), key.view, {kp.data(), kp.size()},
                                        {cp.data(), cp.size()}, *t, arena);
  bool rotation = region::rotateAddCached(out.view, asConst(a.view), 5, rk.view, {rp.data(), rp.size()},
                                          {map.data(), map.size()}, &term, 1, *t, arena);
  prepared_memory_probe::enabled = false;
  ASSERT_TRUE(product);
  ASSERT_TRUE(rotation);
  EXPECT_EQ(prepared_memory_probe::allocations, 0U);
  EXPECT_EQ(arena.used(), entry);
  EXPECT_EQ(*guard, 57);
  EXPECT_LE(arena.highWater(), arena.capacity());
}

TEST(PreparedArithmetic, InvalidKeysMetadataAndAliasesRejectBeforeAnyWrite) {
  auto t = EvaluationTables::create(parameters(2));
  ASSERT_TRUE(t);
  Cipher a(*t, 3, 2), b(*t, 3, 2), out(*t, 3, 2);
  fill(a, *t, 17);
  fill(b, *t, 91);
  fill(out, *t, 49);
  SyntheticKey key(*t), rk(*t, SwitchKeyKind::Galois, 5);
  auto kp = keyProducts(key.view, *t), rp = keyProducts(rk.view, *t), cp = cipherProducts(asConst(a.view), *t);
  auto map = rotationMap(5, *t);
  Arena arena(region::scratchBytes(t->params()));
  auto before = std::vector<uint64_t>(out.slab.data(), out.slab.data() + out.slab.capacityWords());
  auto layout = out.view.layout;
  {
    auto bad = key.view;
    bad.log_degree = 99;
    EXPECT_FALSE(region::multiplyCached(out.view, asConst(a.view), asConst(b.view), bad, {kp.data(), kp.size()},
                                        {cp.data(), cp.size()}, *t, arena));
    bad = key.view;
    bad.digit_count = std::numeric_limits<uint32_t>::max();
    EXPECT_FALSE(
      region::multiplyCached(out.view, asConst(a.view), asConst(b.view), bad, {kp.data(), kp.size()}, {}, *t, arena));
  }
  auto invalid = asConst(a.view);
  invalid.layout.poly.basis.main_end = 99;
  EXPECT_FALSE(
    region::multiplyCached(out.view, invalid, asConst(b.view), key.view, {kp.data(), kp.size()}, {}, *t, arena));
  auto short_out = out.view;
  short_out.capacity_words = 1;
  EXPECT_FALSE(region::multiplyCached(short_out, asConst(a.view), asConst(b.view), key.view, {kp.data(), kp.size()}, {},
                                      *t, arena));
  auto a_before = std::vector<uint64_t>(a.slab.data(), a.slab.data() + a.slab.capacityWords());
  EXPECT_FALSE(
    region::multiplyCached(a.view, asConst(a.view), asConst(b.view), key.view, {kp.data(), kp.size()}, {}, *t, arena));
  EXPECT_FALSE(region::rotateAddCached(a.view, asConst(a.view), 5, rk.view, {rp.data(), rp.size()},
                                       {map.data(), map.size()}, nullptr, 0, *t, arena));
  EXPECT_TRUE(std::equal(a_before.begin(), a_before.end(), a.slab.data()));
  auto key_before = std::vector<uint64_t>(key.slabs[0].data(), key.slabs[0].data() + key.slabs[0].capacityWords());
  auto key_out = out.view;
  key_out.data = key.slabs[0].data();
  key_out.capacity_words = key.slabs[0].capacityWords();
  EXPECT_FALSE(
    region::multiplyCached(key_out, asConst(a.view), asConst(b.view), key.view, {kp.data(), kp.size()}, {}, *t, arena));
  EXPECT_TRUE(std::equal(key_before.begin(), key_before.end(), key.slabs[0].data()));
  auto badkey = rk.view;
  badkey.log_degree = 99;
  EXPECT_FALSE(region::rotateAddCached(out.view, asConst(a.view), 5, badkey, {rp.data(), rp.size()},
                                       {map.data(), map.size()}, nullptr, 0, *t, arena));
  EXPECT_FALSE(
    region::rotateAddCached(out.view, asConst(a.view), 5, {}, {}, {map.data(), map.size()}, nullptr, 0, *t, arena));
  EXPECT_FALSE(
    region::rotateAddCached(out.view, asConst(a.view), 1, {}, {}, {nullptr, map.size()}, nullptr, 0, *t, arena));
  EXPECT_FALSE(region::rotateAddCached(out.view, asConst(a.view), 1, {}, {nullptr, 1}, {}, nullptr, 0, *t, arena));
  EXPECT_EQ(out.view.layout, layout);
  EXPECT_TRUE(std::equal(before.begin(), before.end(), out.slab.data()));
  ASSERT_TRUE(region::rotateAddCached(out.view, asConst(a.view), 1, {}, {}, {}, nullptr, 0, *t, arena));
  equal(asConst(out.view), asConst(a.view), *t);
}
