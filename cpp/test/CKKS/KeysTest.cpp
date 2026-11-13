#include "CKKS/Keys.h"
#include "CKKS/Arena.h"
#include "CKKS/Encoding.h"
#include "CKKS/Eval.h"
#include "CKKS/EvalTables.h"
#include "CKKS/ModArith.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>
// Allocation and destruction probes belong only to this test executable.
namespace memory_probe {
thread_local bool count = false;
thread_local std::size_t allocations = 0;
thread_local const void *secret = nullptr;
thread_local std::size_t secret_bytes = 0;
thread_local bool wiped = false;
void check(void *p) noexcept {
  if (p != secret || secret == nullptr)
    return;
  const auto *bytes = static_cast<const unsigned char *>(p);
  wiped = true;
  for (std::size_t i = 0; i < secret_bytes; ++i)
    if (bytes[i] != 0)
      wiped = false;
  secret = nullptr;
}
} // namespace memory_probe
void *operator new(std::size_t bytes) {
  if (memory_probe::count)
    ++memory_probe::allocations;
  if (void *p = std::malloc(bytes == 0 ? 1 : bytes))
    return p;
  throw std::bad_alloc();
}
void *operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void *p) noexcept {
  memory_probe::check(p);
  std::free(p);
}
void operator delete[](void *p) noexcept { ::operator delete(p); }
void operator delete(void *p, std::size_t) noexcept { ::operator delete(p); }
void operator delete[](void *p, std::size_t) noexcept { ::operator delete(p); }
void *operator new(std::size_t bytes, std::align_val_t alignment) {
  if (memory_probe::count)
    ++memory_probe::allocations;
  void *p = nullptr;
  if (posix_memalign(&p, static_cast<std::size_t>(alignment), bytes == 0 ? 1 : bytes) != 0)
    throw std::bad_alloc();
  return p;
}
void *operator new[](std::size_t bytes, std::align_val_t alignment) { return ::operator new(bytes, alignment); }
void operator delete(void *p, std::align_val_t) noexcept {
  memory_probe::check(p);
  std::free(p);
}
void operator delete[](void *p, std::align_val_t alignment) noexcept { ::operator delete(p, alignment); }
void operator delete(void *p, std::size_t, std::align_val_t alignment) noexcept { ::operator delete(p, alignment); }
void operator delete[](void *p, std::size_t, std::align_val_t alignment) noexcept { ::operator delete(p, alignment); }

using namespace fhenomenon::ckks;
namespace {
// Deterministic randomness exists only in tests. These small-N tests check
// algebra and integration, and make no security claim for their parameters.
class TestRandom final : public RandomSource {
  public:
  explicit TestRandom(uint64_t seed = 87493827) : state_(seed) {}
  bool fill(uint8_t *out, std::size_t bytes) override {
    if (fail)
      return false;
    for (std::size_t i = 0; i < bytes; ++i) {
      state_ ^= state_ << 13;
      state_ ^= state_ >> 7;
      state_ ^= state_ << 17;
      out[i] = static_cast<uint8_t>(state_ >> 32);
    }
    return true;
  }
  bool fail = false;

  private:
  uint64_t state_;
};
uint64_t primeBelow(uint32_t bits, uint32_t skip = 0) {
  uint64_t p = (uint64_t{1} << bits) - 2047;
  while (!modarith::isPrime(p) || skip-- != 0) {
    p -= 2048;
    while (!modarith::isPrime(p))
      p -= 2048;
  }
  return p;
}
Params profile() {
  Params::Spec spec;
  spec.log_degree = 10;
  spec.main_primes = {primeBelow(50), primeBelow(40), primeBelow(40, 1)};
  spec.aux_primes = {primeBelow(60)};
  spec.dnum = 3;
  spec.log_default_scale = 40;
  return Params::create(spec).value();
}
class CkksKeysTest : public ::testing::Test {
  protected:
  Params params = profile();
  std::unique_ptr<EvaluationTables> tables = EvaluationTables::create(params);
  Arena arena{8 * 1024 * 1024};
  void SetUp() override { ASSERT_TRUE(tables); }
  TestRandom random;
  PolyLayout layout() const {
    return {{ParamsId::of(params), 0, params.levelCount(), 0, 0}, params.logDegree(), NttForm::Evaluation};
  }
  CiphertextView cipher(PolySlab &slab, uint32_t polys = 2) const {
    return {slab.data(),
            slab.capacityWords(),
            params.degree() * (params.levelCount() + params.auxPrimes().size()),
            {layout(), std::ldexp(1.0, 40), polys}};
  }
  RnsPolyView poly(PolySlab &slab) const { return {slab.data(), slab.capacityWords(), layout()}; }
  std::size_t capacity() const { return 3 * params.degree() * (params.levelCount() + params.auxPrimes().size()); }
  std::vector<double> decodeCipher(ConstCiphertextView ct, const SecretKey &secret, uint32_t count = 8) {
    PolySlab plain(params.degree() * params.levelCount());
    auto view = poly(plain);
    view.layout = ct.layout.poly;
    EXPECT_TRUE(decrypt(view, ct, secret, *tables, arena));
    std::vector<double> result(count);
    EXPECT_TRUE(encoding::decode(result.data(), count, asConst(view), ct.layout.scale, *tables, arena));
    return result;
  }
};
TEST_F(CkksKeysTest, FractionalPackingRepeatsLogicalSlots) {
  PolySlab storage(params.degree() * params.levelCount());
  auto output = poly(storage);
  const double input[] = {0.125, -1.75, 2.5, 0.0, 4.125, -0.375, 1.25, 3.5};
  ASSERT_TRUE(encoding::encode(output, input, 8, std::ldexp(1.0, 40), *tables, arena));
  std::vector<double> decoded(params.numSlots());
  ASSERT_TRUE(encoding::decode(decoded.data(), static_cast<uint32_t>(decoded.size()), asConst(output),
                               std::ldexp(1.0, 40), *tables, arena));
  for (std::size_t i = 0; i < decoded.size(); ++i)
    EXPECT_NEAR(decoded[i], input[i % 8], 1e-8);
}
TEST_F(CkksKeysTest, DecodeUsesEveryActiveMainPrime) {
  PolySlab storage(params.degree() * params.levelCount());
  auto output = poly(storage);
  const double input[] = {1500.125};
  ASSERT_TRUE(encoding::encode(output, input, 1, std::ldexp(1.0, 40), *tables, arena));
  double decoded = 0;
  ASSERT_TRUE(encoding::decode(&decoded, 1, asConst(output), std::ldexp(1.0, 40), *tables, arena));
  EXPECT_NEAR(decoded, input[0], 1e-8);
}
TEST_F(CkksKeysTest, EncodingRejectsInvalidInputWithoutChangingOutput) {
  PolySlab storage(params.degree() * params.levelCount());
  auto output = poly(storage);
  std::fill_n(output.data, output.capacity_words, 77);
  double input[] = {1, 2, 3};
  EXPECT_FALSE(encoding::encode(output, input, 3, 1, *tables, arena));
  input[0] = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(encoding::encode(output, input, 1, 1, *tables, arena));
  input[0] = std::ldexp(1.0, 63);
  EXPECT_FALSE(encoding::encode(output, input, 1, 1, *tables, arena));
  Arena tiny(1);
  input[0] = 1;
  EXPECT_FALSE(encoding::encode(output, input, 1, 1, *tables, tiny));
  EXPECT_TRUE(std::all_of(output.data, output.data + output.capacity_words, [](uint64_t x) { return x == 77; }));
}
TEST_F(CkksKeysTest, EncodingRejectsCenteredModulusOverflowBeforeWriting) {
  Params::Spec spec;
  spec.log_degree = 10;
  spec.main_primes = {12289, 40961};
  spec.aux_primes = {786433};
  spec.dnum = 2;
  spec.log_default_scale = 13;
  auto configured = Params::create(spec);
  ASSERT_TRUE(configured);
  params = *configured;
  tables = EvaluationTables::create(params);
  ASSERT_TRUE(tables);
  PolySlab storage(params.degree() * params.levelCount());
  auto output = poly(storage);
  // Q0/2 floors to 6144; (12289*40961)/2 floors to 251684864.
  const double boundaries[] = {6144.0, 251684864.0};
  for (uint32_t limbs = 1; limbs <= 2; ++limbs) {
    output.layout.basis.main_end = limbs;
    for (double sign : {-1.0, 1.0}) {
      const double accepted = sign * boundaries[limbs - 1];
      double decoded = 0;
      ASSERT_TRUE(encoding::encode(output, &accepted, 1, 1.0, *tables, arena));
      ASSERT_TRUE(encoding::decode(&decoded, 1, asConst(output), 1.0, *tables, arena));
      EXPECT_DOUBLE_EQ(decoded, accepted);
      std::fill_n(output.data, output.capacity_words, 77);
      const auto before = output.layout;
      const double rejected = sign * (boundaries[limbs - 1] + 1.0);
      memory_probe::allocations = 0;
      memory_probe::count = true;
      const bool encoded = encoding::encode(output, &rejected, 1, 1.0, *tables, arena);
      memory_probe::count = false;
      EXPECT_FALSE(encoded) << "active limbs=" << limbs << " value=" << rejected;
      EXPECT_EQ(memory_probe::allocations, 0u);
      EXPECT_EQ(output.layout, before);
      EXPECT_TRUE(std::all_of(output.data, output.data + output.capacity_words, [](uint64_t x) { return x == 77; }));
    }
  }
}
TEST_F(CkksKeysTest, EncodingAcceptsInt64CoefficientsWhenModulusProductExceedsUint64) {
  // The default test chain has 130 modulus bits. Exact multiplication must
  // detect uint64 overflow instead of wrapping and spuriously rejecting input.
  PolySlab storage(params.degree() * params.levelCount());
  auto output = poly(storage);
  for (double input : {-0x1p62, 0x1p62}) {
    double decoded = 0;
    memory_probe::allocations = 0;
    memory_probe::count = true;
    const bool encoded = encoding::encode(output, &input, 1, 1.0, *tables, arena);
    memory_probe::count = false;
    ASSERT_TRUE(encoded);
    EXPECT_EQ(memory_probe::allocations, 0u);
    ASSERT_TRUE(encoding::decode(&decoded, 1, asConst(output), 1.0, *tables, arena));
    EXPECT_DOUBLE_EQ(decoded, input);
  }
}
TEST_F(CkksKeysTest, PublicEncryptionIsRandomAndWrongSecretDoesNotDecrypt) {
  auto secret = generateSecretKey(*tables, random, arena);
  ASSERT_TRUE(secret);
  auto public_key = generatePublicKey(*secret, *tables, random, arena);
  ASSERT_TRUE(public_key);
  auto wrong = generateSecretKey(*tables, random, arena);
  ASSERT_TRUE(wrong);
  PolySlab plain(params.degree() * params.levelCount());
  auto pv = poly(plain);
  const double input[] = {0.125, -1.75, 2.5, 0, 4.125, -0.375, 1.25, 3.5};
  ASSERT_TRUE(encoding::encode(pv, input, 8, std::ldexp(1.0, 40), *tables, arena));
  PolySlab first(capacity()), second(capacity());
  auto a = cipher(first), b = cipher(second);
  ASSERT_TRUE(encrypt(a, asConst(pv), std::ldexp(1.0, 40), *public_key, random, *tables, arena));
  ASSERT_TRUE(encrypt(b, asConst(pv), std::ldexp(1.0, 40), *public_key, random, *tables, arena));
  EXPECT_NE(std::memcmp(a.poly(0).data, b.poly(0).data, pv.layout.byteCount()), 0);
  auto clear = decodeCipher(asConst(a), *secret);
  for (uint32_t i = 0; i < 8; ++i)
    EXPECT_NEAR(clear[i], input[i], 1e-6);
  auto bad = decodeCipher(asConst(a), *wrong);
  EXPECT_GT(std::abs(bad[0] - input[0]), 1.0);
  // The public key remains sufficient after the originating secret is gone.
  secret.reset();
  EXPECT_TRUE(encrypt(b, asConst(pv), std::ldexp(1.0, 40), *public_key, random, *tables, arena));
}
TEST_F(CkksKeysTest, EntropyAndScratchFailuresAreClosedAndTransactional) {
  random.fail = true;
  EXPECT_FALSE(generateSecretKey(*tables, random, arena));
  random.fail = false;
  auto secret = generateSecretKey(*tables, random, arena);
  ASSERT_TRUE(secret);
  auto public_key = generatePublicKey(*secret, *tables, random, arena);
  ASSERT_TRUE(public_key);
  PolySlab plain(params.degree() * params.levelCount());
  auto pv = poly(plain);
  const double one = 1;
  ASSERT_TRUE(encoding::encode(pv, &one, 1, std::ldexp(1.0, 40), *tables, arena));
  PolySlab storage(capacity());
  auto output = cipher(storage);
  std::fill_n(output.data, output.capacity_words, 77);
  random.fail = true;
  EXPECT_FALSE(generatePublicKey(*secret, *tables, random, arena));
  EXPECT_FALSE(generateRelinearizationKey(*secret, *tables, random, arena));
  EXPECT_FALSE(encrypt(output, asConst(pv), std::ldexp(1.0, 40), *public_key, random, *tables, arena));
  random.fail = false;
  Arena tiny(1);
  EXPECT_FALSE(encrypt(output, asConst(pv), std::ldexp(1.0, 40), *public_key, random, *tables, tiny));
  EXPECT_TRUE(std::all_of(output.data, output.data + output.capacity_words, [](uint64_t x) { return x == 77; }));
  EXPECT_EQ(arena.used(), 0u);
}
TEST_F(CkksKeysTest, RealTwoLevelMultiplicationAndSignedRotations) {
  auto secret = generateSecretKey(*tables, random, arena);
  ASSERT_TRUE(secret);
  auto public_key = generatePublicKey(*secret, *tables, random, arena);
  ASSERT_TRUE(public_key);
  auto relin = generateRelinearizationKey(*secret, *tables, random, arena);
  ASSERT_TRUE(relin);
  PolySlab plain(params.degree() * params.levelCount());
  auto pv = poly(plain);
  const double input[] = {0.125, -0.75, 1.5, 0.25, 1.125, -0.375, 0.5, 1.25};
  ASSERT_TRUE(encoding::encode(pv, input, 8, std::ldexp(1.0, 40), *tables, arena));
  PolySlab source(capacity()), product(capacity()), output(capacity());
  auto a = cipher(source), b = cipher(product), c = cipher(output);
  ASSERT_TRUE(encrypt(a, asConst(pv), std::ldexp(1.0, 40), *public_key, random, *tables, arena));
  ASSERT_TRUE(eval::hmult(b, asConst(a), asConst(a), relin->view(), *tables, arena));
  EXPECT_EQ(b.layout.num_polys, 2u);
  EXPECT_EQ(b.layout.level(), 1);
  auto square = decodeCipher(asConst(b), *secret);
  for (uint32_t i = 0; i < 8; ++i)
    EXPECT_NEAR(square[i], input[i] * input[i], 2e-6);
  ASSERT_TRUE(eval::hmult(c, asConst(b), asConst(b), relin->view(), *tables, arena));
  EXPECT_EQ(c.layout.level(), 0);
  auto fourth = decodeCipher(asConst(c), *secret);
  for (uint32_t i = 0; i < 8; ++i)
    EXPECT_NEAR(fourth[i], std::pow(input[i], 4), 2e-5);
  for (int64_t step : {int64_t{1}, int64_t{-1}, int64_t{0}}) {
    const uint64_t g = eval::galoisElement(step, params.logDegree());
    auto key = generateGaloisKey(*secret, g, *tables, random, arena);
    ASSERT_TRUE(key);
    c = cipher(output);
    ASSERT_TRUE(eval::rotate(c, asConst(a), g, key->view(), *tables, arena));
    auto rotated = decodeCipher(asConst(c), *secret);
    for (int64_t i = 0; i < 8; ++i)
      EXPECT_NEAR(rotated[static_cast<std::size_t>(i)], input[(i + step + 8) % 8], 2e-6);
  }
}
TEST_F(CkksKeysTest, HostScratchBoundsCoverNamedOperations) {
  Arena keys(keyScratchBytes(params));
  auto secret = generateSecretKey(*tables, random, keys);
  ASSERT_TRUE(secret);
  auto public_key = generatePublicKey(*secret, *tables, random, keys);
  ASSERT_TRUE(public_key);
  auto relin = generateRelinearizationKey(*secret, *tables, random, keys);
  ASSERT_TRUE(relin);
  auto galois = generateGaloisKey(*secret, 5, *tables, random, keys);
  ASSERT_TRUE(galois);
  Arena host(encoding::scratchBytes(params));
  PolySlab storage(params.degree() * params.levelCount());
  auto pv = poly(storage);
  const double input = 0.375;
  double output = 0;
  ASSERT_TRUE(encoding::encode(pv, &input, 1, std::ldexp(1.0, 40), *tables, host));
  ASSERT_TRUE(encoding::decode(&output, 1, asConst(pv), std::ldexp(1.0, 40), *tables, host));
  EXPECT_NEAR(output, input, 1e-8);
  EXPECT_LE(keys.highWater(), keyScratchBytes(params));
  EXPECT_LE(host.highWater(), encoding::scratchBytes(params));
}

TEST_F(CkksKeysTest, SmallSecretsAndKeyErrorsAreConsistentAcrossRnsLimbs) {
  auto secret = generateSecretKey(*tables, random, arena);
  ASSERT_TRUE(secret);
  auto pk = generatePublicKey(*secret, *tables, random, arena);
  ASSERT_TRUE(pk);
  auto ek = generateRelinearizationKey(*secret, *tables, random, arena);
  ASSERT_TRUE(ek);
  const std::size_t n = params.degree();
  std::vector<int64_t> reference(n), error_reference(n);
  std::vector<uint64_t> coefficient(n);
  const auto sv = secret->view();
  for (uint32_t limb = 0; limb < sv.layout.basis.limbCount(); ++limb) {
    const auto &nt = limb < params.levelCount() ? tables->mainNtt(limb) : tables->auxNtt(limb - params.levelCount());
    const uint64_t q = nt.modulus();
    std::copy_n(sv.limb(limb), n, coefficient.data());
    ntt::inverse(coefficient.data(), nt);
    for (std::size_t j = 0; j < n; ++j) {
      const int64_t value =
        coefficient[j] > q / 2 ? -static_cast<int64_t>(q - coefficient[j]) : static_cast<int64_t>(coefficient[j]);
      EXPECT_GE(value, -1);
      EXPECT_LE(value, 1);
      if (limb == 0)
        reference[j] = value;
      else
        EXPECT_EQ(value, reference[j]);
    }
  }
  const auto check_error = [&](ConstCiphertextView key, uint32_t digit, bool projector) {
    for (uint32_t limb = 0; limb < key.layout.poly.basis.limbCount(); ++limb) {
      const auto &nt = limb < params.levelCount() ? tables->mainNtt(limb) : tables->auxNtt(limb - params.levelCount());
      const uint64_t q = nt.modulus();
      for (std::size_t j = 0; j < n; ++j) {
        const uint64_t sum =
          (key.poly(0).limb(limb)[j] + modarith::mulMod(key.poly(1).limb(limb)[j], sv.limb(limb)[j], q)) % q;
        uint64_t message = 0;
        if (projector && limb == digit) {
          uint64_t p = 1;
          for (uint64_t factor : params.auxPrimes())
            p = modarith::mulMod(p, factor % q, q);
          message = modarith::mulMod(p, modarith::mulMod(sv.limb(limb)[j], sv.limb(limb)[j], q), q);
        }
        coefficient[j] = (sum + q - message) % q;
      }
      ntt::inverse(coefficient.data(), nt);
      for (std::size_t j = 0; j < n; ++j) {
        const int64_t value =
          coefficient[j] > q / 2 ? -static_cast<int64_t>(q - coefficient[j]) : static_cast<int64_t>(coefficient[j]);
        EXPECT_GE(value, -21);
        EXPECT_LE(value, 21);
        if (limb == 0)
          error_reference[j] = value;
        else
          EXPECT_EQ(value, error_reference[j]);
      }
    }
    EXPECT_TRUE(std::any_of(error_reference.begin(), error_reference.end(), [](int64_t x) { return x != 0; }));
  };
  check_error(pk->view(), 0, false);
  const auto ev = ek->view();
  for (uint32_t digit = 0; digit < ev.digit_count; ++digit)
    check_error(ev.digits[digit], digit, true);
}
TEST_F(CkksKeysTest, SecretStorageIsWipedBeforeRelease) {
  auto secret = generateSecretKey(*tables, random, arena);
  ASSERT_TRUE(secret);
  memory_probe::secret = secret->view().data;
  memory_probe::secret_bytes = secret->view().capacity_words * sizeof(uint64_t);
  memory_probe::wiped = false;
  secret.reset();
  EXPECT_TRUE(memory_probe::wiped);
}
TEST_F(CkksKeysTest, HostKernelsDoNotAllocateAfterPreparation) {
  auto secret = generateSecretKey(*tables, random, arena);
  ASSERT_TRUE(secret);
  auto pk = generatePublicKey(*secret, *tables, random, arena);
  ASSERT_TRUE(pk);
  PolySlab storage(params.degree() * params.levelCount()), ct(capacity()),
    decoded(params.degree() * params.levelCount());
  auto pv = poly(storage), dv = poly(decoded);
  auto cv = cipher(ct);
  const double input[] = {0.125, -0.5};
  double output[2] = {};
  memory_probe::allocations = 0;
  memory_probe::count = true;
  const bool encoded = encoding::encode(pv, input, 2, std::ldexp(1.0, 40), *tables, arena);
  const bool encrypted = encrypt(cv, asConst(pv), std::ldexp(1.0, 40), *pk, random, *tables, arena);
  const bool decrypted = decrypt(dv, asConst(cv), *secret, *tables, arena);
  const bool decoded_ok = encoding::decode(output, 2, asConst(dv), std::ldexp(1.0, 40), *tables, arena);
  memory_probe::count = false;
  EXPECT_TRUE(encoded);
  EXPECT_TRUE(encrypted);
  EXPECT_TRUE(decrypted);
  EXPECT_TRUE(decoded_ok);
  EXPECT_EQ(memory_probe::allocations, 0u);
  EXPECT_NEAR(output[0], 0.125, 1e-6);
  EXPECT_NEAR(output[1], -0.5, 1e-6);
}
TEST_F(CkksKeysTest, DifferentParameterContextsCannotShareKeys) {
  auto secret = generateSecretKey(*tables, random, arena);
  ASSERT_TRUE(secret);
  Params::Spec spec;
  spec.log_degree = 10;
  spec.main_primes = params.mainPrimes();
  spec.aux_primes = {primeBelow(60, 1)};
  spec.dnum = 3;
  spec.log_default_scale = 40;
  auto other = Params::create(spec);
  ASSERT_TRUE(other);
  auto other_tables = EvaluationTables::create(*other);
  ASSERT_TRUE(other_tables);
  EXPECT_FALSE(generatePublicKey(*secret, *other_tables, random, arena));
  EXPECT_FALSE(generateRelinearizationKey(*secret, *other_tables, random, arena));
  auto other_secret = generateSecretKey(*other_tables, random, arena);
  ASSERT_TRUE(other_secret);
  EXPECT_TRUE(generatePublicKey(*other_secret, *other_tables, random, arena));
}

TEST_F(CkksKeysTest, PublicEncryptionAndDecryptionSupportAliasedStorage) {
  auto secret = generateSecretKey(*tables, random, arena);
  ASSERT_TRUE(secret);
  auto pk = generatePublicKey(*secret, *tables, random, arena);
  ASSERT_TRUE(pk);
  PolySlab storage(capacity());
  auto pv = poly(storage);
  auto cv = cipher(storage);
  const double input[] = {-0.125, 0.75};
  double output[2]{};
  ASSERT_TRUE(encoding::encode(pv, input, 2, std::ldexp(1.0, 40), *tables, arena));
  ASSERT_TRUE(encrypt(cv, asConst(pv), std::ldexp(1.0, 40), *pk, random, *tables, arena));
  ASSERT_TRUE(decrypt(pv, asConst(cv), *secret, *tables, arena));
  ASSERT_TRUE(encoding::decode(output, 2, asConst(pv), cv.layout.scale, *tables, arena));
  EXPECT_NEAR(output[0], input[0], 1e-6);
  EXPECT_NEAR(output[1], input[1], 1e-6);
}
TEST_F(CkksKeysTest, HybridDigitsWithMultipleMainAndAuxPrimesWorkAtLowerLevels) {
  Params::Spec spec;
  spec.log_degree = 10;
  spec.main_primes = params.mainPrimes();
  spec.aux_primes = {primeBelow(60), primeBelow(60, 1)};
  spec.dnum = 2;
  spec.log_default_scale = 40;
  auto configured = Params::create(spec);
  ASSERT_TRUE(configured);
  params = *configured;
  tables = EvaluationTables::create(params);
  ASSERT_TRUE(tables);
  auto secret = generateSecretKey(*tables, random, arena);
  ASSERT_TRUE(secret);
  auto pk = generatePublicKey(*secret, *tables, random, arena);
  ASSERT_TRUE(pk);
  auto relin = generateRelinearizationKey(*secret, *tables, random, arena);
  ASSERT_TRUE(relin);
  PolySlab storage(params.degree() * params.levelCount()), source(capacity()), result(capacity());
  auto pv = poly(storage);
  auto a = cipher(source), b = cipher(result);
  const double input[] = {-0.5, 0.125, 1.25, 0.75};
  ASSERT_TRUE(encoding::encode(pv, input, 4, std::ldexp(1.0, 40), *tables, arena));
  ASSERT_TRUE(encrypt(a, asConst(pv), std::ldexp(1.0, 40), *pk, random, *tables, arena));
  ASSERT_TRUE(eval::hmult(b, asConst(a), asConst(a), relin->view(), *tables, arena));
  ASSERT_TRUE(eval::hmult(b, asConst(b), asConst(b), relin->view(), *tables, arena));
  auto clear = decodeCipher(asConst(b), *secret, 4);
  for (uint32_t i = 0; i < 4; ++i)
    EXPECT_NEAR(clear[i], std::pow(input[i], 4), 2e-5);
}
TEST_F(CkksKeysTest, RealArithmeticStagesAndRotationsUseNoHeap) {
  auto secret = generateSecretKey(*tables, random, arena);
  ASSERT_TRUE(secret);
  auto pk = generatePublicKey(*secret, *tables, random, arena);
  ASSERT_TRUE(pk);
  auto relin = generateRelinearizationKey(*secret, *tables, random, arena);
  ASSERT_TRUE(relin);
  auto rotation = generateGaloisKey(*secret, 5, *tables, random, arena);
  ASSERT_TRUE(rotation);
  auto conjugate = generateGaloisKey(*secret, 2 * params.degree() - 1, *tables, random, arena);
  ASSERT_TRUE(conjugate);
  PolySlab storage(params.degree() * params.levelCount()), source(capacity()), result(capacity()), stages(capacity());
  auto pv = poly(storage);
  auto a = cipher(source), b = cipher(result), c = cipher(stages);
  const double input[] = {-0.5, 0.125, 1.25, 0.75};
  ASSERT_TRUE(encoding::encode(pv, input, 4, std::ldexp(1.0, 40), *tables, arena));
  ASSERT_TRUE(encrypt(a, asConst(pv), std::ldexp(1.0, 40), *pk, random, *tables, arena));
  memory_probe::allocations = 0;
  memory_probe::count = true;
  const bool added = eval::add(b, asConst(a), asConst(a), *tables, arena);
  const bool tensor = eval::tensor(c, asConst(a), asConst(a), *tables, arena);
  const bool relinearized = eval::relinearize(c, asConst(c), relin->view(), *tables, arena);
  const bool rescaled = eval::rescale(c, asConst(c), *tables, arena);
  memory_probe::count = false;
  ASSERT_TRUE(added);
  ASSERT_TRUE(tensor);
  ASSERT_TRUE(relinearized);
  ASSERT_TRUE(rescaled);
  EXPECT_EQ(memory_probe::allocations, 0u);
  auto sum = decodeCipher(asConst(b), *secret, 4), square = decodeCipher(asConst(c), *secret, 4);
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(sum[i], 2 * input[i], 2e-6);
    EXPECT_NEAR(square[i], input[i] * input[i], 2e-6);
  }
  memory_probe::allocations = 0;
  memory_probe::count = true;
  const bool rotated = eval::rotate(b, asConst(a), 5, rotation->view(), *tables, arena);
  const bool conjugated = eval::rotate(c, asConst(a), 2 * params.degree() - 1, conjugate->view(), *tables, arena);
  memory_probe::count = false;
  ASSERT_TRUE(rotated);
  ASSERT_TRUE(conjugated);
  EXPECT_EQ(memory_probe::allocations, 0u);
  auto shifted = decodeCipher(asConst(b), *secret, 4), conjugated_clear = decodeCipher(asConst(c), *secret, 4);
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_NEAR(shifted[i], input[(i + 1) % 4], 2e-6);
    EXPECT_NEAR(conjugated_clear[i], input[i], 2e-6);
  }
}
TEST(CkksSystemRandomTest, FillsFromOsAndRejectsInvalidOutput) {
  SystemRandom random;
  uint8_t a[32]{}, b[32]{};
  ASSERT_TRUE(random.fill(a, sizeof(a)));
  ASSERT_TRUE(random.fill(b, sizeof(b)));
  EXPECT_NE(std::memcmp(a, b, sizeof(a)), 0);
  EXPECT_TRUE(random.fill(nullptr, 0));
  EXPECT_FALSE(random.fill(nullptr, 1));
}
static_assert(!std::is_copy_constructible_v<SecretKey>);
static_assert(!std::is_copy_constructible_v<PublicKey>);
} // namespace
