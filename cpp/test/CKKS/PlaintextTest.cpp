#include "CKKS/Plaintext.h"
#include "../../src/CKKS/backend/BackendInternal.h"
#include "CKKS/Backend.h"
#include "CKKS/Encoding.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <new>
#include <vector>

namespace plaintext_allocations {
thread_local bool track = false;
thread_local size_t count = 0;
thread_local size_t fail_at = 0;
void allocation() {
  if (track && ++count == fail_at)
    throw std::bad_alloc();
}
} // namespace plaintext_allocations
void *operator new(size_t size) {
  plaintext_allocations::allocation();
  if (void *p = std::malloc(size ? size : 1))
    return p;
  throw std::bad_alloc();
}
void *operator new[](size_t size) { return ::operator new(size); }
void operator delete(void *p) noexcept { std::free(p); }
void operator delete[](void *p) noexcept { ::operator delete(p); }
void operator delete(void *p, size_t) noexcept { ::operator delete(p); }
void operator delete[](void *p, size_t) noexcept { ::operator delete(p); }
void *operator new(size_t size, std::align_val_t alignment) {
  plaintext_allocations::allocation();
  void *p = nullptr;
  if (posix_memalign(&p, static_cast<size_t>(alignment), size ? size : 1) != 0)
    throw std::bad_alloc();
  return p;
}
void *operator new[](size_t size, std::align_val_t alignment) { return ::operator new(size, alignment); }
void operator delete(void *p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void *p, std::align_val_t alignment) noexcept { ::operator delete(p, alignment); }
void operator delete(void *p, size_t, std::align_val_t alignment) noexcept { ::operator delete(p, alignment); }
void operator delete[](void *p, size_t, std::align_val_t alignment) noexcept { ::operator delete(p, alignment); }

using namespace fhenomenon::ckks;
namespace {
using Context = std::unique_ptr<FhnBackendCtx, decltype(&ckks_fhn_destroy)>;
using Client = std::unique_ptr<FhnClientCtx, decltype(&ckks_fhn_client_destroy)>;
using Plaintext = std::unique_ptr<FhnPlaintext, decltype(&ckks_fhn_plaintext_free)>;
struct Buffer {
  FhnBackendCtx *ctx;
  FhnBuffer *ptr;
  explicit Buffer(FhnBackendCtx *c) : ctx(c), ptr(ckks_fhn_buffer_alloc(c)) {}
  ~Buffer() { ckks_fhn_buffer_free(ctx, ptr); }
};
struct Snapshot {
  CiphertextLayout layout;
  uint32_t slots;
  bool certified;
  std::vector<uint64_t> words;
  explicit Snapshot(const FhnBuffer *value)
    : layout(value->layout), slots(value->logical_slots), certified(value->certified) {
    for (uint32_t c = 0; c < layout.num_polys; ++c) {
      const auto *data = value->view().poly(c).data;
      words.insert(words.end(), data, data + layout.poly.wordCount());
    }
  }
  void expectUnchanged(const FhnBuffer *value) const {
    EXPECT_EQ(value->layout, layout);
    EXPECT_EQ(value->logical_slots, slots);
    EXPECT_EQ(value->certified, certified);
    EXPECT_EQ(Snapshot(value).words, words);
  }
};
class PackedPlaintext : public ::testing::Test {
  protected:
  Context ctx{ckks_fhn_create(nullptr), ckks_fhn_destroy};
  Client client{ckks_fhn_client_create(ctx.get()), ckks_fhn_client_destroy};
  const std::array<double, 8> values{0.5, -1.25, 2, 0.125, 0.25, -0.5, 1.5, 0.75};
  const std::array<double, 8> weights{-0.75, 0.5, 0, -4, 1.25, 2, -0.125, 3};
  void encrypt(FhnBuffer *buffer) {
    ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), buffer, values.data(), values.size()), 0);
  }
  Plaintext prepare(const FhnBuffer *prototype, FhnPlaintextPurpose purpose) {
    FhnPlaintext *raw = nullptr;
    EXPECT_EQ(ckks_fhn_plaintext_encode(ctx.get(), prototype, weights.data(), weights.size(), purpose, &raw), 0);
    return Plaintext(raw, ckks_fhn_plaintext_free);
  }
  template <typename Expected> void expectPhysical(const FhnBuffer *value, Expected expected) {
    const auto &params = ctx->tables->params();
    // Trusted validation explicitly inspects every physical slot, including
    // repeated slots beyond the public logical decrypt API's requested count.
    Arena arena(keyScratchBytes(params) + encoding::scratchBytes(params) + value->layout.poly.byteCount() + 512);
    PolySlab slab(value->layout.poly.wordCount());
    RnsPolyView plain{slab.data(), slab.capacityWords(), value->layout.poly};
    ASSERT_TRUE(decrypt(plain, value->view(), *client->secret, *ctx->tables, arena));
    std::vector<double> actual(params.numSlots());
    ASSERT_TRUE(encoding::decode(actual.data(), static_cast<uint32_t>(actual.size()), asConst(plain),
                                 value->layout.scale, *ctx->tables, arena));
    for (size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected(i), 2e-5) << "physical slot " << i;
  }
};

TEST_F(PackedPlaintext, PackedMultiplyConsumesOneLevelAndPreservesEveryPhysicalSlot) {
  Buffer input(ctx.get()), output(ctx.get()), second(ctx.get());
  encrypt(input.ptr);
  encrypt(second.ptr);
  const Snapshot original(input.ptr);
  auto plain = prepare(input.ptr, FHN_PLAINTEXT_MULTIPLY);
  ASSERT_TRUE(plain);
  const auto required = ckks_fhn_plaintext_required_bytes(ctx.get(), input.ptr, FHN_PLAINTEXT_MULTIPLY);
  EXPECT_EQ(required, input.ptr->layout.poly.byteCount());
  EXPECT_EQ(ckks_fhn_plaintext_bytes(plain.get()), required);
  ASSERT_EQ(ckks_fhn_plaintext_multiply(ctx.get(), output.ptr, input.ptr, plain.get()), 0);
  EXPECT_EQ(output.ptr->layout.level(), original.layout.level() - 1);
  EXPECT_EQ(output.ptr->layout.scale, original.layout.scale);
  EXPECT_EQ(output.ptr->logical_slots, values.size());
  expectPhysical(output.ptr, [&](size_t i) { return values[i % 8] * weights[i % 8]; });
  original.expectUnchanged(input.ptr);
  // Fresh encryption has a different ciphertext identity, but can reuse weights.
  ASSERT_EQ(ckks_fhn_plaintext_multiply(ctx.get(), second.ptr, second.ptr, plain.get()), 0);
  expectPhysical(second.ptr, [&](size_t i) { return values[i % 8] * weights[i % 8]; });
}

TEST_F(PackedPlaintext, PackedAddPreservesLevelAndSupportsAliasesAtBothLevels) {
  Buffer input(ctx.get()), output(ctx.get());
  encrypt(input.ptr);
  auto add = prepare(input.ptr, FHN_PLAINTEXT_ADD);
  ASSERT_TRUE(add);
  const Snapshot original(input.ptr);
  ASSERT_EQ(ckks_fhn_plaintext_add(ctx.get(), output.ptr, input.ptr, add.get()), 0);
  EXPECT_EQ(output.ptr->layout, original.layout);
  expectPhysical(output.ptr, [&](size_t i) { return values[i % 8] + weights[i % 8]; });
  original.expectUnchanged(input.ptr);
  ASSERT_EQ(ckks_fhn_plaintext_add(ctx.get(), input.ptr, input.ptr, add.get()), 0);
  expectPhysical(input.ptr, [&](size_t i) { return values[i % 8] + weights[i % 8]; });
  auto multiply = prepare(output.ptr, FHN_PLAINTEXT_MULTIPLY);
  ASSERT_TRUE(multiply);
  ASSERT_EQ(ckks_fhn_plaintext_multiply(ctx.get(), output.ptr, output.ptr, multiply.get()), 0);
  auto lower_add = prepare(output.ptr, FHN_PLAINTEXT_ADD);
  ASSERT_TRUE(lower_add);
  const auto lower_layout = output.ptr->layout;
  ASSERT_EQ(ckks_fhn_plaintext_add(ctx.get(), output.ptr, output.ptr, lower_add.get()), 0);
  EXPECT_EQ(output.ptr->layout, lower_layout);
  expectPhysical(output.ptr,
                 [&](size_t i) { return (values[i % 8] + weights[i % 8]) * weights[i % 8] + weights[i % 8]; });
}

TEST_F(PackedPlaintext, FullWidthDistinctPublicWeightsAreNotRepeatedOrTruncated) {
  Buffer input(ctx.get());
  const auto slots = ctx->tables->params().numSlots();
  std::vector<double> x(slots), w(slots);
  for (size_t i = 0; i < slots; ++i) {
    x[i] = static_cast<double>(i % 37) / 19 - 1;
    w[i] = static_cast<double>((i * 13) % 89) / 43 - 1;
  }
  ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), input.ptr, x.data(), slots), 0);
  FhnPlaintext *raw = nullptr;
  ASSERT_EQ(ckks_fhn_plaintext_encode(ctx.get(), input.ptr, w.data(), slots, FHN_PLAINTEXT_MULTIPLY, &raw), 0);
  Plaintext plain(raw, ckks_fhn_plaintext_free);
  ASSERT_EQ(ckks_fhn_plaintext_multiply(ctx.get(), input.ptr, input.ptr, plain.get()), 0);
  expectPhysical(input.ptr, [&](size_t i) { return x[i] * w[i]; });
}

TEST_F(PackedPlaintext, InvalidPreparationLeavesExistingHandleAndPrototypeIntact) {
  Buffer input(ctx.get()), empty(ctx.get());
  encrypt(input.ptr);
  auto valid = prepare(input.ptr, FHN_PLAINTEXT_MULTIPLY);
  ASSERT_TRUE(valid);
  const Snapshot original(input.ptr);
  FhnPlaintext *out = valid.get();
  auto reject = [&](FhnBackendCtx *context, const FhnBuffer *prototype, const double *data, uint32_t count,
                    FhnPlaintextPurpose purpose) {
    EXPECT_NE(ckks_fhn_plaintext_encode(context, prototype, data, count, purpose, &out), 0);
    EXPECT_EQ(out, valid.get());
    original.expectUnchanged(input.ptr);
  };
  reject(nullptr, input.ptr, weights.data(), 8, FHN_PLAINTEXT_MULTIPLY);
  reject(ctx.get(), nullptr, weights.data(), 8, FHN_PLAINTEXT_MULTIPLY);
  reject(ctx.get(), empty.ptr, weights.data(), 8, FHN_PLAINTEXT_MULTIPLY);
  reject(ctx.get(), input.ptr, nullptr, 8, FHN_PLAINTEXT_MULTIPLY);
  for (uint32_t count : {0u, 3u, 4u, 16u})
    reject(ctx.get(), input.ptr, weights.data(), count, FHN_PLAINTEXT_MULTIPLY);
  reject(ctx.get(), input.ptr, weights.data(), 8, static_cast<FhnPlaintextPurpose>(7));
  for (double value : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(), 0x1p63}) {
    auto invalid = weights;
    invalid.back() = value;
    reject(ctx.get(), input.ptr, invalid.data(), 8, FHN_PLAINTEXT_MULTIPLY);
  }
  EXPECT_NE(ckks_fhn_plaintext_encode(ctx.get(), input.ptr, weights.data(), 8, FHN_PLAINTEXT_ADD, nullptr), 0);
  EXPECT_EQ(ckks_fhn_plaintext_required_bytes(nullptr, input.ptr, FHN_PLAINTEXT_ADD), 0u);
  EXPECT_EQ(ckks_fhn_plaintext_required_bytes(ctx.get(), empty.ptr, FHN_PLAINTEXT_ADD), 0u);
  EXPECT_EQ(ckks_fhn_plaintext_required_bytes(ctx.get(), input.ptr, static_cast<FhnPlaintextPurpose>(7)), 0u);
  EXPECT_EQ(ckks_fhn_plaintext_bytes(nullptr), 0u);
  ckks_fhn_plaintext_free(nullptr);
}

TEST_F(PackedPlaintext, IncompatibleContextLevelScaleSlotsAndPurposeRejectBeforeMutation) {
  Context other(ckks_fhn_create(nullptr), ckks_fhn_destroy);
  Buffer input(ctx.get()), output(ctx.get()), foreign(other.get());
  encrypt(input.ptr);
  encrypt(output.ptr);
  auto multiply = prepare(input.ptr, FHN_PLAINTEXT_MULTIPLY);
  auto add = prepare(input.ptr, FHN_PLAINTEXT_ADD);
  ASSERT_TRUE(multiply);
  ASSERT_TRUE(add);
  const Snapshot original(input.ptr), destination(output.ptr);
  auto reject = [&](FhnBackendCtx *context, FhnBuffer *out, const FhnBuffer *in, const FhnPlaintext *plain,
                    FhnPlaintextApplyFn apply) {
    const Snapshot source(in);
    EXPECT_NE(apply(context, out, in, plain), 0);
    source.expectUnchanged(in);
    destination.expectUnchanged(output.ptr);
  };
  reject(nullptr, output.ptr, input.ptr, multiply.get(), ckks_fhn_plaintext_multiply);
  reject(other.get(), output.ptr, input.ptr, multiply.get(), ckks_fhn_plaintext_multiply);
  reject(ctx.get(), foreign.ptr, input.ptr, multiply.get(), ckks_fhn_plaintext_multiply);
  reject(ctx.get(), output.ptr, input.ptr, nullptr, ckks_fhn_plaintext_multiply);
  reject(ctx.get(), output.ptr, input.ptr, add.get(), ckks_fhn_plaintext_multiply);
  reject(ctx.get(), output.ptr, input.ptr, multiply.get(), ckks_fhn_plaintext_add);
  input.ptr->layout.scale *= 2;
  reject(ctx.get(), output.ptr, input.ptr, multiply.get(), ckks_fhn_plaintext_multiply);
  reject(ctx.get(), input.ptr, input.ptr, add.get(), ckks_fhn_plaintext_add);
  input.ptr->layout.scale /= 2;
  input.ptr->logical_slots /= 2;
  reject(ctx.get(), output.ptr, input.ptr, multiply.get(), ckks_fhn_plaintext_multiply);
  input.ptr->logical_slots *= 2;
  --input.ptr->layout.poly.basis.main_end;
  reject(ctx.get(), output.ptr, input.ptr, multiply.get(), ckks_fhn_plaintext_multiply);
  ++input.ptr->layout.poly.basis.main_end;
  original.expectUnchanged(input.ptr);
  FhnPlaintext *raw = nullptr;
  EXPECT_NE(ckks_fhn_plaintext_encode(other.get(), input.ptr, weights.data(), 8, FHN_PLAINTEXT_ADD, &raw), 0);
  EXPECT_EQ(raw, nullptr);
  EXPECT_EQ(ckks_fhn_plaintext_required_bytes(other.get(), input.ptr, FHN_PLAINTEXT_ADD), 0u);
}

TEST_F(PackedPlaintext, ZeroAndIntegerWeightsConsumeLevelsAndExhaustedMultiplyIsTransactional) {
  Buffer input(ctx.get());
  encrypt(input.ptr);
  for (double scalar : {2.0, 0.0}) {
    std::array<double, 8> w;
    w.fill(scalar);
    FhnPlaintext *raw = nullptr;
    ASSERT_EQ(ckks_fhn_plaintext_encode(ctx.get(), input.ptr, w.data(), w.size(), FHN_PLAINTEXT_MULTIPLY, &raw), 0);
    Plaintext plain(raw, ckks_fhn_plaintext_free);
    const auto level = input.ptr->layout.level();
    ASSERT_EQ(ckks_fhn_plaintext_multiply(ctx.get(), input.ptr, input.ptr, plain.get()), 0);
    EXPECT_EQ(input.ptr->layout.level(), level - 1);
    expectPhysical(input.ptr, [&](size_t i) { return scalar == 0 ? 0 : values[i % 8] * 2; });
  }
  const Snapshot exhausted(input.ptr);
  FhnPlaintext *raw = nullptr;
  EXPECT_EQ(ckks_fhn_plaintext_required_bytes(ctx.get(), input.ptr, FHN_PLAINTEXT_MULTIPLY), 0u);
  EXPECT_NE(ckks_fhn_plaintext_encode(ctx.get(), input.ptr, weights.data(), 8, FHN_PLAINTEXT_MULTIPLY, &raw), 0);
  EXPECT_EQ(raw, nullptr);
  exhausted.expectUnchanged(input.ptr);
  auto add = prepare(input.ptr, FHN_PLAINTEXT_ADD);
  ASSERT_TRUE(add);
  ASSERT_EQ(ckks_fhn_plaintext_add(ctx.get(), input.ptr, input.ptr, add.get()), 0);
  EXPECT_EQ(input.ptr->layout.level(), 0);
  expectPhysical(input.ptr, [&](size_t i) { return weights[i % 8]; });
}

TEST_F(PackedPlaintext, PreparedValuesAndComputeDoNotRetainClientOrSourceIdentity) {
  Buffer source(ctx.get()), input(ctx.get()), output(ctx.get());
  encrypt(source.ptr);
  encrypt(input.ptr);
  auto plain = prepare(source.ptr, FHN_PLAINTEXT_MULTIPLY);
  ASSERT_TRUE(plain);
  ASSERT_EQ(ckks_fhn_plaintext_multiply(ctx.get(), output.ptr, input.ptr, plain.get()), 0);
  const Snapshot expected(output.ptr);
  ckks_fhn_buffer_free(ctx.get(), source.ptr);
  source.ptr = nullptr;
  client.reset();
  ASSERT_EQ(ckks_fhn_plaintext_multiply(ctx.get(), output.ptr, input.ptr, plain.get()), 0);
  expected.expectUnchanged(output.ptr);
}

TEST_F(PackedPlaintext, ComputeAndByteQueriesAllocateNothingAndPreserveActualCompatibleScale) {
  Buffer input(ctx.get()), product(ctx.get()), sum(ctx.get());
  encrypt(input.ptr);
  auto multiply = prepare(input.ptr, FHN_PLAINTEXT_MULTIPLY);
  auto add = prepare(input.ptr, FHN_PLAINTEXT_ADD);
  ASSERT_TRUE(multiply);
  ASSERT_TRUE(add);
  input.ptr->layout.scale *= 1 + 2e-13;
  plaintext_allocations::count = 0;
  plaintext_allocations::track = true;
  const auto required = ckks_fhn_plaintext_required_bytes(ctx.get(), input.ptr, FHN_PLAINTEXT_MULTIPLY);
  const auto retained = ckks_fhn_plaintext_bytes(multiply.get());
  const int m = ckks_fhn_plaintext_multiply(ctx.get(), product.ptr, input.ptr, multiply.get());
  const int a = ckks_fhn_plaintext_add(ctx.get(), sum.ptr, input.ptr, add.get());
  plaintext_allocations::track = false;
  EXPECT_EQ(m, 0);
  EXPECT_EQ(a, 0);
  EXPECT_EQ(required, retained);
  EXPECT_EQ(plaintext_allocations::count, 0u);
  EXPECT_EQ(product.ptr->layout.scale, input.ptr->layout.scale);
  EXPECT_EQ(sum.ptr->layout.scale, input.ptr->layout.scale);
  expectPhysical(product.ptr, [&](size_t i) { return values[i % 8] * weights[i % 8]; });
  expectPhysical(sum.ptr, [&](size_t i) { return values[i % 8] + weights[i % 8]; });
}

TEST_F(PackedPlaintext, AllocationAndScratchFailuresLeaveOutputAndCallerScratchIntact) {
  Buffer input(ctx.get()), output(ctx.get());
  encrypt(input.ptr);
  encrypt(output.ptr);
  auto valid = prepare(input.ptr, FHN_PLAINTEXT_MULTIPLY);
  auto add = prepare(input.ptr, FHN_PLAINTEXT_ADD);
  ASSERT_TRUE(valid);
  ASSERT_TRUE(add);
  const Snapshot original(input.ptr), destination(output.ptr);
  // Fail each durable allocation: first the handle, then its coefficient slab.
  for (size_t fail = 1; fail <= 2; ++fail) {
    FhnPlaintext *result = valid.get();
    plaintext_allocations::count = 0;
    plaintext_allocations::fail_at = fail;
    plaintext_allocations::track = true;
    const int rc = ckks_fhn_plaintext_encode(ctx.get(), input.ptr, weights.data(), 8, FHN_PLAINTEXT_ADD, &result);
    plaintext_allocations::track = false;
    plaintext_allocations::fail_at = 0;
    EXPECT_NE(rc, 0);
    EXPECT_EQ(result, valid.get());
    original.expectUnchanged(input.ptr);
  }
  {
    Arena::Scope scope(ctx->scratch);
    ASSERT_NE(ctx->scratch.allocateBytes(ctx->scratch.remaining()), nullptr);
    const auto used = ctx->scratch.used();
    FhnPlaintext *result = valid.get();
    EXPECT_NE(ckks_fhn_plaintext_encode(ctx.get(), input.ptr, weights.data(), 8, FHN_PLAINTEXT_ADD, &result), 0);
    EXPECT_EQ(result, valid.get());
    EXPECT_NE(ckks_fhn_plaintext_multiply(ctx.get(), output.ptr, input.ptr, valid.get()), 0);
    EXPECT_NE(ckks_fhn_plaintext_add(ctx.get(), input.ptr, input.ptr, add.get()), 0);
    EXPECT_EQ(ctx->scratch.used(), used);
  }
  original.expectUnchanged(input.ptr);
  destination.expectUnchanged(output.ptr);
  EXPECT_EQ(ctx->scratch.used(), 0u);
  // A non-aligned caller allocation must fit within the reported scratch bound.
  {
    Arena::Scope scope(ctx->scratch);
    ASSERT_NE(ctx->scratch.allocate<unsigned char>(1), nullptr);
    const auto used = ctx->scratch.used();
    ASSERT_EQ(ckks_fhn_plaintext_multiply(ctx.get(), output.ptr, input.ptr, valid.get()), 0);
    EXPECT_EQ(ctx->scratch.used(), used);
  }
}
} // namespace
