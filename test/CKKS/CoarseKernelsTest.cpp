#include "CKKS/CoarseKernels.h"
#include "../../src/CKKS/backend/BackendInternal.h"
#include "CKKS/Backend.h"
#include "CKKS/Eval.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <gtest/gtest.h>
#include <memory>
#include <new>
#include <numeric>
#include <vector>
namespace coarse_allocations {
thread_local bool track = false;
thread_local size_t count = 0;
thread_local size_t fail_at = 0;
void allocation() {
  if (track && ++count == fail_at)
    throw std::bad_alloc();
}
} // namespace coarse_allocations
void *operator new(size_t size) {
  coarse_allocations::allocation();
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
  coarse_allocations::allocation();
  void *p = nullptr;
  if (posix_memalign(&p, static_cast<size_t>(alignment), size ? size : 1) != 0)
    throw std::bad_alloc();
  return p;
}
void *operator new[](size_t size, std::align_val_t a) { return ::operator new(size, a); }
void operator delete(void *p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void *p, std::align_val_t a) noexcept { ::operator delete(p, a); }
void operator delete(void *p, size_t, std::align_val_t a) noexcept { ::operator delete(p, a); }
void operator delete[](void *p, size_t, std::align_val_t a) noexcept { ::operator delete(p, a); }

using namespace fhenomenon::ckks;
namespace {
using Context = std::unique_ptr<FhnBackendCtx, decltype(&ckks_fhn_destroy)>;
using Client = std::unique_ptr<FhnClientCtx, decltype(&ckks_fhn_client_destroy)>;
struct Buffer {
  FhnBackendCtx *ctx;
  FhnBuffer *ptr;
  explicit Buffer(FhnBackendCtx *c) : ctx(c), ptr(ckks_fhn_buffer_alloc(c)) {}
  ~Buffer() { ckks_fhn_buffer_free(ctx, ptr); }
};
class CoarseKernels : public ::testing::Test {
  protected:
  Context ctx{ckks_fhn_create(nullptr), ckks_fhn_destroy};
  Client client{ckks_fhn_client_create(ctx.get()), ckks_fhn_client_destroy};
  void prepare(uint32_t width, bool multiplication = true) {
    std::vector<FhnInstruction> instructions;
    if (multiplication) {
      FhnInstruction i{};
      i.opcode = FHN_HMULT;
      instructions.push_back(i);
    }
    for (uint32_t d = width / 2; d; d /= 2) {
      FhnInstruction i{};
      i.opcode = FHN_ROTATE;
      i.params[0] = d;
      instructions.push_back(i);
    }
    FhnProgram p{};
    p.version = FHN_ABI_VERSION;
    p.instructions = instructions.data();
    p.num_instructions = static_cast<uint32_t>(instructions.size());
    ASSERT_EQ(ckks_fhn_client_prepare(client.get(), &p), 0);
  }
  int op(FhnOpCode code, FhnBuffer *out, const FhnBuffer *a, const FhnBuffer *b = nullptr, int64_t p = 0) {
    const auto *table = ckks_fhn_get_kernels(ctx.get());
    const FhnBuffer *args[]{a, b};
    std::array<int64_t, 4> ip{p, 0, 0, 0};
    std::array<double, 2> fp{};
    for (uint32_t i = 0; i < table->num_kernels; ++i)
      if (table->kernels[i].opcode == code)
        return table->kernels[i].fn(ctx.get(), out, args, ip.data(), fp.data());
    return -1;
  }
  void encrypt(FhnBuffer *out, const std::vector<double> &v) {
    ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), out, v.data(), static_cast<uint32_t>(v.size())), 0);
  }
  std::vector<double> decrypt(const FhnBuffer *in) {
    std::vector<double> v(in->logical_slots);
    EXPECT_EQ(ckks_fhn_client_decrypt_vec_f64(client.get(), in, v.data(), static_cast<uint32_t>(v.size())), 0);
    return v;
  }
  void expect(const FhnBuffer *in, const std::vector<double> &want) {
    auto v = decrypt(in);
    ASSERT_EQ(v.size(), want.size());
    for (size_t i = 0; i < v.size(); ++i)
      EXPECT_NEAR(v[i], want[i], 1e-4 + 1e-6 * std::abs(want[i])) << "slot " << i;
  }
  void ordinaryReduce(FhnBuffer *out, const FhnBuffer *in, uint32_t width) {
    ASSERT_EQ(op(FHN_ADD_CS, out, in), 0);
    Buffer tmp(ctx.get());
    for (uint32_t d = width / 2; d; d /= 2) {
      ASSERT_EQ(op(FHN_ROTATE, tmp.ptr, out, nullptr, d), 0);
      ASSERT_EQ(op(FHN_ADD_CC, out, out, tmp.ptr), 0);
    }
  }
};
// Omitting centering, multiplying the supplied sum twice, or changing the tree
// direction breaks the independent plaintext oracle and the ordinary expression.
TEST_F(CoarseKernels, EncryptedStatisticsMatchesOriginalExpressionAcrossEveryPhysicalSlot) {
  prepare(64);
  Buffer x(ctx.get()), sum(ctx.get()), square(ctx.get()), reference(ctx.get()), sum_square(ctx.get()), out(ctx.get());
  std::vector<double> values(ctx->tables->params().numSlots());
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = static_cast<double>(static_cast<int>((40 * 131 + i % 64) * 7 + 3) % 17 - 8);
  encrypt(x.ptr, values);
  ordinaryReduce(sum.ptr, x.ptr, 64);
  ASSERT_EQ(op(FHN_HMULT, square.ptr, x.ptr, x.ptr), 0);
  ordinaryReduce(reference.ptr, square.ptr, 64);
  for (unsigned i = 0; i < 6; ++i)
    ASSERT_EQ(op(FHN_ADD_CC, reference.ptr, reference.ptr, reference.ptr), 0);
  ASSERT_EQ(op(FHN_HMULT, sum_square.ptr, sum.ptr, sum.ptr), 0);
  ASSERT_EQ(op(FHN_SUB_CC, reference.ptr, reference.ptr, sum_square.ptr), 0);
  ASSERT_EQ(ckks_fhn_centered_variance(ctx.get(), out.ptr, x.ptr, sum.ptr, 64), 0);
  const double s = std::accumulate(values.begin(), values.begin() + 64, 0.0);
  const double ss = std::inner_product(values.begin(), values.begin() + 64, values.begin(), 0.0);
  const std::vector<double> want(values.size(), 64 * ss - s * s);
  expect(out.ptr, want);
  expect(reference.ptr, want);
  EXPECT_EQ(out.ptr->layout.poly, reference.ptr->layout.poly);
  EXPECT_EQ(out.ptr->layout.num_polys, reference.ptr->layout.num_polys);
  EXPECT_DOUBLE_EQ(out.ptr->layout.scale, 64 * reference.ptr->layout.scale);
}
TEST_F(CoarseKernels, EnergyCrossesOrdinaryBoundaryAndPartialReductionCompletesIt) {
  prepare(8);
  Buffer x(ctx.get()), sum(ctx.get()), energy(ctx.get()), rotated(ctx.get()), out(ctx.get());
  const std::vector<double> values{1, -2, 3, 0, -1, 2, 1, -3};
  encrypt(x.ptr, values);
  ordinaryReduce(sum.ptr, x.ptr, 8);
  ASSERT_EQ(ckks_fhn_centered_energy(ctx.get(), energy.ptr, x.ptr, sum.ptr, 8), 0);
  std::vector<double> want;
  const double s = std::accumulate(values.begin(), values.end(), 0.0);
  for (auto x : values)
    want.push_back((8 * x - s) * (8 * x - s) / 8);
  expect(energy.ptr, want);
  ASSERT_EQ(op(FHN_ROTATE, rotated.ptr, energy.ptr, nullptr, 4), 0);
  ASSERT_EQ(op(FHN_ADD_CC, energy.ptr, energy.ptr, rotated.ptr), 0);
  ASSERT_EQ(ckks_fhn_packed_reduce(ctx.get(), out.ptr, energy.ptr, 4), 0);
  expect(out.ptr, std::vector<double>(8, std::accumulate(want.begin(), want.end(), 0.0)));
}
TEST_F(CoarseKernels, PackedDotAndReductionSupportAliasingAndEveryValidLevel) {
  prepare(8);
  Buffer x(ctx.get()), y(ctx.get()), a(ctx.get()), b(ctx.get()), reference(ctx.get());
  const std::vector<double> left{1, -2, 3, 0, -1, 2, 1, -3}, right{2, -1, 1, 0, 1, 2, -1, 1};
  encrypt(x.ptr, left);
  encrypt(y.ptr, right);
  const double dot = std::inner_product(left.begin(), left.end(), right.begin(), 0.0);
  for (int64_t level = 0; level <= ckks_fhn_fresh_level(ctx.get()); ++level) {
    ASSERT_EQ(op(FHN_LEVEL_DOWN, a.ptr, x.ptr, nullptr, level), 0);
    ASSERT_EQ(op(FHN_LEVEL_DOWN, b.ptr, y.ptr, nullptr, level), 0);
    ASSERT_EQ(ckks_fhn_packed_reduce(ctx.get(), reference.ptr, a.ptr, 8), 0);
    expect(reference.ptr, std::vector<double>(8, 1));
    if (!level) {
      EXPECT_EQ(ckks_fhn_packed_dot(ctx.get(), a.ptr, a.ptr, b.ptr, 8), -1);
      continue;
    }
    ASSERT_EQ(ckks_fhn_packed_dot(ctx.get(), b.ptr, a.ptr, b.ptr, 8), 0);
    expect(b.ptr, std::vector<double>(8, dot));
    ASSERT_EQ(ckks_fhn_packed_reduce(ctx.get(), a.ptr, a.ptr, 4), 0);
    std::vector<double> partial(8);
    for (size_t i = 0; i < 8; ++i)
      for (size_t j = 0; j < 4; ++j)
        partial[i] += left[(i + j) % 8];
    expect(a.ptr, partial);
  }
}
TEST_F(CoarseKernels, ConstantAndZeroVectorsPreserveAbsoluteAccuracyAndRejectLowLevels) {
  prepare(64);
  Buffer x(ctx.get()), a(ctx.get()), sum(ctx.get()), out(ctx.get()), square(ctx.get()), reference(ctx.get()),
    sum_square(ctx.get());
  for (double value : {0.0, 1.0, -8.0}) {
    encrypt(x.ptr, std::vector<double>(64, value));
    for (int64_t level = 1; level <= ckks_fhn_fresh_level(ctx.get()); ++level) {
      ASSERT_EQ(op(FHN_LEVEL_DOWN, a.ptr, x.ptr, nullptr, level), 0);
      ordinaryReduce(sum.ptr, a.ptr, 64);
      if (level < 2) {
        encrypt(out.ptr, {99});
        const auto layout = out.ptr->layout;
        const std::vector<uint64_t> before(out.ptr->slab.data(), out.ptr->slab.data() + out.ptr->slab.capacityWords());
        EXPECT_EQ(ckks_fhn_centered_energy(ctx.get(), out.ptr, a.ptr, sum.ptr, 64), -1);
        EXPECT_EQ(ckks_fhn_centered_variance(ctx.get(), out.ptr, a.ptr, sum.ptr, 64), -1);
        EXPECT_EQ(out.ptr->layout, layout);
        EXPECT_TRUE(std::equal(before.begin(), before.end(), out.ptr->slab.data()));
        continue;
      }
      ASSERT_EQ(ckks_fhn_centered_variance(ctx.get(), out.ptr, a.ptr, sum.ptr, 64), 0);
      SCOPED_TRACE(::testing::Message() << "constant=" << value << " level=" << level);
      ASSERT_EQ(op(FHN_HMULT, square.ptr, a.ptr, a.ptr), 0);
      ordinaryReduce(reference.ptr, square.ptr, 64);
      for (unsigned i = 0; i < 6; ++i)
        ASSERT_EQ(op(FHN_ADD_CC, reference.ptr, reference.ptr, reference.ptr), 0);
      ASSERT_EQ(op(FHN_HMULT, sum_square.ptr, sum.ptr, sum.ptr), 0);
      ASSERT_EQ(op(FHN_SUB_CC, reference.ptr, reference.ptr, sum_square.ptr), 0);
      auto actual = decrypt(out.ptr), original = decrypt(reference.ptr);
      double actual_max = 0, original_max = 0;
      for (size_t i = 0; i < actual.size(); ++i) {
        actual_max = std::max(actual_max, std::abs(actual[i]));
        original_max = std::max(original_max, std::abs(original[i]));
      }
      std::cout << "constant=" << value << " level=" << level << " coarse_max=" << actual_max
                << " original_max=" << original_max << '\n';
      expect(out.ptr, std::vector<double>(64, 0));
      expect(reference.ptr, std::vector<double>(64, 0));
    }
  }
}
TEST_F(CoarseKernels, InvalidInputsKeysAndScratchNeverPublishAndIdentityNeedsNoKeys) {
  Buffer x(ctx.get()), sum(ctx.get()), out(ctx.get());
  encrypt(x.ptr, {1, -2, 3, 0});
  encrypt(sum.ptr, {2, 2, 2, 2});
  encrypt(out.ptr, {99});
  const auto layout = out.ptr->layout;
  const std::vector<uint64_t> bytes(out.ptr->slab.data(), out.ptr->slab.data() + out.ptr->slab.capacityWords());
  auto unchanged = [&] {
    EXPECT_EQ(out.ptr->layout, layout);
    EXPECT_EQ(out.ptr->logical_slots, 1u);
    EXPECT_TRUE(out.ptr->certified);
    EXPECT_TRUE(std::equal(bytes.begin(), bytes.end(), out.ptr->slab.data()));
  };
  EXPECT_EQ(ckks_fhn_centered_variance(ctx.get(), out.ptr, x.ptr, sum.ptr, 4), -1);
  unchanged();
  prepare(4);
  for (uint32_t width : {0u, 3u, 8u}) {
    EXPECT_EQ(ckks_fhn_centered_energy(ctx.get(), out.ptr, x.ptr, sum.ptr, width), -1);
    EXPECT_EQ(ckks_fhn_centered_variance(ctx.get(), out.ptr, x.ptr, sum.ptr, width), -1);
    EXPECT_EQ(ckks_fhn_packed_dot(ctx.get(), out.ptr, x.ptr, sum.ptr, width), -1);
    EXPECT_EQ(ckks_fhn_packed_reduce(ctx.get(), out.ptr, x.ptr, width), -1);
    unchanged();
  }
  Context other(ckks_fhn_create(nullptr), ckks_fhn_destroy);
  EXPECT_EQ(ckks_fhn_packed_reduce(other.get(), out.ptr, x.ptr, 4), -1);
  EXPECT_EQ(ckks_fhn_packed_reduce(nullptr, out.ptr, x.ptr, 4), -1);
  EXPECT_EQ(ckks_fhn_packed_reduce(ctx.get(), nullptr, x.ptr, 4), -1);
  EXPECT_EQ(ckks_fhn_packed_reduce(ctx.get(), out.ptr, nullptr, 4), -1);
  sum.ptr->layout.scale *= 2;
  EXPECT_EQ(ckks_fhn_centered_energy(ctx.get(), out.ptr, x.ptr, sum.ptr, 4), -1);
  sum.ptr->layout.scale /= 2;
  x.ptr->certified = false;
  EXPECT_EQ(ckks_fhn_packed_reduce(ctx.get(), out.ptr, x.ptr, 4), -1);
  x.ptr->certified = true;
  const auto saved = x.ptr->slab.data()[0];
  x.ptr->slab.data()[0] = ctx->tables->params().mainPrimes()[0];
  EXPECT_EQ(ckks_fhn_packed_reduce(ctx.get(), out.ptr, x.ptr, 4), -1);
  x.ptr->slab.data()[0] = saved;
  unchanged();
  {
    Arena::Scope scope(ctx->scratch);
    ASSERT_NE(ctx->scratch.allocateBytes(ctx->scratch.remaining()), nullptr);
    const auto used = ctx->scratch.used();
    EXPECT_EQ(ckks_fhn_centered_variance(ctx.get(), out.ptr, x.ptr, sum.ptr, 4), -1);
    EXPECT_EQ(ctx->scratch.used(), used);
    unchanged();
  }
  client.reset();
  EXPECT_EQ(ckks_fhn_packed_reduce(ctx.get(), out.ptr, x.ptr, 4), -1);
  EXPECT_EQ(ckks_fhn_centered_energy(ctx.get(), out.ptr, x.ptr, sum.ptr, 1), -1);
  unchanged();
  ASSERT_EQ(ckks_fhn_packed_reduce(ctx.get(), out.ptr, x.ptr, 1), 0);
  EXPECT_EQ(out.ptr->layout, x.ptr->layout);
}
// An incidental vector allocation in a coarse stage breaks the caller-owned
// scratch contract; all four functions must execute real work without one.
TEST_F(CoarseKernels, SuccessfulCallsAllocateNothingAndWidthOneMatchesPointwiseProducts) {
  prepare(8);
  Buffer x(ctx.get()), sum(ctx.get()), energy(ctx.get()), variance(ctx.get()), dot(ctx.get()), reduced(ctx.get());
  encrypt(x.ptr, {1, -2, 3, 0, -1, 2, 1, -3});
  ordinaryReduce(sum.ptr, x.ptr, 8);
  Arena::Scope scope(ctx->scratch);
  auto *guard = ctx->scratch.allocate<unsigned char>(1);
  ASSERT_NE(guard, nullptr);
  *guard = 73;
  const auto used = ctx->scratch.used();
  coarse_allocations::count = 0;
  coarse_allocations::track = true;
  const std::array<int, 4> results{ckks_fhn_centered_energy(ctx.get(), energy.ptr, x.ptr, sum.ptr, 8),
                                   ckks_fhn_centered_variance(ctx.get(), variance.ptr, x.ptr, sum.ptr, 8),
                                   ckks_fhn_packed_dot(ctx.get(), dot.ptr, x.ptr, x.ptr, 8),
                                   ckks_fhn_packed_reduce(ctx.get(), reduced.ptr, x.ptr, 8)};
  coarse_allocations::track = false;
  for (auto result : results)
    ASSERT_EQ(result, 0);
  EXPECT_EQ(coarse_allocations::count, 0u);
  EXPECT_EQ(ctx->scratch.used(), used);
  EXPECT_EQ(*guard, 73);
  EXPECT_LE(ctx->scratch.highWater(), ctx->scratch.capacity());
  ASSERT_EQ(ckks_fhn_packed_dot(ctx.get(), dot.ptr, x.ptr, x.ptr, 1), 0);
  expect(dot.ptr, {1, 4, 9, 0, 1, 4, 1, 9});
  ASSERT_EQ(ckks_fhn_centered_variance(ctx.get(), x.ptr, x.ptr, x.ptr, 1), 0);
  expect(x.ptr, std::vector<double>(8, 0));
}
TEST_F(CoarseKernels, LateMissingRotationAndMalformedKeyLeaveAliasedInputUntouched) {
  prepare(4);
  Buffer x(ctx.get()), sum(ctx.get());
  encrypt(x.ptr, {1, -2, 3, 0});
  ordinaryReduce(sum.ptr, x.ptr, 4);
  const auto layout = x.ptr->layout;
  const std::vector<uint64_t> before(x.ptr->slab.data(), x.ptr->slab.data() + x.ptr->slab.capacityWords());
  const auto g = eval::galoisElement(1, ctx->tables->params().logDegree());
  auto saved = ctx->rotations.at(g);
  ctx->rotations.erase(g);
  EXPECT_EQ(ckks_fhn_centered_variance(ctx.get(), x.ptr, x.ptr, sum.ptr, 4), -1);
  EXPECT_EQ(x.ptr->layout, layout);
  EXPECT_TRUE(std::equal(before.begin(), before.end(), x.ptr->slab.data()));
  ctx->rotations.emplace(g, saved);
  ctx->rotations[g] = ctx->relin;
  EXPECT_EQ(ckks_fhn_packed_reduce(ctx.get(), x.ptr, x.ptr, 4), -1);
  EXPECT_TRUE(std::equal(before.begin(), before.end(), x.ptr->slab.data()));
}

// A large common offset must not turn a tiny variance into first-order
// key-switch/encoding noise, nor should the divided scale be hidden.
TEST_F(CoarseKernels, NearConstantFreshInputsMeetOriginalOracleAtEveryPhysicalSlot) {
  prepare(64);
  Buffer x(ctx.get()), sum(ctx.get()), out(ctx.get());
  std::vector<double> values(ctx->tables->params().numSlots());
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = 8.0 - static_cast<double>(i % 7) / 1000.0;
  // Repeat the same 64-slot dataset across the complete physical vector.
  for (size_t i = 64; i < values.size(); ++i)
    values[i] = values[i % 64];
  encrypt(x.ptr, values);
  ordinaryReduce(sum.ptr, x.ptr, 64);
  ASSERT_EQ(ckks_fhn_centered_variance(ctx.get(), out.ptr, x.ptr, sum.ptr, 64), 0);
  const double s = std::accumulate(values.begin(), values.begin() + 64, 0.0);
  const double ss = std::inner_product(values.begin(), values.begin() + 64, values.begin(), 0.0);
  expect(out.ptr, std::vector<double>(values.size(), 64 * ss - s * s));
}

} // namespace
