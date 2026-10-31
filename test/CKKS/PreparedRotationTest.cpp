#include "CKKS/PreparedRotation.h"
#include "../../src/CKKS/backend/BackendInternal.h"
#include "CKKS/Backend.h"
#include "CKKS/detail/Region.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <new>
#include <vector>
namespace preparation_allocations {
thread_local bool track = false;
thread_local size_t count = 0;
thread_local size_t fail_at = 0;
void allocation() {
  if (track && ++count == fail_at)
    throw std::bad_alloc();
}
} // namespace preparation_allocations
void *operator new(size_t size) {
  preparation_allocations::allocation();
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
  preparation_allocations::allocation();
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
using Preparation = std::unique_ptr<FhnRotationPreparation, decltype(&ckks_fhn_rotation_preparation_free)>;
using Context = std::unique_ptr<FhnBackendCtx, decltype(&ckks_fhn_destroy)>;
using Client = std::unique_ptr<FhnClientCtx, decltype(&ckks_fhn_client_destroy)>;
struct Buffer {
  FhnBackendCtx *ctx;
  FhnBuffer *ptr;
  explicit Buffer(FhnBackendCtx *c) : ctx(c), ptr(ckks_fhn_buffer_alloc(c)) {}
  ~Buffer() { ckks_fhn_buffer_free(ctx, ptr); }
};
Params parameters(uint32_t dnum) {
  Params::Spec spec;
  spec.log_degree = 10;
  spec.main_primes = {12289, 40961, 61441, 65537};
  spec.aux_primes = {786433, 1179649};
  if (dnum == 1)
    spec.aux_primes.push_back(2013265921);
  spec.dnum = dnum;
  spec.log_default_scale = 13;
  return Params::create(spec).value();
}
FhnKernelFn kernel(FhnBackendCtx *ctx, FhnOpCode op) {
  auto *table = ckks_fhn_get_kernels(ctx);
  for (uint32_t i = 0; i < table->num_kernels; ++i)
    if (table->kernels[i].opcode == op)
      return table->kernels[i].fn;
  return nullptr;
}
void prepareKeys(FhnClientCtx *client, const std::vector<int64_t> &rotations) {
  std::vector<FhnInstruction> instructions(rotations.size());
  for (size_t i = 0; i < rotations.size(); ++i) {
    instructions[i].opcode = FHN_ROTATE;
    instructions[i].params[0] = rotations[i];
  }
  FhnProgram program{};
  program.version = FHN_ABI_VERSION;
  program.num_instructions = static_cast<uint32_t>(instructions.size());
  program.instructions = instructions.data();
  ASSERT_EQ(ckks_fhn_client_prepare(client, &program), 0);
}
void equal(const FhnBuffer *a, const FhnBuffer *b) {
  ASSERT_EQ(a->layout, b->layout);
  ASSERT_EQ(a->logical_slots, b->logical_slots);
  ASSERT_EQ(a->certified, b->certified);
  for (uint32_t c = 0; c < a->layout.num_polys; ++c)
    for (uint32_t l = 0; l < a->layout.poly.basis.main_end; ++l)
      EXPECT_TRUE(std::equal(a->view().poly(c).limb(l), a->view().poly(c).limb(l) + a->owner->tables->params().degree(),
                             b->view().poly(c).limb(l)))
        << "component " << c << " limb " << l;
}
class PreparedRotation : public ::testing::Test {
  protected:
  Context ctx{ckks_fhn_create(nullptr), ckks_fhn_destroy};
  Client client{ckks_fhn_client_create(ctx.get()), ckks_fhn_client_destroy};
  void encrypt(FhnBuffer *buffer, double value) {
    ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), buffer, &value, 1), 0);
  }
};

// Replacing preparation with a pointer to the source, skipping coefficient
// signs, or moving approximate conversion before automorphism breaks this test.
TEST(PreparedRotationArithmetic, ExactResiduesAtEveryLevelAndDigitPartition) {
  for (uint32_t digits : {1u, 2u, 4u}) {
    std::string error;
    auto tables = EvaluationTables::create(parameters(digits), &error);
    ASSERT_TRUE(tables) << error;
    Context ctx(new FhnBackendCtx(std::move(tables)), ckks_fhn_destroy);
    Client client(ckks_fhn_client_create(ctx.get()), ckks_fhn_client_destroy);
    ASSERT_TRUE(client);
    const std::vector<int64_t> rotations{0, 1, -1, 7, -7, 512, -512, INT64_MIN, INT64_MAX};
    prepareKeys(client.get(), rotations);
    Buffer source(ctx.get()), out(ctx.get()), reference(ctx.get()), region_reference(ctx.get());
    const auto &p = ctx->tables->params();
    for (uint32_t level = 0; level < p.levelCount(); ++level) {
      // Canonical but independently varying full RNS residues deliberately
      // stress approximate conversion and centered sign boundaries.
      source.ptr->layout = {
        {{ctx->tables->paramsId(), 0, level + 1, 0, 0}, p.logDegree(), NttForm::Evaluation}, 8192, 2};
      source.ptr->logical_slots = 8;
      source.ptr->certified = true;
      for (uint32_t c = 0; c < 2; ++c)
        for (uint32_t l = 0; l <= level; ++l)
          for (size_t i = 0; i < p.degree(); ++i)
            source.ptr->view().poly(c).limb(l)[i] = (i * 7937 + (c + 1) * 17413 + l * 93) % p.mainPrimes()[l];
      Preparation preparation(ckks_fhn_prepare_rotation_source(ctx.get(), source.ptr),
                              ckks_fhn_rotation_preparation_free);
      ASSERT_TRUE(preparation);
      EXPECT_EQ(ckks_fhn_rotation_preparation_required_bytes(ctx.get(), source.ptr),
                ckks_fhn_rotation_preparation_bytes(preparation.get()));
      EXPECT_EQ(ckks_fhn_rotation_preparation_bytes(preparation.get()),
                3 * (level + 1) * p.degree() * sizeof(uint64_t));
      for (auto rotation : rotations) {
        SCOPED_TRACE(::testing::Message() << "digits=" << digits << " level=" << level << " rotation=" << rotation);
        const FhnBuffer *args[]{source.ptr};
        ASSERT_EQ(kernel(ctx.get(), FHN_ROTATE)(ctx.get(), reference.ptr, args, &rotation, nullptr), 0);
        const auto g = eval::galoisElement(rotation, p.logDegree());
        auto key = g == 1 ? nullptr : ctx->rotations.at(g).lock();
        auto view = region_reference.ptr->view();
        Arena scratch(region::scratchBytes(p));
        ASSERT_TRUE(region::rotateAdd(view, asConst(source.ptr->view()), g, key ? key->view() : EvaluationKeyView{},
                                      nullptr, 0, *ctx->tables, scratch));
        region_reference.ptr->layout = view.layout;
        region_reference.ptr->logical_slots = 8;
        region_reference.ptr->certified = true;
        ASSERT_EQ(ckks_fhn_rotate_prepared(ctx.get(), out.ptr, preparation.get(), rotation), 0);
        equal(out.ptr, reference.ptr);
        equal(out.ptr, region_reference.ptr);
      }
    }
  }
}

TEST_F(PreparedRotation, EncryptedDefaultProfileMatchesAtEveryLevelAndAllLogicalSlots) {
  prepareKeys(client.get(), {1, -1, 7});
  Buffer source(ctx.get()), leveled(ctx.get()), out(ctx.get()), reference(ctx.get());
  std::vector<double> values(ctx->tables->params().numSlots());
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = (static_cast<int>(i % 17) - 8) / 16.0;
  ASSERT_EQ(
    ckks_fhn_client_encrypt_vec_f64(client.get(), source.ptr, values.data(), static_cast<uint32_t>(values.size())), 0);
  for (int64_t level = 0; level <= ckks_fhn_fresh_level(ctx.get()); ++level) {
    const FhnBuffer *input[]{source.ptr};
    ASSERT_EQ(kernel(ctx.get(), FHN_LEVEL_DOWN)(ctx.get(), leveled.ptr, input, &level, nullptr), 0);
    Preparation prepared(ckks_fhn_prepare_rotation_source(ctx.get(), leveled.ptr), ckks_fhn_rotation_preparation_free);
    ASSERT_TRUE(prepared);
    for (int64_t rotation : {int64_t{0}, int64_t{1}, int64_t{-1}, int64_t{7}}) {
      const FhnBuffer *args[]{leveled.ptr};
      ASSERT_EQ(kernel(ctx.get(), FHN_ROTATE)(ctx.get(), reference.ptr, args, &rotation, nullptr), 0);
      ASSERT_EQ(ckks_fhn_rotate_prepared(ctx.get(), out.ptr, prepared.get(), rotation), 0);
      equal(out.ptr, reference.ptr);
      std::vector<double> actual(values.size());
      ASSERT_EQ(
        ckks_fhn_client_decrypt_vec_f64(client.get(), out.ptr, actual.data(), static_cast<uint32_t>(actual.size())), 0);
      for (size_t i = 0; i < values.size(); ++i) {
        const auto index = (static_cast<int64_t>(i) + rotation + static_cast<int64_t>(values.size())) %
                           static_cast<int64_t>(values.size());
        EXPECT_NEAR(actual[i], values[static_cast<size_t>(index)], 1e-6) << "level " << level << " slot " << i;
      }
    }
  }
}

TEST_F(PreparedRotation, SnapshotSurvivesSourceMutationFreeAndAliasing) {
  prepareKeys(client.get(), {1, -1});
  Buffer out(ctx.get()), reference(ctx.get());
  auto source = std::make_unique<Buffer>(ctx.get());
  const double values[]{0.5, -1.25, 2.0, 0.125};
  ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), source->ptr, values, 4), 0);
  Preparation preparation(ckks_fhn_prepare_rotation_source(ctx.get(), source->ptr), ckks_fhn_rotation_preparation_free);
  ASSERT_TRUE(preparation);
  const FhnBuffer *args[]{source->ptr};
  const int64_t rotation = 1;
  ASSERT_EQ(kernel(ctx.get(), FHN_ROTATE)(ctx.get(), reference.ptr, args, &rotation, nullptr), 0);
  encrypt(source->ptr, 99);
  ASSERT_EQ(ckks_fhn_rotate_prepared(ctx.get(), source->ptr, preparation.get(), rotation), 0);
  equal(source->ptr, reference.ptr);
  source.reset();
  ASSERT_EQ(ckks_fhn_rotate_prepared(ctx.get(), out.ptr, preparation.get(), rotation), 0);
  equal(out.ptr, reference.ptr);
  std::array<double, 4> actual{};
  ASSERT_EQ(ckks_fhn_client_decrypt_vec_f64(client.get(), out.ptr, actual.data(), 4), 0);
  for (size_t i = 0; i < actual.size(); ++i)
    EXPECT_NEAR(actual[i], values[(i + 1) % 4], 1e-6);
}

TEST_F(PreparedRotation, RejectsMissingExpiredKeysWrongContextAndScratchWithoutPublication) {
  Buffer source(ctx.get()), out(ctx.get());
  encrypt(source.ptr, 2);
  encrypt(out.ptr, 99);
  Preparation preparation(ckks_fhn_prepare_rotation_source(ctx.get(), source.ptr), ckks_fhn_rotation_preparation_free);
  ASSERT_TRUE(preparation);
  const auto before_layout = out.ptr->layout;
  const std::vector<uint64_t> before(out.ptr->slab.data(), out.ptr->slab.data() + out.ptr->slab.capacityWords());
  auto unchanged = [&] {
    EXPECT_EQ(out.ptr->layout, before_layout);
    EXPECT_EQ(out.ptr->logical_slots, 1u);
    EXPECT_TRUE(out.ptr->certified);
    EXPECT_TRUE(std::equal(before.begin(), before.end(), out.ptr->slab.data()));
  };
  EXPECT_EQ(ckks_fhn_rotate_prepared(ctx.get(), out.ptr, preparation.get(), 1), -1);
  unchanged();
  Context other(ckks_fhn_create(nullptr), ckks_fhn_destroy);
  Buffer foreign_out(other.get());
  EXPECT_EQ(ckks_fhn_rotate_prepared(ctx.get(), foreign_out.ptr, preparation.get(), 0), -1);
  EXPECT_EQ(ckks_fhn_rotate_prepared(other.get(), out.ptr, preparation.get(), 0), -1);
  EXPECT_EQ(ckks_fhn_rotate_prepared(nullptr, out.ptr, preparation.get(), 0), -1);
  EXPECT_EQ(ckks_fhn_rotate_prepared(ctx.get(), nullptr, preparation.get(), 0), -1);
  EXPECT_EQ(ckks_fhn_rotate_prepared(ctx.get(), out.ptr, nullptr, 0), -1);
  unchanged();
  prepareKeys(client.get(), {1});
  {
    Arena::Scope scope(ctx->scratch);
    ASSERT_NE(ctx->scratch.allocateBytes(ctx->scratch.remaining()), nullptr);
    const auto used = ctx->scratch.used();
    EXPECT_EQ(ckks_fhn_rotate_prepared(ctx.get(), out.ptr, preparation.get(), 1), -1);
    EXPECT_EQ(ctx->scratch.used(), used);
    unchanged();
  }
  client.reset();
  EXPECT_EQ(ckks_fhn_rotate_prepared(ctx.get(), out.ptr, preparation.get(), 1), -1);
  unchanged();
  EXPECT_EQ(ckks_fhn_rotate_prepared(ctx.get(), out.ptr, preparation.get(), 0), 0);
  equal(out.ptr, source.ptr);
}

TEST_F(PreparedRotation, RejectsInvalidUncertifiedNoncanonicalAndThreeComponentInputs) {
  Buffer source(ctx.get());
  EXPECT_EQ(ckks_fhn_rotation_preparation_required_bytes(ctx.get(), source.ptr), 0u);
  EXPECT_EQ(ckks_fhn_prepare_rotation_source(ctx.get(), source.ptr), nullptr);
  EXPECT_EQ(ckks_fhn_prepare_rotation_source(nullptr, source.ptr), nullptr);
  EXPECT_EQ(ckks_fhn_prepare_rotation_source(ctx.get(), nullptr), nullptr);
  encrypt(source.ptr, 2);
  Context other(ckks_fhn_create(nullptr), ckks_fhn_destroy);
  EXPECT_EQ(ckks_fhn_prepare_rotation_source(other.get(), source.ptr), nullptr);
  source.ptr->certified = false;
  EXPECT_EQ(ckks_fhn_rotation_preparation_required_bytes(ctx.get(), source.ptr), 0u);
  EXPECT_EQ(ckks_fhn_prepare_rotation_source(ctx.get(), source.ptr), nullptr);
  source.ptr->certified = true;
  source.ptr->slab.data()[0] = ctx->tables->params().mainPrimes()[0];
  EXPECT_EQ(ckks_fhn_rotation_preparation_required_bytes(ctx.get(), source.ptr), 0u);
  EXPECT_EQ(ckks_fhn_prepare_rotation_source(ctx.get(), source.ptr), nullptr);
  encrypt(source.ptr, 2);
  source.ptr->layout.num_polys = 3;
  EXPECT_EQ(ckks_fhn_rotation_preparation_required_bytes(ctx.get(), source.ptr), 0u);
  EXPECT_EQ(ckks_fhn_prepare_rotation_source(ctx.get(), source.ptr), nullptr);
  EXPECT_EQ(ckks_fhn_rotation_preparation_bytes(nullptr), 0u);
  ckks_fhn_rotation_preparation_free(nullptr);
}

// Estimation and reuse may not secretly allocate before the runtime's budget
// check; every construction allocation failure must leave the source intact.
TEST_F(PreparedRotation, EstimateAndConsumeAllocateNothingAndPreparationFailureIsRetryable) {
  Buffer source(ctx.get()), out(ctx.get());
  encrypt(source.ptr, 2);
  prepareKeys(client.get(), {1});
  const std::vector<uint64_t> before(source.ptr->slab.data(),
                                     source.ptr->slab.data() + source.ptr->slab.capacityWords());
  preparation_allocations::count = 0;
  preparation_allocations::track = true;
  const auto estimate = ckks_fhn_rotation_preparation_required_bytes(ctx.get(), source.ptr);
  const auto invalid_estimate = ckks_fhn_rotation_preparation_required_bytes(nullptr, source.ptr);
  preparation_allocations::track = false;
  EXPECT_EQ(preparation_allocations::count, 0u);
  EXPECT_EQ(estimate, 3u * 3u * 8192u * sizeof(uint64_t));
  EXPECT_EQ(invalid_estimate, 0u);
  bool reached_success = false;
  for (size_t fail = 1; fail <= 8; ++fail) {
    preparation_allocations::count = 0;
    preparation_allocations::fail_at = fail;
    preparation_allocations::track = true;
    auto *raw = ckks_fhn_prepare_rotation_source(ctx.get(), source.ptr);
    preparation_allocations::track = false;
    const bool injected = preparation_allocations::count >= fail;
    preparation_allocations::fail_at = 0;
    Preparation prepared(raw, ckks_fhn_rotation_preparation_free);
    EXPECT_EQ(bool(prepared), !injected);
    EXPECT_TRUE(std::equal(before.begin(), before.end(), source.ptr->slab.data()));
    if (!injected) {
      reached_success = true;
      ASSERT_TRUE(prepared);
      EXPECT_EQ(estimate, ckks_fhn_rotation_preparation_bytes(prepared.get()));
      Arena::Scope scope(ctx->scratch);
      auto *guard = ctx->scratch.allocate<unsigned char>(1);
      ASSERT_NE(guard, nullptr);
      *guard = 73;
      const auto used = ctx->scratch.used();
      preparation_allocations::count = 0;
      preparation_allocations::track = true;
      const auto result = ckks_fhn_rotate_prepared(ctx.get(), out.ptr, prepared.get(), 1);
      preparation_allocations::track = false;
      EXPECT_EQ(result, 0);
      EXPECT_EQ(preparation_allocations::count, 0u);
      EXPECT_EQ(ctx->scratch.used(), used);
      EXPECT_EQ(*guard, 73);
      EXPECT_LE(ctx->scratch.highWater(), ctx->scratch.capacity());
      break;
    }
  }
  EXPECT_TRUE(reached_success);
}
} // namespace
