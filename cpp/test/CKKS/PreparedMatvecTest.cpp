#include "CKKS/PreparedMatvec.h"
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
namespace matvec_allocations {
thread_local bool track = false;
thread_local size_t count = 0;
thread_local size_t fail_at = 0;
void allocation() {
  if (track && ++count == fail_at)
    throw std::bad_alloc();
}
} // namespace matvec_allocations
void *operator new(size_t size) {
  matvec_allocations::allocation();
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
  matvec_allocations::allocation();
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

#include "CKKS/CoarseKernels.h"
using namespace fhenomenon::ckks;
namespace {
using Preparation = std::unique_ptr<FhnMatvecPreparation, decltype(&ckks_fhn_matvec_preparation_free)>;
using Context = std::unique_ptr<FhnBackendCtx, decltype(&ckks_fhn_destroy)>;
using Client = std::unique_ptr<FhnClientCtx, decltype(&ckks_fhn_client_destroy)>;
struct Buffer {
  FhnBackendCtx *ctx;
  FhnBuffer *ptr;
  explicit Buffer(FhnBackendCtx *c) : ctx(c), ptr(ckks_fhn_buffer_alloc(c)) {}
  ~Buffer() { ckks_fhn_buffer_free(ctx, ptr); }
};
void prepareKeys(FhnClientCtx *client, uint32_t width) {
  std::vector<FhnInstruction> instructions(1);
  instructions[0].opcode = FHN_HMULT;
  for (uint32_t d = width / 2; d; d /= 2) {
    FhnInstruction instruction{};
    instruction.opcode = FHN_ROTATE;
    instruction.params[0] = d;
    instructions.push_back(instruction);
  }
  FhnProgram p{};
  p.version = FHN_ABI_VERSION;
  p.instructions = instructions.data();
  p.num_instructions = static_cast<uint32_t>(instructions.size());
  ASSERT_EQ(ckks_fhn_client_prepare(client, &p), 0);
}
void equal(const FhnBuffer *a, const FhnBuffer *b) {
  ASSERT_EQ(a->layout, b->layout);
  ASSERT_EQ(a->logical_slots, b->logical_slots);
  ASSERT_EQ(a->certified, b->certified);
  for (uint32_t c = 0; c < a->layout.num_polys; ++c)
    for (uint32_t l = 0; l < a->layout.poly.basis.main_end; ++l)
      EXPECT_TRUE(std::equal(a->view().poly(c).limb(l), a->view().poly(c).limb(l) + a->owner->tables->params().degree(),
                             b->view().poly(c).limb(l)))
        << c << ":" << l;
}
class PreparedMatvec : public ::testing::Test {
  protected:
  Context ctx{ckks_fhn_create(nullptr), ckks_fhn_destroy};
  Client client{ckks_fhn_client_create(ctx.get()), ckks_fhn_client_destroy};
  const std::array<double, 8> values{0.5, -1.25, 2, 0.125, 0.25, -0.5, 1.5, 0.75};
  void encrypt(FhnBuffer *buffer) {
    ASSERT_EQ(ckks_fhn_client_encrypt_vec_f64(client.get(), buffer, values.data(), values.size()), 0);
  }
  Preparation prepare(FhnBuffer *source, uint32_t flags = FHN_MATVEC_PREPARE_ALL, uint32_t width = 8) {
    FhnMatvecPrepareOptions options{sizeof(FhnMatvecPrepareOptions), flags};
    return Preparation(ckks_fhn_prepare_matvec_source(ctx.get(), source, width, &options),
                       ckks_fhn_matvec_preparation_free);
  }
  int op(FhnOpCode code, FhnBuffer *out, const FhnBuffer *a, const FhnBuffer *b = nullptr, int64_t param = 0) {
    const auto *table = ckks_fhn_get_kernels(ctx.get());
    const FhnBuffer *args[]{a, b};
    const std::array<int64_t, 4> ip{param, 0, 0, 0};
    const std::array<double, 2> fp{};
    for (uint32_t i = 0; i < table->num_kernels; ++i)
      if (table->kernels[i].opcode == code)
        return table->kernels[i].fn(ctx.get(), out, args, ip.data(), fp.data());
    return -1;
  }
};
TEST_F(PreparedMatvec, FlagsMatchPackedDotAtEveryValidDefaultLevel) {
  prepareKeys(client.get(), 8);
  Buffer x(ctx.get()), row(ctx.get()), lowx(ctx.get()), lowrow(ctx.get()), out(ctx.get()), ref(ctx.get());
  encrypt(x.ptr);
  encrypt(row.ptr);
  for (int64_t level = 1; level <= ckks_fhn_fresh_level(ctx.get()); ++level) {
    ASSERT_EQ(op(FHN_LEVEL_DOWN, lowx.ptr, x.ptr, nullptr, level), 0);
    ASSERT_EQ(op(FHN_LEVEL_DOWN, lowrow.ptr, row.ptr, nullptr, level), 0);
    ASSERT_EQ(ckks_fhn_packed_dot(ctx.get(), ref.ptr, lowx.ptr, lowrow.ptr, 8), 0);
    for (uint32_t flags = 0; flags <= FHN_MATVEC_PREPARE_ALL; ++flags) {
      SCOPED_TRACE(::testing::Message() << "level=" << level << " flags=" << flags);
      auto prepared = prepare(lowx.ptr, flags);
      ASSERT_TRUE(prepared);
      FhnMatvecPrepareOptions options{sizeof(FhnMatvecPrepareOptions), flags};
      EXPECT_EQ(ckks_fhn_matvec_preparation_required_bytes(ctx.get(), lowx.ptr, 8, &options),
                ckks_fhn_matvec_preparation_bytes(prepared.get()));
      ASSERT_EQ(ckks_fhn_matvec_dot(ctx.get(), out.ptr, prepared.get(), lowrow.ptr), 0);
      equal(out.ptr, ref.ptr);
    }
  }
}

TEST_F(PreparedMatvec, ProductOrdinaryRotateAddAndPartialReduceMatchExactly) {
  prepareKeys(client.get(), 8);
  Buffer x(ctx.get()), row(ctx.get()), out(ctx.get()), ref(ctx.get()), product(ctx.get()), rotated(ctx.get());
  encrypt(x.ptr);
  encrypt(row.ptr);
  auto prepared = prepare(x.ptr);
  ASSERT_TRUE(prepared);
  ASSERT_EQ(ckks_fhn_matvec_product(ctx.get(), product.ptr, prepared.get(), row.ptr), 0);
  ASSERT_EQ(ckks_fhn_packed_dot(ctx.get(), ref.ptr, x.ptr, row.ptr, 1), 0);
  equal(product.ptr, ref.ptr);
  // The ordinary public normalized multiplication also feeds prepared reduction.
  ASSERT_EQ(op(FHN_HMULT, product.ptr, x.ptr, row.ptr), 0);
  equal(product.ptr, ref.ptr);
  ASSERT_EQ(op(FHN_ROTATE, rotated.ptr, product.ptr, nullptr, 4), 0);
  ASSERT_EQ(op(FHN_ADD_CC, product.ptr, product.ptr, rotated.ptr), 0);
  // Ordinary paths admit tiny scale drift; reduction must preserve actual metadata.
  product.ptr->layout.scale *= 1.0 + 2e-13;
  ASSERT_EQ(ckks_fhn_packed_reduce(ctx.get(), ref.ptr, product.ptr, 4), 0);
  ASSERT_EQ(ckks_fhn_matvec_reduce(ctx.get(), out.ptr, prepared.get(), product.ptr, 4), 0);
  equal(out.ptr, ref.ptr);
  ASSERT_EQ(ckks_fhn_matvec_reduce(ctx.get(), product.ptr, prepared.get(), product.ptr, 4), 0);
  equal(product.ptr, ref.ptr);
}
TEST_F(PreparedMatvec, SnapshotSurvivesSourceMutationFreeAndOutputAliasesRow) {
  prepareKeys(client.get(), 8);
  auto x = std::make_unique<Buffer>(ctx.get());
  Buffer row(ctx.get()), out(ctx.get()), ref(ctx.get());
  encrypt(x->ptr);
  encrypt(row.ptr);
  auto prepared = prepare(x->ptr);
  ASSERT_TRUE(prepared);
  ASSERT_EQ(ckks_fhn_packed_dot(ctx.get(), ref.ptr, x->ptr, row.ptr, 8), 0);
  std::fill_n(x->ptr->slab.data(), x->ptr->slab.capacityWords(), uint64_t{0});
  x.reset();
  ASSERT_EQ(ckks_fhn_matvec_dot(ctx.get(), out.ptr, prepared.get(), row.ptr), 0);
  equal(out.ptr, ref.ptr);
  ASSERT_EQ(ckks_fhn_matvec_dot(ctx.get(), row.ptr, prepared.get(), row.ptr), 0);
  equal(row.ptr, ref.ptr);
}
TEST_F(PreparedMatvec, InvalidRequestsAndScratchFailuresLeaveDestinationUntouched) {
  prepareKeys(client.get(), 8);
  Buffer x(ctx.get()), row(ctx.get()), out(ctx.get());
  encrypt(x.ptr);
  encrypt(row.ptr);
  encrypt(out.ptr);
  auto prepared = prepare(x.ptr);
  ASSERT_TRUE(prepared);
  const auto layout = out.ptr->layout;
  const auto slots = out.ptr->logical_slots;
  const std::vector<uint64_t> before(out.ptr->slab.data(), out.ptr->slab.data() + out.ptr->slab.capacityWords());
  const auto unchanged = [&] {
    EXPECT_EQ(out.ptr->layout, layout);
    EXPECT_EQ(out.ptr->logical_slots, slots);
    EXPECT_TRUE(out.ptr->certified);
    EXPECT_TRUE(std::equal(before.begin(), before.end(), out.ptr->slab.data()));
  };
  EXPECT_EQ(ckks_fhn_matvec_reduce(ctx.get(), out.ptr, prepared.get(), x.ptr, 4), -1);
  unchanged();
  for (uint32_t width : {0u, 3u, 16u})
    EXPECT_EQ(ckks_fhn_matvec_reduce(ctx.get(), out.ptr, prepared.get(), x.ptr, width), -1);
  row.ptr->layout.scale *= 2;
  EXPECT_EQ(ckks_fhn_matvec_dot(ctx.get(), out.ptr, prepared.get(), row.ptr), -1);
  row.ptr->layout.scale /= 2;
  unchanged();
  row.ptr->slab.data()[0] = ctx->tables->params().mainPrimes()[0];
  EXPECT_EQ(ckks_fhn_matvec_product(ctx.get(), out.ptr, prepared.get(), row.ptr), -1);
  encrypt(row.ptr);
  unchanged();
  Context other(ckks_fhn_create(nullptr), ckks_fhn_destroy);
  Buffer foreign(other.get());
  EXPECT_EQ(ckks_fhn_matvec_dot(other.get(), foreign.ptr, prepared.get(), row.ptr), -1);
  EXPECT_EQ(ckks_fhn_matvec_dot(ctx.get(), foreign.ptr, prepared.get(), row.ptr), -1);
  EXPECT_EQ(ckks_fhn_matvec_dot(nullptr, out.ptr, prepared.get(), row.ptr), -1);
  unchanged();
  {
    Arena::Scope scope(ctx->scratch);
    ASSERT_NE(ctx->scratch.allocateBytes(ctx->scratch.remaining()), nullptr);
    auto used = ctx->scratch.used();
    EXPECT_EQ(ckks_fhn_matvec_dot(ctx.get(), out.ptr, prepared.get(), row.ptr), -1);
    EXPECT_EQ(ctx->scratch.used(), used);
    unchanged();
  }
  auto original = std::move(out.ptr->slab);
  out.ptr->slab = PolySlab(1);
  out.ptr->slab.data()[0] = 73;
  EXPECT_EQ(ckks_fhn_matvec_dot(ctx.get(), out.ptr, prepared.get(), row.ptr), -1);
  EXPECT_EQ(out.ptr->slab.data()[0], 73u);
  out.ptr->slab = std::move(original);
  unchanged();
}
TEST_F(PreparedMatvec, ExpiredAndReplacedKeysRejectWithoutRetainingClientKeys) {
  prepareKeys(client.get(), 8);
  Buffer x(ctx.get()), row(ctx.get()), out(ctx.get());
  encrypt(x.ptr);
  encrypt(row.ptr);
  auto prepared = prepare(x.ptr);
  ASSERT_TRUE(prepared);
  auto previous = client->relin;
  client->relin = generateRelinearizationKey(*client->secret, *ctx->tables, client->random, client->scratch);
  ctx->relin = client->relin;
  ASSERT_TRUE(client->relin);
  EXPECT_EQ(ckks_fhn_matvec_dot(ctx.get(), out.ptr, prepared.get(), row.ptr), -1);
  client->relin = previous;
  ctx->relin = previous;
  previous.reset();
  ASSERT_EQ(ckks_fhn_matvec_dot(ctx.get(), out.ptr, prepared.get(), row.ptr), 0);
  auto g = eval::galoisElement(1, ctx->tables->params().logDegree());
  auto oldRotation = client->rotations.at(g);
  auto replacement = generateGaloisKey(*client->secret, g, *ctx->tables, client->random, client->scratch);
  ASSERT_TRUE(replacement);
  ctx->rotations[g] = replacement;
  EXPECT_EQ(ckks_fhn_matvec_dot(ctx.get(), out.ptr, prepared.get(), row.ptr), -1);
  ctx->rotations[g] = oldRotation;
  oldRotation.reset();
  replacement.reset();
  std::weak_ptr<const EvaluationKey> weak = client->relin;
  client.reset();
  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(ckks_fhn_matvec_product(ctx.get(), out.ptr, prepared.get(), row.ptr), -1);
}
TEST_F(PreparedMatvec, EstimateAndConsumeAllocateNothingAndSetupRollsBackEveryAllocationFailure) {
  prepareKeys(client.get(), 8);
  Buffer x(ctx.get()), row(ctx.get()), out(ctx.get()), product(ctx.get());
  encrypt(x.ptr);
  encrypt(row.ptr);
  matvec_allocations::count = 0;
  matvec_allocations::track = true;
  const auto estimate = ckks_fhn_matvec_preparation_required_bytes(ctx.get(), x.ptr, 8, nullptr);
  matvec_allocations::track = false;
  EXPECT_EQ(matvec_allocations::count, 0u);
  ASSERT_GT(estimate, 0u);
  const auto &p = ctx->tables->params();
  EXPECT_EQ(estimate, 4 * p.degree() * p.levelCount() * sizeof(uint64_t) +
                        4 * 2 * p.dnum() * p.degree() * (p.levelCount() + p.auxPrimes().size()) * sizeof(uint64_t) +
                        3 * p.degree() * sizeof(uint32_t));
  bool succeeded = false;
  for (size_t fail = 1; fail <= 8; ++fail) {
    matvec_allocations::count = 0;
    matvec_allocations::fail_at = fail;
    matvec_allocations::track = true;
    auto *raw = ckks_fhn_prepare_matvec_source(ctx.get(), x.ptr, 8, nullptr);
    matvec_allocations::track = false;
    bool injected = matvec_allocations::count >= fail;
    matvec_allocations::fail_at = 0;
    Preparation prepared(raw, ckks_fhn_matvec_preparation_free);
    EXPECT_EQ(bool(prepared), !injected);
    if (injected)
      continue;
    succeeded = true;
    ASSERT_TRUE(prepared);
    EXPECT_EQ(ckks_fhn_matvec_preparation_bytes(prepared.get()), estimate);
    Arena::Scope scope(ctx->scratch);
    auto *guard = ctx->scratch.allocate<unsigned char>(1);
    ASSERT_NE(guard, nullptr);
    *guard = 73;
    auto used = ctx->scratch.used();
    matvec_allocations::count = 0;
    matvec_allocations::track = true;
    int a = ckks_fhn_matvec_product(ctx.get(), product.ptr, prepared.get(), row.ptr);
    int b = ckks_fhn_matvec_reduce(ctx.get(), out.ptr, prepared.get(), product.ptr, 8);
    int c = ckks_fhn_matvec_dot(ctx.get(), out.ptr, prepared.get(), row.ptr);
    matvec_allocations::track = false;
    EXPECT_EQ(a, 0);
    EXPECT_EQ(b, 0);
    EXPECT_EQ(c, 0);
    EXPECT_EQ(matvec_allocations::count, 0u);
    EXPECT_EQ(ctx->scratch.used(), used);
    EXPECT_EQ(*guard, 73);
    break;
  }
  EXPECT_TRUE(succeeded);
}
TEST_F(PreparedMatvec, InvalidPreparationsAreRejectedBeforeAllocation) {
  Buffer x(ctx.get());
  encrypt(x.ptr);
  EXPECT_EQ(ckks_fhn_prepare_matvec_source(ctx.get(), x.ptr, 8, nullptr), nullptr);
  prepareKeys(client.get(), 8);
  for (uint32_t width : {0u, 3u, 16u})
    EXPECT_EQ(ckks_fhn_prepare_matvec_source(ctx.get(), x.ptr, width, nullptr), nullptr);
  for (FhnMatvecPrepareOptions options :
       {FhnMatvecPrepareOptions{0, 7}, FhnMatvecPrepareOptions{sizeof(FhnMatvecPrepareOptions), 8}})
    EXPECT_EQ(ckks_fhn_prepare_matvec_source(ctx.get(), x.ptr, 8, &options), nullptr);
  x.ptr->certified = false;
  EXPECT_EQ(ckks_fhn_prepare_matvec_source(ctx.get(), x.ptr, 8, nullptr), nullptr);
  x.ptr->certified = true;
  x.ptr->layout.num_polys = 3;
  EXPECT_EQ(ckks_fhn_prepare_matvec_source(ctx.get(), x.ptr, 8, nullptr), nullptr);
  x.ptr->layout.num_polys = 2;
  x.ptr->layout.poly.basis.main_end = 1;
  EXPECT_EQ(ckks_fhn_prepare_matvec_source(ctx.get(), x.ptr, 8, nullptr), nullptr);
  EXPECT_EQ(ckks_fhn_matvec_preparation_bytes(nullptr), 0u);
  ckks_fhn_matvec_preparation_free(nullptr);
}

TEST_F(PreparedMatvec, WidthOneAllFlagsAreExactAndScaleOverflowCannotAdmitReduction) {
  prepareKeys(client.get(), 8);
  Buffer x(ctx.get()), row(ctx.get()), out(ctx.get()), ref(ctx.get()), product(ctx.get());
  encrypt(x.ptr);
  encrypt(row.ptr);
  ASSERT_EQ(ckks_fhn_packed_dot(ctx.get(), ref.ptr, x.ptr, row.ptr, 1), 0);
  for (uint32_t flags = 0; flags <= FHN_MATVEC_PREPARE_ALL; ++flags) {
    auto prepared = prepare(x.ptr, flags, 1);
    ASSERT_TRUE(prepared);
    ASSERT_EQ(ckks_fhn_matvec_dot(ctx.get(), out.ptr, prepared.get(), row.ptr), 0);
    equal(out.ptr, ref.ptr);
    ASSERT_EQ(ckks_fhn_matvec_reduce(ctx.get(), product.ptr, prepared.get(), out.ptr, 1), 0);
    equal(product.ptr, ref.ptr);
  }
  x.ptr->layout.scale = std::numeric_limits<double>::max();
  auto overflow = prepare(x.ptr);
  ASSERT_TRUE(overflow);
  EXPECT_EQ(ckks_fhn_matvec_reduce(ctx.get(), out.ptr, overflow.get(), product.ptr, 1), -1);
  x.ptr->layout.scale = std::numeric_limits<double>::min();
  auto underflow = prepare(x.ptr);
  ASSERT_TRUE(underflow);
  EXPECT_EQ(ckks_fhn_matvec_reduce(ctx.get(), out.ptr, underflow.get(), product.ptr, 1), -1);
}
TEST_F(PreparedMatvec, CloseRowScaleFlowsThroughProductToReduceWithoutRelabeling) {
  prepareKeys(client.get(), 8);
  Buffer x(ctx.get()), row(ctx.get()), out(ctx.get()), ref(ctx.get()), product(ctx.get());
  encrypt(x.ptr);
  encrypt(row.ptr);
  auto prepared = prepare(x.ptr);
  ASSERT_TRUE(prepared);
  row.ptr->layout.scale *= 1.0 + 5e-13;
  ASSERT_EQ(ckks_fhn_packed_dot(ctx.get(), ref.ptr, x.ptr, row.ptr, 8), 0);
  ASSERT_EQ(ckks_fhn_matvec_product(ctx.get(), product.ptr, prepared.get(), row.ptr), 0);
  const auto scale = product.ptr->layout.scale;
  ASSERT_EQ(ckks_fhn_matvec_reduce(ctx.get(), out.ptr, prepared.get(), product.ptr, 8), 0);
  EXPECT_EQ(out.ptr->layout.scale, scale);
  equal(out.ptr, ref.ptr);
  row.ptr->layout.scale *= 1.0 + 2e-12;
  EXPECT_EQ(ckks_fhn_matvec_product(ctx.get(), out.ptr, prepared.get(), row.ptr), -1);
}

TEST(PreparedMatvecArithmetic, EveryDigitPartitionAndLevelMatchesEveryCanonicalResidue) {
  for (uint32_t digits : {1u, 2u, 4u}) {
    Params::Spec spec;
    spec.log_degree = 10;
    spec.main_primes = {12289, 40961, 61441, 65537};
    spec.aux_primes = {786433, 1179649};
    if (digits == 1)
      spec.aux_primes.push_back(2013265921);
    spec.dnum = digits;
    spec.log_default_scale = 13;
    auto parameters = Params::create(spec);
    ASSERT_TRUE(parameters);
    auto tables = EvaluationTables::create(*parameters);
    ASSERT_TRUE(tables);
    Context ctx(new FhnBackendCtx(std::move(tables)), ckks_fhn_destroy);
    Client client(ckks_fhn_client_create(ctx.get()), ckks_fhn_client_destroy);
    ASSERT_TRUE(client);
    prepareKeys(client.get(), 8);
    Buffer x(ctx.get()), row(ctx.get()), out(ctx.get()), ref(ctx.get());
    const auto &p = ctx->tables->params();
    for (uint32_t level = 1; level < p.levelCount(); ++level) {
      x.ptr->layout = {{{ctx->tables->paramsId(), 0, level + 1, 0, 0}, p.logDegree(), NttForm::Evaluation}, 8192, 2};
      row.ptr->layout = x.ptr->layout;
      x.ptr->logical_slots = row.ptr->logical_slots = 8;
      x.ptr->certified = row.ptr->certified = true;
      for (uint32_t c = 0; c < 2; ++c)
        for (uint32_t l = 0; l <= level; ++l)
          for (size_t i = 0; i < p.degree(); ++i) {
            const auto q = p.mainPrimes()[l];
            x.ptr->view().poly(c).limb(l)[i] = (i * 7937 + (c + 1) * 17413 + l * 93) % q;
            row.ptr->view().poly(c).limb(l)[i] = i % 3 == 0   ? q - 1
                                                 : i % 3 == 1 ? 0
                                                              : (i * 9127 + (c + 3) * 19751 + l * 179) % q;
          }
      ASSERT_EQ(ckks_fhn_packed_dot(ctx.get(), ref.ptr, x.ptr, row.ptr, 8), 0);
      for (uint32_t flags = 0; flags <= FHN_MATVEC_PREPARE_ALL; ++flags) {
        SCOPED_TRACE(::testing::Message() << "digits=" << digits << " level=" << level << " flags=" << flags);
        FhnMatvecPrepareOptions options{sizeof(FhnMatvecPrepareOptions), flags};
        Preparation prep(ckks_fhn_prepare_matvec_source(ctx.get(), x.ptr, 8, &options),
                         ckks_fhn_matvec_preparation_free);
        ASSERT_TRUE(prep);
        const size_t expected =
          2 * (level + 1) * p.degree() * sizeof(uint64_t) +
          (flags & FHN_MATVEC_PREPARE_VECTOR ? 2 * (level + 1) * p.degree() * sizeof(uint64_t) : 0) +
          (flags & FHN_MATVEC_PREPARE_KEYS
             ? 4 * 2 * digits * (p.levelCount() + p.auxPrimes().size()) * p.degree() * sizeof(uint64_t)
             : 0) +
          (flags & FHN_MATVEC_PREPARE_MAPS ? 3 * p.degree() * sizeof(uint32_t) : 0);
        EXPECT_EQ(ckks_fhn_matvec_preparation_bytes(prep.get()), expected);
        ASSERT_EQ(ckks_fhn_matvec_dot(ctx.get(), out.ptr, prep.get(), row.ptr), 0);
        equal(out.ptr, ref.ptr);
      }
    }
  }
}
TEST_F(PreparedMatvec, InvalidEvaluationKeyIsRejectedBeforeDurableAllocation) {
  prepareKeys(client.get(), 8);
  Buffer x(ctx.get());
  encrypt(x.ptr);
  const auto key = client->relin->view();
  auto *data = const_cast<uint64_t *>(key.digits[0].poly(0).limb(0));
  const auto saved = *data;
  *data = ctx->tables->params().mainPrimes()[0];
  matvec_allocations::count = 0;
  matvec_allocations::track = true;
  auto *result = ckks_fhn_prepare_matvec_source(ctx.get(), x.ptr, 8, nullptr);
  matvec_allocations::track = false;
  *data = saved;
  EXPECT_EQ(result, nullptr);
  EXPECT_EQ(matvec_allocations::count, 0u);
  ckks_fhn_matvec_preparation_free(result);
}
} // namespace
