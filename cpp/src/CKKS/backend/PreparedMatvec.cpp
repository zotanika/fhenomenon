#include "CKKS/PreparedMatvec.h"
#include "BackendInternal.h"
#include "CKKS/ModArith.h"
#include "CKKS/detail/Region.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>

using namespace fhenomenon::ckks;
namespace {
constexpr size_t maxStages = NttTables::kMaxLogDegree;
using Leases = std::array<std::shared_ptr<const EvaluationKey>, maxStages + 1>;
bool input(FhnBackendCtx *ctx, const FhnBuffer *value) {
  return value && value->owner == ctx && value->certified && value->logical_slots &&
         !(value->logical_slots & (value->logical_slots - 1)) &&
         value->logical_slots <= ctx->tables->params().numSlots() && value->layout.num_polys == 2 &&
         region::validateInput(value->view(), *ctx->tables);
}
bool sameScale(double a, double b) {
  return std::isfinite(a) && a > 0 && std::isfinite(b) && b > 0 && std::abs(a - b) <= 1e-12 * std::max(a, b);
}
bool checkedAdd(size_t &total, size_t count, size_t factor) {
  if (!factor)
    return true;
  if (count > (std::numeric_limits<size_t>::max() - total) / factor)
    return false;
  total += count * factor;
  return true;
}
struct Plan {
  uint32_t flags = 0;
  size_t stages = 0, words = 0, key_words = 0, total_words = 0;
  std::array<uint64_t, maxStages> galois{};
};
bool plan(FhnBackendCtx *ctx, const FhnBuffer *source, uint32_t width, const FhnMatvecPrepareOptions *options,
          Plan &out) {
  if (!ctx || !input(ctx, source) || source->layout.poly.basis.main_end < 2 || !width || (width & (width - 1)) ||
      width > source->logical_slots ||
      (options && (options->struct_size != sizeof(FhnMatvecPrepareOptions) ||
                   (options->flags & ~uint32_t{FHN_MATVEC_PREPARE_ALL}))))
    return false;
  out.flags = options ? options->flags : uint32_t{FHN_MATVEC_PREPARE_ALL};
  const auto &p = ctx->tables->params();
  for (uint32_t d = width / 2; d; d /= 2) {
    if (out.stages == maxStages)
      return false;
    out.galois[out.stages++] = eval::galoisElement(d, p.logDegree());
  }
  out.words = source->layout.poly.wordCount();
  if (!checkedAdd(out.total_words, out.words, 2))
    return false;
  if ((out.flags & FHN_MATVEC_PREPARE_VECTOR) && !checkedAdd(out.total_words, out.words, 2))
    return false;
  if (out.flags & FHN_MATVEC_PREPARE_KEYS) {
    if (!checkedAdd(out.key_words, p.levelCount() + p.auxPrimes().size(), p.degree()) ||
        out.key_words > std::numeric_limits<size_t>::max() / (2 * p.dnum()))
      return false;
    out.key_words *= 2 * p.dnum();
    if (!checkedAdd(out.total_words, out.key_words, out.stages + 1))
      return false;
  }
  // N is a power of two >=1024, so these uint32 maps occupy exact whole words.
  if ((out.flags & FHN_MATVEC_PREPARE_MAPS) && !checkedAdd(out.total_words, p.degree() / 2, out.stages))
    return false;
  return out.total_words <= std::numeric_limits<size_t>::max() / sizeof(uint64_t) &&
         out.total_words <= std::numeric_limits<uint64_t>::max() / sizeof(uint64_t);
}
bool acquire(FhnBackendCtx *ctx, const Plan &description, Leases &keys, bool validate) {
  keys[0] = ctx->relin.lock();
  if (!keys[0] || (validate && !region::validateKey(keys[0]->view(), SwitchKeyKind::Relinearization, 1, *ctx->tables)))
    return false;
  for (size_t s = 0; s < description.stages; ++s) {
    const auto found = ctx->rotations.find(description.galois[s]);
    if (found == ctx->rotations.end() || !(keys[s + 1] = found->second.lock()) ||
        (validate &&
         !region::validateKey(keys[s + 1]->view(), SwitchKeyKind::Galois, description.galois[s], *ctx->tables)))
      return false;
  }
  return true;
}
uint32_t reverse(uint32_t value, uint32_t bits) {
  uint32_t result = 0;
  for (uint32_t bit = 0; bit < bits; ++bit) {
    result = (result << 1) | (value & 1);
    value >>= 1;
  }
  return result;
}
} // namespace

// The single slab owns snapshot, selected quotients, and maps. Weak identities
// cannot keep the public keys or their client alive. Consumers lease live keys.
struct FhnMatvecPreparation {
  FhnBackendCtx *const owner;
  const CiphertextLayout layout;
  const uint32_t logical_slots, width;
  const Plan description;
  std::array<std::weak_ptr<const EvaluationKey>, maxStages + 1> identities;
  PolySlab storage;
  region::CipherProducts vector_products{};
  std::array<region::KeyProducts, maxStages + 1> key_products{};
  std::array<region::RotationMap, maxStages> maps{};
  FhnMatvecPreparation(FhnBackendCtx *ctx, const FhnBuffer *source, uint32_t reduction_width, const Plan &p,
                       const Leases &keys)
    : owner(ctx), layout(source->layout), logical_slots(source->logical_slots), width(reduction_width), description(p),
      storage(p.total_words) {
    const auto &params = ctx->tables->params();
    const size_t n = params.degree();
    auto *cursor = storage.data();
    for (uint32_t c = 0; c < 2; ++c) {
      std::copy_n(source->view().poly(c).data, p.words, cursor);
      cursor += p.words;
    }
    if (p.flags & FHN_MATVEC_PREPARE_VECTOR) {
      vector_products = {cursor, 2 * p.words};
      for (uint32_t c = 0; c < 2; ++c)
        for (uint32_t l = 0; l < layout.poly.basis.main_end; ++l)
          for (size_t i = 0; i < n; ++i)
            *cursor++ = modarith::shoupFactor(source->view().poly(c).limb(l)[i], params.mainPrimes()[l]);
    }
    for (size_t k = 0; k <= p.stages; ++k) {
      identities[k] = keys[k];
      if (!(p.flags & FHN_MATVEC_PREPARE_KEYS))
        continue;
      key_products[k] = {cursor, p.key_words};
      const auto key = keys[k]->view();
      for (uint32_t d = 0; d < key.digit_count; ++d)
        for (uint32_t c = 0; c < 2; ++c)
          for (uint32_t l = 0; l < key.main_count + key.aux_count; ++l) {
            const auto q = l < key.main_count ? params.mainPrimes()[l] : params.auxPrimes()[l - key.main_count];
            for (size_t i = 0; i < n; ++i)
              *cursor++ = modarith::shoupFactor(key.digits[d].poly(c).limb(l)[i], q);
          }
    }
    if (p.flags & FHN_MATVEC_PREPARE_MAPS)
      for (size_t s = 0; s < p.stages; ++s) {
        auto *map = reinterpret_cast<uint32_t *>(cursor);
        maps[s] = {map, n};
        for (uint32_t i = 0; i < n; ++i) {
          const auto odd = (p.galois[s] * (2 * reverse(i, params.logDegree()) + 1)) % (2 * n);
          map[i] = reverse(static_cast<uint32_t>((odd - 1) / 2), params.logDegree());
        }
        cursor += n / 2;
      }
  }
  ConstCiphertextView source() const { return {storage.data(), 2 * description.words, description.words, layout}; }
};

namespace {
enum class Operation { Product, Dot, Reduce };
int execute(Operation operation, FhnBackendCtx *ctx, FhnBuffer *out, const FhnMatvecPreparation *preparation,
            const FhnBuffer *operand, uint32_t width) {
  if (!ctx || !out || !preparation || preparation->owner != ctx || out->owner != ctx || !input(ctx, operand))
    return -1;
  const bool product = operation != Operation::Reduce;
  if (product)
    width = operation == Operation::Product ? 1 : preparation->width;
  if (!width || (width & (width - 1)) || width > preparation->width ||
      operand->logical_slots != preparation->logical_slots)
    return -1;
  auto expected = preparation->layout;
  if (!product) {
    --expected.poly.basis.main_end;
    expected.scale = expected.scale * expected.scale /
                     static_cast<double>(ctx->tables->params().mainPrimes()[expected.poly.basis.main_end]);
  }
  if (operand->layout.poly != expected.poly || !sameScale(operand->layout.scale, expected.scale))
    return -1;
  auto destination = out->view();
  destination.layout = expected;
  if (!destination.wellFormed())
    return -1;
  Leases keys;
  if (!acquire(ctx, preparation->description, keys, false))
    return -1;
  for (size_t k = 0; k <= preparation->description.stages; ++k) {
    const auto identity = preparation->identities[k].lock();
    if (!identity || identity != keys[k])
      return -1;
  }
  Arena::Scope scope(ctx->scratch);
  const auto words = expected.poly.wordCount();
  CiphertextView first{ctx->scratch.allocate<uint64_t>(2 * words), 2 * words, words, expected};
  CiphertextView second{ctx->scratch.allocate<uint64_t>(2 * words), 2 * words, words, expected};
  if (!first.data || !second.data || ctx->scratch.remaining() < region::scratchBytes(ctx->tables->params()))
    return -1;
  ConstCiphertextView accumulator = operand->view();
  CiphertextView *next = &first;
  if (product) {
    if (!region::multiplyCached(first, preparation->source(), operand->view(), keys[0]->view(),
                                preparation->key_products[0], preparation->vector_products, *ctx->tables, ctx->scratch))
      return -1;
    accumulator = asConst(first);
    next = &second;
  }
  // A narrower reduction uses the tail of the exact same width/2,...,1 tree.
  size_t stage = 0;
  for (uint32_t d = preparation->width; d > width; d /= 2)
    ++stage;
  for (uint32_t d = width / 2; d; d /= 2, ++stage) {
    const region::LinearTerm term{accumulator, 1};
    if (!region::rotateAddCached(*next, accumulator, preparation->description.galois[stage], keys[stage + 1]->view(),
                                 preparation->key_products[stage + 1], preparation->maps[stage], &term, 1, *ctx->tables,
                                 ctx->scratch))
      return -1;
    accumulator = asConst(*next);
    next = next == &first ? &second : &first;
  }
  const auto live = accumulator.layout.poly.wordCount();
  for (uint32_t c = 0; c < 2; ++c)
    std::copy_n(accumulator.poly(c).data, live, out->slab.data() + c * ctx->stride);
  out->layout = accumulator.layout;
  out->logical_slots = operand->logical_slots;
  out->certified = true;
  return 0;
}
int guarded(Operation operation, FhnBackendCtx *ctx, FhnBuffer *out, const FhnMatvecPreparation *preparation,
            const FhnBuffer *operand, uint32_t width) {
  try {
    return execute(operation, ctx, out, preparation, operand, width);
  } catch (...) {
    return -1;
  }
}
} // namespace
extern "C" {
uint32_t ckks_fhn_matvec_api_version() { return 1; }
uint64_t ckks_fhn_matvec_preparation_required_bytes(FhnBackendCtx *ctx, const FhnBuffer *source, uint32_t width,
                                                    const FhnMatvecPrepareOptions *options) {
  try {
    Plan description;
    Leases keys;
    return plan(ctx, source, width, options, description) && acquire(ctx, description, keys, false)
             ? description.total_words * sizeof(uint64_t)
             : 0;
  } catch (...) {
    return 0;
  }
}
FhnMatvecPreparation *ckks_fhn_prepare_matvec_source(FhnBackendCtx *ctx, const FhnBuffer *source, uint32_t width,
                                                     const FhnMatvecPrepareOptions *options) {
  try {
    Plan description;
    Leases keys;
    if (!plan(ctx, source, width, options, description) || !acquire(ctx, description, keys, true))
      return nullptr;
    // Validate and strongly lease every key before the first durable allocation.
    return new FhnMatvecPreparation(ctx, source, width, description, keys);
  } catch (...) {
    return nullptr;
  }
}
uint64_t ckks_fhn_matvec_preparation_bytes(const FhnMatvecPreparation *preparation) {
  return preparation ? preparation->storage.capacityWords() * sizeof(uint64_t) : 0;
}
void ckks_fhn_matvec_preparation_free(FhnMatvecPreparation *preparation) { delete preparation; }
int ckks_fhn_matvec_product(FhnBackendCtx *ctx, FhnBuffer *out, const FhnMatvecPreparation *preparation,
                            const FhnBuffer *row) {
  return guarded(Operation::Product, ctx, out, preparation, row, 1);
}
int ckks_fhn_matvec_dot(FhnBackendCtx *ctx, FhnBuffer *out, const FhnMatvecPreparation *preparation,
                        const FhnBuffer *row) {
  return guarded(Operation::Dot, ctx, out, preparation, row, 0);
}
int ckks_fhn_matvec_reduce(FhnBackendCtx *ctx, FhnBuffer *out, const FhnMatvecPreparation *preparation,
                           const FhnBuffer *value, uint32_t width) {
  return guarded(Operation::Reduce, ctx, out, preparation, value, width);
}
}
