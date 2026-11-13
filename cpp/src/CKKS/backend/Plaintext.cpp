#include "CKKS/Plaintext.h"
#include "BackendInternal.h"
#include "CKKS/Encoding.h"
#include "CKKS/Eval.h"
#include "CKKS/detail/Region.h"
#include <algorithm>
#include <cmath>
#include <memory>

using namespace fhenomenon::ckks;
namespace {
bool compatibleInput(FhnBackendCtx *ctx, const FhnBuffer *input) {
  return ctx && input && input->owner == ctx && input->certified && input->logical_slots &&
         !(input->logical_slots & (input->logical_slots - 1)) &&
         input->logical_slots <= ctx->tables->params().numSlots() && input->layout.num_polys == 2 &&
         region::validateInput(input->view(), *ctx->tables);
}
bool preparationReady(FhnBackendCtx *ctx, const FhnBuffer *input, FhnPlaintextPurpose purpose) {
  if ((purpose != FHN_PLAINTEXT_ADD && purpose != FHN_PLAINTEXT_MULTIPLY) || !compatibleInput(ctx, input))
    return false;
  if (purpose == FHN_PLAINTEXT_ADD)
    return true;
  return input->layout.poly.basis.main_end >= 2 &&
         std::isfinite(input->layout.scale *
                       static_cast<double>(ctx->tables->params().mainPrimes()[input->layout.poly.basis.main_end - 1]));
}
bool sameScale(double a, double b) { return std::abs(a - b) <= 1e-12 * std::max(a, b); }
} // namespace

// The handle owns only public encoded coefficients and compatibility metadata.
// Neither source identity nor a key/client lease is retained.
struct FhnPlaintext {
  FhnBackendCtx *const owner;
  const CiphertextLayout source_layout;
  const uint32_t logical_slots;
  const FhnPlaintextPurpose purpose;
  const double scale;
  PolySlab storage;
  FhnPlaintext(FhnBackendCtx *ctx, const FhnBuffer *prototype, FhnPlaintextPurpose usage)
    : owner(ctx), source_layout(prototype->layout), logical_slots(prototype->logical_slots), purpose(usage),
      scale(usage == FHN_PLAINTEXT_ADD
              ? prototype->layout.scale
              : static_cast<double>(ctx->tables->params().mainPrimes()[prototype->layout.poly.basis.main_end - 1])),
      storage(prototype->layout.poly.wordCount()) {}
  ConstRnsPolyView view() const { return {storage.data(), storage.capacityWords(), source_layout.poly}; }
};

namespace {
int apply(FhnPlaintextPurpose purpose, FhnBackendCtx *ctx, FhnBuffer *out, const FhnBuffer *input,
          const FhnPlaintext *plain) {
  try {
    if (!ctx || !out || out->owner != ctx || !plain || plain->owner != ctx || plain->purpose != purpose ||
        !compatibleInput(ctx, input) || input->logical_slots != plain->logical_slots ||
        input->layout.poly != plain->source_layout.poly || !sameScale(input->layout.scale, plain->source_layout.scale))
      return -1;
    auto destination = out->view();
    const bool success =
      purpose == FHN_PLAINTEXT_ADD
        ? eval::addPlaintext(destination, input->view(), plain->view(), plain->scale, *ctx->tables, ctx->scratch)
        : eval::multiplyPlaintext(destination, input->view(), plain->view(), plain->scale, *ctx->tables, ctx->scratch);
    if (!success)
      return -1;
    out->layout = destination.layout;
    out->logical_slots = input->logical_slots;
    out->certified = true;
    return 0;
  } catch (...) {
    return -1;
  }
}
} // namespace

extern "C" {
uint32_t ckks_fhn_plaintext_api_version() { return 1; }
uint64_t ckks_fhn_plaintext_required_bytes(FhnBackendCtx *ctx, const FhnBuffer *prototype,
                                           FhnPlaintextPurpose purpose) {
  return preparationReady(ctx, prototype, purpose) ? prototype->layout.poly.byteCount() : 0;
}
int ckks_fhn_plaintext_encode(FhnBackendCtx *ctx, const FhnBuffer *prototype, const double *values, uint32_t count,
                              FhnPlaintextPurpose purpose, FhnPlaintext **out) {
  try {
    if (!out || !values || !preparationReady(ctx, prototype, purpose) || count != prototype->logical_slots)
      return -1;
    for (uint32_t i = 0; i < count; ++i)
      if (!std::isfinite(values[i]))
        return -1;
    auto prepared = std::make_unique<FhnPlaintext>(ctx, prototype, purpose);
    RnsPolyView destination{prepared->storage.data(), prepared->storage.capacityWords(), prepared->source_layout.poly};
    if (!encoding::encode(destination, values, count, prepared->scale, *ctx->tables, ctx->scratch))
      return -1;
    *out = prepared.release();
    return 0;
  } catch (...) {
    return -1;
  }
}
int ckks_fhn_plaintext_multiply(FhnBackendCtx *ctx, FhnBuffer *out, const FhnBuffer *input, const FhnPlaintext *plain) {
  return apply(FHN_PLAINTEXT_MULTIPLY, ctx, out, input, plain);
}
int ckks_fhn_plaintext_add(FhnBackendCtx *ctx, FhnBuffer *out, const FhnBuffer *input, const FhnPlaintext *plain) {
  return apply(FHN_PLAINTEXT_ADD, ctx, out, input, plain);
}
uint64_t ckks_fhn_plaintext_bytes(const FhnPlaintext *plain) {
  return plain ? plain->storage.capacityWords() * sizeof(uint64_t) : 0;
}
void ckks_fhn_plaintext_free(FhnPlaintext *plain) { delete plain; }
}
