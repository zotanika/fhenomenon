#include "CKKS/PreparedRotation.h"
#include "BackendInternal.h"
#include "CKKS/detail/Region.h"
#include <algorithm>
#include <limits>
#include <memory>

using namespace fhenomenon::ckks;

// No client, secret, or source-buffer pointer. Only this context's tables and
// currently live public evaluation keys are borrowed when consuming the state.
struct FhnRotationPreparation {
  FhnBackendCtx *const owner;
  const CiphertextLayout layout;
  const uint32_t logical_slots;
  const std::size_t words;
  PolySlab storage;
  FhnRotationPreparation(FhnBackendCtx *ctx, const FhnBuffer *source)
    : owner(ctx), layout(source->layout), logical_slots(source->logical_slots), words(layout.poly.wordCount()),
      storage(3 * words) {
    for (uint32_t c = 0; c < 2; ++c)
      std::copy_n(source->view().poly(c).data, words, storage.data() + c * words);
    std::copy_n(storage.data() + words, words, storage.data() + 2 * words);
    for (uint32_t l = 0; l < layout.poly.basis.main_end; ++l)
      ntt::inverse(storage.data() + 2 * words + l * ctx->tables->params().degree(), ctx->tables->mainNtt(l));
  }
  ConstCiphertextView source() const { return {storage.data(), 2 * words, words, layout}; }
  ConstRnsPolyView coefficients() const {
    auto poly = layout.poly;
    poly.form = NttForm::Coefficient;
    return {storage.data() + 2 * words, words, poly};
  }
};

extern "C" {
uint64_t ckks_fhn_rotation_preparation_required_bytes(FhnBackendCtx *ctx, const FhnBuffer *source) {
  if (!ctx || !source || source->owner != ctx || !source->certified || !source->logical_slots ||
      (source->logical_slots & (source->logical_slots - 1)) ||
      source->logical_slots > ctx->tables->params().numSlots() || source->layout.num_polys != 2 ||
      !region::validateInput(source->view(), *ctx->tables))
    return 0;
  const auto words = source->layout.poly.wordCount();
  if (words > std::numeric_limits<uint64_t>::max() / (3 * sizeof(uint64_t)) ||
      words > std::numeric_limits<std::size_t>::max() / (3 * sizeof(uint64_t)))
    return 0;
  return 3 * words * sizeof(uint64_t);
}
FhnRotationPreparation *ckks_fhn_prepare_rotation_source(FhnBackendCtx *ctx, const FhnBuffer *source) {
  if (!ckks_fhn_rotation_preparation_required_bytes(ctx, source))
    return nullptr;
  try {
    return new FhnRotationPreparation(ctx, source);
  } catch (...) {
    return nullptr;
  }
}
uint64_t ckks_fhn_rotation_preparation_bytes(const FhnRotationPreparation *preparation) {
  return preparation ? preparation->storage.capacityWords() * sizeof(uint64_t) : 0;
}
int ckks_fhn_rotate_prepared(FhnBackendCtx *ctx, FhnBuffer *out, const FhnRotationPreparation *preparation,
                             int64_t rotation) {
  if (!ctx || !out || !preparation || preparation->owner != ctx || out->owner != ctx)
    return -1;
  try {
    const auto g = eval::galoisElement(rotation, ctx->tables->params().logDegree());
    std::shared_ptr<const EvaluationKey> key;
    if (g != 1) {
      const auto found = ctx->rotations.find(g);
      if (found == ctx->rotations.end() || !(key = found->second.lock()))
        return -1;
    }
    Arena::Scope scope(ctx->scratch);
    const auto words = preparation->words;
    auto *data = ctx->scratch.allocate<uint64_t>(2 * words);
    CiphertextView result{data, 2 * words, words, preparation->layout};
    if (!data || !region::rotatePrepared(result, preparation->source(), preparation->coefficients(), g,
                                         key ? key->view() : EvaluationKeyView{}, *ctx->tables, ctx->scratch))
      return -1;
    // The only public writes occur after all validation and arithmetic succeed.
    for (uint32_t c = 0; c < 2; ++c)
      std::copy_n(result.poly(c).data, words, out->slab.data() + c * ctx->stride);
    out->layout = result.layout;
    out->logical_slots = preparation->logical_slots;
    out->certified = true;
    return 0;
  } catch (...) {
    return -1;
  }
}
void ckks_fhn_rotation_preparation_free(FhnRotationPreparation *preparation) { delete preparation; }
}
