#include "CKKS/CoarseKernels.h"
#include "BackendInternal.h"
#include "CKKS/detail/Region.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <memory>

using namespace fhenomenon::ckks;
namespace {
enum class Operation { Energy, Variance, Dot, Reduce };
bool input(FhnBackendCtx *ctx, const FhnBuffer *value) {
  return value && value->owner == ctx && value->certified && value->logical_slots &&
         !(value->logical_slots & (value->logical_slots - 1)) &&
         value->logical_slots <= ctx->tables->params().numSlots() && region::validateInput(value->view(), *ctx->tables);
}
CiphertextView staging(Arena &arena, CiphertextLayout layout) {
  const auto words = layout.poly.wordCount();
  return {arena.allocate<uint64_t>(2 * words), 2 * words, words, layout};
}
int execute(Operation operation, FhnBackendCtx *ctx, FhnBuffer *out, const FhnBuffer *a, const FhnBuffer *b,
            uint32_t width) {
  if (!ctx || !out || out->owner != ctx || !input(ctx, a) || !width || (width & (width - 1)) ||
      width > a->logical_slots)
    return -1;
  const bool product = operation != Operation::Reduce;
  const bool centered = operation == Operation::Energy || operation == Operation::Variance;
  const bool reduction = operation != Operation::Energy;
  const auto &tables = *ctx->tables;
  const auto layout = a->layout;
  // Centering and the width-scaled output need additional plaintext range.
  // The contributed statistics profile admits only at least three active Q
  // primes; dot/reduce retain the underlying primitives' level requirements.
  if (centered && layout.poly.basis.main_end < 3)
    return -1;
  if (product && (!input(ctx, b) || a->logical_slots != b->logical_slots || layout.poly != b->layout.poly ||
                  std::abs(layout.scale - b->layout.scale) > 1e-12 * std::max(layout.scale, b->layout.scale) ||
                  layout.poly.basis.main_end < 2))
    return -1;
  // Validate destination capacity before any arithmetic. Its existing payload
  // may have any valid state, including never initialized or aliasing an input.
  auto destination = out->view();
  destination.layout = layout;
  if (!destination.wellFormed())
    return -1;

  std::shared_ptr<const EvaluationKey> relin;
  // A physical CKKS vector has at most logDegree-1 reduction stages. Use the
  // parameter system's maximum, including contexts constructed by native hosts.
  std::array<std::shared_ptr<const EvaluationKey>, NttTables::kMaxLogDegree> rotations{};
  std::array<uint64_t, NttTables::kMaxLogDegree> galois{};
  size_t stages = 0;
  if (product) {
    relin = ctx->relin.lock();
    if (!relin || !region::validateKey(relin->view(), SwitchKeyKind::Relinearization, 1, tables))
      return -1;
  }
  if (reduction)
    for (uint32_t distance = width / 2; distance; distance /= 2) {
      if (stages == rotations.size())
        return -1;
      const auto g = eval::galoisElement(distance, tables.params().logDegree());
      const auto found = ctx->rotations.find(g);
      if (found == ctx->rotations.end() || !(rotations[stages] = found->second.lock()) ||
          !region::validateKey(rotations[stages]->view(), SwitchKeyKind::Galois, g, tables))
        return -1;
      galois[stages++] = g;
    }

  Arena::Scope scope(ctx->scratch);
  // Only two compact active-Q ciphertexts remain live across the region calls.
  // The default profile's existing context scratch covers these plus the full
  // conservative region bound. Smaller caller-supplied arenas fail atomically.
  auto first = staging(ctx->scratch, layout);
  auto second = staging(ctx->scratch, layout);
  if (!first.data || !second.data || ctx->scratch.remaining() < region::scratchBytes(tables.params()))
    return -1;
  ConstCiphertextView accumulator = a->view();
  CiphertextView *next = &first;
  if (centered) {
    const region::LinearTerm terms[]{{a->view(), static_cast<int64_t>(width)}, {b->view(), -1}};
    if (!region::linear(first, terms, 2, 0, layout, tables, ctx->scratch) ||
        !region::multiply(second, asConst(first), asConst(first), relin->view(), tables, ctx->scratch))
      return -1;
    // Dividing the centered square by width changes the represented value:
    // report that operation through its actual scale, never a forced scale join.
    // This retains the Q level but uses log2(width) additional bits of range.
    const auto divided_scale = second.layout.scale * static_cast<double>(width);
    if (!std::isfinite(divided_scale) || divided_scale <= 0)
      return -1;
    second.layout.scale = divided_scale;
    accumulator = asConst(second);
    next = &first;
  } else if (product) {
    if (!region::multiply(first, a->view(), b->view(), relin->view(), tables, ctx->scratch))
      return -1;
    accumulator = asConst(first);
    next = &second;
  }
  for (size_t stage = 0; stage < stages; ++stage) {
    const region::LinearTerm term{accumulator, 1};
    if (!region::rotateAdd(*next, accumulator, galois[stage], rotations[stage]->view(), &term, 1, tables, ctx->scratch))
      return -1;
    accumulator = asConst(*next);
    next = next == &first ? &second : &first;
  }
  // Publish only after the entire operation succeeds. All keys are strongly
  // leased until this point, and neither escaped input was mutated while read.
  const auto words = accumulator.layout.poly.wordCount();
  for (uint32_t component = 0; component < 2; ++component)
    std::copy_n(accumulator.poly(component).data, words, out->slab.data() + component * ctx->stride);
  out->layout = accumulator.layout;
  out->logical_slots = a->logical_slots;
  out->certified = true;
  return 0;
}
int guarded(Operation operation, FhnBackendCtx *ctx, FhnBuffer *out, const FhnBuffer *a, const FhnBuffer *b,
            uint32_t width) {
  try {
    return execute(operation, ctx, out, a, b, width);
  } catch (...) {
    return -1;
  }
}
} // namespace
extern "C" {
uint32_t ckks_fhn_coarse_api_version() { return 1; }
int ckks_fhn_centered_energy(FhnBackendCtx *ctx, FhnBuffer *out, const FhnBuffer *x, const FhnBuffer *sum,
                             uint32_t width) {
  return guarded(Operation::Energy, ctx, out, x, sum, width);
}
int ckks_fhn_centered_variance(FhnBackendCtx *ctx, FhnBuffer *out, const FhnBuffer *x, const FhnBuffer *sum,
                               uint32_t width) {
  return guarded(Operation::Variance, ctx, out, x, sum, width);
}
int ckks_fhn_packed_dot(FhnBackendCtx *ctx, FhnBuffer *out, const FhnBuffer *a, const FhnBuffer *b, uint32_t width) {
  return guarded(Operation::Dot, ctx, out, a, b, width);
}
int ckks_fhn_packed_reduce(FhnBackendCtx *ctx, FhnBuffer *out, const FhnBuffer *input, uint32_t width) {
  return guarded(Operation::Reduce, ctx, out, input, nullptr, width);
}
}
