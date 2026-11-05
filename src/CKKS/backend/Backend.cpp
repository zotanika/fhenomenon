#include "CKKS/Backend.h"
#include "BackendInternal.h"
#include "CKKS/Arena.h"
#include "CKKS/Encoding.h"
#include "CKKS/Eval.h"
#include "CKKS/Keys.h"
#include "CKKS/ModArith.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <set>
#include <unordered_map>
#include <vector>

using namespace fhenomenon::ckks;

namespace {
std::size_t strideWords(const Params &p) { return (p.mainPrimes().size() + p.auxPrimes().size()) * p.degree(); }
std::size_t computeScratch(const Params &p) {
  std::size_t bound = 0;
  for (uint32_t l = 0; l < p.levelCount(); ++l)
    for (auto op : {eval::EvalOperation::Add, eval::EvalOperation::Subtract, eval::EvalOperation::Negate,
                    eval::EvalOperation::Tensor, eval::EvalOperation::Relinearize, eval::EvalOperation::Rescale,
                    eval::EvalOperation::Rotate, eval::EvalOperation::HMultiply, eval::EvalOperation::AddScalar,
                    eval::EvalOperation::MultiplyScalar, eval::EvalOperation::LevelDown})
      bound = std::max(bound, eval::scratchBytes(op, p, l));
  // Two adapter staging ciphertexts make fused operations and arbitrary
  // output aliases transactional. Bound includes alignment, not key/table RAM.
  return bound + 6 * strideWords(p) * sizeof(uint64_t) + 512;
}
std::size_t clientScratch(const Params &p) {
  const std::size_t complex_slots = p.numSlots() * sizeof(std::complex<double>);
  return std::max(keyScratchBytes(p), encoding::scratchBytes(p) + complex_slots) +
         4 * strideWords(p) * sizeof(uint64_t) + 512;
}
uint64_t primeBelow(uint32_t bits, uint32_t log_degree, const std::vector<uint64_t> &used) {
  const uint64_t step = uint64_t{1} << (log_degree + 1);
  uint64_t candidate = ((uint64_t{1} << bits) - 1) / step * step + 1;
  if (candidate >= (uint64_t{1} << bits))
    candidate -= step;
  for (; candidate > step; candidate -= step)
    if (std::find(used.begin(), used.end(), candidate) == used.end() && modarith::isPrime(candidate))
      return candidate;
  return 0;
}
uint32_t unsignedField(const nlohmann::json &j, const char *name, uint32_t fallback) {
  if (!j.contains(name))
    return fallback;
  const auto &v = j.at(name);
  if (!v.is_number_unsigned() || v.get<uint64_t>() > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("invalid unsigned configuration field");
  return v.get<uint32_t>();
}
std::optional<Params> configuration(const char *text) {
  const auto j = text ? nlohmann::json::parse(text) : nlohmann::json::object();
  if (!j.is_object())
    return std::nullopt;
  const std::set<std::string> allowed{"log_degree", "main_primes", "aux_primes", "dnum", "log_default_scale"};
  for (auto it = j.begin(); it != j.end(); ++it)
    if (!allowed.count(it.key()))
      return std::nullopt;
  Params::Spec s;
  s.log_degree = unsignedField(j, "log_degree", 13);
  // Published ternary bounds exist for this range. Higher degrees and sparse
  // secrets require a separately justified profile before this adapter accepts them.
  if (s.log_degree < 10 || s.log_degree > 15)
    return std::nullopt;
  s.dnum = unsignedField(j, "dnum", 3);
  s.log_default_scale = unsignedField(j, "log_default_scale", 40);
  for (const auto *name : {"main_primes", "aux_primes"})
    if (j.contains(name)) {
      if (!j.at(name).is_array())
        return std::nullopt;
      for (const auto &v : j.at(name))
        if (!v.is_number_unsigned())
          return std::nullopt;
    }
  if (j.contains("main_primes"))
    s.main_primes = j.at("main_primes").get<std::vector<uint64_t>>();
  else
    for (uint32_t bits : {50u, 40u, 40u})
      s.main_primes.push_back(primeBelow(bits, s.log_degree, s.main_primes));
  if (j.contains("aux_primes"))
    s.aux_primes = j.at("aux_primes").get<std::vector<uint64_t>>();
  else
    s.aux_primes.push_back(primeBelow(60, s.log_degree, s.main_primes));
  auto p = Params::create(std::move(s));
  if (!p)
    return std::nullopt;
  for (uint32_t d = 0; d < p->dnum(); ++d) {
    const auto range = p->digitRange(d);
    if (range.first == range.second)
      return std::nullopt;
  }
  // Conservative sum of prime bit lengths, including key-switch P. Values
  // follow SEAL util/hestdparms.h's 128-bit classical uniform-ternary table.
  // This guard is not a security certification of this implementation.
  constexpr uint32_t max_bits[]{27, 54, 109, 218, 438, 881};
  uint32_t total_bits = 0;
  for (const auto &primes : {p->mainPrimes(), p->auxPrimes()})
    for (uint64_t prime : primes)
      for (; prime; prime >>= 1)
        ++total_bits;
  if (total_bits > max_bits[p->logDegree() - 10])
    return std::nullopt;
  return p;
}
} // namespace

// Evaluation context: no SecretKey, PublicKey, client reference, or callable
// encrypt/decrypt closure. Only immutable tables, explicit workspace and weak
// observers of evaluation keys. Calls on one context must be serialized.
FhnBackendCtx::FhnBackendCtx(std::unique_ptr<EvaluationTables> t)
  : tables(std::move(t)), scratch(computeScratch(tables->params())), stride(strideWords(tables->params())) {}
FhnClientCtx::FhnClientCtx(FhnBackendCtx *ctx) : backend(ctx), scratch(clientScratch(ctx->tables->params())) {}

namespace {
PolyLayout freshPoly(const Params &p) {
  return {{ParamsId::of(p), 0, p.levelCount(), 0, 0}, p.logDegree(), NttForm::Evaluation};
}
CiphertextView staging(FhnBackendCtx *ctx, Arena &arena) {
  auto *data = arena.allocate<uint64_t>(3 * ctx->stride);
  return {data, data ? 3 * ctx->stride : 0, ctx->stride, {}};
}
bool ready(FhnBackendCtx *ctx, const FhnBuffer *b) {
  return b && b->owner == ctx && b->logical_slots && b->certified && b->view().wellFormed();
}
void commit(FhnBuffer *out, ConstCiphertextView result, uint32_t slots) {
  const auto bytes = result.layout.poly.wordCount() * sizeof(uint64_t);
  for (uint32_t k = 0; k < result.layout.num_polys; ++k)
    std::memcpy(out->slab.data() + k * out->owner->stride, result.poly(k).data, bytes);
  out->layout = result.layout;
  out->logical_slots = slots;
  out->certified = true;
}
std::shared_ptr<const EvaluationKey> rotationKey(FhnBackendCtx *ctx, uint64_t g) {
  const auto it = ctx->rotations.find(g);
  return it == ctx->rotations.end() ? nullptr : it->second.lock();
}
int execute(FhnOpCode op, FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *args, const int64_t *ip,
            const double *fp) {
  if (!ctx || !result || result->owner != ctx || !args || !ready(ctx, args[0]))
    return -1;
  const bool binary = op == FHN_ADD_CC || op == FHN_SUB_CC || op == FHN_MULT_CC || op == FHN_HMULT ||
                      op == FHN_HROT_ADD || op == FHN_HCONJ_ADD;
  if (binary && (!ready(ctx, args[1]) || args[0]->logical_slots != args[1]->logical_slots))
    return -1;
  if ((op == FHN_ROTATE || op == FHN_HROT || op == FHN_HROT_ADD || op == FHN_LEVEL_DOWN) && !ip)
    return -1;
  if ((op == FHN_ADD_CS || op == FHN_MULT_CS || op == FHN_SUB_SC) && (!fp || !std::isfinite(fp[0])))
    return -1;
  auto &arena = ctx->scratch;
  Arena::Scope scope(arena);
  auto out = staging(ctx, arena);
  if (!out.data)
    return -1;
  const auto a = args[0]->view();
  const auto b = binary ? args[1]->view() : ConstCiphertextView{};
  const auto &tables = *ctx->tables;
  bool ok = false;
  switch (op) {
  case FHN_NOP:
    ok = eval::levelDown(out, a, a.layout.poly.basis.main_end - 1, tables, arena);
    break;
  case FHN_ADD_CC:
    ok = eval::add(out, a, b, tables, arena);
    break;
  case FHN_SUB_CC:
    ok = eval::subtract(out, a, b, tables, arena);
    break;
  case FHN_NEGATE:
    ok = eval::negate(out, a, tables, arena);
    break;
  case FHN_ADD_CS:
    ok = eval::addScalar(out, a, fp[0], tables, arena);
    break;
  case FHN_MULT_CS:
    ok = eval::multiplyScalar(out, a, fp[0], tables, arena);
    break;
  case FHN_SUB_SC: {
    auto temp = staging(ctx, arena);
    ok = temp.data && eval::negate(temp, a, tables, arena) && eval::addScalar(out, asConst(temp), fp[0], tables, arena);
    break;
  }
  case FHN_MULT_CC:
    ok = eval::tensor(out, a, b, tables, arena);
    break;
  case FHN_RESCALE:
    ok = eval::rescale(out, a, tables, arena);
    break;
  case FHN_LEVEL_DOWN:
    ok =
      ip[0] >= 0 && ip[0] <= a.layout.level() && eval::levelDown(out, a, static_cast<uint32_t>(ip[0]), tables, arena);
    break;
  case FHN_RELINEARIZE:
  case FHN_HMULT: {
    const auto key = ctx->relin.lock();
    if (key)
      ok = op == FHN_HMULT ? eval::hmult(out, a, b, key->view(), tables, arena)
                           : eval::relinearize(out, a, key->view(), tables, arena);
    break;
  }
  case FHN_ROTATE:
  case FHN_HROT:
  case FHN_CONJUGATE:
  case FHN_HROT_ADD:
  case FHN_HCONJ_ADD: {
    const bool conjugate = op == FHN_CONJUGATE || op == FHN_HCONJ_ADD;
    const uint64_t g =
      conjugate ? 2 * tables.params().degree() - 1 : eval::galoisElement(ip[0], tables.params().logDegree());
    const auto key = rotationKey(ctx, g);
    if (g != 1 && !key)
      break;
    const auto view = key ? key->view() : EvaluationKeyView{};
    if (binary) {
      auto temp = staging(ctx, arena);
      ok =
        temp.data && eval::rotate(temp, a, g, view, tables, arena) && eval::add(out, asConst(temp), b, tables, arena);
    } else
      ok = eval::rotate(out, a, g, view, tables, arena);
    break;
  }
  default:
    return -1;
  }
  if (!ok)
    return -1;
  commit(result, asConst(out), args[0]->logical_slots);
  return 0;
}
template <FhnOpCode Op>
int kernel(FhnBackendCtx *ctx, FhnBuffer *out, const FhnBuffer *const *a, const int64_t *p, const double *f) {
  try {
    return execute(Op, ctx, out, a, p, f);
  } catch (...) {
    return -1;
  }
}
FhnKernelEntry entries[]{{FHN_NOP, kernel<FHN_NOP>, "copy"},
                         {FHN_ADD_CC, kernel<FHN_ADD_CC>, "add"},
                         {FHN_SUB_CC, kernel<FHN_SUB_CC>, "subtract"},
                         {FHN_NEGATE, kernel<FHN_NEGATE>, "negate"},
                         {FHN_ADD_CS, kernel<FHN_ADD_CS>, "add_scalar"},
                         {FHN_SUB_SC, kernel<FHN_SUB_SC>, "scalar_subtract"},
                         {FHN_MULT_CS, kernel<FHN_MULT_CS>, "multiply_scalar_rescale"},
                         {FHN_MULT_CC, kernel<FHN_MULT_CC>, "tensor"},
                         {FHN_RELINEARIZE, kernel<FHN_RELINEARIZE>, "hybrid_relinearize"},
                         {FHN_RESCALE, kernel<FHN_RESCALE>, "rescale"},
                         {FHN_LEVEL_DOWN, kernel<FHN_LEVEL_DOWN>, "level_down"},
                         {FHN_ROTATE, kernel<FHN_ROTATE>, "hybrid_rotate"},
                         {FHN_CONJUGATE, kernel<FHN_CONJUGATE>, "hybrid_conjugate"},
                         {FHN_HMULT, kernel<FHN_HMULT>, "tensor_relinearize_rescale"},
                         {FHN_HROT, kernel<FHN_HROT>, "hybrid_rotate"},
                         {FHN_HROT_ADD, kernel<FHN_HROT_ADD>, "rotate_add"},
                         {FHN_HCONJ_ADD, kernel<FHN_HCONJ_ADD>, "conjugate_add"}};
FhnKernelTable kernel_table{static_cast<uint32_t>(sizeof(entries) / sizeof(entries[0])), entries};

bool encodeVector(RnsPolyView &plain, const double *values, uint32_t count, double scale,
                  const EvaluationTables &tables, Arena &arena) {
  return encoding::encode(plain, values, count, scale, tables, arena);
}
bool encodeVector(RnsPolyView &plain, const FhnComplex64 *values, uint32_t count, double scale,
                  const EvaluationTables &tables, Arena &arena) {
  Arena::Scope scope(arena);
  auto *slots = arena.allocate<std::complex<double>>(count);
  if (!slots)
    return false;
  for (uint32_t i = 0; i < count; ++i)
    new (slots + i) std::complex<double>(values[i].real, values[i].imag);
  return encoding::encode(plain, slots, count, scale, tables, arena);
}
bool decodeVector(double *values, uint32_t count, ConstRnsPolyView plain, double scale, const EvaluationTables &tables,
                  Arena &arena) {
  return encoding::decode(values, count, plain, scale, tables, arena);
}
bool decodeVector(FhnComplex64 *values, uint32_t count, ConstRnsPolyView plain, double scale,
                  const EvaluationTables &tables, Arena &arena) {
  Arena::Scope scope(arena);
  auto *slots = arena.allocate<std::complex<double>>(count);
  if (!slots)
    return false;
  for (uint32_t i = 0; i < count; ++i)
    new (slots + i) std::complex<double>();
  if (!encoding::decode(slots, count, plain, scale, tables, arena))
    return false;
  for (uint32_t i = 0; i < count; ++i)
    values[i] = {slots[i].real(), slots[i].imag()};
  return true;
}

template <typename Slot> int encryptVector(FhnClientCtx *client, FhnBuffer *out, const Slot *values, uint32_t count) {
  if (!client || !out || out->owner != client->backend || !values || !count || (count & (count - 1)))
    return -1;
  try {
    auto *ctx = client->backend;
    const auto &p = ctx->tables->params();
    if (count > p.numSlots())
      return -1;
    Arena::Scope scope(client->scratch);
    auto layout = freshPoly(p);
    auto *data = client->scratch.allocate<uint64_t>(layout.wordCount());
    RnsPolyView plain{data, layout.wordCount(), layout};
    auto output = staging(ctx, client->scratch);
    const double scale = std::ldexp(1.0, static_cast<int>(p.logDefaultScale()));
    if (!data || !output.data || !encodeVector(plain, values, count, scale, *ctx->tables, client->scratch) ||
        !encrypt(output, asConst(plain), scale, *client->public_key, client->random, *ctx->tables, client->scratch))
      return -1;
    commit(out, asConst(output), count);
    return 0;
  } catch (...) {
    return -1;
  }
}
template <typename Slot> int decryptVector(FhnClientCtx *client, const FhnBuffer *in, Slot *values, uint32_t count) {
  if (!client || !ready(client->backend, in) || !values || count != in->logical_slots)
    return -1;
  try {
    auto *ctx = client->backend;
    Arena::Scope scope(client->scratch);
    const auto layout = in->layout.poly;
    auto *data = client->scratch.allocate<uint64_t>(layout.wordCount());
    RnsPolyView plain{data, layout.wordCount(), layout};
    return data && decrypt(plain, in->view(), *client->secret, *ctx->tables, client->scratch) &&
               decodeVector(values, count, asConst(plain), in->layout.scale, *ctx->tables, client->scratch)
             ? 0
             : -1;
  } catch (...) {
    return -1;
  }
}
} // namespace

extern "C" {
FhnBackendInfo *ckks_fhn_get_info() {
  static FhnBackendInfo info{"Native RNS-CKKS (CPU, leveled)", "0.1.0", FHN_DEVICE_CPU, 0};
  return &info;
}
FhnBackendCtx *ckks_fhn_create(const char *json) {
  try {
    auto params = configuration(json);
    if (!params)
      return nullptr;
    auto tables = EvaluationTables::create(*params);
    return tables ? new FhnBackendCtx(std::move(tables)) : nullptr;
  } catch (...) {
    return nullptr;
  }
}
void ckks_fhn_destroy(FhnBackendCtx *ctx) { delete ctx; }
FhnKernelTable *ckks_fhn_get_kernels(FhnBackendCtx *ctx) { return ctx ? &kernel_table : nullptr; }
FhnBuffer *ckks_fhn_buffer_alloc(FhnBackendCtx *ctx) {
  try {
    return ctx ? new FhnBuffer(ctx) : nullptr;
  } catch (...) {
    return nullptr;
  }
}
void ckks_fhn_buffer_free(FhnBackendCtx *ctx, FhnBuffer *buffer) {
  if (buffer && buffer->owner == ctx)
    delete buffer;
}
int64_t ckks_fhn_fresh_level(FhnBackendCtx *ctx) { return ctx ? ctx->tables->params().freshLevel() : -1; }
uint64_t ckks_fhn_level_bytes(FhnBackendCtx *ctx, int64_t level) {
  if (!ctx || level < 0 || level > ctx->tables->params().freshLevel())
    return 0;
  // Report physical reserve for the existing movement budget. A lower level
  // shrinks live payload but cannot release a fixed slab's allocation.
  return 3 * ctx->stride * sizeof(uint64_t);
}
FhnLevelEffect ckks_fhn_opcode_level_effect(FhnBackendCtx *, FhnOpCode op) {
  if (op == FHN_HMULT || op == FHN_RESCALE || op == FHN_MULT_CS)
    return FHN_LEVEL_CONSUME;
  return op == FHN_LEVEL_DOWN ? FHN_LEVEL_SET_PARAM0 : FHN_LEVEL_PRESERVE;
}
FhnClientCtx *ckks_fhn_client_create(FhnBackendCtx *ctx) {
  if (!ctx || ctx->client_created)
    return nullptr;
  try {
    auto client = std::make_unique<FhnClientCtx>(ctx);
    client->secret = generateSecretKey(*ctx->tables, client->random, client->scratch);
    if (!client->secret)
      return nullptr;
    client->public_key = generatePublicKey(*client->secret, *ctx->tables, client->random, client->scratch);
    if (!client->public_key)
      return nullptr;
    ctx->client_created = true;
    return client.release();
  } catch (...) {
    return nullptr;
  }
}
void ckks_fhn_client_destroy(FhnClientCtx *client) { delete client; }
int ckks_fhn_client_prepare(FhnClientCtx *client, const FhnProgram *program) {
  if (!client || !program || program->version != FHN_ABI_VERSION ||
      (program->num_instructions && !program->instructions))
    return -1;
  try {
    auto *ctx = client->backend;
    bool needs_relin = false;
    std::set<uint64_t> rotations;
    for (uint32_t i = 0; i < program->num_instructions; ++i) {
      const auto &inst = program->instructions[i];
      needs_relin = needs_relin || inst.opcode == FHN_HMULT || inst.opcode == FHN_RELINEARIZE;
      if (inst.opcode == FHN_ROTATE || inst.opcode == FHN_HROT || inst.opcode == FHN_HROT_ADD)
        rotations.insert(eval::galoisElement(inst.params[0], ctx->tables->params().logDegree()));
      if (inst.opcode == FHN_CONJUGATE || inst.opcode == FHN_HCONJ_ADD)
        rotations.insert(2 * ctx->tables->params().degree() - 1);
    }
    if (needs_relin && !client->relin) {
      client->relin = generateRelinearizationKey(*client->secret, *ctx->tables, client->random, client->scratch);
      if (!client->relin)
        return -1;
      ctx->relin = client->relin;
    }
    for (uint64_t g : rotations) {
      if (g == 1)
        continue;
      auto found = client->rotations.find(g);
      if (found == client->rotations.end()) {
        auto key = generateGaloisKey(*client->secret, g, *ctx->tables, client->random, client->scratch);
        if (!key)
          return -1;
        found = client->rotations.emplace(g, std::move(key)).first;
      }
      // Registration may have thrown after the client retained this key.
      // A retry must restore the observer even when generation is complete.
      ctx->rotations.insert_or_assign(g, found->second);
    }
    return 0;
  } catch (...) {
    return -1;
  }
}
int ckks_fhn_client_encrypt_vec_f64(FhnClientCtx *client, FhnBuffer *out, const double *values, uint32_t count) {
  return encryptVector(client, out, values, count);
}
int ckks_fhn_client_decrypt_vec_f64(FhnClientCtx *client, const FhnBuffer *in, double *values, uint32_t count) {
  return decryptVector(client, in, values, count);
}
int ckks_fhn_client_encrypt_vec_c64(FhnClientCtx *client, FhnBuffer *out, const FhnComplex64 *values, uint32_t count) {
  return encryptVector(client, out, values, count);
}
int ckks_fhn_client_decrypt_vec_c64(FhnClientCtx *client, const FhnBuffer *in, FhnComplex64 *values, uint32_t count) {
  return decryptVector(client, in, values, count);
}
int ckks_fhn_buffer_info(FhnBackendCtx *ctx, const FhnBuffer *buffer, FhnCkksBufferInfo *out) {
  if (!ctx || !buffer || buffer->owner != ctx || !out)
    return -1;
  *out = {buffer->slab.capacityWords() * sizeof(uint64_t),
          buffer->logical_slots ? buffer->layout.byteCount() : 0,
          ctx->scratch.capacity(),
          ctx->scratch.highWater(),
          buffer->logical_slots ? buffer->layout.level() : -1,
          buffer->layout.scale,
          buffer->layout.num_polys,
          buffer->logical_slots};
  return 0;
}
}
