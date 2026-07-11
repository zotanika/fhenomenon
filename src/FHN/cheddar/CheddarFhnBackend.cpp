// ---------------------------------------------------------------------------
// Cheddar-FHE GPU backend for the FHN kernel API.
//
// This file implements the 4 required fhn_* exports and the host-side data
// plane (fhn_buffer_alloc/free, fhn_encrypt_f64/fhn_decrypt_f64) for
// cheddar-fhe (CKKS). It is compiled into libcheddar_fhn.so and loaded via
// ExternalBackend.
// ---------------------------------------------------------------------------

#include "FHN/CheddarFhnBackend.h"

#include "ScalarOps.h"
#include "UserInterface.h"
#include "core/Context.h"
#include "core/Encode.h"
#include "core/Parameter.h"
#include "core/Type.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <complex>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

using json = nlohmann::json;
using word = uint32_t;

// --- Concrete types behind the opaque handles ---

struct FhnBackendCtx {
  std::unique_ptr<cheddar::Parameter<word>> param;
  cheddar::ContextPtr<word> context;
  std::unique_ptr<cheddar::UserInterface<word>> ui;
  int default_encryption_level;
  double default_scale;
  int log_degree;
  // Rotation distances (normalized to [1, degree/2)) whose evaluation keys
  // were prepared via UserInterface::PrepareRotationKey. Kernels must consult
  // this set before any GetRotationKey call: cheddar's EvkMap AssertTrue()s
  // on a missing key or rot_idx <= 0, which std::exit()s the whole process.
  std::set<int> prepared_rotation_keys;
};

enum class CheddarBufKind {
  Empty,
  Ciphertext,
  Plaintext,
  Message, // decoded complex vector
};

struct FhnBuffer {
  CheddarBufKind kind = CheddarBufKind::Empty;
  cheddar::Ciphertext<word> ct;
  cheddar::Plaintext<word> pt;
  std::vector<cheddar::Complex> msg;
};

// --- Helper: parse JSON config and create Parameter ---

static FhnBackendCtx *createFromJson(const char *config_json) {
  auto *ctx = new FhnBackendCtx();

  json j;
  if (config_json && strlen(config_json) > 0) {
    // Try as file path first, then as raw JSON
    std::ifstream f(config_json);
    if (f.is_open()) {
      j = json::parse(f);
    } else {
      j = json::parse(config_json);
    }
  } else {
    // Default: try to find parameter file
    throw std::runtime_error("cheddar fhn_create: config_json is required");
  }

  ctx->log_degree = j.at("log_degree").get<int>();
  int log_default_scale = j.at("log_default_scale").get<int>();
  ctx->default_scale = static_cast<double>(UINT64_C(1) << log_default_scale);
  ctx->default_encryption_level = j.at("default_encryption_level").get<int>();

  std::vector<word> main_primes;
  for (auto &p : j.at("main_primes"))
    main_primes.push_back(p.get<word>());

  std::vector<word> aux_primes;
  for (auto &p : j.at("auxiliary_primes"))
    aux_primes.push_back(p.get<word>());

  std::vector<word> ter_primes;
  if (j.contains("terminal_primes")) {
    for (auto &p : j["terminal_primes"])
      ter_primes.push_back(p.get<word>());
  }

  std::vector<std::pair<int, int>> level_config;
  for (auto &pair : j.at("level_config"))
    level_config.emplace_back(pair[0].get<int>(), pair[1].get<int>());

  std::pair<int, int> additional_base = {0, 0};
  if (j.contains("additional_base")) {
    additional_base = {j["additional_base"][0].get<int>(), j["additional_base"][1].get<int>()};
  }

  ctx->param =
    std::make_unique<cheddar::Parameter<word>>(ctx->log_degree, ctx->default_scale, ctx->default_encryption_level,
                                               level_config, main_primes, aux_primes, ter_primes, additional_base);

  if (j.contains("dense_hamming_weight"))
    ctx->param->SetDenseHammingWeight(j["dense_hamming_weight"].get<int>());
  if (j.contains("sparse_hamming_weight"))
    ctx->param->SetSparseHammingWeight(j["sparse_hamming_weight"].get<int>());

  ctx->context = cheddar::Context<word>::Create(*ctx->param);
  ctx->ui = std::make_unique<cheddar::UserInterface<word>>(ctx->context);

  // Optional config key: "rotation_keys": [1, 2, 4, ...]
  // Rotation distances to prepare evaluation keys for. Entries are normalized
  // modulo the full slot count (degree / 2); entries that normalize to 0
  // (identity) are skipped. The UserInterface constructor only auto-prepares
  // the multiplication/conjugation keys, and the rotation kernels below
  // refuse (return non-zero) any distance whose key is not listed here.
  // NOTE: cheddar reduces a rotation distance modulo the *ciphertext's* slot
  // count (Context::HRot/HRotAdd), so for sparse-packed messages with
  // n' < degree/2 slots, list the distance reduced mod n' (e.g. rotate(-1)
  // on a 4-slot message needs key 3).
  if (j.contains("rotation_keys")) {
    const int64_t num_slots = static_cast<int64_t>(ctx->param->degree_) / 2;
    for (const auto &r : j["rotation_keys"]) {
      int64_t raw = r.get<int64_t>();
      int dist = static_cast<int>(((raw % num_slots) + num_slots) % num_slots);
      if (dist == 0 || ctx->prepared_rotation_keys.count(dist))
        continue;
      // PrepareRotationKey asserts 0 < rot_idx < degree/2, which the
      // normalization above guarantees. Pass max_level_ explicitly: despite
      // the header doc ("default: -1 --> param_->max_level_"), GetNPForEvk
      // treats -1 as the dense-to-sparse short-base case and would build the
      // key over the wrong prime set — rotating a normal ciphertext would
      // then hit "Beta mismatch" (AssertTrue -> std::exit) in MultKey.
      ctx->ui->PrepareRotationKey(dist, ctx->param->max_level_);
      ctx->prepared_rotation_keys.insert(dist);
    }
  }

  return ctx;
}

// --- Kernel argument guards ------------------------------------------------
// The executor passes nullptr for unused operand slots (operand id 0) and
// makes no promises about buffer payload kinds, so every kernel validates
// the operands it dereferences and returns non-zero instead of crashing.

static inline bool isCiphertext(const FhnBuffer *buf) { return buf && buf->kind == CheddarBufKind::Ciphertext; }

static inline bool isPlaintext(const FhnBuffer *buf) { return buf && buf->kind == CheddarBufKind::Plaintext; }

// cheddar's Context AssertTrue()s (std::exit) on operand NP or scale
// mismatch; check the same preconditions recoverably instead. Mult-family
// ops (Mult/HMult) assert matching NP only — operand scales may
// legitimately differ, the product scale is their product — while
// Add/Sub/HRotAdd/HConjAdd assert matching NP and scale.
template <typename A, typename B> static bool sameNp(const A &a, const B &b) { return a.GetNP() == b.GetNP(); }

// Margin mirrors Context::AssertSameScale's kScaleErrorMargin.
template <typename A, typename B> static bool sameNpAndScale(const A &a, const B &b) {
  if (!sameNp(a, b))
    return false;
  double diff = a.GetScale() - b.GetScale();
  diff = diff < 0 ? -diff : diff;
  return diff < 1e-12 * a.GetScale();
}

// Normalize a raw rotation distance into [0, num_slots) and check that its
// evaluation key is prepared. cheddar reduces rotation distances modulo the
// ciphertext's slot count (Context::HRot/HRotAdd) and applies the
// automorphism of the reduced index, so the key must be looked up under the
// reduced index as well. Returns the normalized distance (0 = identity, no
// key needed), or -1 if the key for a non-zero distance was never prepared.
// Callers must not call GetRotationKey unless this returns > 0: cheddar's
// EvkMap AssertTrue()s on a missing key or rot_idx <= 0, which std::exit()s
// the whole process.
static int normalizeRotationDistance(const FhnBackendCtx *ctx, int64_t raw_dist, int64_t num_slots) {
  if (num_slots <= 0)
    return -1;
  int dist = static_cast<int>(((raw_dist % num_slots) + num_slots) % num_slots);
  if (dist == 0)
    return 0;
  if (ctx->prepared_rotation_keys.count(dist) == 0)
    return -1;
  return dist;
}

// --- Kernel implementations ------------------------------------------------
// Compute-only: encode/encrypt/decrypt/decode are host-side data-plane
// exports below, never kernel-table entries.

// --- Arithmetic kernels ---

static int cheddar_add_cc(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                          const double *) {
  if (!isCiphertext(operands[0]))
    return -1;
  if (!isCiphertext(operands[1]))
    return -1;
  if (!sameNpAndScale(operands[0]->ct, operands[1]->ct))
    return -1; // Context::Add(Ct,Ct) asserts same NP and scale
  ctx->context->Add(result->ct, operands[0]->ct, operands[1]->ct);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_add_cp(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                          const double *) {
  if (!isCiphertext(operands[0]))
    return -1;
  if (!isPlaintext(operands[1]))
    return -1;
  if (!sameNpAndScale(operands[0]->ct, operands[1]->pt))
    return -1; // Context::Add(Ct,Pt) asserts same NP and scale
  ctx->context->Add(result->ct, operands[0]->ct, operands[1]->pt);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_sub_cc(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                          const double *) {
  if (!isCiphertext(operands[0]))
    return -1;
  if (!isCiphertext(operands[1]))
    return -1;
  if (!sameNpAndScale(operands[0]->ct, operands[1]->ct))
    return -1; // Context::Sub(Ct,Ct) asserts same NP and scale
  ctx->context->Sub(result->ct, operands[0]->ct, operands[1]->ct);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_sub_cp(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                          const double *) {
  if (!isCiphertext(operands[0]))
    return -1;
  if (!isPlaintext(operands[1]))
    return -1;
  if (!sameNpAndScale(operands[0]->ct, operands[1]->pt))
    return -1; // Context::Sub(Ct,Pt) asserts same NP and scale
  ctx->context->Sub(result->ct, operands[0]->ct, operands[1]->pt);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_negate(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                          const double *) {
  if (!isCiphertext(operands[0]))
    return -1;
  ctx->context->Neg(result->ct, operands[0]->ct);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_mult_cc(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                           const double *) {
  if (!isCiphertext(operands[0]))
    return -1;
  if (!isCiphertext(operands[1]))
    return -1;
  if (!sameNp(operands[0]->ct, operands[1]->ct))
    return -1; // Context::Mult(Ct,Ct) asserts same NP (scales may differ)
  ctx->context->Mult(result->ct, operands[0]->ct, operands[1]->ct);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_mult_cp(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                           const double *) {
  if (!isCiphertext(operands[0]))
    return -1;
  if (!isPlaintext(operands[1]))
    return -1;
  if (!sameNp(operands[0]->ct, operands[1]->pt))
    return -1; // Context::Mult(Ct,Pt) asserts same NP (scales may differ)
  ctx->context->Mult(result->ct, operands[0]->ct, operands[1]->pt);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

// --- Scalar kernels (powered by cheddar::ScalarOps) ---
// These use the convenience helpers from ScalarOps.h.
// fparams[0] = the scalar value.

static int cheddar_add_cs(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                          const double *fparams) {
  if (!isCiphertext(operands[0]))
    return -1;
  cheddar::AddScalar(*ctx->context, result->ct, operands[0]->ct, fparams[0]);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_mult_cs(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                           const double *fparams) {
  if (!isCiphertext(operands[0]))
    return -1;
  cheddar::MultScalar(*ctx->context, result->ct, operands[0]->ct, fparams[0]);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_sub_sc(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                          const double *fparams) {
  if (!isCiphertext(operands[0]))
    return -1;
  // FHN_SUB_SC: scalar - ciphertext → negate + add_scalar
  cheddar::Ciphertext<word> neg;
  ctx->context->Neg(neg, operands[0]->ct);
  cheddar::AddScalar(*ctx->context, result->ct, neg, fparams[0]);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

// --- Key-switching kernels ---

static int cheddar_relinearize(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                               const double *) {
  if (!isCiphertext(operands[0]))
    return -1;
  ctx->context->Relinearize(result->ct, operands[0]->ct, ctx->ui->GetMultiplicationKey());
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_rescale(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                           const double *) {
  if (!isCiphertext(operands[0]))
    return -1;
  ctx->context->Rescale(result->ct, operands[0]->ct);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_rotate(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                          const int64_t *params, const double *) {
  if (!isCiphertext(operands[0]))
    return -1;
  int dist = normalizeRotationDistance(ctx, params[0], operands[0]->ct.GetNumSlots());
  if (dist < 0)
    return -1; // rotation key not prepared
  if (dist == 0) {
    // Identity rotation: plain copy, no key lookup.
    ctx->context->Copy(result->ct, operands[0]->ct);
    result->kind = CheddarBufKind::Ciphertext;
    return 0;
  }
  ctx->context->HRot(result->ct, operands[0]->ct, ctx->ui->GetRotationKey(dist), dist);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_conjugate(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                             const double *) {
  if (!isCiphertext(operands[0]))
    return -1;
  ctx->context->HConj(result->ct, operands[0]->ct, ctx->ui->GetConjugationKey());
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_mult_key(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                            const int64_t *params, const double *) {
  if (!isCiphertext(operands[0]))
    return -1;
  // params[0] = rotation index for key lookup (0 = mult key). MultKey is a
  // raw key switch (no slot permutation), so the rotation index is
  // normalized over the full key-index domain [1, degree/2).
  if (params[0] == 0) {
    ctx->context->MultKey(result->ct, operands[0]->ct, ctx->ui->GetMultiplicationKey());
    result->kind = CheddarBufKind::Ciphertext;
    return 0;
  }
  int key_idx = normalizeRotationDistance(ctx, params[0], static_cast<int64_t>(ctx->param->degree_) / 2);
  if (key_idx <= 0)
    return -1; // key not prepared (or index normalizes to 0: no such key)
  ctx->context->MultKey(result->ct, operands[0]->ct, ctx->ui->GetRotationKey(key_idx));
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

// --- Fused composite kernels ---

static int cheddar_hmult(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                         const double *) {
  if (!isCiphertext(operands[0]))
    return -1;
  if (!isCiphertext(operands[1]))
    return -1;
  if (!sameNp(operands[0]->ct, operands[1]->ct))
    return -1; // HMult delegates to Mult(Ct,Ct): same NP (scales may differ)
  ctx->context->HMult(result->ct, operands[0]->ct, operands[1]->ct, ctx->ui->GetMultiplicationKey(), /*rescale=*/true);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_hrot(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *params,
                        const double *) {
  // Same contract and guards as FHN_ROTATE.
  return cheddar_rotate(ctx, result, operands, params, nullptr);
}

static int cheddar_hrot_add(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                            const int64_t *params, const double *) {
  if (!isCiphertext(operands[0]))
    return -1;
  if (!isCiphertext(operands[1]))
    return -1;
  if (!sameNpAndScale(operands[0]->ct, operands[1]->ct))
    return -1; // Context::HRotAdd asserts same NP and scale
  // HRotAdd reduces the distance modulo max(a.slots, b.slots); mirror that.
  int64_t num_slots = std::max(operands[0]->ct.GetNumSlots(), operands[1]->ct.GetNumSlots());
  int dist = normalizeRotationDistance(ctx, params[0], num_slots);
  if (dist < 0)
    return -1; // rotation key not prepared
  if (dist == 0) {
    // rot(a, 0) + b == a + b: plain addition, no key lookup.
    ctx->context->Add(result->ct, operands[0]->ct, operands[1]->ct);
    result->kind = CheddarBufKind::Ciphertext;
    return 0;
  }
  ctx->context->HRotAdd(result->ct, operands[0]->ct, operands[1]->ct, ctx->ui->GetRotationKey(dist), dist);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_hconj_add(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                             const double *) {
  if (!isCiphertext(operands[0]))
    return -1;
  if (!isCiphertext(operands[1]))
    return -1;
  if (!sameNpAndScale(operands[0]->ct, operands[1]->ct))
    return -1; // Context::HConjAdd asserts same NP and scale
  ctx->context->HConjAdd(result->ct, operands[0]->ct, operands[1]->ct, ctx->ui->GetConjugationKey());
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_level_down(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                              const int64_t *params, const double *) {
  if (!isCiphertext(operands[0]))
    return -1;
  int target_level = static_cast<int>(params[0]);
  // Context::LevelDown AssertTrue()s (std::exit) when target_level exceeds
  // the operand's current level, and for negative targets its loop indexes
  // level_down_consts_ out of bounds; validate the range recoverably here.
  int current_level = ctx->param->NPToLevel(operands[0]->ct.GetNP());
  if (target_level < 0 || target_level > current_level)
    return -1;
  ctx->context->LevelDown(result->ct, operands[0]->ct, target_level);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

// --- Kernel table ----------------------------------------------------------

static FhnKernelEntry cheddar_kernels[] = {
  {FHN_ADD_CC, cheddar_add_cc, "add_cc"},
  {FHN_ADD_CP, cheddar_add_cp, "add_cp"},
  {FHN_ADD_CS, cheddar_add_cs, "add_cs"},
  {FHN_SUB_CC, cheddar_sub_cc, "sub_cc"},
  {FHN_SUB_CP, cheddar_sub_cp, "sub_cp"},
  {FHN_SUB_SC, cheddar_sub_sc, "sub_sc"},
  {FHN_NEGATE, cheddar_negate, "negate"},
  {FHN_MULT_CC, cheddar_mult_cc, "mult_cc"},
  {FHN_MULT_CP, cheddar_mult_cp, "mult_cp"},
  {FHN_MULT_CS, cheddar_mult_cs, "mult_cs"},
  {FHN_RELINEARIZE, cheddar_relinearize, "relinearize"},
  {FHN_RESCALE, cheddar_rescale, "rescale"},
  {FHN_ROTATE, cheddar_rotate, "rotate"},
  {FHN_CONJUGATE, cheddar_conjugate, "conjugate"},
  {FHN_MULT_KEY, cheddar_mult_key, "mult_key"},
  {FHN_HMULT, cheddar_hmult, "hmult"},
  {FHN_HROT, cheddar_hrot, "hrot"},
  {FHN_HROT_ADD, cheddar_hrot_add, "hrot_add"},
  {FHN_HCONJ_ADD, cheddar_hconj_add, "hconj_add"},
  {FHN_LEVEL_DOWN, cheddar_level_down, "level_down"},
};

static FhnKernelTable cheddar_kernel_table = {
  sizeof(cheddar_kernels) / sizeof(cheddar_kernels[0]),
  cheddar_kernels,
};

// --- Backend exports -------------------------------------------------------

extern "C" {

uint32_t fhn_get_abi_version(void) { return FHN_ABI_VERSION; }

FhnBackendInfo *fhn_get_info(void) {
  static FhnBackendInfo info = {
    "cheddar-ckks", "1.0", FHN_DEVICE_GPU,
    0, // detected at runtime
  };
  return &info;
}

FhnBackendCtx *fhn_create(const char *config_json) {
  try {
    return createFromJson(config_json);
  } catch (const std::exception &e) {
    std::cerr << "cheddar fhn_create error: " << e.what() << std::endl;
    return nullptr;
  }
}

void fhn_destroy(FhnBackendCtx *ctx) { delete ctx; }

FhnKernelTable *fhn_get_kernels(FhnBackendCtx * /*ctx*/) { return &cheddar_kernel_table; }

// --- Host-side data plane --------------------------------------------------
// These exports handle plaintexts and key material. The trusted host calls
// them directly; they are never dispatched from an FhnProgram. Like the four
// core exports above, they are un-prefixed so a prefix-less ExternalBackend
// resolves the whole data plane.

FhnBuffer *fhn_buffer_alloc(FhnBackendCtx * /*ctx*/) { return new FhnBuffer{}; }

void fhn_buffer_free(FhnBackendCtx * /*ctx*/, FhnBuffer *buf) { delete buf; }

int fhn_encrypt_f64(FhnBackendCtx *ctx, FhnBuffer *out, double value) {
  if (!out)
    return -1;
  // Fused encode + encrypt: single-slot message at the default level/scale.
  int level = ctx->default_encryption_level;
  double scale = ctx->param->GetScale(level);
  std::vector<cheddar::Complex> msg = {{value, 0.0}};
  cheddar::Plaintext<word> pt;
  ctx->context->encoder_.Encode(pt, level, scale, msg);
  ctx->ui->Encrypt(out->ct, pt);
  out->kind = CheddarBufKind::Ciphertext;
  return 0;
}

int fhn_decrypt_f64(FhnBackendCtx *ctx, const FhnBuffer *in, double *value_out) {
  if (!in || !value_out || in->kind != CheddarBufKind::Ciphertext)
    return -1;
  // Fused decrypt + decode: single-value convention is the first slot.
  cheddar::Plaintext<word> pt;
  ctx->ui->Decrypt(pt, in->ct);
  std::vector<cheddar::Complex> msg;
  ctx->context->encoder_.Decode(msg, pt);
  if (msg.empty())
    return -1;
  *value_out = msg[0].real();
  return 0;
}

// --- Buffer helpers (Cheddar-specific, for tests/examples) -----------------

int cheddar_fhn_buffer_encrypt_message(FhnBackendCtx *ctx, FhnBuffer *buf) {
  if (!buf || buf->kind != CheddarBufKind::Message)
    return -1;
  // Fused encode + encrypt of the staged message, in place.
  int level = ctx->default_encryption_level;
  double scale = ctx->param->GetScale(level);
  cheddar::Plaintext<word> pt;
  ctx->context->encoder_.Encode(pt, level, scale, buf->msg);
  ctx->ui->Encrypt(buf->ct, pt);
  buf->kind = CheddarBufKind::Ciphertext;
  return 0;
}

void cheddar_fhn_buffer_read_complex(FhnBackendCtx *ctx, FhnBuffer *buf, double *real_out, double *imag_out,
                                     int max_slots) {
  if (!buf || !real_out || !imag_out)
    return;
  if (buf->kind == CheddarBufKind::Ciphertext) {
    // Fused decrypt + decode of the ciphertext, in place.
    cheddar::Plaintext<word> pt;
    ctx->ui->Decrypt(pt, buf->ct);
    ctx->context->encoder_.Decode(buf->msg, pt);
    buf->kind = CheddarBufKind::Message;
  }
  if (buf->kind != CheddarBufKind::Message)
    return;
  int n = std::min(max_slots, static_cast<int>(buf->msg.size()));
  for (int i = 0; i < n; i++) {
    real_out[i] = buf->msg[i].real();
    imag_out[i] = buf->msg[i].imag();
  }
}

} // extern "C"
