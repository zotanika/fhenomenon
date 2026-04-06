// ---------------------------------------------------------------------------
// Cheddar-FHE GPU backend for the FHN kernel API.
//
// This file implements the 4 required fhn_* exports for cheddar-fhe (CKKS).
// It is compiled into libcheddar_fhn.so and loaded via ExternalBackend.
// ---------------------------------------------------------------------------

#include "FHN/CheddarFhnBackend.h"

#include "ScalarOps.h"
#include "UserInterface.h"
#include "core/Context.h"
#include "core/Encode.h"
#include "core/Parameter.h"
#include "core/Type.h"

#include <nlohmann/json.hpp>

#include <complex>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
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

  return ctx;
}

// --- Kernel implementations ------------------------------------------------

static int cheddar_encode(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                          const int64_t *params, const double *fparams) {
  // params[0] = level, fparams[0] = scale (0 = use default)
  int level = params[0] > 0 ? static_cast<int>(params[0]) : ctx->default_encryption_level;
  double scale = fparams[0] > 0.0 ? fparams[0] : ctx->param->GetScale(level);

  // operands[0] should contain the message
  if (!operands[0] || operands[0]->kind != CheddarBufKind::Message)
    return -1;

  ctx->context->encoder_.Encode(result->pt, level, scale, operands[0]->msg);
  result->kind = CheddarBufKind::Plaintext;
  return 0;
}

static int cheddar_encrypt(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                           const int64_t * /*params*/, const double * /*fparams*/) {
  // operands[0] should contain plaintext
  if (!operands[0] || operands[0]->kind != CheddarBufKind::Plaintext)
    return -1;

  ctx->ui->Encrypt(result->ct, operands[0]->pt);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_decrypt(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                           const int64_t * /*params*/, const double * /*fparams*/) {
  if (!operands[0] || operands[0]->kind != CheddarBufKind::Ciphertext)
    return -1;

  ctx->ui->Decrypt(result->pt, operands[0]->ct);
  result->kind = CheddarBufKind::Plaintext;
  return 0;
}

static int cheddar_decode(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                          const int64_t * /*params*/, const double * /*fparams*/) {
  if (!operands[0] || operands[0]->kind != CheddarBufKind::Plaintext)
    return -1;

  ctx->context->encoder_.Decode(result->msg, operands[0]->pt);
  result->kind = CheddarBufKind::Message;
  return 0;
}

// --- Arithmetic kernels ---

static int cheddar_add_cc(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                          const double *) {
  ctx->context->Add(result->ct, operands[0]->ct, operands[1]->ct);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_add_cp(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                          const double *) {
  ctx->context->Add(result->ct, operands[0]->ct, operands[1]->pt);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_sub_cc(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                          const double *) {
  ctx->context->Sub(result->ct, operands[0]->ct, operands[1]->ct);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_sub_cp(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                          const double *) {
  ctx->context->Sub(result->ct, operands[0]->ct, operands[1]->pt);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_negate(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                          const double *) {
  ctx->context->Neg(result->ct, operands[0]->ct);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_mult_cc(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                           const double *) {
  ctx->context->Mult(result->ct, operands[0]->ct, operands[1]->ct);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_mult_cp(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                           const double *) {
  ctx->context->Mult(result->ct, operands[0]->ct, operands[1]->pt);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

// --- Scalar kernels (powered by cheddar::ScalarOps) ---
// These use the convenience helpers from ScalarOps.h.
// fparams[0] = the scalar value.

static int cheddar_add_cs(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                          const double *fparams) {
  cheddar::AddScalar(*ctx->context, result->ct, operands[0]->ct, fparams[0]);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_mult_cs(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                           const double *fparams) {
  cheddar::MultScalar(*ctx->context, result->ct, operands[0]->ct, fparams[0]);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_sub_sc(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                          const double *fparams) {
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
  ctx->context->Relinearize(result->ct, operands[0]->ct, ctx->ui->GetMultiplicationKey());
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_rescale(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                           const double *) {
  ctx->context->Rescale(result->ct, operands[0]->ct);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_rotate(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                          const int64_t *params, const double *) {
  int dist = static_cast<int>(params[0]);
  ctx->context->HRot(result->ct, operands[0]->ct, ctx->ui->GetRotationKey(dist), dist);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_conjugate(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                             const double *) {
  ctx->context->HConj(result->ct, operands[0]->ct, ctx->ui->GetConjugationKey());
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_mult_key(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                            const int64_t *params, const double *) {
  // params[0] = rotation index for key lookup (0 = mult key)
  int key_idx = static_cast<int>(params[0]);
  const auto &key = (key_idx == 0) ? ctx->ui->GetMultiplicationKey() : ctx->ui->GetRotationKey(key_idx);
  ctx->context->MultKey(result->ct, operands[0]->ct, key);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

// --- Fused composite kernels ---

static int cheddar_hmult(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                         const double *) {
  ctx->context->HMult(result->ct, operands[0]->ct, operands[1]->ct, ctx->ui->GetMultiplicationKey(), /*rescale=*/true);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_hrot(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *params,
                        const double *) {
  int dist = static_cast<int>(params[0]);
  ctx->context->HRot(result->ct, operands[0]->ct, ctx->ui->GetRotationKey(dist), dist);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_hrot_add(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                            const int64_t *params, const double *) {
  int dist = static_cast<int>(params[0]);
  ctx->context->HRotAdd(result->ct, operands[0]->ct, operands[1]->ct, ctx->ui->GetRotationKey(dist), dist);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_hconj_add(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *,
                             const double *) {
  ctx->context->HConjAdd(result->ct, operands[0]->ct, operands[1]->ct, ctx->ui->GetConjugationKey());
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

static int cheddar_level_down(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                              const int64_t *params, const double *) {
  int target_level = static_cast<int>(params[0]);
  ctx->context->LevelDown(result->ct, operands[0]->ct, target_level);
  result->kind = CheddarBufKind::Ciphertext;
  return 0;
}

// --- Kernel table ----------------------------------------------------------

static FhnKernelEntry cheddar_kernels[] = {
  {FHN_ENCODE, cheddar_encode, "encode"},
  {FHN_ENCRYPT, cheddar_encrypt, "encrypt"},
  {FHN_DECRYPT, cheddar_decrypt, "decrypt"},
  {FHN_DECODE, cheddar_decode, "decode"},
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

FhnBackendInfo *fhn_get_info(void) {
  static FhnBackendInfo info = {
    "cheddar-ckks",
    "1.0",
    FHN_DEVICE_GPU,
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

// --- Buffer helpers --------------------------------------------------------

FhnBuffer *cheddar_fhn_buffer_alloc(FhnBackendCtx * /*ctx*/) { return new FhnBuffer{}; }

void cheddar_fhn_buffer_free(FhnBackendCtx * /*ctx*/, FhnBuffer *buf) { delete buf; }

void cheddar_fhn_buffer_read_complex(FhnBackendCtx * /*ctx*/, FhnBuffer *buf, double *real_out, double *imag_out,
                                     int max_slots) {
  if (buf->kind != CheddarBufKind::Message)
    return;
  int n = std::min(max_slots, static_cast<int>(buf->msg.size()));
  for (int i = 0; i < n; i++) {
    real_out[i] = buf->msg[i].real();
    imag_out[i] = buf->msg[i].imag();
  }
}

} // extern "C"
