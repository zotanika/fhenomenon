#include "FHN/ToyFheKernels.h"
#include "Crypto/ToyFHE.h"

#include <cmath>

// ---------------------------------------------------------------------------
// ToyFHE reference backend for the FHN kernel API.
// This file is intentionally kept simple — it is the template that FHE
// library developers copy when integrating their own library.
// ---------------------------------------------------------------------------

// --- Concrete types behind the opaque handles ---

struct FhnBackendCtx {
  fhenomenon::toyfhe::Engine engine;
  fhenomenon::toyfhe::Parameters params;
};

enum class BufKind { Empty, Ciphertext, IntValue, DoubleValue };

struct FhnBuffer {
  BufKind kind = BufKind::Empty;
  fhenomenon::toyfhe::Ciphertext ct;
  int64_t int_val = 0;
  double double_val = 0.0;
};

// --- Kernel implementations ------------------------------------------------

static int toyfhe_add_cc(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                         const int64_t * /*params*/, const double * /*fparams*/) {
  result->ct = ctx->engine.add(operands[0]->ct, operands[1]->ct);
  result->kind = BufKind::Ciphertext;
  return 0;
}

static int toyfhe_add_cs(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                         const int64_t * /*params*/, const double *fparams) {
  result->ct = ctx->engine.addPlain(operands[0]->ct, fparams[0]);
  result->kind = BufKind::Ciphertext;
  return 0;
}

static int toyfhe_sub_cc(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                         const int64_t * /*params*/, const double * /*fparams*/) {
  // ToyFHE has no direct subtract — negate b then add.
  fhenomenon::toyfhe::Ciphertext neg_b = operands[1]->ct;
  neg_b.c0 = -neg_b.c0;
  neg_b.c1 = -neg_b.c1;
  result->ct = ctx->engine.add(operands[0]->ct, neg_b);
  result->kind = BufKind::Ciphertext;
  return 0;
}

static int toyfhe_negate(FhnBackendCtx * /*ctx*/, FhnBuffer *result, const FhnBuffer *const *operands,
                         const int64_t * /*params*/, const double * /*fparams*/) {
  result->ct = operands[0]->ct;
  result->ct.c0 = -result->ct.c0;
  result->ct.c1 = -result->ct.c1;
  result->kind = BufKind::Ciphertext;
  return 0;
}

static int toyfhe_mult_cc(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                          const int64_t * /*params*/, const double * /*fparams*/) {
  result->ct = ctx->engine.multiply(operands[0]->ct, operands[1]->ct);
  result->kind = BufKind::Ciphertext;
  return 0;
}

static int toyfhe_mult_cs(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                          const int64_t * /*params*/, const double *fparams) {
  result->ct = ctx->engine.multiplyPlain(operands[0]->ct, fparams[0]);
  result->kind = BufKind::Ciphertext;
  return 0;
}

// ToyFHE multiply performs relinearization and rescale internally, so the
// fused HMULT (mult + relin + rescale) is exactly MULT_CC.
static int toyfhe_hmult(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands, const int64_t *params,
                        const double *fparams) {
  return toyfhe_mult_cc(ctx, result, operands, params, fparams);
}

// --- Kernel table ----------------------------------------------------------

// Compute-only: encryption and decryption are host-side data-plane exports
// below, never kernel-table entries.
static FhnKernelEntry toyfhe_kernels[] = {
  {FHN_ADD_CC, toyfhe_add_cc, "add_cc"},    {FHN_ADD_CS, toyfhe_add_cs, "add_cs"},
  {FHN_SUB_CC, toyfhe_sub_cc, "sub_cc"},    {FHN_NEGATE, toyfhe_negate, "negate"},
  {FHN_MULT_CC, toyfhe_mult_cc, "mult_cc"}, {FHN_MULT_CS, toyfhe_mult_cs, "mult_cs"},
  {FHN_HMULT, toyfhe_hmult, "hmult"},
};

static FhnKernelTable toyfhe_kernel_table = {
  sizeof(toyfhe_kernels) / sizeof(toyfhe_kernels[0]),
  toyfhe_kernels,
};

// --- Backend exports -------------------------------------------------------

FhnBackendInfo *toyfhe_fhn_get_info(void) {
  static FhnBackendInfo info = {
    "toyfhe-reference",
    "0.1",
    FHN_DEVICE_CPU,
    0,
  };
  return &info;
}

FhnBackendCtx *toyfhe_fhn_create(const char * /*config_json*/) {
  auto *ctx = new FhnBackendCtx();
  ctx->params = fhenomenon::toyfhe::Parameters{}; // defaults
  ctx->engine.initialize(ctx->params);
  ctx->engine.generateKeys();
  return ctx;
}

void toyfhe_fhn_destroy(FhnBackendCtx *ctx) { delete ctx; }

FhnKernelTable *toyfhe_fhn_get_kernels(FhnBackendCtx * /*ctx*/) { return &toyfhe_kernel_table; }

// --- Host-side data plane ---------------------------------------------------
// These exports handle plaintexts and key material. The trusted host calls
// them directly; they are never dispatched from an FhnProgram.

FhnBuffer *toyfhe_fhn_buffer_alloc(FhnBackendCtx * /*ctx*/) { return new FhnBuffer{}; }

void toyfhe_fhn_buffer_free(FhnBackendCtx * /*ctx*/, FhnBuffer *buf) { delete buf; }

int toyfhe_fhn_encrypt_i64(FhnBackendCtx *ctx, FhnBuffer *out, int64_t value) {
  if (!out)
    return -1;
  out->ct = ctx->engine.encryptInt(value);
  out->kind = BufKind::Ciphertext;
  return 0;
}

int toyfhe_fhn_encrypt_f64(FhnBackendCtx *ctx, FhnBuffer *out, double value) {
  if (!out)
    return -1;
  out->ct = ctx->engine.encryptDouble(value);
  out->kind = BufKind::Ciphertext;
  return 0;
}

int toyfhe_fhn_decrypt_i64(FhnBackendCtx *ctx, const FhnBuffer *in, int64_t *value_out) {
  if (!in || !value_out || in->kind != BufKind::Ciphertext)
    return -1;
  // Fixed-point ciphertexts carry a scaled mantissa; decode through the
  // scale-aware path and round.
  if (in->ct.encoding == fhenomenon::toyfhe::Encoding::FixedPoint) {
    *value_out = static_cast<int64_t>(std::llround(ctx->engine.decryptDouble(in->ct)));
  } else {
    *value_out = ctx->engine.decryptInt(in->ct);
  }
  return 0;
}

int toyfhe_fhn_decrypt_f64(FhnBackendCtx *ctx, const FhnBuffer *in, double *value_out) {
  if (!in || !value_out || in->kind != BufKind::Ciphertext)
    return -1;
  *value_out = ctx->engine.decryptDouble(in->ct);
  return 0;
}

int64_t toyfhe_fhn_buffer_read_int(FhnBackendCtx *ctx, FhnBuffer *buf) {
  if (buf->kind == BufKind::IntValue)
    return buf->int_val;
  if (buf->kind == BufKind::DoubleValue)
    return static_cast<int64_t>(std::llround(buf->double_val));
  if (buf->kind == BufKind::Ciphertext)
    return ctx->engine.decryptInt(buf->ct);
  return 0;
}

double toyfhe_fhn_buffer_read_double(FhnBackendCtx *ctx, FhnBuffer *buf) {
  if (buf->kind == BufKind::DoubleValue)
    return buf->double_val;
  if (buf->kind == BufKind::IntValue)
    return static_cast<double>(buf->int_val);
  if (buf->kind == BufKind::Ciphertext)
    return ctx->engine.decryptDouble(buf->ct);
  return 0.0;
}
