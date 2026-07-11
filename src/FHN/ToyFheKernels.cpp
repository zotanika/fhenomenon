#include "FHN/ToyFheKernels.h"
#include "Crypto/ToyFHE.h"

#include <cmath>
#include <cstddef>
#include <vector>

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

enum class BufKind { Empty, Ciphertext, IntValue, DoubleValue, CiphertextVec };

struct FhnBuffer {
  BufKind kind = BufKind::Empty;
  fhenomenon::toyfhe::Ciphertext ct;
  // Multi-slot vector ciphertext: ToyFHE is single-slot, so slot packing is
  // emulated with one independent toy ciphertext per slot.
  std::vector<fhenomenon::toyfhe::Ciphertext> ct_vec;
  int64_t int_val = 0;
  double double_val = 0.0;
};

// --- Vector (multi-slot) helpers ---------------------------------------------

static bool toyfhe_is_vec(const FhnBuffer *buf) { return buf != nullptr && buf->kind == BufKind::CiphertextVec; }

// Normalize a signed rotation distance into [0, n). Positive = left.
static std::size_t toyfhe_norm_rot(int64_t d, std::size_t n) {
  const int64_t sn = static_cast<int64_t>(n);
  return static_cast<std::size_t>(((d % sn) + sn) % sn);
}

typedef fhenomenon::toyfhe::Ciphertext (fhenomenon::toyfhe::Engine::*ToyBinOp)(
  const fhenomenon::toyfhe::Ciphertext &, const fhenomenon::toyfhe::Ciphertext &) const;

// Slot-wise binary engine op over two CiphertextVec operands of equal size.
// Computes into a local vector first so the result may alias either operand —
// the executor's decomposition paths issue in-place calls per the ABI
// contract in fhn_backend_api.h.
static int toyfhe_vec_binary(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *a, const FhnBuffer *b,
                             ToyBinOp op) {
  if (a->ct_vec.empty() || a->ct_vec.size() != b->ct_vec.size())
    return -1;
  std::vector<fhenomenon::toyfhe::Ciphertext> out;
  out.reserve(a->ct_vec.size());
  for (std::size_t i = 0; i < a->ct_vec.size(); ++i) {
    out.push_back((ctx->engine.*op)(a->ct_vec[i], b->ct_vec[i]));
  }
  result->ct_vec = std::move(out);
  result->kind = BufKind::CiphertextVec;
  return 0;
}

// --- Kernel implementations ------------------------------------------------

static int toyfhe_add_cc(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                         const int64_t * /*params*/, const double * /*fparams*/) {
  const FhnBuffer *a = operands[0];
  const FhnBuffer *b = operands[1];
  if (a == nullptr || b == nullptr)
    return -1;
  if (toyfhe_is_vec(a) || toyfhe_is_vec(b)) {
    // Mixed scalar/vector operands are not defined.
    if (!toyfhe_is_vec(a) || !toyfhe_is_vec(b))
      return -1;
    return toyfhe_vec_binary(ctx, result, a, b, &fhenomenon::toyfhe::Engine::add);
  }
  result->ct = ctx->engine.add(a->ct, b->ct);
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
  const FhnBuffer *a = operands[0];
  const FhnBuffer *b = operands[1];
  if (a == nullptr || b == nullptr)
    return -1;
  if (toyfhe_is_vec(a) || toyfhe_is_vec(b)) {
    // Mixed scalar/vector operands are not defined.
    if (!toyfhe_is_vec(a) || !toyfhe_is_vec(b))
      return -1;
    return toyfhe_vec_binary(ctx, result, a, b, &fhenomenon::toyfhe::Engine::multiply);
  }
  result->ct = ctx->engine.multiply(a->ct, b->ct);
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

// Cyclic rotation of a CiphertextVec. params[0] is a signed distance:
// positive rotates left (result[i] = src[(i + d) mod n]), negative rotates
// right. Scalar ciphertexts have no slots to rotate.
static int toyfhe_rotate(FhnBackendCtx * /*ctx*/, FhnBuffer *result, const FhnBuffer *const *operands,
                         const int64_t *params, const double * /*fparams*/) {
  const FhnBuffer *src = operands[0];
  if (!toyfhe_is_vec(src) || src->ct_vec.empty())
    return -1;
  const std::size_t n = src->ct_vec.size();
  const std::size_t d = toyfhe_norm_rot(params[0], n);
  std::vector<fhenomenon::toyfhe::Ciphertext> out;
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    out.push_back(src->ct_vec[(i + d) % n]);
  }
  result->ct_vec = std::move(out);
  result->kind = BufKind::CiphertextVec;
  return 0;
}

// Fused rotate-and-add in one pass: result[i] = a[(i + d) mod n] + b[i].
// This is the fusion the matvec benchmark measures against the executor's
// decomposition into ROTATE + ADD_CC (two full passes over the slots).
static int toyfhe_hrot_add(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                           const int64_t *params, const double * /*fparams*/) {
  const FhnBuffer *a = operands[0];
  const FhnBuffer *b = operands[1];
  if (!toyfhe_is_vec(a) || !toyfhe_is_vec(b) || a->ct_vec.empty() || a->ct_vec.size() != b->ct_vec.size())
    return -1;
  const std::size_t n = a->ct_vec.size();
  const std::size_t d = toyfhe_norm_rot(params[0], n);
  std::vector<fhenomenon::toyfhe::Ciphertext> out;
  out.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    out.push_back(ctx->engine.add(a->ct_vec[(i + d) % n], b->ct_vec[i]));
  }
  result->ct_vec = std::move(out);
  result->kind = BufKind::CiphertextVec;
  return 0;
}

// ToyFHE multiply already relinearizes and rescales internally, so
// RELINEARIZE and RESCALE are no-op pass-throughs. They are registered so
// the default executor's HMULT decomposition (MULT_CC + RELINEARIZE +
// RESCALE) is exercised realistically.
static int toyfhe_passthrough(FhnBackendCtx * /*ctx*/, FhnBuffer *result, const FhnBuffer *const *operands,
                              const int64_t * /*params*/, const double * /*fparams*/) {
  const FhnBuffer *src = operands[0];
  if (src == nullptr || (src->kind != BufKind::Ciphertext && src->kind != BufKind::CiphertextVec))
    return -1;
  if (result != src) {
    *result = *src;
  }
  return 0;
}

// --- Kernel table ----------------------------------------------------------

// Compute-only: encryption and decryption are host-side data-plane exports
// below, never kernel-table entries.
static FhnKernelEntry toyfhe_kernels[] = {
  {FHN_ADD_CC, toyfhe_add_cc, "add_cc"},
  {FHN_ADD_CS, toyfhe_add_cs, "add_cs"},
  {FHN_SUB_CC, toyfhe_sub_cc, "sub_cc"},
  {FHN_NEGATE, toyfhe_negate, "negate"},
  {FHN_MULT_CC, toyfhe_mult_cc, "mult_cc"},
  {FHN_MULT_CS, toyfhe_mult_cs, "mult_cs"},
  {FHN_RELINEARIZE, toyfhe_passthrough, "relinearize"},
  {FHN_RESCALE, toyfhe_passthrough, "rescale"},
  {FHN_ROTATE, toyfhe_rotate, "rotate"},
  {FHN_HMULT, toyfhe_hmult, "hmult"},
  {FHN_HROT_ADD, toyfhe_hrot_add, "hrot_add"},
};

static FhnKernelTable toyfhe_kernel_table = {
  sizeof(toyfhe_kernels) / sizeof(toyfhe_kernels[0]),
  toyfhe_kernels,
};

// --- Backend exports -------------------------------------------------------

uint32_t toyfhe_fhn_get_abi_version(void) { return FHN_ABI_VERSION; }

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

int toyfhe_fhn_encrypt_vec_i64(FhnBackendCtx *ctx, FhnBuffer *out, const int64_t *values, uint32_t n) {
  if (!ctx || !out || !values || n == 0)
    return -1;
  std::vector<fhenomenon::toyfhe::Ciphertext> cts;
  cts.reserve(n);
  for (uint32_t i = 0; i < n; ++i) {
    cts.push_back(ctx->engine.encryptInt(values[i]));
  }
  out->ct_vec = std::move(cts);
  out->kind = BufKind::CiphertextVec;
  return 0;
}

int toyfhe_fhn_decrypt_vec_i64(FhnBackendCtx *ctx, const FhnBuffer *in, int64_t *out, uint32_t n) {
  if (!ctx || !in || !out || in->kind != BufKind::CiphertextVec || in->ct_vec.size() != n)
    return -1;
  for (uint32_t i = 0; i < n; ++i) {
    out[i] = ctx->engine.decryptInt(in->ct_vec[i]);
  }
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
