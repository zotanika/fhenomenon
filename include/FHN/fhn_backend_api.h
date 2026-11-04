#ifndef FHN_BACKEND_API_H
#define FHN_BACKEND_API_H

#include "FHN/fhn_program.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles */
typedef struct FhnBackendCtx FhnBackendCtx;
typedef struct FhnBuffer FhnBuffer;
typedef struct FhnExecHandle FhnExecHandle;
typedef struct FhnClientCtx FhnClientCtx;

/* Device type enumeration */
typedef enum FhnDeviceType {
  FHN_DEVICE_CPU = 0,
  FHN_DEVICE_GPU = 1,
  FHN_DEVICE_FPGA = 2,
  FHN_DEVICE_ASIC = 3
} FhnDeviceType;

/* Backend info returned by fhn_get_info */
typedef struct FhnBackendInfo {
  const char *name;
  const char *version;
  FhnDeviceType device_type;
  uint64_t device_memory; /* bytes, 0 if N/A */
} FhnBackendInfo;

/* Uniform kernel function signature.
   Returns 0 on success, non-zero on error.

   Buffer ownership: buffers are created and destroyed only by the host,
   through fhn_buffer_alloc/fhn_buffer_free. Kernels never allocate or free
   buffers; they read operands and write into the pre-allocated result.

   Aliasing: kernels must tolerate result == operands[i] (in-place update).
   The default executor's decomposition paths rely on this. */
typedef int (*FhnKernelFn)(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                           const int64_t *params, const double *fparams);

/* One entry in the kernel table */
typedef struct FhnKernelEntry {
  FhnOpCode opcode;
  FhnKernelFn fn;   /* NULL = not supported */
  const char *name; /* for debugging/logging */
} FhnKernelEntry;

/* The kernel table */
typedef struct FhnKernelTable {
  uint32_t num_kernels;
  FhnKernelEntry *kernels;
} FhnKernelTable;

/* ── Function pointer typedefs for dlopen resolution ── */
/* Deliberate leaf function with an eternal signature: safe to resolve and
   call before any other symbol. Must return FHN_ABI_VERSION. */
typedef uint32_t (*FhnGetAbiVersionFn)(void);
typedef FhnBackendInfo *(*FhnGetInfoFn)(void);
typedef FhnBackendCtx *(*FhnCreateFn)(const char *config_json);
typedef void (*FhnDestroyFn)(FhnBackendCtx *ctx);
typedef FhnKernelTable *(*FhnGetKernelsFn)(FhnBackendCtx *ctx);

/* ── Host-side data plane (trusted) ──
   Plaintexts and key material cross the trust boundary only through these
   exports, which the trusted host resolves and calls directly. They are not
   kernel-table entries and cannot be reached from an FhnProgram: the
   instruction stream the executor dispatches is public and compute-only.

   buffer_alloc/buffer_free are required — a generic host cannot feed any
   kernel without them. The encrypt/decrypt exports are optional: they exist
   only when the backend holds key material (development and single-process
   deployments). A production evaluation-only executor never holds keys, and
   ciphertexts then enter and leave through serialized buffers instead. */
typedef FhnBuffer *(*FhnBufferAllocFn)(FhnBackendCtx *ctx);
typedef void (*FhnBufferFreeFn)(FhnBackendCtx *ctx, FhnBuffer *buffer);

/* ── Optional movement hooks (data plane) ──
   Host-orchestrated residency transfers for backends with a distinct
   compute memory space (e.g. GPU). Deliberately NOT kernel-table opcodes:
   movement is not compute, and the public instruction stream stays
   compute-only. Optional and additive — absent exports mean a single
   memory space, and their addition does not bump FHN_ABI_VERSION because
   every previously conformant backend remains conformant.

   fhn_buffer_prefetch: ensure the buffer is resident in the backend's
   compute space (H2D). No-op if already resident. 0 on success.
   fhn_buffer_evict: demote the buffer to host memory (D2H), preserving
   contents; a later prefetch restores compute residency. 0 on success.
   fhn_buffer_free must accept a buffer in either residency state. */
typedef int (*FhnBufferPrefetchFn)(FhnBackendCtx *ctx, FhnBuffer *buffer);
typedef int (*FhnBufferEvictFn)(FhnBackendCtx *ctx, FhnBuffer *buffer);

/* Level effect of a compute opcode, declared by CKKS-family backends for
   byte-accurate movement planning (see fhn_fresh_level/fhn_level_bytes/
   fhn_opcode_level_effect, resolved as an all-or-nothing trio).
   C++ uses the ABI's 32-bit representation explicitly so every backend
   bit pattern is a valid enum value; unknown values can then be rejected
   by the planner without an undefined enum load. The C declaration and
   the ABI 1 representation of all existing values remain unchanged. */
typedef enum FhnLevelEffect
#ifdef __cplusplus
  : uint32_t
#endif
{
  FHN_LEVEL_PRESERVE = 0,   /* result level = min over operand levels    */
  FHN_LEVEL_CONSUME = 1,    /* result level = (min over operands) - 1    */
  FHN_LEVEL_SET_PARAM0 = 2, /* result level = params[0] (must not raise) */
} FhnLevelEffect;

/* ── Optional level model (data plane) ──
   All three exports appear TOGETHER or the group is ignored with a
   warning (mirroring the movement-hook half-pair rule). Additive and
   optional: no FHN_ABI_VERSION bump; existing backends stay conformant. */
typedef int64_t (*FhnFreshLevelFn)(FhnBackendCtx *ctx);
typedef uint64_t (*FhnLevelBytesFn)(FhnBackendCtx *ctx, int64_t level);
typedef FhnLevelEffect (*FhnOpcodeLevelEffectFn)(FhnBackendCtx *ctx, FhnOpCode opcode);

typedef int (*FhnEncryptInt64Fn)(FhnBackendCtx *ctx, FhnBuffer *out, int64_t value);
typedef int (*FhnEncryptDoubleFn)(FhnBackendCtx *ctx, FhnBuffer *out, double value);
typedef int (*FhnDecryptInt64Fn)(FhnBackendCtx *ctx, const FhnBuffer *in, int64_t *value_out);
typedef int (*FhnDecryptDoubleFn)(FhnBackendCtx *ctx, const FhnBuffer *in, double *value_out);

/* Optional packed-real exports, resolved independently with dlsym:
   fhn_encrypt_vec_f64 / fhn_decrypt_vec_f64 (plus any backend prefix).
   ABI version 1 and FhnBackendVTable layout are unchanged; scalar-only
   backends remain valid. Both return 0 on success, nonzero on failure.

   count must be positive. Encryption reads exactly count finite doubles;
   decryption requires count to equal the ciphertext's logical slot count
   and writes exactly count doubles. The caller supplies that many elements,
   not merely a capacity; no truncation or implicit padding is allowed.
   Null arguments, nonfinite values, unsupported counts/ranges and kind or
   length mismatches fail. Validation failures leave the output unchanged.
   Backends may restrict supported counts (e.g. power-of-two packing and
   device capacity) and numerical range; callers must handle failure. */
typedef int (*FhnEncryptVectorDoubleFn)(FhnBackendCtx *ctx, FhnBuffer *out, const double *values, uint32_t count);
typedef int (*FhnDecryptVectorDoubleFn)(FhnBackendCtx *ctx, const FhnBuffer *in, double *values, uint32_t count);

/* Optional separate trusted-client group, resolved together by symbol name:
   fhn_client_create/destroy/prepare/encrypt_vec_f64/decrypt_vec_f64.
   This additive group does not change ABI 1 or FhnBackendVTable.
   The client owns secret material; the compute context must not own or
   reference the client or its secret key. prepare derives and registers
   public evaluation keys for a program before execution. Compute-only
   kernels may borrow those keys, but must not invoke these host functions.
   Destroy the client before its backend context. Buffers belong to the
   backend; the packed-real validation/count contract above also applies.
   A backend may allow just one client/key identity per compute context.
   Evaluation keys can expire at client destruction; affected kernels then
   fail. No serialized ciphertext transport is implied by this interface. */
typedef FhnClientCtx *(*FhnClientCreateFn)(FhnBackendCtx *ctx);
typedef void (*FhnClientDestroyFn)(FhnClientCtx *client);
typedef int (*FhnClientPrepareFn)(FhnClientCtx *client, const FhnProgram *program);
typedef int (*FhnClientEncryptVectorDoubleFn)(FhnClientCtx *client, FhnBuffer *out,
                                           const double *values, uint32_t count);
typedef int (*FhnClientDecryptVectorDoubleFn)(FhnClientCtx *client, const FhnBuffer *in,
                                           double *values, uint32_t count);


/* Optional packed-complex trusted-client exports, resolved as a pair:
   fhn_client_encrypt_vec_c64 / fhn_client_decrypt_vec_c64 (plus backend prefix).
   The trusted-client and exact-count contracts above apply. Both components
   must be finite and within the supported encoding range. Real and complex
   entry points use the same ciphertext kind. Complex decoding of a real
   input yields an approximately zero imaginary component, while real
   decoding returns the real projection. ABI 1 and FhnBackendVTable are unchanged. */
typedef struct FhnComplex64 {
  double real;
  double imag;
} FhnComplex64;
typedef int (*FhnClientEncryptVectorComplex64Fn)(FhnClientCtx *client, FhnBuffer *out,
                                               const FhnComplex64 *values, uint32_t count);
typedef int (*FhnClientDecryptVectorComplex64Fn)(FhnClientCtx *client, const FhnBuffer *in,
                                               FhnComplex64 *values, uint32_t count);

/* Optional advanced exports (NULL if not provided) */
typedef FhnExecHandle *(*FhnSubmitFn)(FhnBackendCtx *ctx, const FhnProgram *program, FhnBuffer **inputs,
                                      uint32_t num_inputs);
typedef int (*FhnPollFn)(FhnExecHandle *handle);
typedef int (*FhnWaitFn)(FhnExecHandle *handle);
typedef FhnBuffer **(*FhnGetOutputsFn)(FhnExecHandle *handle, uint32_t *num_outputs);
typedef void (*FhnExecFreeFn)(FhnExecHandle *handle);

/* VTable for resolved backend symbols */
typedef struct FhnBackendVTable {
  FhnGetAbiVersionFn get_abi_version;
  FhnGetInfoFn get_info;
  FhnCreateFn create;
  FhnDestroyFn destroy;
  FhnGetKernelsFn get_kernels;

  /* Host-side data plane: buffers required, key operations optional */
  FhnBufferAllocFn buffer_alloc;
  FhnBufferFreeFn buffer_free;
  /* Optional movement hooks (NULL if not provided by backend) */
  FhnBufferPrefetchFn prefetch;
  FhnBufferEvictFn evict;
  FhnEncryptInt64Fn encrypt_i64;
  FhnEncryptDoubleFn encrypt_f64;
  FhnDecryptInt64Fn decrypt_i64;
  FhnDecryptDoubleFn decrypt_f64;

  /* Optional (NULL if not provided by backend) */
  FhnSubmitFn submit;
  FhnPollFn poll;
  FhnWaitFn wait;
  FhnGetOutputsFn get_outputs;
  FhnExecFreeFn exec_free;

  /* Optional level model trio (NULL if not provided by backend) */
  FhnFreshLevelFn fresh_level;
  FhnLevelBytesFn level_bytes;
  FhnOpcodeLevelEffectFn opcode_level_effect;
} FhnBackendVTable;

#ifdef __cplusplus
}
#endif

#endif /* FHN_BACKEND_API_H */
