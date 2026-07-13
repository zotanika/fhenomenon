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
   fhn_opcode_level_effect, resolved as an all-or-nothing trio). */
typedef enum FhnLevelEffect {
  FHN_LEVEL_PRESERVE = 0,   /* result level = min over operand levels    */
  FHN_LEVEL_CONSUME = 1,    /* result level = (min over operands) - 1    */
  FHN_LEVEL_SET_PARAM0 = 2, /* result level = params[0] (must not raise) */
} FhnLevelEffect;

typedef int (*FhnEncryptInt64Fn)(FhnBackendCtx *ctx, FhnBuffer *out, int64_t value);
typedef int (*FhnEncryptDoubleFn)(FhnBackendCtx *ctx, FhnBuffer *out, double value);
typedef int (*FhnDecryptInt64Fn)(FhnBackendCtx *ctx, const FhnBuffer *in, int64_t *value_out);
typedef int (*FhnDecryptDoubleFn)(FhnBackendCtx *ctx, const FhnBuffer *in, double *value_out);

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
} FhnBackendVTable;

#ifdef __cplusplus
}
#endif

#endif /* FHN_BACKEND_API_H */
