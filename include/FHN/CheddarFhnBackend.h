#pragma once

#include "FHN/fhn_backend_api.h"

// Cheddar FHN backend exports.
// These are compiled into libcheddar_fhn.so and loaded via ExternalBackend.
// They follow the same export pattern as ToyFheKernels.h, un-prefixed so a
// prefix-less ExternalBackend resolves the whole ABI (core + data plane).

#ifdef __cplusplus
extern "C" {
#endif

// Required exports
uint32_t fhn_get_abi_version(void);
FhnBackendInfo *fhn_get_info(void);
FhnBackendCtx *fhn_create(const char *config_json);
void fhn_destroy(FhnBackendCtx *ctx);
FhnKernelTable *fhn_get_kernels(FhnBackendCtx *ctx);

// Host-side data plane (trusted; never dispatched from an FhnProgram)
FhnBuffer *fhn_buffer_alloc(FhnBackendCtx *ctx);
void fhn_buffer_free(FhnBackendCtx *ctx, FhnBuffer *buf);
int fhn_encrypt_f64(FhnBackendCtx *ctx, FhnBuffer *out, double value);
int fhn_decrypt_f64(FhnBackendCtx *ctx, const FhnBuffer *in, double *value_out);

// Buffer helpers (Cheddar-specific, for tests/examples)
// Encrypt helper: encode + encrypt the message staged in the buffer, in place
int cheddar_fhn_buffer_encrypt_message(FhnBackendCtx *ctx, FhnBuffer *buf);

// Read helper: decrypt/decode result from buffer
void cheddar_fhn_buffer_read_complex(FhnBackendCtx *ctx, FhnBuffer *buf, double *real_out, double *imag_out,
                                     int max_slots);

#ifdef __cplusplus
}
#endif
