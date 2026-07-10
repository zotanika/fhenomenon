#pragma once

#include "FHN/fhn_backend_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// The required backend exports for ToyFHE
uint32_t toyfhe_fhn_get_abi_version(void);
FhnBackendInfo *toyfhe_fhn_get_info(void);
FhnBackendCtx *toyfhe_fhn_create(const char *config_json);
void toyfhe_fhn_destroy(FhnBackendCtx *ctx);
FhnKernelTable *toyfhe_fhn_get_kernels(FhnBackendCtx *ctx);

// Host-side data plane (trusted; never dispatched from an FhnProgram)
FhnBuffer *toyfhe_fhn_buffer_alloc(FhnBackendCtx *ctx);
void toyfhe_fhn_buffer_free(FhnBackendCtx *ctx, FhnBuffer *buf);
int toyfhe_fhn_encrypt_i64(FhnBackendCtx *ctx, FhnBuffer *out, int64_t value);
int toyfhe_fhn_encrypt_f64(FhnBackendCtx *ctx, FhnBuffer *out, double value);
int toyfhe_fhn_decrypt_i64(FhnBackendCtx *ctx, const FhnBuffer *in, int64_t *value_out);
int toyfhe_fhn_decrypt_f64(FhnBackendCtx *ctx, const FhnBuffer *in, double *value_out);

// Convenience readers for tests/examples (decrypt ciphertext buffers in place)
int64_t toyfhe_fhn_buffer_read_int(FhnBackendCtx *ctx, FhnBuffer *buf);
double toyfhe_fhn_buffer_read_double(FhnBackendCtx *ctx, FhnBuffer *buf);

#ifdef __cplusplus
}
#endif
