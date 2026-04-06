#pragma once

#include "FHN/fhn_backend_api.h"

// Cheddar FHN backend exports.
// These are compiled into libcheddar_fhn.so and loaded via ExternalBackend.
// They follow the same 4-function pattern as ToyFheKernels.h.

#ifdef __cplusplus
extern "C" {
#endif

// 4 required exports
FhnBackendInfo  *fhn_get_info(void);
FhnBackendCtx   *fhn_create(const char *config_json);
void             fhn_destroy(FhnBackendCtx *ctx);
FhnKernelTable  *fhn_get_kernels(FhnBackendCtx *ctx);

// Buffer helpers (Cheddar-specific)
FhnBuffer       *cheddar_fhn_buffer_alloc(FhnBackendCtx *ctx);
void             cheddar_fhn_buffer_free(FhnBackendCtx *ctx, FhnBuffer *buf);

// Read helpers: decode/decrypt result from buffer
void             cheddar_fhn_buffer_read_complex(FhnBackendCtx *ctx,
                                                  FhnBuffer *buf,
                                                  double *real_out,
                                                  double *imag_out,
                                                  int max_slots);

#ifdef __cplusplus
}
#endif
