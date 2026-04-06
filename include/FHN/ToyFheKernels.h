#pragma once

#include "FHN/fhn_backend_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// The 4 required backend exports for ToyFHE
FhnBackendInfo *toyfhe_fhn_get_info(void);
FhnBackendCtx *toyfhe_fhn_create(const char *config_json);
void toyfhe_fhn_destroy(FhnBackendCtx *ctx);
FhnKernelTable *toyfhe_fhn_get_kernels(FhnBackendCtx *ctx);

// Buffer helpers (ToyFHE-specific)
FhnBuffer *toyfhe_fhn_buffer_alloc(FhnBackendCtx *ctx);
void toyfhe_fhn_buffer_free(FhnBackendCtx *ctx, FhnBuffer *buf);
int64_t toyfhe_fhn_buffer_read_int(FhnBackendCtx *ctx, FhnBuffer *buf);
double toyfhe_fhn_buffer_read_double(FhnBackendCtx *ctx, FhnBuffer *buf);

#ifdef __cplusplus
}
#endif
