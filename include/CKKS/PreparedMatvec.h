#pragma once
#include "FHN/fhn_matvec_api.h"
#ifdef __cplusplus
extern "C" {
#endif
uint32_t ckks_fhn_matvec_api_version(void);
uint64_t ckks_fhn_matvec_preparation_required_bytes(FhnBackendCtx *, const FhnBuffer *, uint32_t,
                                                    const FhnMatvecPrepareOptions *);
FhnMatvecPreparation *ckks_fhn_prepare_matvec_source(FhnBackendCtx *, const FhnBuffer *, uint32_t,
                                                     const FhnMatvecPrepareOptions *);
uint64_t ckks_fhn_matvec_preparation_bytes(const FhnMatvecPreparation *);
void ckks_fhn_matvec_preparation_free(FhnMatvecPreparation *);
int ckks_fhn_matvec_product(FhnBackendCtx *, FhnBuffer *, const FhnMatvecPreparation *, const FhnBuffer *);
int ckks_fhn_matvec_dot(FhnBackendCtx *, FhnBuffer *, const FhnMatvecPreparation *, const FhnBuffer *);
int ckks_fhn_matvec_reduce(FhnBackendCtx *, FhnBuffer *, const FhnMatvecPreparation *, const FhnBuffer *, uint32_t);
#ifdef __cplusplus
}
#endif
