#pragma once
#include "FHN/fhn_prepared_api.h"
#ifdef __cplusplus
extern "C" {
#endif
uint64_t ckks_fhn_rotation_preparation_required_bytes(FhnBackendCtx *, const FhnBuffer *);
FhnRotationPreparation *ckks_fhn_prepare_rotation_source(FhnBackendCtx *, const FhnBuffer *);
uint64_t ckks_fhn_rotation_preparation_bytes(const FhnRotationPreparation *);
int ckks_fhn_rotate_prepared(FhnBackendCtx *, FhnBuffer *, const FhnRotationPreparation *, int64_t);
void ckks_fhn_rotation_preparation_free(FhnRotationPreparation *);
#ifdef __cplusplus
}
#endif
