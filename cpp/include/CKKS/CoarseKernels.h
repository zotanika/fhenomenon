#pragma once
#include "FHN/fhn_coarse_api.h"
extern "C" {
uint32_t ckks_fhn_coarse_api_version();
int ckks_fhn_centered_energy(FhnBackendCtx *, FhnBuffer *, const FhnBuffer *, const FhnBuffer *, uint32_t);
int ckks_fhn_centered_variance(FhnBackendCtx *, FhnBuffer *, const FhnBuffer *, const FhnBuffer *, uint32_t);
int ckks_fhn_packed_dot(FhnBackendCtx *, FhnBuffer *, const FhnBuffer *, const FhnBuffer *, uint32_t);
int ckks_fhn_packed_reduce(FhnBackendCtx *, FhnBuffer *, const FhnBuffer *, uint32_t);
}
