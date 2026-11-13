#pragma once
#include "FHN/fhn_plaintext_api.h"
#ifdef __cplusplus
extern "C" {
#endif
uint32_t ckks_fhn_plaintext_api_version(void);
uint64_t ckks_fhn_plaintext_required_bytes(FhnBackendCtx *, const FhnBuffer *, FhnPlaintextPurpose);
int ckks_fhn_plaintext_encode(FhnBackendCtx *, const FhnBuffer *, const double *, uint32_t, FhnPlaintextPurpose,
                              FhnPlaintext **);
int ckks_fhn_plaintext_multiply(FhnBackendCtx *, FhnBuffer *, const FhnBuffer *, const FhnPlaintext *);
int ckks_fhn_plaintext_add(FhnBackendCtx *, FhnBuffer *, const FhnBuffer *, const FhnPlaintext *);
uint64_t ckks_fhn_plaintext_bytes(const FhnPlaintext *);
void ckks_fhn_plaintext_free(FhnPlaintext *);
#ifdef __cplusplus
}
#endif
