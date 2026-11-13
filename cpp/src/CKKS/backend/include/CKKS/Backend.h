#pragma once
#include "FHN/fhn_backend_api.h"
#include "FHN/fhn_native_api.h"

// Native plugin host ABI. Compute functions are returned by get_kernels;
// there are deliberately no same-context encrypt/decrypt exports.
extern "C" {
FhnBackendInfo *ckks_fhn_get_info();
FhnBackendCtx *ckks_fhn_create(const char *config_json);
void ckks_fhn_destroy(FhnBackendCtx *);
FhnKernelTable *ckks_fhn_get_kernels(FhnBackendCtx *);
FhnBuffer *ckks_fhn_buffer_alloc(FhnBackendCtx *);
void ckks_fhn_buffer_free(FhnBackendCtx *, FhnBuffer *);
int64_t ckks_fhn_fresh_level(FhnBackendCtx *);
uint64_t ckks_fhn_level_bytes(FhnBackendCtx *, int64_t);
FhnLevelEffect ckks_fhn_opcode_level_effect(FhnBackendCtx *, FhnOpCode);
FhnClientCtx *ckks_fhn_client_create(FhnBackendCtx *);
void ckks_fhn_client_destroy(FhnClientCtx *);
int ckks_fhn_client_prepare(FhnClientCtx *, const FhnProgram *);
int ckks_fhn_client_encrypt_vec_f64(FhnClientCtx *, FhnBuffer *, const double *, uint32_t);
int ckks_fhn_client_decrypt_vec_f64(FhnClientCtx *, const FhnBuffer *, double *, uint32_t);

int ckks_fhn_client_encrypt_vec_c64(FhnClientCtx *, FhnBuffer *, const FhnComplex64 *, uint32_t);
int ckks_fhn_client_decrypt_vec_c64(FhnClientCtx *, const FhnBuffer *, FhnComplex64 *, uint32_t);

// Diagnostic metadata only: no plaintext or secret-key access. The movement
// model reserves full buffer capacity at every level; live payload can shrink.

int ckks_fhn_buffer_info(FhnBackendCtx *, const FhnBuffer *, FhnCkksBufferInfo *);
}
