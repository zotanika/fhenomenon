#pragma once
#include <stddef.h>
#include <stdint.h>
#define FHN_TENSOR_PROVIDER_ABI_VERSION 1u
#ifdef __cplusplus
extern "C" {
#endif
/* All operation results are new immutable handles. Return 0 on success. */
uint32_t fn_get_abi_version(void);
const char *fn_last_error(void);
/* Process-local identity of this loaded provider implementation. */
const void *fn_library_identity(void);
int fn_create(const int32_t *rotation_steps, size_t rotation_count, void **client, void **evaluator);
/* profile_id: 0 = Native128 modulus-budget profile; 1 = insecure NativeTest numerical fixture. */
int fn_create_profile(int profile_id, const int32_t *rotation_steps, size_t rotation_count, void **client,
                      void **evaluator);
int fn_select_device(void *client, int use_cuda);
int fn_synchronize(void *evaluator);
const char *fn_device_info(void *evaluator);
int fn_prepare_mod_raise(void *client, uint32_t target_level);
const char *fn_refresh_plan(void *evaluator, uint32_t target_level);
void fn_client_free(void *client);
void fn_evaluator_free(void *evaluator);
void fn_ciphertext_free(void *ciphertext);
int fn_encrypt(void *client, const double *values, size_t count, void **result);
int fn_decrypt(void *client, void *ciphertext, double *values, size_t count);
int fn_decrypt_bounded(void *client, void *ciphertext, double *values, size_t count, double public_bound);
int fn_binary(void *evaluator, int op, void *a, void *b, void **result); /* 0 add, 1 sub, 2 mul */
int fn_plain(void *evaluator, int op, void *a, const double *values, size_t count, void **result); /* 0 add, 2 mul */
int fn_neg(void *evaluator, void *a, void **result);
int fn_rotate(void *evaluator, void *a, int32_t step, void **result);
int fn_level_down(void *evaluator, void *a, uint32_t level, void **result);
int fn_mod_raise(void *evaluator, void *a, uint32_t target_level, void **result);
int fn_mod_raise_bounded(void *evaluator, void *a, uint32_t target_level, double public_bound, void **result);
/* Full evaluation-top reference baseline; normal execution uses fn_mod_raise. */
int fn_bootstrap(void *evaluator, void *a, void **result);
int fn_slot_to_coeff(void *evaluator, void *a, void **result);
int fn_coeff_to_slot(void *evaluator, void *a, void **result);
const char *fn_profile(void *evaluator);
const char *fn_stats(void *evaluator);
const char *fn_client_stats(void *client);
const char *fn_ciphertext_info(void *ciphertext);
#ifdef __cplusplus
}
#endif
