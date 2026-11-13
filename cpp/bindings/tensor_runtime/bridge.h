#pragma once
#include <stddef.h>
#include <stdint.h>
#define FHN_TENSOR_RUNTIME_ABI_VERSION 1u
#ifdef __cplusplus
extern "C" {
#endif

/* The adapter owns only an evaluator and ciphertexts, never a client or key.
 * create takes ownership of evaluator on success only. Calls are serialized.
 * Eager results and completed scope exports are immutable Runtime Values.
 * Captured results are opaque scope-local symbols until successful completion. */
const char *fr_last_error(void);
uint32_t fr_get_abi_version(void);
/* Identity of the provider implementation actually linked by this Runtime. */
const void *fr_provider_library_identity(void);
int fr_create(void *native_evaluator, size_t max_live_bytes, void **runtime);
void fr_runtime_free(void *runtime);
int fr_import(void *runtime, void *borrowed_native_ciphertext, void **value);
void fr_value_free(void *value);
/* Borrowed ciphertext: the caller must keep value alive for the whole call.
 * Never pass this pointer to fn_ciphertext_free or a mutating external API. */
void *fr_value_borrow(void *value);
/* Symbolic info, including scale, comes from the core Scope descriptor.
 * Runtime identity is null until successful export; symbolic is then false. */
const char *fr_value_info(void *value);
/* Capture a complete graph exactly once, validate it, then execute one runScope.
 * Existing arithmetic APIs record symbolic operations during capture; none run.
 * Callback returns zero on success and registers exports with fr_scope_export.
 * Structural caps must be positive; duration zero means unlimited. Duration is
 * checked at command/span boundaries and cannot preempt a provider operation.
 * No nested scope, import, borrow of symbols, or
 * synchronize is allowed. An unsuccessful arithmetic call poisons capture even
 * if the callback ignores its status. Failure never publishes partial exports.
 * Nonexported symbols expire on return; exported handle states become concrete.
 * A callback may release handles after recording uses/export registration.
 * Context and callback must stay alive; calls on one context are serialized. */
int fr_run_scope(void *runtime, int (*capture)(void *), void *userdata, size_t max_commands, size_t max_values,
                 size_t max_input_edges, size_t max_exports, size_t max_duration_ms);
int fr_scope_export(void *runtime, void *value);
int fr_binary(void *runtime, int opcode, void *a, void *b, void **value);
int fr_plain(void *runtime, int opcode, void *a, const double *plain, size_t count, void **value);
/* 0 negate; 1 rotate; 2 level-down; 3 ModRaise(target); 4 StC; 5 CtS. */
int fr_unary(void *runtime, int opcode, void *a, int64_t parameter, void **value);
/* The public magnitude bound is a caller declaration, never a secret scan.
 * It must be finite and >= 1; the legacy unary ModRaise uses bound 1. */
int fr_mod_raise_bounded(void *runtime, void *a, uint32_t target_level, double public_bound, void **value);
/* JSON-only semantic-kernel contribution catalog. No ciphertext contents,
 * public weights, model identifier or secret key material is returned. */
const char *fr_kernel_catalog(void *runtime);
/* Public semantic origin for subsequently submitted operations. Strings are
 * copied into each captured command and never passed to kernel arithmetic.
 * All-null resets the origin. Reset/replace is allowed in eager mode or during
 * capture, never during native scope execution. Measurements aggregate only
 * successfully executed commands, excluding capture and unattributed commands;
 * each admitted scope clears the previous semantic measurements. */
int fr_set_semantic_origin(void *runtime, const char *name, const char *signature, const char *current_implementation);
/* Mark successful frontend lowering, including a node that emits no commands.
 * Such zero-command rows become visible only when its entire scope succeeds. */
int fr_complete_semantic_origin(void *runtime);
/* One semantic operation, with arbitrary contract-declared input/output arity.
 * attributes_json is copied during capture. output_count is zero on failure;
 * output slots are initialized to null and published only as one successful
 * group. Each returned handle is owned by the caller and freed by fr_value_free.
 * Insufficient output capacity fails before any provider arithmetic. During
 * capture any rejection poisons the entire scope, even if its status is ignored.
 * Captured handles follow the same export/lifetime rules as scalar operations. */
int fr_tensor_kernel(void *runtime, const char *operation, void *const *inputs, size_t input_count,
                     const char *attributes_json, void **outputs, size_t output_capacity, size_t *output_count);
const char *fr_profile(void *runtime);
const char *fr_provider_stats(void *runtime);
const char *fr_runtime_stats(void *runtime);
const char *fr_refresh_plan(void *runtime, uint32_t target_level);
int fr_synchronize(void *runtime);
const char *fr_device_info(void *runtime);

#ifdef __cplusplus
}
#endif
