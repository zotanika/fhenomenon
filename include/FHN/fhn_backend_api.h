#ifndef FHN_BACKEND_API_H
#define FHN_BACKEND_API_H

#include "FHN/fhn_program.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles */
typedef struct FhnBackendCtx FhnBackendCtx;
typedef struct FhnBuffer FhnBuffer;
typedef struct FhnExecHandle FhnExecHandle;

/* Device type enumeration */
typedef enum FhnDeviceType {
  FHN_DEVICE_CPU = 0,
  FHN_DEVICE_GPU = 1,
  FHN_DEVICE_FPGA = 2,
  FHN_DEVICE_ASIC = 3
} FhnDeviceType;

/* Backend info returned by fhn_get_info */
typedef struct FhnBackendInfo {
  const char *name;
  const char *version;
  FhnDeviceType device_type;
  uint64_t device_memory; /* bytes, 0 if N/A */
} FhnBackendInfo;

/* Uniform kernel function signature.
   Returns 0 on success, non-zero on error. */
typedef int (*FhnKernelFn)(FhnBackendCtx *ctx, FhnBuffer *result, const FhnBuffer *const *operands,
                           const int64_t *params, const double *fparams);

/* One entry in the kernel table */
typedef struct FhnKernelEntry {
  FhnOpCode opcode;
  FhnKernelFn fn;   /* NULL = not supported */
  const char *name; /* for debugging/logging */
} FhnKernelEntry;

/* The kernel table */
typedef struct FhnKernelTable {
  uint32_t num_kernels;
  FhnKernelEntry *kernels;
} FhnKernelTable;

/* ── Function pointer typedefs for dlopen resolution ── */
typedef FhnBackendInfo *(*FhnGetInfoFn)(void);
typedef FhnBackendCtx *(*FhnCreateFn)(const char *config_json);
typedef void (*FhnDestroyFn)(FhnBackendCtx *ctx);
typedef FhnKernelTable *(*FhnGetKernelsFn)(FhnBackendCtx *ctx);

/* Optional advanced exports (NULL if not provided) */
typedef FhnExecHandle *(*FhnSubmitFn)(FhnBackendCtx *ctx, const FhnProgram *program, FhnBuffer **inputs,
                                      uint32_t num_inputs);
typedef int (*FhnPollFn)(FhnExecHandle *handle);
typedef int (*FhnWaitFn)(FhnExecHandle *handle);
typedef FhnBuffer **(*FhnGetOutputsFn)(FhnExecHandle *handle, uint32_t *num_outputs);
typedef void (*FhnExecFreeFn)(FhnExecHandle *handle);

/* VTable for resolved backend symbols */
typedef struct FhnBackendVTable {
  FhnGetInfoFn get_info;
  FhnCreateFn create;
  FhnDestroyFn destroy;
  FhnGetKernelsFn get_kernels;

  /* Optional (NULL if not provided by backend) */
  FhnSubmitFn submit;
  FhnPollFn poll;
  FhnWaitFn wait;
  FhnGetOutputsFn get_outputs;
  FhnExecFreeFn exec_free;
} FhnBackendVTable;

#ifdef __cplusplus
}
#endif

#endif /* FHN_BACKEND_API_H */
