#ifndef FHN_PREPARED_API_H
#define FHN_PREPARED_API_H
#include "FHN/fhn_backend_api.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct FhnRotationPreparation FhnRotationPreparation;

/* Optional additive group: resolve all five functions together. Preparation
   copies a canonical two-component source and retains its inverse-NTT c1.
   It owns no keys or client. Destroy it before its backend context. Calls on
   one context are serialized by the caller. Source buffers may subsequently
   change or be freed. Rotation requires matching context and live registered
   evaluation keys (identity requires no key). Failure leaves output unchanged.
   bytes reports reserved coefficient/ciphertext slabs, excluding object and
   allocator overhead, keys, tables and context scratch; null reports zero.
   For active Q limb count k and degree N this is exactly 3*k*N*sizeof(uint64_t).
   prepare returns null on error; rotate returns zero on success, -1 on error. */
/* Nonallocating estimate; zero for invalid input. Same reserve as bytes. */
typedef uint64_t (*FhnRotationPreparationRequiredBytesFn)(FhnBackendCtx *, const FhnBuffer *);
typedef FhnRotationPreparation *(*FhnPrepareRotationSourceFn)(FhnBackendCtx *, const FhnBuffer *);
typedef uint64_t (*FhnRotationPreparationBytesFn)(const FhnRotationPreparation *);
typedef int (*FhnRotatePreparedFn)(FhnBackendCtx *, FhnBuffer *, const FhnRotationPreparation *, int64_t);
typedef void (*FhnRotationPreparationFreeFn)(FhnRotationPreparation *);
#ifdef __cplusplus
}
#endif
#endif
