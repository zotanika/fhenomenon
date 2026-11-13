#ifndef FHN_MATVEC_API_H
#define FHN_MATVEC_API_H
#include "FHN/fhn_backend_api.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct FhnMatvecPreparation FhnMatvecPreparation;
enum FhnMatvecPrepareFlags {
  FHN_MATVEC_PREPARE_KEYS = 1,
  FHN_MATVEC_PREPARE_VECTOR = 2,
  FHN_MATVEC_PREPARE_MAPS = 4,
  FHN_MATVEC_PREPARE_ALL = 7
};
typedef struct FhnMatvecPrepareOptions {
  uint32_t struct_size;
  uint32_t flags;
} FhnMatvecPrepareOptions;

/* Optional additive version-1 group. Preparation snapshots a canonical source
   at level >=1, binds its context and current immutable evaluation-key identities,
   and precomputes only the selected fixed multipliers/index maps. A null options
   pointer selects ALL; flags=0 retains the snapshot and key bindings as a control.
   width must be a power of two <= logical_slots. Source may later change or die.
   No client or strong key lease is retained. Consumers require the same live,
   currently registered keys; destroy preparation before its backend context.
   Calls on a context are serialized by the caller. Consumers allocate no heap.
   Required/retained bytes cover exact coefficient and map slabs, excluding object
   and allocator metadata, context scratch, keys and tables. Required bytes does
   not allocate; zero means invalid. Prepare returns null on failure.
   Product consumes one level and dot additionally reduces width/2,...,1. Reduce
   accepts a matching product/intermediate at the expected product level/scale,
   with power-of-two width <= prepared width, and preserves actual input metadata.
   Consumers return 0 on success, -1 on failure, leave output unchanged on failure,
   and allow output/input aliases. Resolve the complete group together. */
typedef uint32_t (*FhnMatvecApiVersionFn)(void);
typedef uint64_t (*FhnMatvecPreparationRequiredBytesFn)(FhnBackendCtx *, const FhnBuffer *, uint32_t,
                                                        const FhnMatvecPrepareOptions *);
typedef FhnMatvecPreparation *(*FhnPrepareMatvecSourceFn)(FhnBackendCtx *, const FhnBuffer *, uint32_t,
                                                          const FhnMatvecPrepareOptions *);
typedef uint64_t (*FhnMatvecPreparationBytesFn)(const FhnMatvecPreparation *);
typedef void (*FhnMatvecPreparationFreeFn)(FhnMatvecPreparation *);
typedef int (*FhnMatvecProductFn)(FhnBackendCtx *, FhnBuffer *, const FhnMatvecPreparation *, const FhnBuffer *);
typedef int (*FhnMatvecDotFn)(FhnBackendCtx *, FhnBuffer *, const FhnMatvecPreparation *, const FhnBuffer *);
typedef int (*FhnMatvecReduceFn)(FhnBackendCtx *, FhnBuffer *, const FhnMatvecPreparation *, const FhnBuffer *,
                                 uint32_t);
#ifdef __cplusplus
}
#endif
#endif
