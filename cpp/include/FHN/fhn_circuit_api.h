#ifndef FHN_CIRCUIT_API_H
#define FHN_CIRCUIT_API_H

#include "FHN/fhn_backend_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define FHN_CIRCUIT_VERSION 1u

/* Value semantics. MULTIPLY means a normalized encrypted product; SCALE
   permits exact integer arithmetic without consuming a modulus. These are
   independent of the physical operations in FhnProgram. */
typedef enum FhnCircuitOp
#ifdef __cplusplus
  : uint32_t
#endif
{ FHN_CIRCUIT_ADD = 0,
  FHN_CIRCUIT_SUBTRACT,
  FHN_CIRCUIT_MULTIPLY,
  FHN_CIRCUIT_SCALE,
  FHN_CIRCUIT_ADD_CONSTANT,
  FHN_CIRCUIT_NEGATE,
  FHN_CIRCUIT_ROTATE,
  FHN_CIRCUIT_CONJUGATE } FhnCircuitOp;

typedef struct FhnCircuitNode {
  FhnCircuitOp op;
  uint32_t result_id;
  uint32_t operands[2];
  int64_t rotation;
  double scalar;
} FhnCircuitNode;

typedef struct FhnCircuit {
  uint32_t version;
  uint32_t num_nodes;
  const FhnCircuitNode *nodes;
  uint32_t num_inputs;
  const uint32_t *input_ids;
  uint32_t num_outputs;
  const uint32_t *output_ids;
} FhnCircuit;

typedef struct FhnExecutable FhnExecutable;

typedef struct FhnCompileOptions {
  uint32_t struct_size;
  /* 0 means unlimited. Bounds reserved internal ciphertext slots + scratch,
     excluding keys, tables, graph metadata and caller-owned I/O buffers. */
  uint64_t workspace_limit_bytes;
} FhnCompileOptions;

typedef struct FhnExecutableStats {
  uint32_t struct_size;
  uint32_t source_nodes;
  uint32_t regions;
  uint32_t linear_regions;
  uint32_t product_regions;
  uint32_t rotation_regions;
  uint32_t storage_slots;
  uint64_t storage_bytes;
  uint64_t scratch_bytes;
} FhnExecutableStats;

/* Optional additive export group: resolve all five functions together.
   Compile copies the graph and prototype metadata, never input values. The
   trusted client prepares public evaluation keys separately. The executable
   must die before its backend context; buffers must outlive each run. Calls
   on one executable/context are serialized by the caller.

   Run may rebind matching input signatures and permits input/output aliasing.
   Distinct output value IDs need distinct destination buffers. Failure leaves
   all public outputs unchanged. Prepared run does not allocate heap memory.
   Key ownership stays with the client; key expiration makes execution fail. */
typedef FhnExecutable *(*FhnCompileCircuitFn)(FhnBackendCtx *, const FhnCircuit *, const FhnBuffer *const *, uint32_t,
                                              const FhnCompileOptions *, char *, uint32_t);
typedef int (*FhnClientPrepareExecutableFn)(FhnClientCtx *, FhnExecutable *);
typedef int (*FhnRunExecutableFn)(FhnExecutable *, const FhnBuffer *const *, uint32_t, FhnBuffer *const *, uint32_t);
typedef int (*FhnExecutableStatsFn)(const FhnExecutable *, FhnExecutableStats *);
typedef void (*FhnExecutableFreeFn)(FhnExecutable *);

#ifdef __cplusplus
}
#endif
#endif
