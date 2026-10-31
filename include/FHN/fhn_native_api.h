#ifndef FHN_NATIVE_API_H
#define FHN_NATIVE_API_H
#include "FHN/fhn_backend_api.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Public diagnostic metadata, with no plaintext or key access. Reserved bytes
   describe full allocated ciphertext capacity; live payload may shrink. */
typedef struct FhnCkksBufferInfo {
  uint64_t reserved_bytes;
  uint64_t live_payload_bytes;
  uint64_t scratch_capacity_bytes;
  uint64_t scratch_high_water_bytes;
  int64_t level;
  double scale;
  uint32_t components;
  uint32_t logical_slots;
} FhnCkksBufferInfo;
typedef int (*FhnCkksBufferInfoFn)(FhnBackendCtx *, const FhnBuffer *, FhnCkksBufferInfo *);
#ifdef __cplusplus
}
#endif
#endif
