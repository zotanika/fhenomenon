#ifndef FHN_PLAINTEXT_API_H
#define FHN_PLAINTEXT_API_H
#include "FHN/fhn_backend_api.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct FhnPlaintext FhnPlaintext;
/* The sentinel widens the C++ value range to int, so validating an arbitrary
   caller value is defined behaviour. It is never a valid purpose. */
typedef enum FhnPlaintextPurpose {
  FHN_PLAINTEXT_ADD = 0,
  FHN_PLAINTEXT_MULTIPLY = 1,
  FHN_PLAINTEXT_PURPOSE_MAX_ENUM = 0x7FFFFFFF
} FhnPlaintextPurpose;

/* Optional additive version-1 group. Resolve all symbols together. Preparation
   encodes public real values; it holds no key material or ciphertext snapshot.
   count must equal the prototype's power-of-two logical_slots. Short vectors
   repeat across all physical CKKS slots. The immutable result binds context,
   level, scale (1e-12 relative tolerance), and logical_slots, but not input
   identity. Destroy prepared values before their context. Serialize context calls.
   MULTIPLY encodes at the dropped prime, consumes exactly one level (also for
   zero/integer weights), and preserves input scale. ADD encodes at prototype
   scale and preserves level. Applying the wrong purpose is an error.
   Required/retained bytes count the exact owned coefficient slab, excluding
   object/allocator metadata and context scratch/tables. Required bytes allocates
   nothing; zero means invalid. Encode/apply return 0 on success and -1 on failure.
   Failed encode leaves *out unchanged; failed apply leaves input/output intact.
   Apply supports exact input/output aliases and allocates no heap. Free(NULL)
   is harmless, and bytes(NULL) returns zero. */
typedef uint32_t (*FhnPlaintextApiVersionFn)(void);
typedef uint64_t (*FhnPlaintextRequiredBytesFn)(FhnBackendCtx *, const FhnBuffer *, FhnPlaintextPurpose);
typedef int (*FhnPlaintextEncodeFn)(FhnBackendCtx *, const FhnBuffer *, const double *, uint32_t, FhnPlaintextPurpose,
                                    FhnPlaintext **);
typedef int (*FhnPlaintextApplyFn)(FhnBackendCtx *, FhnBuffer *, const FhnBuffer *, const FhnPlaintext *);
typedef uint64_t (*FhnPlaintextBytesFn)(const FhnPlaintext *);
typedef void (*FhnPlaintextFreeFn)(FhnPlaintext *);
#ifdef __cplusplus
}
#endif
#endif
