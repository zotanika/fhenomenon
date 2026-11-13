#ifndef FHN_COARSE_API_H
#define FHN_COARSE_API_H
#include "FHN/fhn_backend_api.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Optional compiler-free native contribution, version 1. Calls on one context
   must be serialized. Inputs are canonical two-component ciphertexts belonging
   to that context; binary inputs have equal basis, scale and logical slot count.
   width is a nonzero power of two no greater than logical_slots. Reduction uses
   rotations width/2,...,1 and preserves logical_slots, including partial-width
   reductions. Products consume one Q prime; integer centering consumes none.
   Centered energy/variance require at least three active Q primes (level>=2);
   packed dot requires two. This admission does not establish a universal
   precision bound for arbitrary parameters or plaintext magnitudes.
   All functions return 0 on success and -1 on failure; failure leaves output
   unchanged. Output may alias either input. No client/secret or heap allocation
   is used; registered evaluation keys must remain live during execution.
   Centered energy computes (width*x-sum)^2/width; centered variance reduces
   it. Its actual output scale is width*input_scale^2/q_drop, while packed dot
   uses input_scale^2/q_drop. The centered form uses log2(width) additional bits
   of representable range. Callers must provision scale/modulus headroom for
   their data; it is not interchangeable with an equal-level lower-scale value.
   Centered variance is a variance proxy only when x is period-width and sum is
   its corresponding replicated full-width sum. Metadata does not establish
   these semantic preconditions. */
typedef uint32_t (*FhnCoarseApiVersionFn)(void);
typedef int (*FhnCenteredEnergyFn)(FhnBackendCtx *, FhnBuffer *, const FhnBuffer *, const FhnBuffer *, uint32_t);
typedef int (*FhnCenteredVarianceFn)(FhnBackendCtx *, FhnBuffer *, const FhnBuffer *, const FhnBuffer *, uint32_t);
typedef int (*FhnPackedDotFn)(FhnBackendCtx *, FhnBuffer *, const FhnBuffer *, const FhnBuffer *, uint32_t);
typedef int (*FhnPackedReduceFn)(FhnBackendCtx *, FhnBuffer *, const FhnBuffer *, uint32_t);
#ifdef __cplusplus
}
#endif
#endif
