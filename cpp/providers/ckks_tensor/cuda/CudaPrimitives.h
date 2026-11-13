#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Canonical Fhenomenon residues; Cheddar's MIT Montgomery arithmetic is reused
// internally. No dependency on the Cheddar context, allocator or key generator.
namespace fhn_cuda {
struct Modulus {
  uint64_t q;
  int64_t inverse; // q^-1 mod 2^64, interpreted as a signed word by Cheddar.
  uint64_t r2;     // 2^128 mod q.
};
// Public CKKS modulus contract: odd 3 <= q < 2^62.
Modulus makeModulus(uint64_t q);
struct NttDescriptor {
  Modulus mod;
  const uint64_t *forwardRoots;
  const uint64_t *forwardQuotients; // Retained ABI field; reference path does not use it.
  const uint64_t *inverseRoots;
  const uint64_t *inverseQuotients;
  uint64_t degreeInverse;
  uint64_t degreeInverseQuotient;
  unsigned logDegree;
};
cudaError_t configureNtt(unsigned max_log_degree = 17);
// Asynchronous transform on the supplied stream. Native bit-reversed root
// order, canonical inputs/outputs, [poly_count][table_count][2^log_degree].
// Tables and buffers must remain alive through stream completion.
cudaError_t launchNtt(uint64_t *data, const NttDescriptor *device_tables, unsigned table_count, unsigned poly_count,
                      unsigned log_degree, bool inverse, cudaStream_t stream);
} // namespace fhn_cuda

#if defined(__CUDACC__)
#include "common/Basic.cuh"
namespace fhn_cuda {
__device__ __forceinline__ unsigned bitReverse(unsigned x, unsigned logN) {
  return cheddar::basic::BitReverse(x, logN);
}
__device__ __forceinline__ uint64_t neg(uint64_t a, const Modulus &m) { return cheddar::basic::Negate(a, m.q); }
__device__ __forceinline__ uint64_t reduce(uint64_t a, const Modulus &m) { return a % m.q; }
__device__ __forceinline__ uint64_t mul(uint64_t a, uint64_t b, const Modulus &m) {
  const auto t = cheddar::basic::MultMontgomery<uint64_t>(a, b, m.q, m.inverse);
  return cheddar::basic::MultMontgomery<uint64_t>(t, m.r2, m.q, m.inverse);
}
__device__ __forceinline__ uint64_t add(uint64_t a, uint64_t b, const Modulus &m) {
  return cheddar::basic::Add(a, b, m.q);
}
__device__ __forceinline__ uint64_t sub(uint64_t a, uint64_t b, const Modulus &m) {
  return cheddar::basic::Sub(a, b, m.q);
}
// The quotient belongs to the host table ABI; this implementation deliberately
// uses the public Montgomery path instead of a reciprocal/Shoup fast path.
__device__ __forceinline__ uint64_t multiplyConstant(uint64_t a, uint64_t b, uint64_t, const Modulus &m) {
  return mul(reduce(a, m), b, m);
}
} // namespace fhn_cuda
#endif
