#include "CheddarButterfly.cuh"
#include "CudaPrimitives.h"
#include <cassert>
#include <limits>
#include <stdexcept>

namespace fhn_cuda {
Modulus makeModulus(uint64_t q) {
  if (q < 3 || q >= (uint64_t{1} << 62) || !(q & 1))
    throw std::invalid_argument("CUDA requires an odd modulus in [3, 2^62)");
  uint64_t inverse = 1;
  for (unsigned i = 0; i < 6; ++i)
    inverse *= 2 - q * inverse;
  using Wide = unsigned __int128;
  const auto r = static_cast<uint64_t>((Wide{1} << 64) % q);
  return {q, static_cast<int64_t>(inverse), static_cast<uint64_t>(Wide{r} * r % q)};
}
namespace {
__device__ uint64_t canonical(int64_t a, uint64_t q) { return a < 0 ? uint64_t(a + int64_t(q)) : uint64_t(a); }
__device__ void butterfly(int64_t &a, int64_t &b, uint64_t root, const Modulus &m, bool inverse) {
  const auto montgomery_root = cheddar::basic::MultMontgomery<uint64_t>(root, m.r2, m.q, m.inverse);
  if (inverse)
    cheddar::kernel::ButterflyINTT<uint64_t>(a, b, montgomery_root, m.q, m.inverse);
  else
    cheddar::kernel::ButterflyNTT<uint64_t>(a, b, montgomery_root, m.q, m.inverse);
}
// Small transforms keep signed Cheddar butterfly values in one block. Larger
// transforms use independently ordered global stages; no cross-block spinlock.
__global__ void smallTransform(uint64_t *data, const NttDescriptor *tables, unsigned table_count, unsigned logn,
                               bool inverse) {
  extern __shared__ int64_t values[];
  const unsigned n = 1u << logn;
  const auto d = tables[blockIdx.x % table_count];
  assert(d.logDegree == logn);
  auto *poly = data + size_t(blockIdx.x) * n;
  for (unsigned i = threadIdx.x; i < n; i += blockDim.x)
    values[i] = int64_t(poly[i]);
  __syncthreads();
  for (unsigned stage = 0; stage < logn; ++stage) {
    const unsigned gap = inverse ? (1u << stage) : (n >> (stage + 1));
    const unsigned m = n / (2 * gap);
    for (unsigned k = threadIdx.x; k < n / 2; k += blockDim.x) {
      const unsigned group = k / gap, j = 2 * group * gap + k % gap;
      const auto root = inverse ? d.inverseRoots[m + group] : d.forwardRoots[m + group];
      butterfly(values[j], values[j + gap], root, d.mod, inverse);
    }
    __syncthreads();
  }
  for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
    auto value = canonical(values[i], d.mod.q);
    poly[i] = inverse ? mul(value, d.degreeInverse, d.mod) : value;
  }
}
__global__ void globalStage(uint64_t *data, const NttDescriptor *tables, unsigned table_count, unsigned logn,
                            unsigned stage, bool inverse) {
  const unsigned n = 1u << logn, k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n / 2)
    return;
  const auto d = tables[blockIdx.y % table_count];
  assert(d.logDegree == logn);
  auto *poly = data + size_t(blockIdx.y) * n;
  const unsigned gap = inverse ? (1u << stage) : (n >> (stage + 1));
  const unsigned group = k / gap, j = 2 * group * gap + k % gap, m = n / (2 * gap);
  int64_t a = int64_t(poly[j]), b = int64_t(poly[j + gap]);
  butterfly(a, b, inverse ? d.inverseRoots[m + group] : d.forwardRoots[m + group], d.mod, inverse);
  auto left = canonical(a, d.mod.q), right = canonical(b, d.mod.q);
  if (inverse && stage + 1 == logn) {
    left = mul(left, d.degreeInverse, d.mod);
    right = mul(right, d.degreeInverse, d.mod);
  }
  poly[j] = left;
  poly[j + gap] = right;
}
} // namespace
cudaError_t configureNtt(unsigned max_log_degree) {
  return max_log_degree >= 1 && max_log_degree <= 17 ? cudaSuccess : cudaErrorInvalidValue;
}
cudaError_t launchNtt(uint64_t *data, const NttDescriptor *tables, unsigned table_count, unsigned poly_count,
                      unsigned logn, bool inverse, cudaStream_t stream) {
  if (!data || !tables || !table_count || !poly_count || logn < 1 || logn > 17)
    return cudaErrorInvalidValue;
  const uint64_t transforms = uint64_t(table_count) * poly_count;
  if (transforms > (logn <= 10 ? uint64_t(std::numeric_limits<int>::max()) : uint64_t(65535)))
    return cudaErrorInvalidValue;
  const unsigned n = 1u << logn;
  if (logn <= 10) {
    smallTransform<<<unsigned(transforms), 256, n * sizeof(int64_t), stream>>>(data, tables, table_count, logn,
                                                                               inverse);
    return cudaPeekAtLastError();
  }
  for (unsigned stage = 0; stage < logn; ++stage) {
    globalStage<<<dim3((n / 2 + 255) / 256, unsigned(transforms)), 256, 0, stream>>>(data, tables, table_count, logn,
                                                                                     stage, inverse);
    const auto status = cudaPeekAtLastError();
    if (status != cudaSuccess)
      return status;
  }
  return cudaSuccess;
}
} // namespace fhn_cuda
