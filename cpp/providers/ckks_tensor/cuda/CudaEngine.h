#pragma once
// Independent Fhenomenon CUDA backend. No secret-key type or external
// FHE provider enters this interface. Client metadata/capacity validation belongs
// to the owner; this engine owns immutable device tables/keys and scratch.
#include "CKKS/EvalTables.h"
#include "CKKS/Storage.h"
#include "CudaPrimitives.h"
#include <map>
#include <memory>
#include <vector>

namespace fhn_cuda {
void check(cudaError_t status);
// Bind CUDA work to an owning device and restore the caller's thread-local
// selection on scope exit. A negative target disables the scope for CPU paths.
class ScopedDevice {
  public:
  explicit ScopedDevice(int target);
  ~ScopedDevice() noexcept;
  ScopedDevice(const ScopedDevice &) = delete;
  ScopedDevice &operator=(const ScopedDevice &) = delete;

  private:
  int previous_ = -1;
  bool changed_ = false;
};
struct DeviceWords {
  uint64_t *data = nullptr;
  size_t words = 0;
  explicit DeviceWords(size_t count);
  ~DeviceWords();
  DeviceWords(const DeviceWords &) = delete;
  DeviceWords &operator=(const DeviceWords &) = delete;
  // Completes device DMA before returning, including pageable host sources;
  // immediately safe for consumers on independent nonblocking streams.
  void upload(const uint64_t *source, size_t count);
  std::vector<uint64_t> download(size_t count, cudaStream_t stream) const;

  private:
  int device_id_ = 0;
};

// Canonical dense [2][activeQ][N] ciphertexts and [activeQ][N] plaintexts.
// Operations enqueue only: no ciphertext host copy/evaluation, allocation, or
// synchronization. Bind to one explicit nondefault stream; caller retains all
// buffers and this engine until completion. Exact aliases permitted only by
// add/subtract/negate/addPlaintext/bias/multiplyScalarResidue. Other outputs must
// be disjoint. Not thread-safe: serialize each complete operation enqueue.
class CudaEngine {
  public:
  CudaEngine(const fhenomenon::ckks::EvaluationTables &tables,
             const std::map<uint64_t, fhenomenon::ckks::EvaluationKeyView> &galoisKeys,
             fhenomenon::ckks::EvaluationKeyView relinKey = {});
  ~CudaEngine();
  CudaEngine(const CudaEngine &) = delete;
  CudaEngine &operator=(const CudaEngine &) = delete;
  void rotate(uint64_t *out, const uint64_t *in, unsigned k, uint64_t galois, cudaStream_t stream);
  void add(uint64_t *out, const uint64_t *a, const uint64_t *b, unsigned k, cudaStream_t stream);
  void subtract(uint64_t *out, const uint64_t *a, const uint64_t *b, unsigned k, cudaStream_t stream);
  void negate(uint64_t *out, const uint64_t *in, unsigned k, cudaStream_t stream);
  void addPlaintext(uint64_t *out, const uint64_t *in, const uint64_t *plain, unsigned k, cudaStream_t stream);
  void bias(uint64_t *out, const uint64_t *in, unsigned k, int64_t encoded, cudaStream_t stream);
  void multiplyScalarResidue(uint64_t *out, const uint64_t *in, unsigned k, int64_t integer, cudaStream_t stream);
  // When rescale=true output active Q is k-1; otherwise k. Multiplication
  // always includes relinearization. Rescale requires k>=2.
  void multiplyPlaintext(uint64_t *out, const uint64_t *in, const uint64_t *plain, unsigned k, cudaStream_t stream,
                         bool rescale = true);
  void multiply(uint64_t *out, const uint64_t *a, const uint64_t *b, unsigned k, cudaStream_t stream,
                bool rescale = true);
  void rescale(uint64_t *out, const uint64_t *in, unsigned k, cudaStream_t stream);
  void levelDown(uint64_t *out, const uint64_t *in, unsigned k, unsigned targetK, cudaStream_t stream);
  // Input is bottom-Q (k=1); lift each centered coefficient to targetK Q limbs.
  // This is the modulus-raising primitive, not a standalone bootstrap.
  void centeredLift(uint64_t *out, const uint64_t *in, unsigned targetK, cudaStream_t stream);
  size_t residentBytes() const;

  private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace fhn_cuda
