// Contract fixture: emulate CUDA's documented pageable staging return using
// pinned staging and delayed real DMA, then consume from a nonblocking stream.
#include "CudaEngine.h"
#include <chrono>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <thread>
#include <vector>
using namespace fhn_cuda;
namespace {
void *staging = nullptr;
bool simulate = false, omit_completion = false;
void CUDART_CB delay(void *) { std::this_thread::sleep_for(std::chrono::milliseconds(150)); }
template <class F> F runtime(const char *name) {
  auto function = reinterpret_cast<F>(dlsym(RTLD_NEXT, name));
  if (!function)
    std::terminate();
  return function;
}
} // namespace
extern "C" cudaError_t cudaMemcpy(void *destination, const void *source, size_t bytes, cudaMemcpyKind kind) {
  if (simulate && kind == cudaMemcpyHostToDevice) {
    std::memcpy(staging, source, bytes);
    auto result = cudaLaunchHostFunc(nullptr, delay, nullptr);
    if (result != cudaSuccess)
      return result;
    return cudaMemcpyAsync(destination, staging, bytes, kind, nullptr);
  }
  return runtime<cudaError_t (*)(void *, const void *, size_t, cudaMemcpyKind)>("cudaMemcpy")(destination, source,
                                                                                              bytes, kind);
}
extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  if (simulate && omit_completion && !stream)
    return cudaSuccess;
  return runtime<cudaError_t (*)(cudaStream_t)>("cudaStreamSynchronize")(stream);
}
__global__ void inspect(const uint64_t *words, size_t n, uint64_t expected, unsigned *bad) {
  const auto i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n && words[i] != expected)
    atomicExch(bad, 1U);
}
int main(int argc, char **) {
  try {
    const size_t n = 128 * 1024;
    const uint64_t value = 0xcafe1234567890abULL;
    DeviceWords words(n), failure(1);
    std::vector<uint64_t> source(n, value);
    check(cudaMallocHost(&staging, n * 8));
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaMemsetAsync(words.data, 0, n * 8, stream));
    check(cudaMemsetAsync(failure.data, 0, 8, stream));
    inspect<<<1, 1, 0, stream>>>(words.data, 0, value, reinterpret_cast<unsigned *>(failure.data));
    check(cudaStreamSynchronize(stream));
    omit_completion = argc > 1;
    simulate = true;
    words.upload(source.data(), n);
    inspect<<<unsigned((n + 255) / 256), 256, 0, stream>>>(words.data, n, value,
                                                           reinterpret_cast<unsigned *>(failure.data));
    check(cudaGetLastError());
    const auto bad = failure.download(1, stream)[0];
    simulate = false;
    check(cudaStreamSynchronize(nullptr));
    check(cudaFreeHost(staging));
    check(cudaStreamDestroy(stream));
    if (bad) {
      std::cerr << "FAIL: independent evaluator stream consumed an incomplete pageable upload\n";
      return 1;
    }
    std::cout << "PASS: upload publishes device contents only after delayed real DMA completes\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 2;
  }
}
