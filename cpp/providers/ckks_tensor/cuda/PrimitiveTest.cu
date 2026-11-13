#include "CKKS/ModArith.h"
#include "CKKS/NttTables.h"
#include "CudaPrimitives.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using fhenomenon::ckks::NttTables;
using fhenomenon::ckks::modarith::uint128;
namespace p = fhn_cuda;

void check(cudaError_t status, const char *operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
template <class T> struct Device {
  T *ptr = nullptr;
  size_t count;
  explicit Device(size_t n) : count(n) { check(cudaMalloc(&ptr, n * sizeof(T)), "cudaMalloc"); }
  ~Device() {
    if (ptr)
      cudaFree(ptr);
  }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  void upload(const std::vector<T> &data, cudaStream_t stream) {
    if (data.size() != count)
      throw std::runtime_error("upload size");
    check(cudaMemcpyAsync(ptr, data.data(), count * sizeof(T), cudaMemcpyHostToDevice, stream), "upload");
  }
  std::vector<T> download(cudaStream_t stream) {
    std::vector<T> result(count);
    check(cudaMemcpyAsync(result.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost, stream), "download");
    check(cudaStreamSynchronize(stream), "download wait");
    return result;
  }
};
struct ArithmeticInput {
  p::Modulus mod;
  uint64_t arbitrary, a, b, quotient;
};
struct ArithmeticOutput {
  uint64_t reduced, multiplied, added, subtracted, shoup, lazy;
};
__global__ void arithmeticKernel(const ArithmeticInput *in, ArithmeticOutput *out, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += size_t(gridDim.x) * blockDim.x) {
    const auto v = in[i];
    out[i] = {p::reduce(v.arbitrary, v.mod),
              p::mul(v.a, v.b, v.mod),
              p::add(v.a, v.b, v.mod),
              p::sub(v.a, v.b, v.mod),
              p::multiplyConstant(v.arbitrary, v.b, v.quotient, v.mod),
              p::multiplyConstant(v.arbitrary, v.b, v.quotient, v.mod)};
  }
}
uint64_t assertions = 0;
void require(bool condition, const std::string &label) {
  ++assertions;
  if (!condition)
    throw std::runtime_error(label);
}
uint64_t nttPrimeNear(uint64_t limit, unsigned log_degree) {
  const uint64_t step = uint64_t{2} << log_degree;
  uint64_t q = limit - ((limit - 1) % step);
  while (!fhenomenon::ckks::modarith::isPrime(q))
    q -= step;
  return q;
}
void testArithmetic(cudaStream_t stream) {
  std::mt19937_64 random(0x547abeef1234ULL);
  std::vector<uint64_t> moduli{2,
                               3,
                               4,
                               5,
                               17,
                               127,
                               128,
                               129,
                               65537,
                               1125899906826241ULL,
                               1099511480321ULL,
                               1099510890497ULL,
                               1152921504606830593ULL,
                               4109006241429359928ULL,
                               (uint64_t{1} << 62) - 1,
                               (uint64_t{1} << 62) - 2,
                               (uint64_t{1} << 61),
                               (uint64_t{1} << 61) - 1,
                               (uint64_t{1} << 61) + 1};
  for (unsigned i = 0; i < 128; ++i)
    moduli.push_back(2 + random() % ((uint64_t{1} << 62) - 2));
  std::vector<ArithmeticInput> inputs;
  moduli.erase(std::remove_if(moduli.begin(), moduli.end(), [](uint64_t q) { return q < 3 || !(q & 1); }),
               moduli.end());
  for (auto q : moduli) {
    const auto mod = p::makeModulus(q);
    const std::array<uint64_t, 8> residues{0, 1, q / 2, q / 2 + 1, q - 1, q - 2, random() % q, random() % q};
    const std::array<uint64_t, 8> arbitrary{0, 1, q - 1, q, 2 * q - 1, 4 * q - 1, UINT64_MAX, random()};
    for (auto raw_a : residues)
      for (auto raw_b : residues)
        for (auto x : arbitrary) {
          const auto a = raw_a % q, b = raw_b % q;
          inputs.push_back({mod, x, a, b, static_cast<uint64_t>((uint128{b} << 64) / q)});
        }
    for (unsigned i = 0; i < 4096; ++i) {
      const auto a = random() % q, b = random() % q, x = random();
      inputs.push_back({mod, x, a, b, static_cast<uint64_t>((uint128{b} << 64) / q)});
    }
  }
  Device<ArithmeticInput> input(inputs.size());
  Device<ArithmeticOutput> output(inputs.size());
  input.upload(inputs, stream);
  arithmeticKernel<<<256, 256, 0, stream>>>(input.ptr, output.ptr, inputs.size());
  check(cudaPeekAtLastError(), "arithmetic launch");
  const auto result = output.download(stream);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &v = inputs[i];
    const auto &r = result[i];
    const auto q = v.mod.q;
    const auto label = "arithmetic item " + std::to_string(i) + " q " + std::to_string(q);
    require(r.reduced == v.arbitrary % q, label + " reduce");
    require(r.multiplied == uint64_t(uint128{v.a} * v.b % q), label + " mul");
    require(r.added == (v.a + v.b) % q, label + " add");
    require(r.subtracted == (v.a + q - v.b) % q, label + " sub");
    require(r.shoup == uint64_t(uint128{v.arbitrary} * v.b % q), label + " shoup");
    require(r.lazy < 2 * q && r.lazy % q == r.shoup, label + " lazy");
  }
  for (auto q : {uint64_t{0}, uint64_t{1}, uint64_t{2}, uint64_t{4}, uint64_t{128}, uint64_t{1} << 62, UINT64_MAX}) {
    bool rejected = false;
    try {
      (void)p::makeModulus(q);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "invalid modulus rejected");
  }
  std::printf("arithmetic PASS: %zu cases, %zu moduli\n", inputs.size(), moduli.size());
}
void compare(const std::vector<uint64_t> &actual, const std::vector<uint64_t> &expected, const std::string &label) {
  require(actual.size() == expected.size(), label + " size");
  for (size_t i = 0; i < actual.size(); ++i)
    require(actual[i] == expected[i], label + " coefficient " + std::to_string(i));
}
void testNtt(cudaStream_t stream) {
  std::mt19937_64 random(0x98bead0cc771ULL);
  check(p::configureNtt(17), "configureNtt");
  check(p::configureNtt(10), "smaller context must preserve larger opt-in");
  require(p::configureNtt(0) == cudaErrorInvalidValue, "reject configure zero degree");
  require(p::configureNtt(18) == cudaErrorInvalidValue, "reject configure excessive degree");
  unsigned transform_cases = 0;
  for (unsigned log_degree : {1u, 3u, 10u, 13u, 15u, 16u, 17u}) {
    const size_t n = size_t{1} << log_degree;
    const unsigned poly_count = 3;
    const std::vector<uint64_t> moduli{
      nttPrimeNear((uint64_t{1} << 50) - 1, log_degree), nttPrimeNear((uint64_t{1} << 40) - 1, log_degree),
      nttPrimeNear((uint64_t{1} << 60) - 1, log_degree), nttPrimeNear((uint64_t{1} << 62) - 1, log_degree)};
    const unsigned table_count = static_cast<unsigned>(moduli.size());
    std::vector<NttTables> cpu_tables;
    std::vector<uint64_t> root_data;
    for (auto q : moduli) {
      std::string error;
      auto table = NttTables::create(q, log_degree, &error);
      if (!table)
        throw std::runtime_error(error);
      cpu_tables.push_back(*table);
      for (auto ptr : {table->forwardRoots().value, table->forwardRoots().quotient, table->inverseRoots().value,
                       table->inverseRoots().quotient})
        root_data.insert(root_data.end(), ptr, ptr + n);
    }
    Device<uint64_t> device_roots(root_data.size());
    device_roots.upload(root_data, stream);
    std::vector<p::NttDescriptor> descriptors;
    for (unsigned t = 0; t < table_count; ++t) {
      const auto &ct = cpu_tables[t];
      const auto base = device_roots.ptr + t * 4 * n;
      descriptors.push_back({p::makeModulus(moduli[t]), base, base + n, base + 2 * n, base + 3 * n, ct.degreeInverse(),
                             ct.degreeInverseQuotient(), log_degree});
    }
    Device<p::NttDescriptor> device_tables(table_count);
    device_tables.upload(descriptors, stream);
    Device<uint64_t> data(poly_count * table_count * n);
    for (unsigned pattern = 0; pattern < 4; ++pattern) {
      std::vector<uint64_t> original(data.count);
      for (unsigned poly = 0; poly < poly_count; ++poly)
        for (unsigned t = 0; t < table_count; ++t)
          for (size_t i = 0; i < n; ++i) {
            auto &value = original[(poly * table_count + t) * n + i];
            const auto q = moduli[t];
            value = pattern == 0 ? 0 : pattern == 1 ? q - 1 : pattern == 2 ? random() % q : (i % 2 ? q - 1 : 0);
          }
      auto expected = original;
      for (unsigned pidx = 0; pidx < poly_count; ++pidx)
        for (unsigned t = 0; t < table_count; ++t)
          fhenomenon::ckks::ntt::forward(expected.data() + (pidx * table_count + t) * n, cpu_tables[t]);
      data.upload(original, stream);
      check(p::launchNtt(data.ptr, device_tables.ptr, table_count, poly_count, log_degree, false, stream),
            "forward launch");
      compare(data.download(stream), expected,
              "forward N=" + std::to_string(n) + " pattern=" + std::to_string(pattern));
      check(p::launchNtt(data.ptr, device_tables.ptr, table_count, poly_count, log_degree, true, stream),
            "inverse roundtrip launch");
      compare(data.download(stream), original, "roundtrip N=" + std::to_string(n));
      expected = original;
      for (unsigned pidx = 0; pidx < poly_count; ++pidx)
        for (unsigned t = 0; t < table_count; ++t)
          fhenomenon::ckks::ntt::inverse(expected.data() + (pidx * table_count + t) * n, cpu_tables[t]);
      data.upload(original, stream);
      check(p::launchNtt(data.ptr, device_tables.ptr, table_count, poly_count, log_degree, true, stream),
            "inverse independent launch");
      compare(data.download(stream), expected, "inverse independent N=" + std::to_string(n));
      transform_cases += poly_count * table_count * 3;
    }
    require(p::launchNtt(nullptr, device_tables.ptr, table_count, poly_count, log_degree, false, stream) ==
              cudaErrorInvalidValue,
            "reject null data");
    require(p::launchNtt(data.ptr, nullptr, table_count, poly_count, log_degree, false, stream) ==
              cudaErrorInvalidValue,
            "reject null table");
    require(p::launchNtt(data.ptr, device_tables.ptr, 0, poly_count, log_degree, false, stream) ==
              cudaErrorInvalidValue,
            "reject empty tables");
    require(p::launchNtt(data.ptr, device_tables.ptr, table_count, 0, log_degree, false, stream) ==
              cudaErrorInvalidValue,
            "reject empty polynomials");
    require(p::launchNtt(data.ptr, device_tables.ptr, table_count, poly_count, 18, false, stream) ==
              cudaErrorInvalidValue,
            "reject excessive degree");
    require(p::launchNtt(data.ptr, device_tables.ptr, UINT32_MAX, UINT32_MAX, log_degree, false, stream) ==
              cudaErrorInvalidValue,
            "reject grid overflow");
    std::printf("NTT PASS: N=%zu, tables=%u, polys=%u, patterns=4, near62prime=%llu\n", n, table_count, poly_count,
                (unsigned long long)moduli.back());
  }
  std::printf("NTT PASS: %u polynomial transforms compared\n", transform_cases);
}
int main() {
  try {
    int device = 0;
    check(cudaGetDevice(&device), "get device");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "get properties");
    std::printf("GPU=%s cc=%d.%d shared=%zu sharedOptin=%zu\n", prop.name, prop.major, prop.minor,
                prop.sharedMemPerBlock, prop.sharedMemPerBlockOptin);
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");
    testArithmetic(stream);
    testNtt(stream);
    check(cudaStreamDestroy(stream), "stream destroy");
    std::printf("PASS assertions=%llu\n", (unsigned long long)assertions);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
