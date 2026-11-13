#include "CKKS/ModArith.h"
#include "CudaEngine.h"
#include <algorithm>
#include <stdexcept>

namespace fhn_cuda {
using namespace fhenomenon::ckks;
void check(cudaError_t s) {
  if (s != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(s));
}
ScopedDevice::ScopedDevice(int target) {
  if (target < 0)
    return;
  check(cudaGetDevice(&previous_));
  if (previous_ != target) {
    check(cudaSetDevice(target));
    changed_ = true;
  }
}
ScopedDevice::~ScopedDevice() noexcept {
  if (changed_)
    (void)cudaSetDevice(previous_);
}
DeviceWords::DeviceWords(size_t count) : words(count) {
  if (!count || count > SIZE_MAX / sizeof(uint64_t))
    throw std::invalid_argument("invalid device allocation size");
  check(cudaGetDevice(&device_id_));
  check(cudaMalloc(&data, count * sizeof(uint64_t)));
}
DeviceWords::~DeviceWords() {
  if (data)
    try {
      ScopedDevice device(device_id_);
      (void)cudaFree(data);
    } catch (...) {
    }
}
void DeviceWords::upload(const uint64_t *src, size_t count) {
  if (!src || count > words)
    throw std::out_of_range("device upload");
  ScopedDevice device(device_id_);
  check(cudaMemcpy(data, src, count * sizeof(uint64_t), cudaMemcpyHostToDevice));
  // Pageable host copies may return once staging finishes, before device DMA.
  // Evaluators use a nonblocking stream, so default-stream ordering is not an
  // implicit dependency: publish the allocation only after the copy completes.
  check(cudaStreamSynchronize(nullptr));
}
std::vector<uint64_t> DeviceWords::download(size_t count, cudaStream_t stream) const {
  if (count > words)
    throw std::out_of_range("device download");
  ScopedDevice device(device_id_);
  check(cudaStreamSynchronize(stream));
  std::vector<uint64_t> result(count);
  check(cudaMemcpy(result.data(), data, count * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  return result;
}
namespace {
struct Factors {
  uint64_t pMod, pInv, pShoup, dropMod, dropInv, dropShoup;
};
struct DigitPlan {
  unsigned begin, end, ext;
  const uint64_t *inverse;
  const uint64_t *inverseShoup;
  // Source-major, includes zero entries for source destinations (not read).
  const uint64_t *residue;
  const uint64_t *residueShoup;
};
unsigned blocks(size_t words) { return unsigned((words + 255) / 256); }
__device__ unsigned reverse(unsigned x, unsigned logN) { return fhn_cuda::bitReverse(x, logN); }
__global__ void permute(uint64_t *out, const uint64_t *in, unsigned n, unsigned k, unsigned logN, uint64_t g) {
  const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (x >= size_t(2) * k * n)
    return;
  const unsigned i = x % n,
                 source = reverse(unsigned(((g * (2 * uint64_t(reverse(i, logN)) + 1)) & (2 * n - 1)) - 1) / 2, logN);
  out[x] = in[(x / n) * n + source];
}
__global__ void extendDigit(uint64_t *out, const uint64_t *coeff, unsigned n, const NttDescriptor *tables,
                            DigitPlan plan) {
  const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (x >= size_t(plan.ext) * n)
    return;
  const unsigned l = x / n, i = x % n;
  if (l >= plan.begin && l < plan.end) {
    out[x] = coeff[x];
    return;
  }
  const auto m = tables[l].mod;
  uint64_t result = 0;
  for (unsigned s = plan.begin; s < plan.end; ++s) {
    const unsigned source = s - plan.begin, slot = source * plan.ext + l;
    const uint64_t value =
      multiplyConstant(coeff[size_t(s) * n + i], plan.inverse[source], plan.inverseShoup[source], tables[s].mod);
    result = fhn_cuda::add(result, multiplyConstant(value, plan.residue[slot], plan.residueShoup[slot], m), m);
  }
  out[x] = result;
}
__global__ void accumulateDigit(uint64_t *out, const uint64_t *extended, const uint64_t *key, unsigned n, unsigned k,
                                unsigned p, unsigned fullK, unsigned digit, const NttDescriptor *tables) {
  const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned ext = k + p;
  if (x >= size_t(2) * ext * n)
    return;
  const unsigned c = x / (size_t(ext) * n), l = (x / n) % ext, i = x % n, keyL = l < k ? l : fullK + l - k;
  const auto m = tables[l].mod;
  const uint64_t value =
    mul(extended[size_t(l) * n + i], key[((size_t(digit) * 2 + c) * (fullK + p) + keyL) * n + i], m);
  out[x] = fhn_cuda::add(out[x], value, m);
}
__global__ void extract(uint64_t *out, const uint64_t *in, unsigned n, unsigned limbs, unsigned begin, unsigned count) {
  const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (x >= size_t(2) * count * n)
    return;
  const unsigned c = x / (size_t(count) * n), l = (x / n) % count, i = x % n;
  out[x] = in[(size_t(c) * limbs + begin + l) * n + i];
}
// One thread owns all Garner digits of one coefficient; no inter-thread state.
__global__ void mixedRadix(uint64_t *mixed, const uint64_t *residues, unsigned n, unsigned p, const NttDescriptor *aux,
                           const uint64_t *inverses) {
  const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= 2 * n)
    return;
  const unsigned c = x / n, i = x % n;
  for (unsigned l = 0; l < p; ++l) {
    const auto m = aux[l].mod;
    uint64_t value = residues[(size_t(c) * p + l) * n + i];
    for (unsigned j = 0; j < l; ++j)
      value = mul(sub(value, reduce(mixed[(size_t(c) * p + j) * n + i], m), m), inverses[size_t(l) * p + j], m);
    mixed[(size_t(c) * p + l) * n + i] = value;
  }
}
__global__ void centeredAux(uint64_t *out, const uint64_t *mixed, unsigned n, unsigned k, unsigned p,
                            const NttDescriptor *tables, const Factors *f) {
  const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (x >= size_t(2) * k * n)
    return;
  const unsigned c = x / (size_t(k) * n), l = (x / n) % k, i = x % n;
  const auto m = tables[l].mod;
  uint64_t value = 0;
  bool negative = false, decided = false;
  for (unsigned j = p; j > 0; --j) {
    const uint64_t q = tables[k + j - 1].mod.q, digit = mixed[(size_t(c) * p + j - 1) * n + i];
    value = fhn_cuda::add(mul(value, reduce(q, m), m), reduce(digit, m), m);
    if (!decided && digit != q / 2) {
      negative = digit > q / 2;
      decided = true;
    }
  }
  out[x] = negative ? sub(value, f[l].pMod, m) : value;
}
__global__ void centeredOne(uint64_t *out, const uint64_t *removed, unsigned n, unsigned k, uint64_t divisor,
                            const NttDescriptor *tables) {
  const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (x >= size_t(2) * k * n)
    return;
  const unsigned c = x / (size_t(k) * n), l = (x / n) % k;
  const auto m = tables[l].mod;
  const uint64_t raw = removed[size_t(c) * n + x % n], value = reduce(raw, m);
  out[x] = raw > divisor / 2 ? sub(value, reduce(divisor, m), m) : value;
}
__global__ void finish(uint64_t *out, const uint64_t *in, const uint64_t *correction, unsigned n, unsigned k,
                       unsigned inLimbs, const NttDescriptor *tables, const Factors *f, bool auxiliary) {
  const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (x >= size_t(2) * k * n)
    return;
  const unsigned c = x / (size_t(k) * n), l = (x / n) % k, i = x % n;
  const auto m = tables[l].mod;
  out[x] = multiplyConstant(sub(in[(size_t(c) * inLimbs + l) * n + i], correction[x], m),
                            auxiliary ? f[l].pInv : f[l].dropInv, auxiliary ? f[l].pShoup : f[l].dropShoup, m);
}
// operation: 0 add, 1 subtract, 2 negate, 3 plaintext add, 4 plaintext multiply,
// 5 tensor-base add (both components), 6 rotation-base add (only c0).
__global__ void pointwise(uint64_t *out, const uint64_t *a, const uint64_t *b, unsigned n, unsigned k,
                          const NttDescriptor *tables, unsigned operation) {
  const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (x >= size_t(2) * k * n)
    return;
  const auto m = tables[(x / n) % k].mod;
  const uint64_t value = a[x];
  uint64_t result;
  if (operation == 2)
    result = fhn_cuda::neg(value, m);
  else if (operation == 3)
    result = x < size_t(k) * n ? fhn_cuda::add(value, b[x], m) : value;
  else if (operation == 4)
    result = mul(value, b[x % (size_t(k) * n)], m);
  else if (operation == 6)
    result = x < size_t(k) * n ? fhn_cuda::add(value, b[x], m) : value;
  else if (operation == 1)
    result = sub(value, b[x], m);
  else
    result = fhn_cuda::add(value, b[x], m);
  out[x] = result;
}
__global__ void tensor(uint64_t *out, const uint64_t *a, const uint64_t *b, unsigned n, unsigned k,
                       const NttDescriptor *tables) {
  const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = size_t(k) * n;
  if (x >= stride)
    return;
  const auto m = tables[x / n].mod;
  out[x] = mul(a[x], b[x], m);
  out[stride + x] = fhn_cuda::add(mul(a[x], b[stride + x], m), mul(a[stride + x], b[x], m), m);
  out[2 * stride + x] = mul(a[stride + x], b[stride + x], m);
}
__global__ void biasKernel(uint64_t *out, const uint64_t *in, unsigned n, unsigned k, int64_t encoded,
                           const NttDescriptor *tables, bool multiply = false) {
  const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (x >= size_t(2) * k * n)
    return;
  const auto m = tables[(x / n) % k].mod;
  uint64_t value = in[x];
  if (multiply || x < size_t(k) * n) {
    const uint64_t mag = encoded < 0 ? uint64_t(0) - uint64_t(encoded) : uint64_t(encoded);
    uint64_t r = reduce(mag, m);
    if (encoded < 0 && r)
      r = m.q - r;
    value = multiply ? mul(value, r, m) : fhn_cuda::add(value, r, m);
  }
  out[x] = value;
}
} // namespace
struct CudaEngine::Impl {
  unsigned n, logN, fullK, p, dnum;
  std::vector<uint64_t> primes;
  std::vector<std::shared_ptr<DeviceWords>> allocations;
  std::vector<NttDescriptor *> tables;
  std::vector<Factors *> factors;
  std::vector<std::vector<DigitPlan>> digits;
  std::map<uint64_t, std::shared_ptr<DeviceWords>> keys;
  std::shared_ptr<DeviceWords> relin, permutation, coefficients, extended, accumulation, removed, mixed, correction,
    product, switched;
  uint64_t *crtInverses = nullptr;
  cudaStream_t bound = nullptr;
  size_t bytes = 0;
  std::shared_ptr<DeviceWords> allocate(size_t words) {
    auto a = std::make_shared<DeviceWords>(words);
    allocations.push_back(a);
    bytes += words * 8;
    return a;
  }
  template <typename T> T *upload(const T *src, size_t count) {
    auto a = allocate((count * sizeof(T) + 7) / 8);
    check(cudaMemcpy(a->data, src, count * sizeof(T), cudaMemcpyHostToDevice));
    check(cudaStreamSynchronize(nullptr));
    return reinterpret_cast<T *>(a->data);
  }
  void bind(unsigned k, cudaStream_t stream) {
    if (!k || k > fullK || !stream)
      throw std::invalid_argument("invalid active Q or default stream");
    if (bound && bound != stream)
      throw std::invalid_argument("workspace already bound to another stream");
    bound = stream;
  }
  std::shared_ptr<DeviceWords> key(const EvaluationTables &host, EvaluationKeyView view, SwitchKeyKind kind,
                                   uint64_t g) {
    if (!view.digits || view.digit_count != dnum || view.main_count != fullK || view.aux_count != p ||
        view.params_id != host.paramsId() || view.log_degree != logN || view.kind != kind || view.galois_element != g)
      throw std::invalid_argument("incompatible evaluation key");
    std::vector<uint64_t> words;
    words.reserve(size_t(dnum) * 2 * (fullK + p) * n);
    for (unsigned d = 0; d < dnum; ++d) {
      const auto digit = view.digits[d];
      const PolyLayout expected{{host.paramsId(), 0, fullK, 0, p}, logN, NttForm::Evaluation};
      if (!digit.wellFormed() || digit.layout.num_polys != 2 || digit.layout.scale != 1.0 ||
          digit.layout.poly != expected)
        throw std::invalid_argument("invalid evaluation-key digit storage");
      for (unsigned c = 0; c < 2; ++c)
        for (unsigned l = 0; l < fullK + p; ++l) {
          const uint64_t q = l < fullK ? primes[l] : host.params().auxPrimes()[l - fullK];
          const auto *values = digit.poly(c).limb(l);
          for (unsigned i = 0; i < n; ++i) {
            if (values[i] >= q)
              throw std::invalid_argument("noncanonical evaluation key");
            words.push_back(values[i]);
          }
        }
    }
    auto result = allocate(words.size());
    result->upload(words.data(), words.size());
    return result;
  }
  Impl(const EvaluationTables &host, const std::map<uint64_t, EvaluationKeyView> &hostKeys,
       EvaluationKeyView relinKey) {
    const auto &params = host.params();
    n = params.degree();
    logN = params.logDegree();
    fullK = params.levelCount();
    p = params.auxPrimes().size();
    dnum = params.dnum();
    primes = params.mainPrimes();
    if (logN < 1 || logN > 17 || !p)
      throw std::invalid_argument("native CUDA requires N<=131072 and a nonempty auxiliary basis");
    check(configureNtt(logN));
    std::vector<NttDescriptor> all;
    for (unsigned l = 0; l < fullK + p; ++l) {
      const auto &t = l < fullK ? host.mainNtt(l) : host.auxNtt(l - fullK);
      const auto fw = t.forwardRoots(), inv = t.inverseRoots();
      all.push_back({makeModulus(t.modulus()), upload(fw.value, n), upload(fw.quotient, n), upload(inv.value, n),
                     upload(inv.quotient, n), t.degreeInverse(), t.degreeInverseQuotient(), logN});
    }
    tables.resize(fullK + 1);
    factors.resize(fullK + 1);
    digits.resize(fullK + 1);
    for (unsigned k = 1; k <= fullK; ++k) {
      std::vector<NttDescriptor> active(all.begin(), all.begin() + k);
      active.insert(active.end(), all.begin() + fullK, all.end());
      tables[k] = upload(active.data(), active.size());
      std::vector<Factors> f(k);
      for (unsigned l = 0; l < k; ++l) {
        auto &v = f[l];
        const auto q = primes[l];
        v.pMod = host.auxProductMod(l);
        v.pInv = host.auxProductInverse(l);
        v.pShoup = modarith::shoupFactor(v.pInv, q);
        if (l + 1 < k) {
          v.dropMod = primes[k - 1] % q;
          v.dropInv = host.dropInverse(k - 1, l);
          v.dropShoup = modarith::shoupFactor(v.dropInv, q);
        }
      }
      factors[k] = upload(f.data(), f.size());
      for (unsigned d = 0; d < dnum; ++d) {
        const auto &plan = host.digit(k - 1, d);
        DigitPlan result{plan.begin, plan.end, k + p, nullptr, nullptr, nullptr, nullptr};
        if (plan.begin != plan.end) {
          if (!plan.conversion)
            throw std::invalid_argument("missing digit conversion");
          const auto &conversion = *plan.conversion;
          const unsigned sources = plan.end - plan.begin;
          const auto inverses = conversion.hatInverses(), residues = conversion.hatResidues();
          result.inverse = upload(inverses.value, sources);
          result.inverseShoup = upload(inverses.quotient, sources);
          std::vector<uint64_t> r(size_t(sources) * (k + p)), s(r.size());
          for (unsigned a = 0; a < sources; ++a)
            for (unsigned j = 0; j < plan.destinations.size(); ++j) {
              const auto destination = plan.destinations[j];
              r[size_t(a) * (k + p) + destination] = residues.value[size_t(a) * plan.destinations.size() + j];
              s[size_t(a) * (k + p) + destination] = residues.quotient[size_t(a) * plan.destinations.size() + j];
            }
          result.residue = upload(r.data(), r.size());
          result.residueShoup = upload(s.data(), s.size());
        }
        digits[k].push_back(result);
      }
    }
    std::vector<uint64_t> inverse(size_t(p) * p);
    for (unsigned l = 0; l < p; ++l)
      for (unsigned j = 0; j < l; ++j)
        inverse[size_t(l) * p + j] =
          modarith::invMod(params.auxPrimes()[j] % params.auxPrimes()[l], params.auxPrimes()[l]);
    crtInverses = upload(inverse.data(), inverse.size());
    for (const auto &entry : hostKeys) {
      if (!(entry.first & 1) || entry.first >= 2 * n)
        throw std::invalid_argument("invalid Galois map element");
      keys.emplace(entry.first, key(host, entry.second, SwitchKeyKind::Galois, entry.first));
    }
    if (relinKey.digits)
      relin = key(host, relinKey, SwitchKeyKind::Relinearization, 1);
    permutation = allocate(size_t(2) * fullK * n);
    coefficients = allocate(size_t(fullK) * n);
    extended = allocate(size_t(fullK + p) * n);
    accumulation = allocate(size_t(2) * (fullK + p) * n);
    removed = allocate(size_t(2) * p * n);
    mixed = allocate(size_t(2) * p * n);
    correction = allocate(size_t(2) * fullK * n);
    product = allocate(size_t(3) * fullK * n);
    switched = allocate(size_t(2) * fullK * n);
  }
  void switchKey(uint64_t *out, const uint64_t *in, unsigned k, const uint64_t *evalKey, cudaStream_t stream) {
    const auto t = tables[k];
    const unsigned ext = k + p;
    check(cudaMemcpyAsync(coefficients->data, in, size_t(k) * n * 8, cudaMemcpyDeviceToDevice, stream));
    check(launchNtt(coefficients->data, t, k, 1, logN, true, stream));
    check(cudaMemsetAsync(accumulation->data, 0, size_t(2) * ext * n * 8, stream));
    for (unsigned d = 0; d < dnum; ++d) {
      const auto plan = digits[k][d];
      if (plan.begin == plan.end)
        continue;
      extendDigit<<<blocks(size_t(ext) * n), 256, 0, stream>>>(extended->data, coefficients->data, n, t, plan);
      check(cudaPeekAtLastError());
      check(launchNtt(extended->data, t, ext, 1, logN, false, stream));
      accumulateDigit<<<blocks(size_t(2) * ext * n), 256, 0, stream>>>(accumulation->data, extended->data, evalKey, n,
                                                                       k, p, fullK, d, t);
      check(cudaPeekAtLastError());
    }
    extract<<<blocks(size_t(2) * p * n), 256, 0, stream>>>(removed->data, accumulation->data, n, ext, k, p);
    check(cudaPeekAtLastError());
    check(launchNtt(removed->data, t + k, p, 2, logN, true, stream));
    mixedRadix<<<blocks(size_t(2) * n), 256, 0, stream>>>(mixed->data, removed->data, n, p, t + k, crtInverses);
    check(cudaPeekAtLastError());
    centeredAux<<<blocks(size_t(2) * k * n), 256, 0, stream>>>(correction->data, mixed->data, n, k, p, t, factors[k]);
    check(cudaPeekAtLastError());
    check(launchNtt(correction->data, t, k, 2, logN, false, stream));
    finish<<<blocks(size_t(2) * k * n), 256, 0, stream>>>(out, accumulation->data, correction->data, n, k, ext, t,
                                                          factors[k], true);
    check(cudaPeekAtLastError());
  }
  void drop(uint64_t *out, const uint64_t *in, unsigned k, cudaStream_t stream) {
    const auto t = tables[k];
    extract<<<blocks(size_t(2) * n), 256, 0, stream>>>(removed->data, in, n, k, k - 1, 1);
    check(cudaPeekAtLastError());
    check(launchNtt(removed->data, t + k - 1, 1, 2, logN, true, stream));
    centeredOne<<<blocks(size_t(2) * (k - 1) * n), 256, 0, stream>>>(correction->data, removed->data, n, k - 1,
                                                                     primes[k - 1], t);
    check(cudaPeekAtLastError());
    check(launchNtt(correction->data, t, k - 1, 2, logN, false, stream));
    finish<<<blocks(size_t(2) * (k - 1) * n), 256, 0, stream>>>(out, in, correction->data, n, k - 1, k, t, factors[k],
                                                                false);
    check(cudaPeekAtLastError());
  }
  void point(uint64_t *out, const uint64_t *a, const uint64_t *b, unsigned k, cudaStream_t stream, unsigned op) {
    bind(k, stream);
    if (!out || !a || (op != 2 && !b))
      throw std::invalid_argument("invalid pointwise operands");
    pointwise<<<blocks(size_t(2) * k * n), 256, 0, stream>>>(out, a, b, n, k, tables[k], op);
    check(cudaPeekAtLastError());
  }
};
CudaEngine::CudaEngine(const EvaluationTables &tables, const std::map<uint64_t, EvaluationKeyView> &keys,
                       EvaluationKeyView relin)
  : impl_(std::make_unique<Impl>(tables, keys, relin)) {}
CudaEngine::~CudaEngine() = default;
size_t CudaEngine::residentBytes() const { return impl_->bytes; }
void CudaEngine::add(uint64_t *out, const uint64_t *a, const uint64_t *b, unsigned k, cudaStream_t s) {
  impl_->point(out, a, b, k, s, 0);
}
void CudaEngine::subtract(uint64_t *out, const uint64_t *a, const uint64_t *b, unsigned k, cudaStream_t s) {
  impl_->point(out, a, b, k, s, 1);
}
void CudaEngine::negate(uint64_t *out, const uint64_t *in, unsigned k, cudaStream_t s) {
  impl_->point(out, in, nullptr, k, s, 2);
}
void CudaEngine::addPlaintext(uint64_t *out, const uint64_t *in, const uint64_t *plain, unsigned k, cudaStream_t s) {
  if (out == plain)
    throw std::invalid_argument("plaintext output alias");
  impl_->point(out, in, plain, k, s, 3);
}
void CudaEngine::bias(uint64_t *out, const uint64_t *in, unsigned k, int64_t encoded, cudaStream_t stream) {
  auto &s = *impl_;
  s.bind(k, stream);
  if (!out || !in)
    throw std::invalid_argument("invalid bias");
  biasKernel<<<blocks(size_t(2) * k * s.n), 256, 0, stream>>>(out, in, s.n, k, encoded, s.tables[k]);
  check(cudaPeekAtLastError());
}
void CudaEngine::multiplyScalarResidue(uint64_t *out, const uint64_t *in, unsigned k, int64_t integer,
                                       cudaStream_t stream) {
  auto &s = *impl_;
  s.bind(k, stream);
  if (!out || !in)
    throw std::invalid_argument("invalid integer multiply");
  biasKernel<<<blocks(size_t(2) * k * s.n), 256, 0, stream>>>(out, in, s.n, k, integer, s.tables[k], true);
  check(cudaPeekAtLastError());
}
void CudaEngine::rotate(uint64_t *out, const uint64_t *in, unsigned k, uint64_t g, cudaStream_t stream) {
  auto &s = *impl_;
  s.bind(k, stream);
  if (!out || !in || out == in || !(g & 1) || g >= 2 * s.n)
    throw std::invalid_argument("invalid rotation");
  if (g == 1) {
    check(cudaMemcpyAsync(out, in, size_t(2) * k * s.n * 8, cudaMemcpyDeviceToDevice, stream));
    return;
  }
  auto key = s.keys.find(g);
  if (key == s.keys.end())
    throw std::invalid_argument("rotation key missing");
  permute<<<blocks(size_t(2) * k * s.n), 256, 0, stream>>>(s.permutation->data, in, s.n, k, s.logN, g);
  check(cudaPeekAtLastError());
  s.switchKey(out, s.permutation->data + size_t(k) * s.n, k, key->second->data, stream);
  s.point(out, out, s.permutation->data, k, stream, 6);
}
void CudaEngine::multiplyPlaintext(uint64_t *out, const uint64_t *in, const uint64_t *plain, unsigned k,
                                   cudaStream_t stream, bool normalize) {
  auto &s = *impl_;
  s.bind(k, stream);
  if (!out || !in || !plain || out == in || out == plain || (normalize && k < 2))
    throw std::invalid_argument("invalid plaintext multiply");
  s.point(normalize ? s.product->data : out, in, plain, k, stream, 4);
  if (normalize)
    s.drop(out, s.product->data, k, stream);
}
void CudaEngine::multiply(uint64_t *out, const uint64_t *a, const uint64_t *b, unsigned k, cudaStream_t stream,
                          bool normalize) {
  auto &s = *impl_;
  s.bind(k, stream);
  if (!out || !a || !b || out == a || out == b || (normalize && k < 2) || !s.relin)
    throw std::invalid_argument("invalid multiply or missing relinearization key");
  tensor<<<blocks(size_t(k) * s.n), 256, 0, stream>>>(s.product->data, a, b, s.n, k, s.tables[k]);
  check(cudaPeekAtLastError());
  s.switchKey(s.switched->data, s.product->data + size_t(2) * k * s.n, k, s.relin->data, stream);
  s.point(normalize ? s.switched->data : out, s.switched->data, s.product->data, k, stream, 5);
  if (normalize)
    s.drop(out, s.switched->data, k, stream);
}
void CudaEngine::rescale(uint64_t *out, const uint64_t *in, unsigned k, cudaStream_t stream) {
  auto &s = *impl_;
  s.bind(k, stream);
  if (k < 2 || !out || !in || out == in)
    throw std::invalid_argument("invalid rescale");
  s.drop(out, in, k, stream);
}
void CudaEngine::levelDown(uint64_t *out, const uint64_t *in, unsigned k, unsigned targetK, cudaStream_t stream) {
  auto &s = *impl_;
  s.bind(k, stream);
  if (!targetK || targetK > k || !out || !in || out == in)
    throw std::invalid_argument("invalid level down");
  check(cudaMemcpy2DAsync(out, size_t(targetK) * s.n * 8, in, size_t(k) * s.n * 8, size_t(targetK) * s.n * 8, 2,
                          cudaMemcpyDeviceToDevice, stream));
}
void CudaEngine::centeredLift(uint64_t *out, const uint64_t *in, unsigned targetK, cudaStream_t stream) {
  auto &s = *impl_;
  s.bind(targetK, stream);
  if (!out || !in || out == in)
    throw std::invalid_argument("invalid centered lift");
  check(cudaMemcpyAsync(s.removed->data, in, size_t(2) * s.n * 8, cudaMemcpyDeviceToDevice, stream));
  check(launchNtt(s.removed->data, s.tables[1], 1, 2, s.logN, true, stream));
  centeredOne<<<blocks(size_t(2) * targetK * s.n), 256, 0, stream>>>(out, s.removed->data, s.n, targetK, s.primes[0],
                                                                     s.tables[targetK]);
  check(cudaPeekAtLastError());
  check(launchNtt(out, s.tables[targetK], targetK, 2, s.logN, false, stream));
}
} // namespace fhn_cuda
