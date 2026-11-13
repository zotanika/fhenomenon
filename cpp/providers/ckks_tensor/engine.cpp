#include "engine.hpp"
#include "CKKS/Arena.h"
#include "CKKS/Encoding.h"
#include "CKKS/Eval.h"
#include "CKKS/Keys.h"
#include "CKKS/ModArith.h"
#include "CKKS/Params.h"
#ifdef FHN_NATIVE_CUDA
#include "cuda/CudaEngine.h"
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <utility>

namespace fhenomenon::tensor_ckks {
namespace ck = fhenomenon::ckks;
using Complex = std::complex<double>;
namespace {
void require(bool ok, const char *message) {
  if (!ok)
    throw std::invalid_argument(message);
}
void success(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
bool prime(std::uint64_t n) {
  for (auto a : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL, 29ULL, 31ULL, 37ULL}) {
    if (n % a == 0)
      return n == a;
    auto d = n - 1;
    unsigned r = 0;
    while (!(d & 1)) {
      d >>= 1;
      ++r;
    }
    auto x = ck::modarith::powMod(a, d, n);
    if (x == 1 || x == n - 1)
      continue;
    bool passed = false;
    for (unsigned i = 1; i < r; ++i) {
      x = ck::modarith::mulMod(x, x, n);
      if (x == n - 1) {
        passed = true;
        break;
      }
    }
    if (!passed)
      return false;
  }
  return true;
}
std::uint64_t nextPrime(unsigned bits, unsigned logn, std::set<std::uint64_t> &used) {
  require(bits > logn + 2 && bits <= 62, "invalid prime width");
  const std::uint64_t stride = std::uint64_t{2} << logn;
  auto q = (std::uint64_t{1} << bits) - stride + 1;
  for (; q > (std::uint64_t{1} << (bits - 1)); q -= stride)
    if (!used.count(q) && prime(q)) {
      used.insert(q);
      return q;
    }
  throw std::runtime_error("no unused NTT prime in requested interval");
}
ck::Params parameters(const Config &c) {
  require(c.levels > 1 && c.levels <= 64 && c.evaluation_top < c.levels, "invalid native level budget");
  require(c.log_degree >= 10 && c.log_degree <= 17, "invalid native ring degree");
  require(c.logical_slots && !(c.logical_slots & (c.logical_slots - 1)) &&
            c.logical_slots <= (1U << (c.log_degree - 1)),
          "invalid native packing period");
  require(c.digit_width && c.digit_width <= 4, "digit width must be in [1,4]");
  require(c.log_scale >= 30 && c.log_scale <= 54 && c.bottom_bits >= c.log_scale && c.bottom_bits <= 62 &&
            c.bootstrap_prime_bits >= c.log_scale && c.bootstrap_prime_bits <= 62,
          "invalid native scale/bottom modulus");
  ck::Params::Spec spec;
  spec.log_degree = c.log_degree;
  spec.log_default_scale = c.log_scale;
  spec.dnum = (c.levels + c.digit_width - 1) / c.digit_width;
  std::set<std::uint64_t> used;
  spec.main_primes.push_back(nextPrime(c.bottom_bits, c.log_degree, used));
  for (unsigned i = 1; i < c.levels; ++i)
    spec.main_primes.push_back(
      nextPrime(i <= c.evaluation_top ? c.log_scale : c.bootstrap_prime_bits, c.log_degree, used));
  const unsigned alpha = (c.levels + spec.dnum - 1) / spec.dnum;
  const unsigned max_digit_bits = std::max(c.bottom_bits + (alpha - 1) * c.log_scale, alpha * c.bootstrap_prime_bits);
  const unsigned aux_count = (max_digit_bits + 54) / 54;
  for (unsigned i = 0; i < aux_count; ++i)
    spec.aux_primes.push_back(nextPrime(54, c.log_degree, used));
  // Conservative standard 128-bit classical modulus-budget policy for dense
  // ternary RLWE; not an independent security certification or noise proof.
  static constexpr unsigned limits[] = {27, 54, 109, 218, 438, 881, 1761, 3529};
  double bits = 0;
  for (auto q : spec.main_primes)
    bits += std::log2(static_cast<double>(q));
  for (auto q : spec.aux_primes)
    bits += std::log2(static_cast<double>(q));
  require(c.test_only || bits <= limits[c.log_degree - 10],
          "native profile exceeds 128-bit modulus-budget policy; test-only opt-in required");
  std::string error;
  auto result = ck::Params::create(std::move(spec), &error);
  if (!result)
    throw std::invalid_argument(error);
  return std::move(*result);
}
unsigned padded(unsigned n) {
  unsigned p = 1;
  while (p < n)
    p *= 2;
  return p;
}
} // namespace

struct Cipher {
  std::shared_ptr<void> owner;
  std::shared_ptr<ck::PolySlab> host;
#ifdef FHN_NATIVE_CUDA
  std::shared_ptr<fhn_cuda::DeviceWords> gpu;
#endif
  ck::CiphertextLayout layout;
  unsigned logical_period = 0, logical_count = 0;
  std::string representation = "slots";
  ck::CiphertextView view() const {
    require(bool(host), "host ciphertext view is unavailable for resident CUDA value");
    return {host->data(), host->capacityWords(), layout.poly.wordCount(), layout};
  }
};

struct Engine::Impl {
  struct Plaintext {
    std::shared_ptr<ck::PolySlab> host;
#ifdef FHN_NATIVE_CUDA
    std::shared_ptr<fhn_cuda::DeviceWords> gpu;
#endif
    std::size_t bytes = 0;
    std::list<std::string>::iterator recency;
  };
  Config config;
  mutable std::recursive_mutex mutex;
  std::unique_ptr<ck::EvaluationTables> tables;
  std::unique_ptr<ck::Arena> workspace;
  std::shared_ptr<const ck::EvaluationKey> relin;
  std::map<std::uint64_t, std::shared_ptr<const ck::EvaluationKey>> rotations;
  std::vector<double> scales;
  bool keys_ready = false, cuda = false, ciphertext_created = false;
  std::uint64_t calls = 0;
  std::size_t keys_bytes = 0;
  std::map<std::string, std::shared_ptr<Plaintext>> plaintexts;
  std::list<std::string> recency;
  std::size_t plaintext_bytes = 0;
  std::uint64_t plaintext_hits = 0;
  static constexpr std::size_t plaintext_limit = 256ULL * 1024 * 1024;
#ifdef FHN_NATIVE_CUDA
  std::unique_ptr<fhn_cuda::CudaEngine> device;
  cudaStream_t stream = nullptr;
#endif
  explicit Impl(Config c) : config(c), tables(ck::EvaluationTables::create(parameters(c))) {
    success(bool(tables), "native tables construction failed");
    auto &p = tables->params();
    std::size_t bytes = std::max(ck::keyScratchBytes(p), ck::encoding::scratchBytes(p));
    for (auto op :
         {ck::eval::EvalOperation::Add, ck::eval::EvalOperation::Subtract, ck::eval::EvalOperation::Negate,
          ck::eval::EvalOperation::HMultiply, ck::eval::EvalOperation::Rotate, ck::eval::EvalOperation::Rescale,
          ck::eval::EvalOperation::AddPlaintext, ck::eval::EvalOperation::LevelDown})
      for (unsigned l = 0; l < c.levels; ++l)
        bytes = std::max(bytes, ck::eval::scratchBytes(op, p, l));
    workspace = std::make_unique<ck::Arena>(bytes);
    scales.push_back(std::ldexp(1., c.log_scale));
    for (unsigned l = 1; l < c.levels; ++l)
      scales.push_back(std::sqrt(static_cast<double>(p.mainPrimes()[l]) * scales.back()));
  }
  ~Impl() {
#ifdef FHN_NATIVE_CUDA
    if (stream) {
      // Cached plaintext allocations can outlive the explicit device engine.
      // Destroy every GPU owner before restoring the caller's selected device.
      try {
        fhn_cuda::ScopedDevice selected(0);
        cudaStreamSynchronize(stream);
        device.reset();
        plaintexts.clear();
        cudaStreamDestroy(stream);
      } catch (...) {
      } // Destructors cannot report an already-failed CUDA runtime.
    }
#endif
  }
  void check(const Value &a) const {
    require(a && a->owner.get() == this, "ciphertext belongs to a different native key context");
    require(a->logical_period && a->logical_count, "invalid ciphertext shape");
  }
  void pair(const Value &a, const Value &b) const {
    check(a);
    check(b);
    require(a->logical_period == b->logical_period && a->logical_count == b->logical_count,
            "ciphertext shape mismatch");
    require(a->representation == b->representation, "ciphertext message representation mismatch");
  }
  void sync() {
#ifdef FHN_NATIVE_CUDA
    if (cuda)
      fhn_cuda::check(cudaStreamSynchronize(stream));
#endif
  }
  Value value(const std::shared_ptr<Impl> &self, unsigned level, double scale, unsigned period, unsigned count) {
    require(level < config.levels && std::isfinite(scale) && scale > 0, "invalid native ciphertext level/scale");
    ciphertext_created = true;
    auto out = std::make_shared<Cipher>();
    out->owner = self;
    out->logical_period = period;
    out->logical_count = count;
    out->layout.poly = {{tables->paramsId(), 0, level + 1, 0, 0}, config.log_degree, ck::NttForm::Evaluation};
    out->layout.num_polys = 2;
    out->layout.scale = scale;
#ifdef FHN_NATIVE_CUDA
    if (cuda)
      out->gpu = std::make_shared<fhn_cuda::DeviceWords>(out->layout.wordCount());
    else
#endif
      out->host = std::make_shared<ck::PolySlab>(out->layout.wordCount());
    return out;
  }
  Value copy(const std::shared_ptr<Impl> &self, const Value &a) {
    check(a);
    auto out = value(self, a->layout.poly.basis.main_end - 1, a->layout.scale, a->logical_period, a->logical_count);
    out->representation = a->representation;
#ifdef FHN_NATIVE_CUDA
    if (cuda) {
      fhn_cuda::check(
        cudaMemcpyAsync(out->gpu->data, a->gpu->data, a->layout.wordCount() * 8, cudaMemcpyDeviceToDevice, stream));
      sync();
      return out;
    }
#endif
    std::copy_n(a->host->data(), a->layout.wordCount(), out->host->data());
    return out;
  }
  Value drop(const std::shared_ptr<Impl> &self, const Value &a, unsigned target) {
    check(a);
    auto l = a->layout.poly.basis.main_end - 1;
    require(target <= l, "cannot level-down upward");
    if (target == l)
      return a;
    auto out = value(self, target, a->layout.scale, a->logical_period, a->logical_count);
    out->representation = a->representation;
#ifdef FHN_NATIVE_CUDA
    if (cuda) {
      device->levelDown(out->gpu->data, a->gpu->data, l + 1, target + 1, stream);
      sync();
      return out;
    }
#endif
    auto v = out->view();
    workspace->reset();
    success(ck::eval::levelDown(v, ck::asConst(a->view()), target, *tables, *workspace), "native level-down failed");
    out->layout = v.layout;
    return out;
  }
  std::shared_ptr<Plaintext> encoded(const Value &a, const std::vector<Complex> &values, double pscale) {
    // Only explicitly public constants enter this cache. Ciphertext contents
    // and client plaintext inputs are never keys or cached payloads.
    std::string key;
    const unsigned limbs = a->layout.poly.basis.main_end;
    const std::size_t count = values.size();
    auto append = [&](const auto &v) { key.append(reinterpret_cast<const char *>(&v), sizeof(v)); };
    append(limbs);
    append(pscale);
    append(count);
    for (auto z : values) {
      const double real = z.real(), imag = z.imag();
      append(real);
      append(imag);
    }
    auto found = plaintexts.find(key);
    if (found != plaintexts.end()) {
      ++plaintext_hits;
      recency.splice(recency.end(), recency, found->second->recency);
      return found->second;
    }
    auto entry = std::make_shared<Plaintext>();
    entry->host = std::make_shared<ck::PolySlab>(a->layout.poly.wordCount());
    ck::RnsPolyView pv{entry->host->data(), entry->host->capacityWords(), a->layout.poly};
    workspace->reset();
    success(ck::encoding::encode(pv, values.data(), values.size(), pscale, *tables, *workspace),
            "native public plaintext encoding failed");
    entry->bytes = entry->host->capacityWords() * 8;
#ifdef FHN_NATIVE_CUDA
    if (cuda) {
      entry->gpu = std::make_shared<fhn_cuda::DeviceWords>(entry->host->capacityWords());
      entry->gpu->upload(entry->host->data(), entry->host->capacityWords());
      entry->bytes *= 2;
    }
#endif
    if (entry->bytes <= plaintext_limit) {
      while (plaintext_bytes + entry->bytes > plaintext_limit && !recency.empty()) {
        auto old = plaintexts.find(recency.front());
        plaintext_bytes -= old->second->bytes;
        plaintexts.erase(old);
        recency.pop_front();
      }
      recency.push_back(key);
      entry->recency = std::prev(recency.end());
      plaintexts.emplace(std::move(key), entry);
      plaintext_bytes += entry->bytes;
    }
    return entry;
  }
  Value plain(const std::shared_ptr<Impl> &self, const Value &a, const std::vector<Complex> &values, bool multiply,
              double output_scale) {
    check(a);
    require(!values.empty() && values.size() <= a->logical_period && !(values.size() & (values.size() - 1)),
            "public plaintext must have a power-of-two period");
    for (auto v : values)
      require(std::isfinite(v.real()) && std::isfinite(v.imag()), "nonfinite public plaintext");
    const unsigned l = a->layout.poly.basis.main_end - 1;
    require(!multiply || l > 0, "native multiplication exhausted its level budget");
    const double pscale = multiply
                            ? static_cast<double>(tables->params().mainPrimes()[l]) * output_scale / a->layout.scale
                            : a->layout.scale;
    auto plain = encoded(a, values, pscale);
    ck::RnsPolyView pv{plain->host->data(), plain->host->capacityWords(), a->layout.poly};
    auto out =
      value(self, multiply ? l - 1 : l, multiply ? output_scale : a->layout.scale, a->logical_period, a->logical_count);
    out->representation = a->representation;
#ifdef FHN_NATIVE_CUDA
    if (cuda) {
      if (multiply)
        device->multiplyPlaintext(out->gpu->data, a->gpu->data, plain->gpu->data, l + 1, stream);
      else
        device->addPlaintext(out->gpu->data, a->gpu->data, plain->gpu->data, l + 1, stream);
      sync();
      ++calls;
      return out;
    }
#endif
    auto v = out->view();
    workspace->reset();
    if (!multiply)
      success(ck::eval::addPlaintext(v, ck::asConst(a->view()), ck::asConst(pv), pscale, *tables, *workspace),
              "native plaintext addition failed");
    else {
      auto raw = value(self, l, a->layout.scale * pscale, a->logical_period, a->logical_count);
      const auto n = tables->params().degree();
      for (unsigned p = 0; p < 2; ++p)
        for (unsigned j = 0; j <= l; ++j) {
          auto q = tables->params().mainPrimes()[j];
          auto *dst = raw->view().poly(p).limb(j);
          const auto *src = a->view().poly(p).limb(j), *pt = pv.limb(j);
          for (std::size_t i = 0; i < n; ++i)
            dst[i] = ck::modarith::mulMod(src[i], pt[i], q);
        }
      success(ck::eval::rescale(v, ck::asConst(raw->view()), *tables, *workspace), "native plaintext rescale failed");
    }
    out->layout = v.layout;
    ++calls;
    return out;
  }
};

Engine::Engine(Config config) : impl_(std::make_shared<Impl>(config)) {}
Engine::~Engine() = default;
const Config &Engine::config() const { return impl_->config; }
unsigned Engine::physicalSlots() const { return impl_->tables->params().numSlots(); }
unsigned Engine::period(const Value &a) const {
  impl_->check(a);
  return a->logical_period;
}
unsigned Engine::count(const Value &a) const {
  impl_->check(a);
  return a->logical_count;
}
unsigned Engine::level(const Value &a) const {
  impl_->check(a);
  return a->layout.poly.basis.main_end - 1;
}
double Engine::scale(const Value &a) const {
  impl_->check(a);
  return a->layout.scale;
}
double Engine::valueScale(unsigned l) const { return impl_->scales.at(l); }
std::uint64_t Engine::evaluationModulus(unsigned l) const {
  require(l <= config().evaluation_top, "evaluation modulus exceeds public evaluation chain");
  return impl_->tables->params().mainPrimes().at(l);
}
std::uint64_t Engine::bottomModulus() const { return impl_->tables->params().mainPrimes()[0]; }
std::size_t Engine::bytes(const Value &a) const {
  impl_->check(a);
  return a->layout.wordCount() * 8;
}
std::string Engine::representation(const Value &a) const {
  impl_->check(a);
  return a->representation;
}
Value Engine::withRepresentation(const Value &a, const std::string &r) {
  std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
#ifdef FHN_NATIVE_CUDA
  fhn_cuda::ScopedDevice selected(impl_->cuda ? 0 : -1);
#endif
  impl_->check(a);
  require(r == "slots" || r == "coefficients", "unknown message representation");
  auto out = std::make_shared<Cipher>(*a);
  out->representation = r;
  return out;
}
Value Engine::broadcastTo(const Value &a, unsigned p, unsigned n) {
  std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
#ifdef FHN_NATIVE_CUDA
  fhn_cuda::ScopedDevice selected(impl_->cuda ? 0 : -1);
#endif
  impl_->check(a);
  require(p && !(p & (p - 1)) && p <= config().logical_slots && n && n <= p, "invalid scalar broadcast shape");
  require(a->representation == "slots" && a->logical_period == 1 && a->logical_count == 1,
          "only periodically encoded slot scalars can broadcast");
  auto out = std::make_shared<Cipher>(*a);
  out->logical_period = p;
  out->logical_count = n;
  return out;
}
Value Engine::cloneWithScale(const Value &a, double s) {
  std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
#ifdef FHN_NATIVE_CUDA
  fhn_cuda::ScopedDevice selected(impl_->cuda ? 0 : -1);
#endif
  impl_->check(a);
  require(std::isfinite(s) && s > 0, "invalid interpretation scale");
  auto out = std::make_shared<Cipher>(*a);
  out->layout.scale = s;
  return out;
}
Value Engine::plain(const Value &a, const std::vector<Complex> &v, bool multiply) {
  std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
#ifdef FHN_NATIVE_CUDA
  fhn_cuda::ScopedDevice selected(impl_->cuda ? 0 : -1);
#endif
  auto l = level(a);
  require(!multiply || l > 0, "native multiplication exhausted its level budget");
  return impl_->plain(impl_, a, v, multiply, multiply ? valueScale(l - 1) : scale(a));
}
Value Engine::levelDown(const Value &a, unsigned target) {
  std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
#ifdef FHN_NATIVE_CUDA
  fhn_cuda::ScopedDevice selected(impl_->cuda ? 0 : -1);
#endif
  const auto l = level(a);
  require(target <= l, "cannot level-down upward");
  if (target == l)
    return a;
  auto normalized = impl_->plain(impl_, a, {Complex(1, 0)}, true, valueScale(target));
  return impl_->drop(impl_, normalized, target);
}
Value Engine::add(const Value &left, const Value &right) {
  std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
#ifdef FHN_NATIVE_CUDA
  fhn_cuda::ScopedDevice selected(impl_->cuda ? 0 : -1);
#endif
  if (period(left) == 1 && period(right) > 1)
    return add(broadcastTo(left, period(right), count(right)), right);
  if (period(right) == 1 && period(left) > 1)
    return add(left, broadcastTo(right, period(left), count(left)));
  impl_->pair(left, right);
  auto l = std::min(level(left), level(right));
  auto a = levelDown(left, l), b = levelDown(right, l);
  require(std::abs(scale(a) - scale(b)) <= 1e-12 * std::max(scale(a), scale(b)),
          "add requires arithmetically aligned scales");
  auto out = impl_->value(impl_, l, scale(a), period(a), count(a));
  out->representation = a->representation;
#ifdef FHN_NATIVE_CUDA
  if (impl_->cuda) {
    impl_->device->add(out->gpu->data, a->gpu->data, b->gpu->data, l + 1, impl_->stream);
    impl_->sync();
    ++impl_->calls;
    return out;
  }
#endif
  auto v = out->view();
  impl_->workspace->reset();
  success(ck::eval::add(v, ck::asConst(a->view()), ck::asConst(b->view()), *impl_->tables, *impl_->workspace),
          "native addition failed");
  out->layout = v.layout;
  ++impl_->calls;
  return out;
}
Value Engine::sub(const Value &a, const Value &b) {
  std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
#ifdef FHN_NATIVE_CUDA
  fhn_cuda::ScopedDevice selected(impl_->cuda ? 0 : -1);
#endif
  return add(a, neg(b));
}
Value Engine::mul(const Value &left, const Value &right) {
  std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
#ifdef FHN_NATIVE_CUDA
  fhn_cuda::ScopedDevice selected(impl_->cuda ? 0 : -1);
#endif
  if (period(left) == 1 && period(right) > 1)
    return mul(broadcastTo(left, period(right), count(right)), right);
  if (period(right) == 1 && period(left) > 1)
    return mul(left, broadcastTo(right, period(left), count(left)));
  impl_->pair(left, right);
  auto l = std::min(level(left), level(right));
  require(l > 0, "native multiplication exhausted its level budget");
  auto a = levelDown(left, l), b = levelDown(right, l);
  const double result_scale = scale(a) * scale(b) / static_cast<double>(impl_->tables->params().mainPrimes()[l]);
  auto out = impl_->value(impl_, l - 1, result_scale, period(a), count(a));
  out->representation = a->representation;
  require(bool(impl_->relin), "client has not prepared multiplication keys");
#ifdef FHN_NATIVE_CUDA
  if (impl_->cuda) {
    impl_->device->multiply(out->gpu->data, a->gpu->data, b->gpu->data, l + 1, impl_->stream);
    impl_->sync();
    ++impl_->calls;
    return out;
  }
#endif
  auto v = out->view();
  impl_->workspace->reset();
  success(ck::eval::hmult(v, ck::asConst(a->view()), ck::asConst(b->view()), impl_->relin->view(), *impl_->tables,
                          *impl_->workspace),
          "native ciphertext multiplication failed");
  out->layout = v.layout;
  ++impl_->calls;
  return out;
}
Value Engine::neg(const Value &a) {
  std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
#ifdef FHN_NATIVE_CUDA
  fhn_cuda::ScopedDevice selected(impl_->cuda ? 0 : -1);
#endif
  auto l = level(a);
  auto out = impl_->value(impl_, l, scale(a), period(a), count(a));
  out->representation = a->representation;
#ifdef FHN_NATIVE_CUDA
  if (impl_->cuda) {
    impl_->device->negate(out->gpu->data, a->gpu->data, l + 1, impl_->stream);
    impl_->sync();
    ++impl_->calls;
    return out;
  }
#endif
  auto v = out->view();
  impl_->workspace->reset();
  success(ck::eval::negate(v, ck::asConst(a->view()), *impl_->tables, *impl_->workspace), "native negation failed");
  out->layout = v.layout;
  ++impl_->calls;
  return out;
}
Value Engine::rotate(const Value &a, std::int64_t step) {
  std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
#ifdef FHN_NATIVE_CUDA
  fhn_cuda::ScopedDevice selected(impl_->cuda ? 0 : -1);
#endif
  impl_->check(a);
  const auto slots = static_cast<std::int64_t>(physicalSlots());
  step %= slots;
  if (step < 0)
    step += slots;
  if (!step)
    return a;
  auto out = a;
  for (std::int64_t bit = 1; step; bit *= 2, step >>= 1)
    if (step & 1) {
      const auto g = ck::eval::galoisElement(bit, config().log_degree);
      auto key = impl_->rotations.find(g);
      require(key != impl_->rotations.end(), "missing prepared native rotation key");
      auto result = impl_->value(impl_, level(out), scale(out), period(out), count(out));
      result->representation = out->representation;
#ifdef FHN_NATIVE_CUDA
      if (impl_->cuda) {
        impl_->device->rotate(result->gpu->data, out->gpu->data, level(out) + 1, g, impl_->stream);
        impl_->sync();
      } else
#endif
      {
        auto v = result->view();
        impl_->workspace->reset();
        success(
          ck::eval::rotate(v, ck::asConst(out->view()), g, key->second->view(), *impl_->tables, *impl_->workspace),
          "native rotation failed");
        result->layout = v.layout;
      }
      out = result;
      ++impl_->calls;
    }
  return out;
}
Value Engine::conjugate(const Value &a) {
  std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
#ifdef FHN_NATIVE_CUDA
  fhn_cuda::ScopedDevice selected(impl_->cuda ? 0 : -1);
#endif
  auto l = level(a);
  const auto g = (std::uint64_t{2} << config().log_degree) - 1;
  auto key = impl_->rotations.find(g);
  require(key != impl_->rotations.end(), "missing native conjugation key");
  auto out = impl_->value(impl_, l, scale(a), period(a), count(a));
  out->representation = a->representation;
#ifdef FHN_NATIVE_CUDA
  if (impl_->cuda) {
    impl_->device->rotate(out->gpu->data, a->gpu->data, l + 1, g, impl_->stream);
    impl_->sync();
    ++impl_->calls;
    return out;
  }
#endif
  auto v = out->view();
  impl_->workspace->reset();
  success(ck::eval::rotate(v, ck::asConst(a->view()), g, key->second->view(), *impl_->tables, *impl_->workspace),
          "native conjugation failed");
  out->layout = v.layout;
  ++impl_->calls;
  return out;
}
Value Engine::multiplyInteger(const Value &a, std::int64_t factor) {
  std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
#ifdef FHN_NATIVE_CUDA
  fhn_cuda::ScopedDevice selected(impl_->cuda ? 0 : -1);
#endif
  auto out = impl_->value(impl_, level(a), scale(a), period(a), count(a));
  out->representation = a->representation;
#ifdef FHN_NATIVE_CUDA
  if (impl_->cuda) {
    impl_->device->multiplyScalarResidue(out->gpu->data, a->gpu->data, level(a) + 1, factor, impl_->stream);
    impl_->sync();
    ++impl_->calls;
    return out;
  }
#endif
  auto n = impl_->tables->params().degree();
  for (unsigned p = 0; p < 2; ++p)
    for (unsigned l = 0; l <= level(a); ++l) {
      auto q = impl_->tables->params().mainPrimes()[l];
      const auto abs =
        factor < 0 ? std::uint64_t{0} - static_cast<std::uint64_t>(factor) : static_cast<std::uint64_t>(factor);
      auto f = abs % q;
      if (factor < 0 && f)
        f = q - f;
      auto *dst = out->view().poly(p).limb(l);
      auto *src = a->view().poly(p).limb(l);
      for (std::size_t i = 0; i < n; ++i)
        dst[i] = ck::modarith::mulMod(src[i], f, q);
    }
  ++impl_->calls;
  return out;
}
Value Engine::multiplyI(const Value &a) {
  std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
#ifdef FHN_NATIVE_CUDA
  fhn_cuda::ScopedDevice selected(impl_->cuda ? 0 : -1);
#endif
  // Canonical exponents5^j are1 mod4; X^(N/2) evaluates to i in every slot.
  auto out = impl_->value(impl_, level(a), scale(a), period(a), count(a));
  out->representation = a->representation;
  ck::PolySlab plain(a->layout.poly.wordCount());
  ck::RnsPolyView pv{plain.data(), plain.capacityWords(), a->layout.poly};
  impl_->workspace->reset();
  Complex value(0, 1);
  success(ck::encoding::encode(pv, &value, 1, 1., *impl_->tables, *impl_->workspace), "native i encoding failed");
#ifdef FHN_NATIVE_CUDA
  if (impl_->cuda) {
    fhn_cuda::DeviceWords dp(plain.capacityWords());
    dp.upload(plain.data(), plain.capacityWords());
    impl_->device->multiplyPlaintext(out->gpu->data, a->gpu->data, dp.data, level(a) + 1, impl_->stream, false);
    impl_->sync();
    ++impl_->calls;
    return out;
  }
#endif
  auto n = impl_->tables->params().degree();
  for (unsigned p = 0; p < 2; ++p)
    for (unsigned l = 0; l <= level(a); ++l) {
      auto q = impl_->tables->params().mainPrimes()[l];
      auto *dst = out->view().poly(p).limb(l);
      auto *src = a->view().poly(p).limb(l);
      auto *pt = pv.limb(l);
      for (std::size_t i = 0; i < n; ++i)
        dst[i] = ck::modarith::mulMod(src[i], pt[i], q);
    }
  ++impl_->calls;
  return out;
}
Value Engine::liftBottom(const Value &input, unsigned top) {
  std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
#ifdef FHN_NATIVE_CUDA
  fhn_cuda::ScopedDevice selected(impl_->cuda ? 0 : -1);
#endif
  impl_->check(input);
  require(top < config().levels, "lift target exceeds prepared chain");
  auto a = impl_->drop(impl_, input, 0);
  auto out = impl_->value(impl_, top, scale(a), period(a), count(a));
  out->representation = a->representation;
#ifdef FHN_NATIVE_CUDA
  if (impl_->cuda) {
    impl_->device->centeredLift(out->gpu->data, a->gpu->data, top + 1, impl_->stream);
    impl_->sync();
    ++impl_->calls;
    return out;
  }
#endif
  const auto n = impl_->tables->params().degree(), q0 = bottomModulus();
  std::vector<std::uint64_t> coeff(n);
  for (unsigned p = 0; p < 2; ++p) {
    std::copy_n(a->view().poly(p).limb(0), n, coeff.data());
    ck::ntt::inverse(coeff.data(), impl_->tables->mainNtt(0));
    for (unsigned l = 0; l <= top; ++l) {
      auto q = impl_->tables->params().mainPrimes()[l];
      auto *dst = out->view().poly(p).limb(l);
      for (std::size_t i = 0; i < n; ++i) {
        const bool negative = coeff[i] > q0 / 2;
        const auto mag = negative ? q0 - coeff[i] : coeff[i];
        const auto rem = mag % q;
        dst[i] = negative && rem ? q - rem : rem;
      }
      ck::ntt::forward(dst, impl_->tables->mainNtt(l));
    }
  }
  ++impl_->calls;
  return out;
}
void Engine::enableCuda() {
  std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
#ifdef FHN_NATIVE_CUDA
  fhn_cuda::ScopedDevice selected(0);
#endif
  require(impl_->keys_ready, "prepare client evaluation keys before selecting CUDA");
  require(!impl_->cuda, "CUDA already selected");
  require(!impl_->ciphertext_created, "select CUDA before creating any native ciphertext");
#ifdef FHN_NATIVE_CUDA
  std::map<std::uint64_t, ck::EvaluationKeyView> keys;
  for (auto &entry : impl_->rotations)
    keys.emplace(entry.first, entry.second->view());
  auto device = std::make_unique<fhn_cuda::CudaEngine>(*impl_->tables, keys, impl_->relin->view());
  fhn_cuda::check(cudaStreamCreateWithFlags(&impl_->stream, cudaStreamNonBlocking));
  impl_->device = std::move(device);
  impl_->cuda = true;
#else
  throw std::runtime_error("native CUDA support was not built; CPU fallback is forbidden");
#endif
}
bool Engine::isCuda() const { return impl_->cuda; }
void Engine::synchronize() {
  std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
#ifdef FHN_NATIVE_CUDA
  fhn_cuda::ScopedDevice selected(impl_->cuda ? 0 : -1);
#endif
  impl_->sync();
}
std::size_t Engine::workspaceBytes() const { return impl_->workspace->capacity(); }
std::size_t Engine::keyBytes() const {
  auto bytes = impl_->keys_bytes;
#ifdef FHN_NATIVE_CUDA
  if (impl_->cuda)
    bytes += impl_->device->residentBytes();
#endif
  return bytes;
}
std::uint64_t Engine::arithmeticCalls() const { return impl_->calls; }
std::size_t Engine::plaintextCacheBytes() const {
  std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
  return impl_->plaintext_bytes;
}
std::uint64_t Engine::plaintextCacheHits() const {
  std::lock_guard<std::recursive_mutex> guard(impl_->mutex);
  return impl_->plaintext_hits;
}

struct Client::Impl {
  std::shared_ptr<Engine::Impl> state;
  std::unique_ptr<ck::SecretKey> secret;
  std::unique_ptr<ck::PublicKey> public_key;
  std::uint64_t encrypts = 0, decrypts = 0;
  unsigned last_egress = 0;
  explicit Impl(std::shared_ptr<Engine::Impl> s) : state(std::move(s)) {}
};
Client::Client(Engine &e, const std::vector<std::int32_t> &requested) : impl_(std::make_unique<Impl>(e.impl_)) {
  auto &s = *impl_->state;
  std::lock_guard<std::recursive_mutex> guard(s.mutex);
  require(!s.keys_ready, "native key context already initialized");
  ck::SystemRandom random;
  s.workspace->reset();
  impl_->secret = ck::generateSecretKey(*s.tables, random, *s.workspace);
  success(bool(impl_->secret), "native secret key generation failed");
  impl_->public_key = ck::generatePublicKey(*impl_->secret, *s.tables, random, *s.workspace);
  success(bool(impl_->public_key), "native public key generation failed");
  s.relin = ck::generateRelinearizationKey(*impl_->secret, *s.tables, random, *s.workspace);
  success(bool(s.relin), "native relinearization key generation failed");
  std::set<std::uint64_t> elements;
  const auto n = s.tables->params().degree();
  for (std::uint64_t step = 1; step < n / 2; step *= 2)
    elements.insert(ck::eval::galoisElement(step, s.config.log_degree));
  for (auto step : requested)
    elements.insert(ck::eval::galoisElement(step, s.config.log_degree));
  elements.insert(2 * n - 1);
  elements.erase(1);
  for (auto g : elements) {
    auto key = ck::generateGaloisKey(*impl_->secret, g, *s.tables, random, *s.workspace);
    success(bool(key), "native rotation key generation failed");
    s.rotations.emplace(g, std::move(key));
  }
  auto account = [&](ck::EvaluationKeyView k) {
    for (unsigned i = 0; i < k.digit_count; ++i)
      s.keys_bytes += k.digits[i].layout.wordCount() * 8;
  };
  account(s.relin->view());
  for (auto &entry : s.rotations)
    account(entry.second->view());
  s.keys_ready = true;
}
Client::~Client() = default;
Value Client::encrypt(const std::vector<Complex> &values) {
  auto &s = *impl_->state;
  std::lock_guard<std::recursive_mutex> guard(s.mutex);
#ifdef FHN_NATIVE_CUDA
  fhn_cuda::ScopedDevice selected(s.cuda ? 0 : -1);
#endif
  require(!values.empty() && values.size() <= s.config.logical_slots, "input exceeds configured logical slots");
  const auto period = padded(values.size());
  std::vector<Complex> padded_values = values;
  padded_values.resize(period);
  const auto l = s.config.evaluation_top;
  auto out = s.value(impl_->state, l, s.scales[l], period, values.size());
  ck::PolySlab plain(out->layout.poly.wordCount());
  ck::RnsPolyView pv{plain.data(), plain.capacityWords(), out->layout.poly};
  s.workspace->reset();
  success(ck::encoding::encode(pv, padded_values.data(), period, out->layout.scale, *s.tables, *s.workspace),
          "native client encoding failed");
  ck::PolySlab host(out->layout.wordCount());
  ck::CiphertextView v{host.data(), host.capacityWords(), out->layout.poly.wordCount(), out->layout};
  ck::SystemRandom random;
  success(ck::encrypt(v, ck::asConst(pv), out->layout.scale, *impl_->public_key, random, *s.tables, *s.workspace),
          "native encryption failed");
#ifdef FHN_NATIVE_CUDA
  if (s.cuda)
    out->gpu->upload(host.data(), host.capacityWords());
  else
#endif
    std::copy_n(host.data(), host.capacityWords(), out->host->data());
  ++impl_->encrypts;
  return out;
}
std::vector<Complex> Client::decrypt(const Value &a, double public_bound) {
  auto &s = *impl_->state;
  std::lock_guard<std::recursive_mutex> guard(s.mutex);
#ifdef FHN_NATIVE_CUDA
  fhn_cuda::ScopedDevice selected(s.cuda ? 0 : -1);
#endif
  s.check(a);
  require(a->representation == "slots", "convert coefficient messages to slots before egress");
  require(std::isfinite(public_bound) && public_bound >= 1, "invalid public egress magnitude bound");
  // No observation of private magnitude. The caller's bound must fit the
  // centered one-prime encoding interval at the current scale.
  require(public_bound * a->layout.scale < static_cast<double>(s.tables->params().mainPrimes()[0]) / 4.,
          "public egress bound exceeds minimum-prime dynamic range");
  auto low = s.drop(impl_->state, a, 0);
  ck::PolySlab host(low->layout.wordCount());
#ifdef FHN_NATIVE_CUDA
  if (s.cuda) {
    s.sync();
    fhn_cuda::check(cudaMemcpy(host.data(), low->gpu->data, low->layout.wordCount() * 8, cudaMemcpyDeviceToHost));
  } else
#endif
    std::copy_n(low->host->data(), low->layout.wordCount(), host.data());
  ck::ConstCiphertextView v{host.data(), host.capacityWords(), low->layout.poly.wordCount(), low->layout};
  ck::PolySlab plain(low->layout.poly.wordCount());
  ck::RnsPolyView pv{plain.data(), plain.capacityWords(), low->layout.poly};
  s.workspace->reset();
  success(ck::decrypt(pv, v, *impl_->secret, *s.tables, *s.workspace), "native minimum-prime decryption failed");
  std::vector<Complex> result(a->logical_count);
  success(
    ck::encoding::decode(result.data(), result.size(), ck::asConst(pv), low->layout.scale, *s.tables, *s.workspace),
    "native terminal decode failed");
  ++impl_->decrypts;
  impl_->last_egress = 1;
  return result;
}
std::uint64_t Client::encryptCalls() const { return impl_->encrypts; }
std::uint64_t Client::decryptCalls() const { return impl_->decrypts; }
unsigned Client::lastEgressModuli() const { return impl_->last_egress; }
} // namespace fhenomenon::tensor_ckks
