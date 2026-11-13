// PRIVATE evaluator differential and supported-contract checks. No timing.
#include "CKKS/Encoding.h"
#include "CKKS/Eval.h"
#include "CKKS/Keys.h"
#include "CKKS/ModArith.h"
#include "CudaEngine.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fhenomenon::ckks;
using namespace fhn_cuda;
namespace {
std::size_t checks = 0, wordsCompared = 0, rejections = 0;
void require(bool value, const std::string &message) {
  if (!value)
    throw std::runtime_error(message);
}
void rejected(const std::string &name, const std::function<void()> &operation) {
  bool caught = false;
  try {
    operation();
  } catch (const std::invalid_argument &) {
    caught = true;
  }
  require(caught, name + ": expected invalid_argument rejection");
  ++checks;
  ++rejections;
}
uint64_t randomState = 0xc2b2ae3d27d4eb4fULL;
uint64_t randomWord() {
  randomState += 0x9e3779b97f4a7c15ULL;
  auto v = randomState;
  v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
  v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
  return v ^ (v >> 31);
}
uint64_t primeBelow(unsigned bits, unsigned logN, const std::vector<uint64_t> &used) {
  const auto step = uint64_t{1} << (logN + 1);
  auto p = ((uint64_t{1} << bits) - 1) / step * step + 1;
  if (p >= (uint64_t{1} << bits))
    p -= step;
  for (; p > step; p -= step)
    if (std::find(used.begin(), used.end(), p) == used.end() && modarith::isPrime(p))
      return p;
  throw std::runtime_error("no test prime");
}
Params profile(unsigned kind) {
  Params::Spec spec;
  spec.log_degree = 10;
  spec.dnum = kind == 0 ? 3 : 2;
  spec.log_default_scale = 40;
  std::vector<uint64_t> used;
  const unsigned qCount = kind == 0 ? 3 : 5, pCount = kind == 0 ? 1 : 4;
  for (unsigned i = 0; i < qCount; ++i) {
    used.push_back(primeBelow(kind == 0 ? (i == 0 ? 50 : 40) : 62, spec.log_degree, used));
    spec.main_primes.push_back(used.back());
  }
  for (unsigned i = 0; i < pCount; ++i) {
    used.push_back(primeBelow(kind == 0 ? 60 : 62, spec.log_degree, used));
    spec.aux_primes.push_back(used.back());
  }
  std::string error;
  const auto result = Params::create(spec, &error);
  require(bool(result), "test profile: " + error);
  return *result;
}
uint64_t pattern(uint64_t q, unsigned mode, std::size_t i) {
  if (mode == 0)
    return 0;
  if (mode == 1) {
    const std::array<uint64_t, 8> boundary{0, 1, q - 1, q - 2, q / 2, q / 2 + 1, q / 2 - 1, 2};
    return boundary[i % boundary.size()];
  }
  return randomWord() % q;
}
struct SyntheticKey {
  std::vector<PolySlab> storage;
  std::vector<ConstCiphertextView> digits;
  EvaluationKeyView view;
  explicit SyntheticKey(const EvaluationTables &t) {
    const auto &p = t.params();
    const auto k = p.levelCount();
    const auto n = p.degree();
    const unsigned aux = p.auxPrimes().size(), digitsCount = p.dnum();
    const auto stride = (k + aux) * n;
    storage.reserve(digitsCount);
    digits.reserve(digitsCount);
    for (unsigned d = 0; d < digitsCount; ++d) {
      storage.emplace_back(2 * stride);
      for (unsigned c = 0; c < 2; ++c)
        for (unsigned l = 0; l < k + aux; ++l) {
          const auto q = l < k ? p.mainPrimes()[l] : p.auxPrimes()[l - k];
          for (std::size_t i = 0; i < n; ++i)
            storage.back().data()[c * stride + l * n + i] = pattern(q, i % 3 ? 2 : 1, i + c + d);
        }
      digits.push_back({storage.back().data(),
                        2 * stride,
                        stride,
                        {{{t.paramsId(), 0, k, 0, aux}, p.logDegree(), NttForm::Evaluation}, 1.0, 2}});
    }
    view = {digits.data(), digitsCount, t.paramsId(), p.logDegree(), k, aux, SwitchKeyKind::Galois, 5};
  }
};
void wrongScaleRed() {
  auto tables = EvaluationTables::create(profile(1));
  require(bool(tables), "red test tables");
  SyntheticKey key(*tables);
  key.digits[0].layout.scale = 2.0;
  rejected("wrong-scale evaluation-key digit", [&] { CudaEngine invalid(*tables, {{5, key.view}}); });
}
struct Stream {
  cudaStream_t value = nullptr;
  Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
  ~Stream() {
    if (value) {
      cudaStreamSynchronize(value);
      cudaStreamDestroy(value);
    }
  }
};
// Declare after device resources: it drains work before those resources unwind.
struct Drain {
  cudaStream_t stream;
  ~Drain() { cudaStreamSynchronize(stream); }
};
struct HostCipher {
  PolySlab storage;
  CiphertextView view;
  HostCipher(const EvaluationTables &t, unsigned k, unsigned count = 2)
    : storage(count * k * t.params().degree()),
      view{storage.data(),
           storage.capacityWords(),
           k * t.params().degree(),
           {{{t.paramsId(), 0, k, 0, 0}, t.params().logDegree(), NttForm::Evaluation}, std::ldexp(1.0, 40), count}} {}
  void fill(const EvaluationTables &t, unsigned mode) {
    for (unsigned c = 0; c < view.layout.num_polys; ++c)
      for (unsigned l = 0; l < view.layout.poly.basis.main_end; ++l)
        for (std::size_t i = 0; i < t.params().degree(); ++i)
          view.poly(c).limb(l)[i] = pattern(t.params().mainPrimes()[l], mode, i + c + l);
  }
};
void equal(const std::string &name, const DeviceWords &actual, ConstCiphertextView expected,
           const EvaluationTables &tables, cudaStream_t stream) {
  const auto k = expected.layout.poly.basis.main_end;
  const auto n = tables.params().degree();
  const auto downloaded = actual.download(expected.layout.wordCount(), stream);
  for (unsigned c = 0; c < expected.layout.num_polys; ++c)
    for (unsigned l = 0; l < k; ++l)
      for (std::size_t i = 0; i < n; ++i) {
        const auto value = downloaded[(c * k + l) * n + i];
        if (value != expected.poly(c).limb(l)[i])
          throw std::runtime_error(name + ": exact word mismatch component=" + std::to_string(c) +
                                   " limb=" + std::to_string(l) + " index=" + std::to_string(i));
        require(value < tables.params().mainPrimes()[l], name + ": noncanonical output");
        ++wordsCompared;
      }
  ++checks;
}
void malformedKeys(const EvaluationTables &tables) {
  SyntheticKey key(tables);
  const auto original = key.digits[0];
  auto test = [&](const std::string &name) {
    rejected(name, [&] { CudaEngine invalid(tables, {{5, key.view}}); });
    key.digits[0] = original;
  };
  key.digits[0].layout.scale = 2.0;
  test("key digit scale");
  key.digits[0].layout.poly.log_degree -= 1;
  test("key digit log degree");
  key.digits[0].layout.poly.basis.aux_begin = 1;
  test("key digit auxiliary begin");
  key.digits[0].layout.poly.basis.params_id = ParamsId{42};
  test("key digit parameter identity");
  key.digits[0].layout.poly.form = NttForm::Coefficient;
  test("key digit coefficient form");
  key.digits[0].layout.num_polys = 1;
  test("key digit component count");
  auto bad = key.view;
  bad.digits = nullptr;
  rejected("null key digits", [&] { CudaEngine invalid(tables, {{5, bad}}); });
  bad = key.view;
  bad.kind = SwitchKeyKind::Relinearization;
  rejected("wrong key kind", [&] { CudaEngine invalid(tables, {{5, bad}}); });
  bad = key.view;
  bad.galois_element = 7;
  rejected("wrong key map binding", [&] { CudaEngine invalid(tables, {{5, bad}}); });
  key.storage[0].data()[0] = tables.params().mainPrimes()[0];
  rejected("noncanonical evaluation key", [&] { CudaEngine invalid(tables, {{5, key.view}}); });
}
void requestGuards(const EvaluationTables &tables) {
  Stream stream, other;
  CudaEngine gpu(tables, {});
  const auto n = tables.params().degree();
  const auto k = tables.params().levelCount();
  std::vector<uint64_t> zero(2 * k * n, 0), sentinel(2 * k * n, 0xdecafbad12345678ULL);
  DeviceWords input(zero.size()), out(zero.size()), plain(k * n);
  input.upload(zero.data(), zero.size());
  out.upload(sentinel.data(), sentinel.size());
  plain.upload(zero.data(), k * n);
  Drain drain{stream.value};
  auto reject = [&](const std::string &label, const std::function<void()> &op) {
    rejected(label, op);
    require(out.download(out.words, stream.value) == sentinel, label + ": output changed");
    require(input.download(input.words, stream.value) == zero, label + ": input changed");
  };
  reject("zero k rotate", [&] { gpu.rotate(out.data, input.data, 0, 1, stream.value); });
  reject("too-large k rotate", [&] { gpu.rotate(out.data, input.data, k + 1, 1, stream.value); });
  reject("default stream rotate", [&] { gpu.rotate(out.data, input.data, k, 1, nullptr); });
  reject("default stream add", [&] { gpu.add(out.data, input.data, input.data, k, nullptr); });
  reject("default stream bias", [&] { gpu.bias(out.data, input.data, k, 1, nullptr); });
  reject("default stream multiply", [&] { gpu.multiplyPlaintext(out.data, input.data, plain.data, k, nullptr); });
  reject("rotation key missing", [&] { gpu.rotate(out.data, input.data, k, 5, stream.value); });
  reject("rotate output aliases input", [&] { gpu.rotate(input.data, input.data, k, 1, stream.value); });
  reject("invalid galois even", [&] { gpu.rotate(out.data, input.data, k, 2, stream.value); });
  reject("invalid galois range", [&] { gpu.rotate(out.data, input.data, k, 2 * n + 1, stream.value); });
  reject("different stream", [&] { gpu.rotate(out.data, input.data, k, 1, other.value); });
  reject("multiply at bottom level", [&] { gpu.multiplyPlaintext(out.data, input.data, plain.data, 1, stream.value); });
  reject("multiply output aliases ciphertext",
         [&] { gpu.multiplyPlaintext(input.data, input.data, plain.data, k, stream.value); });
  reject("multiply output aliases plaintext",
         [&] { gpu.multiplyPlaintext(plain.data, input.data, plain.data, k, stream.value); });
  reject("zero k multiply", [&] { gpu.multiplyPlaintext(out.data, input.data, plain.data, 0, stream.value); });
  reject("zero k add", [&] { gpu.add(out.data, input.data, input.data, 0, stream.value); });
  reject("zero k bias", [&] { gpu.bias(out.data, input.data, 0, 1, stream.value); });
  reject("null rotate output", [&] { gpu.rotate(nullptr, input.data, k, 1, stream.value); });
  reject("missing relinearization key", [&] { gpu.multiply(out.data, input.data, input.data, k, stream.value); });
  reject("rescale at bottom level", [&] { gpu.rescale(out.data, input.data, 1, stream.value); });
  reject("rescale exact alias", [&] { gpu.rescale(input.data, input.data, k, stream.value); });
  reject("levelDown zero target", [&] { gpu.levelDown(out.data, input.data, k, 0, stream.value); });
  reject("levelDown excessive target", [&] { gpu.levelDown(out.data, input.data, k, k + 1, stream.value); });
  reject("levelDown exact alias", [&] { gpu.levelDown(input.data, input.data, k, k, stream.value); });
  reject("lift zero target", [&] { gpu.centeredLift(out.data, input.data, 0, stream.value); });
  reject("lift excessive target", [&] { gpu.centeredLift(out.data, input.data, k + 1, stream.value); });
  reject("lift exact alias", [&] { gpu.centeredLift(input.data, input.data, k, stream.value); });
  reject("subtract null operand", [&] { gpu.subtract(out.data, input.data, nullptr, k, stream.value); });
  reject("negate null input", [&] { gpu.negate(out.data, nullptr, k, stream.value); });
  reject("plain add aliases plaintext", [&] { gpu.addPlaintext(plain.data, input.data, plain.data, k, stream.value); });
  reject("integer multiply null input", [&] { gpu.multiplyScalarResidue(out.data, nullptr, k, 7, stream.value); });
  // Identity needs no evaluation key, but still uses a disjoint output.
  gpu.rotate(out.data, input.data, k, 1, stream.value);
  require(out.download(out.words, stream.value) == zero, "keyless identity rotation");
  ++checks;
}
void exercise(unsigned kind) {
  auto tables = EvaluationTables::create(profile(kind));
  require(bool(tables), "table creation");
  const auto &params = tables->params();
  const unsigned fullK = params.levelCount();
  const auto n = params.degree();
  SyntheticKey key(*tables);
  const std::array<uint64_t, 4> rotations{1, eval::galoisElement(1, params.logDegree()),
                                          eval::galoisElement(-1, params.logDegree()), 2 * n - 1};
  std::map<uint64_t, EvaluationKeyView> keys;
  for (auto g : rotations)
    if (g != 1) {
      auto v = key.view;
      v.galois_element = g;
      keys.emplace(g, v);
    }
  Stream stream;
  auto relin = key.view;
  relin.kind = SwitchKeyKind::Relinearization;
  relin.galois_element = 1;
  CudaEngine gpu(*tables, keys, relin);
  for (unsigned k = 1; k <= fullK; ++k) {
    std::size_t scratch = 0;
    for (auto op : {eval::EvalOperation::Rotate, eval::EvalOperation::MultiplyPlaintext, eval::EvalOperation::Add,
                    eval::EvalOperation::HMultiply, eval::EvalOperation::Tensor, eval::EvalOperation::Relinearize,
                    eval::EvalOperation::Rescale, eval::EvalOperation::Subtract, eval::EvalOperation::Negate,
                    eval::EvalOperation::AddPlaintext, eval::EvalOperation::LevelDown})
      scratch = std::max(scratch, eval::scratchBytes(op, params, k - 1));
    Arena arena(scratch);
    for (unsigned mode = 0; mode < 3; ++mode) {
      const auto label =
        "profile=" + std::to_string(kind) + " activeQ=" + std::to_string(k) + " pattern=" + std::to_string(mode);
      HostCipher input(*tables, k), other(*tables, k), plain(*tables, k, 1), reference(*tables, k);
      input.fill(*tables, mode);
      other.fill(*tables, (mode + 1) % 3);
      plain.fill(*tables, (mode + 2) % 3);
      DeviceWords dInput(2 * k * n), dOther(2 * k * n), dPlain(k * n), dOut(2 * k * n);
      Drain drain{stream.value};
      auto reset = [&] {
        dInput.upload(input.storage.data(), dInput.words);
        dOther.upload(other.storage.data(), dOther.words);
        dPlain.upload(plain.storage.data(), dPlain.words);
      };
      reset();
      for (auto g : rotations) {
        const auto rotationKey = g == 1 ? EvaluationKeyView{} : keys.at(g);
        require(eval::rotate(reference.view, asConst(input.view), g, rotationKey, *tables, arena),
                label + " CPU rotate");
        gpu.rotate(dOut.data, dInput.data, k, g, stream.value);
        equal(label + " rotation=" + std::to_string(g), dOut, asConst(reference.view), *tables, stream.value);
      }
      if (k >= 2) {
        HostCipher product(*tables, k - 1);
        require(eval::multiplyPlaintext(product.view, asConst(input.view), asConst(plain.view.poly(0)),
                                        static_cast<double>(params.mainPrimes()[k - 1]), *tables, arena),
                label + " CPU multiply");
        gpu.multiplyPlaintext(dOut.data, dInput.data, dPlain.data, k, stream.value);
        equal(label + " multiplyPlaintext/rescale", dOut, asConst(product.view), *tables, stream.value);
      }
      reset();
      require(eval::subtract(reference.view, asConst(input.view), asConst(other.view), *tables, arena),
              label + " CPU subtract");
      gpu.subtract(dOut.data, dInput.data, dOther.data, k, stream.value);
      equal(label + " subtract", dOut, asConst(reference.view), *tables, stream.value);
      require(eval::negate(reference.view, asConst(input.view), *tables, arena), label + " CPU negate");
      gpu.negate(dOut.data, dInput.data, k, stream.value);
      equal(label + " negate", dOut, asConst(reference.view), *tables, stream.value);
      require(eval::addPlaintext(reference.view, asConst(input.view), asConst(plain.view.poly(0)),
                                 plain.view.layout.scale, *tables, arena),
              label + " CPU add plain");
      gpu.addPlaintext(dOut.data, dInput.data, dPlain.data, k, stream.value);
      equal(label + " addPlaintext", dOut, asConst(reference.view), *tables, stream.value);
      HostCipher tensorProduct(*tables, k, 3), rawProduct(*tables, k);
      require(eval::tensor(tensorProduct.view, asConst(input.view), asConst(other.view), *tables, arena),
              label + " CPU tensor");
      require(eval::relinearize(rawProduct.view, asConst(tensorProduct.view), relin, *tables, arena),
              label + " CPU relin");
      gpu.multiply(dOut.data, dInput.data, dOther.data, k, stream.value, false);
      equal(label + " raw multiply/relinearize", dOut, asConst(rawProduct.view), *tables, stream.value);
      if (k >= 2) {
        HostCipher scaled(*tables, k - 1);
        require(eval::hmult(scaled.view, asConst(input.view), asConst(other.view), relin, *tables, arena),
                label + " CPU hmult");
        gpu.multiply(dOut.data, dInput.data, dOther.data, k, stream.value);
        equal(label + " multiply/relinearize/rescale", dOut, asConst(scaled.view), *tables, stream.value);
        require(eval::rescale(scaled.view, asConst(input.view), *tables, arena), label + " CPU rescale");
        gpu.rescale(dOut.data, dInput.data, k, stream.value);
        equal(label + " rescale", dOut, asConst(scaled.view), *tables, stream.value);
      }
      for (unsigned target = 1; target <= k; ++target) {
        HostCipher lowered(*tables, target);
        require(eval::levelDown(lowered.view, asConst(input.view), target - 1, *tables, arena),
                label + " CPU level down");
        gpu.levelDown(dOut.data, dInput.data, k, target, stream.value);
        equal(label + " levelDown=" + std::to_string(target), dOut, asConst(lowered.view), *tables, stream.value);
      }
      if (k == 1) {
        HostCipher lifted(*tables, fullK);
        for (unsigned c = 0; c < 2; ++c) {
          std::vector<uint64_t> coefficients(input.view.poly(c).limb(0), input.view.poly(c).limb(0) + n);
          ntt::inverse(coefficients.data(), tables->mainNtt(0));
          for (unsigned l = 0; l < fullK; ++l) {
            const auto q = params.mainPrimes()[l], q0 = params.mainPrimes()[0];
            for (unsigned i = 0; i < n; ++i) {
              const auto value = coefficients[i];
              lifted.view.poly(c).limb(l)[i] = value > q0 / 2 ? (value % q + q - q0 % q) % q : value % q;
            }
            ntt::forward(lifted.view.poly(c).limb(l), tables->mainNtt(l));
          }
        }
        DeviceWords dLift(2 * fullK * n);
        gpu.centeredLift(dLift.data, dInput.data, fullK, stream.value);
        equal(label + " centeredLift", dLift, asConst(lifted.view), *tables, stream.value);
      }
      for (int64_t integer : {int64_t(0), int64_t(7), int64_t(-13), std::numeric_limits<int64_t>::min()}) {
        for (unsigned c = 0; c < 2; ++c)
          for (unsigned l = 0; l < k; ++l)
            for (unsigned i = 0; i < n; ++i) {
              const auto q = params.mainPrimes()[l],
                         magnitude = integer < 0 ? uint64_t(0) - uint64_t(integer) : uint64_t(integer);
              const auto residue = integer < 0 && magnitude % q ? q - magnitude % q : magnitude % q;
              reference.view.poly(c).limb(l)[i] = modarith::mulMod(input.view.poly(c).limb(l)[i], residue, q);
            }
        gpu.multiplyScalarResidue(dOut.data, dInput.data, k, integer, stream.value);
        equal(label + " integer multiply", dOut, asConst(reference.view), *tables, stream.value);
      }
      require(eval::add(reference.view, asConst(input.view), asConst(other.view), *tables, arena), label + " CPU add");
      for (unsigned alias = 0; alias < 3; ++alias) {
        reset();
        auto &destination = alias == 0 ? dOut : alias == 1 ? dInput : dOther;
        gpu.add(destination.data, dInput.data, dOther.data, k, stream.value);
        equal(label + " add alias=" + std::to_string(alias), destination, asConst(reference.view), *tables,
              stream.value);
      }
      const std::array<int64_t, 7> biases{
        0, 1, -1, 17, -23, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max() - 1023};
      for (auto encoded : biases) {
        const double scalar = std::ldexp(static_cast<double>(encoded), -40);
        require(eval::addScalar(reference.view, asConst(input.view), scalar, *tables, arena), label + " CPU bias");
        for (unsigned alias = 0; alias < 2; ++alias) {
          reset();
          auto &destination = alias ? dInput : dOut;
          gpu.bias(destination.data, dInput.data, k, encoded, stream.value);
          equal(label + " bias=" + std::to_string(encoded) + " alias=" + std::to_string(alias), destination,
                asConst(reference.view), *tables, stream.value);
        }
      }
    }
  }
  check(cudaStreamSynchronize(stream.value));
  malformedKeys(*tables);
  requestGuards(*tables);
  std::cout << "PASS profile " << kind << " N=" << n << " Q=" << fullK << " P=" << params.auxPrimes().size()
            << " dnum=" << params.dnum() << '\n';
}

void realEncrypted() {
  Params::Spec spec;
  spec.log_degree = 10;
  spec.dnum = 2;
  spec.log_default_scale = 40;
  std::vector<uint64_t> used;
  for (unsigned bits : {50u, 40u, 40u, 40u, 40u}) {
    used.push_back(primeBelow(bits, 10, used));
    spec.main_primes.push_back(used.back());
  }
  for (unsigned i = 0; i < 3; ++i) {
    used.push_back(primeBelow(60, 10, used));
    spec.aux_primes.push_back(used.back());
  }
  std::string error;
  auto params = Params::create(spec, &error);
  require(bool(params), "real params: " + error);
  auto tables = EvaluationTables::create(*params, &error);
  require(bool(tables), "real tables: " + error);
  const unsigned k = params->levelCount(), n = params->degree();
  size_t scratch = std::max(keyScratchBytes(*params), encoding::scratchBytes(*params));
  for (unsigned level = 0; level < k; ++level)
    for (auto op : {eval::EvalOperation::Add, eval::EvalOperation::Subtract, eval::EvalOperation::Negate,
                    eval::EvalOperation::Rotate, eval::EvalOperation::HMultiply, eval::EvalOperation::AddPlaintext,
                    eval::EvalOperation::MultiplyPlaintext})
      scratch = std::max(scratch, eval::scratchBytes(op, *params, level));
  Arena arena(scratch);
  SystemRandom randomness;
  auto secret = generateSecretKey(*tables, randomness, arena);
  require(bool(secret), "real secret generation");
  auto publicKey = generatePublicKey(*secret, *tables, randomness, arena);
  require(bool(publicKey), "real public generation");
  auto relin = generateRelinearizationKey(*secret, *tables, randomness, arena);
  require(bool(relin), "real relin generation");
  std::map<uint64_t, std::shared_ptr<const EvaluationKey>> keyOwners;
  std::map<uint64_t, EvaluationKeyView> keyViews;
  for (uint64_t g : {eval::galoisElement(1, 10), eval::galoisElement(-1, 10), uint64_t(2 * n - 1)}) {
    auto key = generateGaloisKey(*secret, g, *tables, randomness, arena);
    require(bool(key), "real galois generation");
    keyViews.emplace(g, key->view());
    keyOwners.emplace(g, std::move(key));
  }
  std::vector<std::complex<double>> a{{-.4, .05}, {.2, -.1}, {.9, .15},  {-.7, -.2},
                                      {.1, .25},  {.3, -.3}, {-.2, .35}, {.8, -.4}};
  std::vector<std::complex<double>> b{{.6, -.2}, {-.5, .15}, {.1, -.1}, {.3, .05},
                                      {-.8, .2}, {.4, -.15}, {.7, .1},  {-.2, -.05}};
  HostCipher plainA(*tables, k, 1), plainB(*tables, k, 1), encryptedA(*tables, k), encryptedB(*tables, k),
    reference(*tables, k);
  auto pa = plainA.view.poly(0), pb = plainB.view.poly(0);
  require(encoding::encode(pa, a.data(), a.size(), plainA.view.layout.scale, *tables, arena), "encode A");
  require(encoding::encode(pb, b.data(), b.size(), plainB.view.layout.scale, *tables, arena), "encode B");
  require(encrypt(encryptedA.view, asConst(pa), plainA.view.layout.scale, *publicKey, randomness, *tables, arena),
          "encrypt A");
  require(encrypt(encryptedB.view, asConst(pb), plainB.view.layout.scale, *publicKey, randomness, *tables, arena),
          "encrypt B");
  Stream stream;
  CudaEngine gpu(*tables, keyViews, relin->view());
  DeviceWords da(2 * k * n), db(2 * k * n), dp(k * n), out(2 * k * n);
  Drain drain{stream.value};
  da.upload(encryptedA.storage.data(), da.words);
  db.upload(encryptedB.storage.data(), db.words);
  dp.upload(plainB.storage.data(), dp.words);
  double maxError = 0;
  auto verify = [&](const std::string &name, const HostCipher &cpu, const std::vector<std::complex<double>> &expected) {
    equal("real encrypted " + name, out, asConst(cpu.view), *tables, stream.value);
    const unsigned active = cpu.view.layout.poly.basis.main_end;
    HostCipher downloaded(*tables, active), decrypted(*tables, active, 1);
    const auto words = out.download(2 * active * n, stream.value);
    std::copy(words.begin(), words.end(), downloaded.storage.data());
    downloaded.view.layout = cpu.view.layout;
    auto decodedPoly = decrypted.view.poly(0);
    require(decrypt(decodedPoly, asConst(downloaded.view), *secret, *tables, arena), name + " decrypt");
    std::vector<std::complex<double>> decoded(a.size());
    require(encoding::decode(decoded.data(), decoded.size(), asConst(decodedPoly), downloaded.view.layout.scale,
                             *tables, arena),
            name + " decode");
    for (unsigned i = 0; i < decoded.size(); ++i) {
      const double e = std::abs(decoded[i] - expected[i]);
      maxError = std::max(maxError, e);
      require(e < 1e-7, name + " decoded error=" + std::to_string(e));
    }
  };
  std::vector<std::complex<double>> expected(a.size());
  for (unsigned i = 0; i < a.size(); ++i)
    expected[i] = a[i] + b[i];
  require(eval::add(reference.view, asConst(encryptedA.view), asConst(encryptedB.view), *tables, arena),
          "real CPU add");
  gpu.add(out.data, da.data, db.data, k, stream.value);
  verify("add", reference, expected);
  for (unsigned i = 0; i < a.size(); ++i)
    expected[i] = a[i] - b[i];
  require(eval::subtract(reference.view, asConst(encryptedA.view), asConst(encryptedB.view), *tables, arena),
          "real CPU subtract");
  gpu.subtract(out.data, da.data, db.data, k, stream.value);
  verify("subtract", reference, expected);
  for (unsigned i = 0; i < a.size(); ++i)
    expected[i] = -a[i];
  require(eval::negate(reference.view, asConst(encryptedA.view), *tables, arena), "real CPU negate");
  gpu.negate(out.data, da.data, k, stream.value);
  verify("negate", reference, expected);
  for (int step : {1, -1}) {
    const auto g = eval::galoisElement(step, 10);
    for (unsigned i = 0; i < a.size(); ++i)
      expected[i] = a[(i + a.size() + step) % a.size()];
    require(eval::rotate(reference.view, asConst(encryptedA.view), g, keyViews.at(g), *tables, arena),
            "real CPU rotate");
    gpu.rotate(out.data, da.data, k, g, stream.value);
    verify("rotate=" + std::to_string(step), reference, expected);
  }
  for (unsigned i = 0; i < a.size(); ++i)
    expected[i] = std::conj(a[i]);
  require(eval::rotate(reference.view, asConst(encryptedA.view), 2 * n - 1, keyViews.at(2 * n - 1), *tables, arena),
          "real CPU conjugate");
  gpu.rotate(out.data, da.data, k, 2 * n - 1, stream.value);
  verify("conjugate", reference, expected);
  HostCipher product(*tables, k - 1);
  for (unsigned i = 0; i < a.size(); ++i)
    expected[i] = a[i] * b[i];
  require(eval::hmult(product.view, asConst(encryptedA.view), asConst(encryptedB.view), relin->view(), *tables, arena),
          "real CPU hmult");
  gpu.multiply(out.data, da.data, db.data, k, stream.value);
  verify("multiply", product, expected);
  for (unsigned i = 0; i < a.size(); ++i)
    expected[i] = a[i] + b[i];
  require(
    eval::addPlaintext(reference.view, asConst(encryptedA.view), asConst(pb), plainB.view.layout.scale, *tables, arena),
    "real CPU add plain");
  gpu.addPlaintext(out.data, da.data, dp.data, k, stream.value);
  verify("add plaintext", reference, expected);
  // Re-encode public multiplier at the exact dropped-prime scale.
  plainB.view.layout.scale = double(params->mainPrimes().back());
  require(encoding::encode(pb, b.data(), b.size(), plainB.view.layout.scale, *tables, arena), "encode normalized B");
  dp.upload(plainB.storage.data(), dp.words);
  for (unsigned i = 0; i < a.size(); ++i)
    expected[i] = a[i] * b[i];
  require(eval::multiplyPlaintext(product.view, asConst(encryptedA.view), asConst(pb), plainB.view.layout.scale,
                                  *tables, arena),
          "real CPU plain mult");
  gpu.multiplyPlaintext(out.data, da.data, dp.data, k, stream.value);
  verify("multiply plaintext", product, expected);
  // Exact Gaussian-unit plaintext: X^(N/2) evaluates to i in every
  // canonical slot, so this raw multiplication consumes no level or scale.
  HostCipher imaginaryUnit(*tables, k, 1);
  for (unsigned l = 0; l < k; ++l) {
    std::fill_n(imaginaryUnit.view.poly(0).limb(l), n, uint64_t(0));
    imaginaryUnit.view.poly(0).limb(l)[n / 2] = 1;
    ntt::forward(imaginaryUnit.view.poly(0).limb(l), tables->mainNtt(l));
    for (unsigned c = 0; c < 2; ++c)
      for (unsigned i = 0; i < n; ++i)
        reference.view.poly(c).limb(l)[i] = modarith::mulMod(
          encryptedA.view.poly(c).limb(l)[i], imaginaryUnit.view.poly(0).limb(l)[i], params->mainPrimes()[l]);
  }
  reference.view.layout = encryptedA.view.layout;
  dp.upload(imaginaryUnit.storage.data(), dp.words);
  gpu.multiplyPlaintext(out.data, da.data, dp.data, k, stream.value, false);
  for (unsigned i = 0; i < a.size(); ++i)
    expected[i] = a[i] * std::complex<double>(0, 1);
  verify("raw imaginary-unit multiply", reference, expected);
  // GPU-resident dependency chain: multiply, rotation and addition without a
  // download between operations; only the final output crosses the boundary.
  DeviceWords intermediate(2 * (k - 1) * n), rotated(2 * (k - 1) * n);
  gpu.multiply(intermediate.data, da.data, db.data, k, stream.value);
  const auto g = eval::galoisElement(1, 10);
  gpu.rotate(rotated.data, intermediate.data, k - 1, g, stream.value);
  gpu.add(out.data, intermediate.data, rotated.data, k - 1, stream.value);
  HostCipher cpuRot(*tables, k - 1), cpuChain(*tables, k - 1);
  require(eval::hmult(product.view, asConst(encryptedA.view), asConst(encryptedB.view), relin->view(), *tables, arena),
          "chain CPU hmult");
  require(eval::rotate(cpuRot.view, asConst(product.view), g, keyViews.at(g), *tables, arena), "chain CPU rotate");
  require(eval::add(cpuChain.view, asConst(product.view), asConst(cpuRot.view), *tables, arena), "chain CPU add");
  for (unsigned i = 0; i < a.size(); ++i)
    expected[i] = a[i] * b[i] + a[(i + 1) % a.size()] * b[(i + 1) % a.size()];
  verify("resident chain", cpuChain, expected);
  std::cout << "PASS real encrypted N=" << n << " Q=" << k << " P=3 dnum=2 maxDecodedError=" << maxError << '\n';
}
} // namespace
int main(int argc, char **argv) try {
  if (argc == 2 && std::string(argv[1]) == "--red-key-scale") {
    wrongScaleRed();
    std::cout << "PASS wrong-scale key rejected\n";
    return 0;
  }
  check(configureNtt());

  exercise(0);
  exercise(1);
  realEncrypted();
  std::cout << "{\"passed\":true,\"checks\":" << checks << ",\"rejections\":" << rejections
            << ",\"exact_canonical_words\":" << wordsCompared
            << ",\"profiles\":2,"
               "\"scope\":\"private supported-profile kernels; caller owns input canonicality, capacities and "
               "disjointness\"}\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "FAIL " << error.what() << '\n';
  return 1;
}
