#include "cuda/CudaEngine.h"
#include "engine.hpp"
#include <cmath>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>

using namespace fhenomenon::tensor_ckks;
using Complex = std::complex<double>;
namespace {
void require(bool yes, const char *message) {
  if (!yes)
    throw std::runtime_error(message);
}
int current() {
  int device = -1;
  fhn_cuda::check(cudaGetDevice(&device));
  return device;
}
template <class F> auto preserved(F operation) {
  const int before = current();
  if constexpr (std::is_void_v<decltype(operation())>) {
    operation();
    require(current() == before, "operation changed caller CUDA device");
  } else {
    auto value = operation();
    require(current() == before, "operation changed caller CUDA device");
    return value;
  }
}
void close(Client &client, const Value &value, const std::vector<Complex> &expected) {
  const auto actual = preserved([&] { return client.decrypt(value, 32.); });
  require(actual.size() == expected.size(), "decrypted shape");
  for (std::size_t i = 0; i < actual.size(); ++i)
    require(std::abs(actual[i] - expected[i]) < 2e-7, "cross-thread resident arithmetic");
  require(client.lastEgressModuli() == 1, "minimum-prime egress");
}
template <class F> void thread(F operation) {
  std::exception_ptr failure;
  std::thread worker([&] {
    try {
      operation();
    } catch (...) {
      failure = std::current_exception();
    }
  });
  worker.join();
  if (failure)
    std::rethrow_exception(failure);
}
} // namespace
int main() {
  try {
    int count = 0;
    fhn_cuda::check(cudaGetDeviceCount(&count));
    require(count > 0, "CUDA device required");
    const int original = current(), caller = count > 1 ? 1 : 0;
    fhn_cuda::check(cudaSetDevice(caller));
    // Direct allocation ownership remains valid when released on another thread
    // or with a different current device. The pointer itself must belong to 0.
    std::unique_ptr<fhn_cuda::DeviceWords> words;
    {
      fhn_cuda::ScopedDevice selected(0);
      require(current() == 0, "CUDA scope failed to select device 0");
      words = std::make_unique<fhn_cuda::DeviceWords>(4);
    }
    require(current() == caller, "CUDA scope did not restore caller device");
    cudaPointerAttributes attr{};
    fhn_cuda::check(cudaPointerGetAttributes(&attr, words->data));
    require(attr.device == 0, "allocation belongs to wrong CUDA device");
    const std::vector<std::uint64_t> input{1, 2, 3, 4};
    preserved([&] { words->upload(input.data(), input.size()); });
    require(preserved([&] { return words->download(input.size(), nullptr); }) == input, "owned allocation transfer");
    thread([&] {
      fhn_cuda::check(cudaSetDevice(caller));
      preserved([&] { words.reset(); });
    });
    try {
      fhn_cuda::ScopedDevice selected(0);
      throw std::runtime_error("test unwind");
    } catch (const std::runtime_error &) {
    }
    require(current() == caller, "CUDA scope did not restore on exception");

    Config cfg;
    cfg.log_degree = 10;
    cfg.levels = 9;
    cfg.evaluation_top = 8;
    cfg.logical_slots = 4;
    cfg.test_only = true;
    auto engine = std::make_unique<Engine>(cfg);
    auto client = std::make_unique<Client>(*engine);
    preserved([&] { engine->enableCuda(); });
    require(engine->isCuda(), "CUDA evaluator not selected");
    const std::vector<Complex> x{{.1, .03}, {-.2, 0}, {.35, -.04}, {.7, .02}};
    auto a = preserved([&] { return client->encrypt(x); });
    Value survivor;
    // Enabling on caller device 1 and entering on a fresh device-0 thread used
    // to submit work to a stream created on the wrong device. On one-GPU hosts
    // this still checks shared context lifetime and cross-thread serialization.
    thread([&] {
      fhn_cuda::check(cudaSetDevice(0));
      close(*client, a, x);
      fhn_cuda::check(cudaSetDevice(caller));
      auto b = preserved([&] { return client->encrypt({.5}); });
      std::vector<Complex> expected = x;
      for (auto &z : expected)
        z += .5;
      close(*client, preserved([&] { return engine->add(a, b); }), expected);
      expected = x;
      for (auto &z : expected)
        z -= .5;
      close(*client, preserved([&] { return engine->sub(a, b); }), expected);
      expected = x;
      for (auto &z : expected)
        z *= .5;
      close(*client, preserved([&] { return engine->mul(a, b); }), expected);
      expected = x;
      for (auto &z : expected)
        z = -z;
      close(*client, preserved([&] { return engine->neg(a); }), expected);
      close(*client, preserved([&] { return engine->rotate(a, 1); }), {x[1], x[2], x[3], x[0]});
      expected = x;
      for (auto &z : expected)
        z = std::conj(z);
      close(*client, preserved([&] { return engine->conjugate(a); }), expected);
      expected = x;
      for (auto &z : expected)
        z += .125;
      close(*client, preserved([&] { return engine->plain(a, {.125}, false); }), expected);
      expected = x;
      for (auto &z : expected)
        z *= .25;
      close(*client, preserved([&] { return engine->plain(a, {.25}, true); }), expected);
      expected = x;
      for (auto &z : expected)
        z *= -2.;
      close(*client, preserved([&] { return engine->multiplyInteger(a, -2); }), expected);
      expected = x;
      for (auto &z : expected)
        z *= Complex(0, 1);
      close(*client, preserved([&] { return engine->multiplyI(a); }), expected);
      close(*client, preserved([&] { return engine->levelDown(a, 3); }), x);
      auto lifted = preserved([&] { return engine->liftBottom(a, 8); });
      require(engine->level(lifted) == 8, "lift target level");
      close(*client, preserved([&] { return engine->cloneWithScale(a, engine->scale(a)); }), x);
      auto coefficients = preserved([&] { return engine->withRepresentation(a, "coefficients"); });
      require(engine->representation(coefficients) == "coefficients", "representation metadata");
      close(*client, preserved([&] { return engine->broadcastTo(b, 4, 4); }), {.5, .5, .5, .5});
      bool rejected = false;
      try {
        engine->levelDown(a, 99);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      require(rejected && current() == caller, "failed operation changed caller CUDA device");
      preserved([&] { engine->synchronize(); });
      survivor = preserved([&] { return engine->plain(a, {.125}, false); });
    });
    require(current() == caller, "worker changed main thread device");
    // Client and public Engine may be released before the final ciphertext;
    // that ciphertext retains and eventually destroys cached public GPU data.
    thread([&] {
      fhn_cuda::check(cudaSetDevice(caller));
      preserved([&] {
        a.reset();
        client.reset();
        engine.reset();
      });
    });
    preserved([&] { survivor.reset(); });
    fhn_cuda::check(cudaSetDevice(original));
    std::cout << "CUDA device ownership, arithmetic, exception restoration and cross-thread lifetime checks passed; "
              << (count > 1 ? "multi-GPU device switching exercised" : "multi-GPU branch unavailable (one CUDA device)")
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
