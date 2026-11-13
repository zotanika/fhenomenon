#include "Client.h"
#include <cmath>
#include <dlfcn.h>
#include <stdexcept>

namespace test_support {
using namespace fhenomenon::runtime;
namespace {
void check(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
} // namespace
struct Client::Impl {
  std::shared_ptr<void> library;
  NativeServices services;
  NativeRuntime native;
  std::unique_ptr<FhnClientCtx, FhnClientDestroyFn> client;
  FhnClientEncryptVectorDoubleFn encrypt;
  FhnClientDecryptVectorDoubleFn decrypt;
  std::uint32_t slots;
  template <class T> T symbol(const char *name) const {
    auto function = reinterpret_cast<T>(dlsym(library.get(), name));
    check(function != nullptr, name);
    return function;
  }
  Impl(Runtime &runtime, const std::string &path, std::uint32_t count)
    : library(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL),
              [](void *handle) {
                if (handle)
                  dlclose(handle);
              }),
      services(loadNativeServices(path)), native(runtime, services),
      client(nullptr, symbol<FhnClientDestroyFn>("ckks_fhn_client_destroy")),
      encrypt(symbol<FhnClientEncryptVectorDoubleFn>("ckks_fhn_client_encrypt_vec_f64")),
      decrypt(symbol<FhnClientDecryptVectorDoubleFn>("ckks_fhn_client_decrypt_vec_f64")), slots(count) {
    check(count && !(count & (count - 1)) && count <= 4096, "invalid native planner slot count");
    client.reset(symbol<FhnClientCreateFn>("ckks_fhn_client_create")(services.context.get()));
    check(bool(client), "planner client creation failed");
    std::vector<FhnInstruction> rotations;
    for (std::uint32_t step = 1; step < count; step *= 2) {
      FhnInstruction instruction{};
      instruction.opcode = FHN_ROTATE;
      instruction.params[0] = step;
      rotations.push_back(instruction);
    }
    FhnProgram program{FHN_ABI_VERSION, static_cast<std::uint32_t>(rotations.size()), rotations.data(), 0, nullptr, 0,
                       nullptr};
    check(symbol<FhnClientPrepareFn>("ckks_fhn_client_prepare")(client.get(), &program) == 0,
          "planner rotation-key preparation failed");
  }
};
Client::Client(Runtime &runtime, const std::string &path, std::uint32_t slots)
  : impl_(std::make_unique<Impl>(runtime, path, slots)) {}
Client::~Client() = default;
NativeServices Client::services() const { return impl_->services; }
Value Client::encrypt(const std::vector<double> &input) {
  if (input.size() != impl_->slots)
    throw std::invalid_argument("input must fill the declared logical slots (pad explicitly)");
  for (auto value : input)
    if (!std::isfinite(value))
      throw std::invalid_argument("input values must be finite");
  const auto &services = impl_->services;
  auto release = [&](FhnBuffer *buffer) { services.buffer_free(services.context.get(), buffer); };
  std::unique_ptr<FhnBuffer, decltype(release)> buffer(services.buffer_alloc(services.context.get()), release);
  check(bool(buffer), "input ciphertext allocation failed");
  check(impl_->encrypt(impl_->client.get(), buffer.get(), input.data(), input.size()) == 0, "input encryption failed");
  return impl_->native.importCopy(buffer.get());
}
std::vector<double> Client::decrypt(const Value &value) const {
  std::vector<double> result(impl_->slots);
  check(impl_->decrypt(impl_->client.get(), impl_->native.buffer(value), result.data(), result.size()) == 0,
        "trusted planner validation decryption failed");
  return result;
}
} // namespace test_support
