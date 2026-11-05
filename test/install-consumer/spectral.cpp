#include "FHN/NativeRuntime.h"
#include "FHN/contrib/SpectralKernels.h"
#include "Session/Session.h"
#include <array>
#include <cmath>
#include <complex>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
template <class T> T symbol(void *library, const char *name) {
  auto value = reinterpret_cast<T>(dlsym(library, name));
  require(value != nullptr, name);
  return value;
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "expected installed plugin path");
    using namespace fhenomenon::runtime;
    std::unique_ptr<void, decltype(&dlclose)> library(dlopen(argv[1], RTLD_NOW | RTLD_LOCAL), dlclose);
    require(bool(library), "plugin load failed");
    Runtime runtime;
    auto services = loadNativeServices(argv[1]);
    auto create = symbol<FhnClientCreateFn>(library.get(), "ckks_fhn_client_create");
    auto destroy = symbol<FhnClientDestroyFn>(library.get(), "ckks_fhn_client_destroy");
    std::unique_ptr<FhnClientCtx, FhnClientDestroyFn> client(create(services.context.get()), destroy);
    require(bool(client), "client create failed");
    auto prepare = symbol<FhnClientPrepareFn>(library.get(), "ckks_fhn_client_prepare");
    auto encrypt = symbol<FhnClientEncryptVectorComplex64Fn>(library.get(), "ckks_fhn_client_encrypt_vec_c64");
    auto decrypt = symbol<FhnClientDecryptVectorComplex64Fn>(library.get(), "ckks_fhn_client_decrypt_vec_c64");
    auto requirements = fhenomenon::contrib::nativeSpectralKeyRequirements();
    require(requirements.multiplication && requirements.conjugation, "spectral key requirements differ");
    FhnInstruction keys[2]{};
    keys[0].opcode = FHN_HMULT;
    keys[1].opcode = FHN_CONJUGATE;
    FhnProgram program{FHN_ABI_VERSION, 2, keys, 0, nullptr, 0, nullptr};
    require(prepare(client.get(), &program) == 0, "key preparation failed");
    NativeRuntime native(runtime, services);
    fhenomenon::contrib::registerSpectralKernels(runtime, services, argv[1]);
    fhenomenon::contrib::SpectralRuntime spectral(runtime);
    const std::array<FhnComplex64, 4> samples{{{1.25, -0.75}, {0, 2}, {-1, 0.5}, {0.5, -1}}};
    auto free = [&](FhnBuffer *buffer) { services.buffer_free(services.context.get(), buffer); };
    std::unique_ptr<FhnBuffer, decltype(free)> buffer(services.buffer_alloc(services.context.get()), free);
    require(bool(buffer), "buffer allocation failed");
    require(encrypt(client.get(), buffer.get(), samples.data(), samples.size()) == 0, "complex encrypt failed");
    auto source = native.importCopy(buffer.get());
    auto session = fhenomenon::Session::create(runtime);
    auto result = session->run([&](Scope &scope) {
      const auto x = scope.input(source);
      const auto reference = spectral.prepareReference(scope, x);
      const auto power = spectral.pair(scope, x, reference.second);
      const auto conjugateSquared = spectral.pair(scope, reference.first, reference.second);
      const auto sum = native.op(scope, FHN_ADD_CC, {power, conjugateSquared});
      return ScopeValues{reference.second, power, conjugateSquared, sum};
    });
    // A second scope reuses the same installed contribution artifact and an
    // intermediate produced by its previous tile through an ordinary ADD.
    auto continued = session->run([&](Scope &scope) {
      auto product = spectral.pair(scope, scope.input(source), scope.input(result[0]));
      return ScopeValues{native.op(scope, FHN_ADD_CC, {product, scope.input(result[1])})};
    });
    bool selectedTile = false;
    for (const auto &event : runtime.trace())
      selectedTile |= event.kind == TraceKind::Selected && event.recipe == "contrib.spectral.scope.pair-tile";
    require(selectedTile, "installed spectral tile was not selected");
    for (std::size_t which = 0; which < 4; ++which) {
      std::array<FhnComplex64, 4> actual{};
      const auto &value = which < 3 ? result[which + 1] : continued[0];
      require(decrypt(client.get(), native.buffer(value), actual.data(), actual.size()) == 0, "complex decrypt failed");
      for (std::size_t i = 0; i < samples.size(); ++i) {
        const std::complex<double> x(samples[i].real, samples[i].imag);
        const auto power = x * std::conj(x), square = std::conj(x) * std::conj(x);
        const auto expected = which == 0 ? power : which == 1 ? square : which == 2 ? power + square : 2.0 * power;
        const std::complex<double> decoded(actual[i].real, actual[i].imag);
        require(std::isfinite(decoded.real()) && std::isfinite(decoded.imag()) && std::abs(decoded - expected) < 1e-5,
                "installed complex spectrum mismatch");
      }
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
