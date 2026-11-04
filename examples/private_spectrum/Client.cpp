#include "Workflow.h"
#include <dlfcn.h>
#include <stdexcept>

namespace private_spectrum {
namespace {
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

NativeServices computeOnly(const std::string &path) {
  auto services = fhenomenon::runtime::loadNativeServices(path);
  services.compile = nullptr;
  services.run = nullptr;
  services.executable_stats = nullptr;
  services.executable_free = nullptr;
  return services;
}
} // namespace

struct Client::Impl {
  std::unique_ptr<void, decltype(&dlclose)> library;
  NativeServices services;
  std::unique_ptr<FhnClientCtx, FhnClientDestroyFn> owner;
  NativeRuntime native;
  FhnClientEncryptVectorComplex64Fn encrypt;
  FhnClientDecryptVectorComplex64Fn decrypt;
  std::unique_ptr<JointDetector> detector;

  template <class T> T symbol(const char *name) const {
    auto fn = reinterpret_cast<T>(dlsym(library.get(), name));
    if (!fn)
      throw std::runtime_error(std::string("missing client symbol ") + name);
    return fn;
  }

  Impl(Runtime &runtime, const std::string &path)
    : library(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL), dlclose),
      services(computeOnly(path)),
      owner(nullptr, symbol<FhnClientDestroyFn>("ckks_fhn_client_destroy")),
      native(runtime, services),
      encrypt(symbol<FhnClientEncryptVectorComplex64Fn>("ckks_fhn_client_encrypt_vec_c64")),
      decrypt(symbol<FhnClientDecryptVectorComplex64Fn>("ckks_fhn_client_decrypt_vec_c64")) {
    require(bool(library), "cannot load native CKKS library");
    owner.reset(symbol<FhnClientCreateFn>("ckks_fhn_client_create")(services.context.get()));
    require(bool(owner), "client creation failed");

    // Both evaluator modes use these public evaluation keys. Each bin occupies
    // a separate slot, so this circuit requires no rotation keys.
    const auto keys = fhenomenon::contrib::nativeSpectralKeyRequirements();
    std::vector<FhnInstruction> requests;
    if (keys.multiplication) {
      FhnInstruction multiply{};
      multiply.opcode = FHN_HMULT;
      requests.push_back(multiply);
    }
    if (keys.conjugation) {
      FhnInstruction conjugate{};
      conjugate.opcode = FHN_CONJUGATE;
      requests.push_back(conjugate);
    }
    const FhnProgram request{
      FHN_ABI_VERSION, static_cast<std::uint32_t>(requests.size()), requests.data(), 0, nullptr, 0, nullptr};
    require(symbol<FhnClientPrepareFn>("ckks_fhn_client_prepare")(owner.get(), &request) == 0,
            "evaluation key preparation failed");
  }

  Value encode(const Bins &bins) {
    std::array<FhnComplex64, binCount> values{};
    for (unsigned bin = 0; bin < binCount; ++bin) {
      const auto value = bins[bin];
      if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        throw std::invalid_argument("frame bins must be finite");
      values[bin] = {value.real(), value.imag()};
    }
    auto release = [&](FhnBuffer *buffer) { services.buffer_free(services.context.get(), buffer); };
    std::unique_ptr<FhnBuffer, decltype(release)> buffer(services.buffer_alloc(services.context.get()), release);
    require(bool(buffer), "ciphertext allocation failed");
    require(encrypt(owner.get(), buffer.get(), values.data(), values.size()) == 0, "client encryption failed");
    return native.importCopy(buffer.get());
  }

  Bins decode(const Value &value) const {
    std::array<FhnComplex64, binCount> decoded{};
    require(decrypt(owner.get(), native.buffer(value), decoded.data(), decoded.size()) == 0, "client decryption failed");
    Bins result{};
    for (unsigned bin = 0; bin < binCount; ++bin)
      result[bin] = {decoded[bin].real, decoded[bin].imag};
    return result;
  }

  Spectrum decode(const EncryptedSpectrum &encrypted) const {
    Spectrum result;
    for (unsigned sensor = 0; sensor < sensorCount; ++sensor) {
      result.cross[sensor] = decode(encrypted.cross[sensor]);
      result.power[sensor] = decode(encrypted.power[sensor]);
    }
    return result;
  }
};

Client::Client(Runtime &runtime, const std::string &path) : impl_(std::make_unique<Impl>(runtime, path)) {}
Client::~Client() = default;
NativeServices Client::computeServices() const { return impl_->services; }
EncryptedFrame Client::encrypt(const Frame &frame) {
  EncryptedFrame encrypted;
  for (unsigned sensor = 0; sensor < sensorCount; ++sensor)
    encrypted[sensor] = impl_->encode(frame[sensor]);
  return encrypted;
}
void Client::startMonitoring(const EncryptedSpectrum &baseline, unsigned frames) {
  impl_->detector = std::make_unique<JointDetector>(impl_->decode(baseline), frames);
}
Alarm Client::observe(const EncryptedSpectrum &window, unsigned frames) {
  if (!impl_->detector)
    throw std::logic_error("monitoring baseline has not been established");
  return impl_->detector->observe(impl_->decode(window), frames);
}
double Client::verify(const EncryptedSpectrum &encrypted, const Spectrum &expected) {
  return maxComplexError(impl_->decode(encrypted), expected);
}
Spectrum Client::decrypt(const EncryptedSpectrum &encrypted) { return impl_->decode(encrypted); }
} // namespace private_spectrum
