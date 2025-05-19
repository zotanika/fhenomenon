#include "Backend/Builtin.h"
#include "Backend/External.h"

namespace fhenomenon {

Backend &Backend::getInstance(std::string_view libPath, std::string_view configPath) {
  static std::unique_ptr<Backend> backend;
  if (!backend) {
    if (libPath.empty()) {
      // instance.swap(createBuiltinBackend());
      backend = createBuiltinBackend();
    } else {
      // instance = createExternalBackend(libPath);
      backend = createExternalBackend(libPath);
    }

    // Initialize KeyManager
    if (!configPath.empty()) {
      Configuration config(configPath);
      // (TODO)
      // std::shared_ptr<seal::SEALContext> context =
      // std::make_shared<seal::SEALContext>(
      //    seal::EncryptionParameters(seal::scheme_type::ckks));
      // keyManager.reset(new KeyManager(context, config));
    }

    // Register scheduling strategies
    // scheduler.registerStrategy(std::make_shared<scheduler::PrintOperationsStrategy<T>>());
    // scheduler.registerStrategy(std::make_shared<scheduler::FuseOperationsStrategy<T>>());
  }
  // return *instance;
  return *backend;
}

std::unique_ptr<Backend> Backend::createBuiltinBackend() {
  return std::make_unique<BuiltinBackendMockup>();
  // return std::unique_ptr<Backend>(new BuiltinBackendMockup());
}

std::unique_ptr<Backend> Backend::createExternalBackend(std::string_view libPath) {
  return std::make_unique<ExternalBackend>(libPath.data());
}

} // namespace fhenomenon
