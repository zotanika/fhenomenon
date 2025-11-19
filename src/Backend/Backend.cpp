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
      [[maybe_unused]] Configuration config(configPath);
      // TODO: Wire configuration into backend selection/key loading flow.
    }

    // Register scheduling strategies
    // scheduler.registerStrategy(std::make_shared<scheduler::PrintOperationsStrategy<T>>());
    // scheduler.registerStrategy(std::make_shared<scheduler::FuseOperationsStrategy<T>>());
  }
  // return *instance;
  return *backend;
}

std::unique_ptr<Backend> Backend::createBuiltinBackend() {
  return std::make_unique<BuiltinBackend>();
}

std::unique_ptr<Backend> Backend::createExternalBackend(std::string_view libPath) {
  return std::make_unique<ExternalBackend>(libPath.data());
}

} // namespace fhenomenon
