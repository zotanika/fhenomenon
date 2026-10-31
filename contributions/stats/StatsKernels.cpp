#include "FHN/contrib/StatsKernels.h"
#include "FHN/fhn_coarse_api.h"
#include <cmath>
#include <dlfcn.h>
#include <limits>
#include <utility>

namespace fhenomenon::contrib {
namespace {
using namespace runtime;
enum class Operation { Energy, Variance, Dot, Reduce };

struct Provider {
  // Services and their context are destroyed before the extra library lease.
  std::shared_ptr<void> library;
  NativeServices services;
  std::shared_ptr<NativeRuntime> native;
  FhnCenteredEnergyFn energy = nullptr;
  FhnCenteredVarianceFn variance = nullptr;
  FhnPackedDotFn dot = nullptr;
  FhnPackedReduceFn reduce = nullptr;

  bool available(Operation operation) const {
    switch (operation) {
    case Operation::Energy:
      return energy != nullptr;
    case Operation::Variance:
      return variance != nullptr;
    case Operation::Dot:
      return dot != nullptr;
    case Operation::Reduce:
      return reduce != nullptr;
    }
    return false;
  }
};

struct StagingBuffer {
  std::shared_ptr<const Provider> provider;
  FhnBuffer *buffer = nullptr;
  explicit StagingBuffer(std::shared_ptr<const Provider> owner) : provider(std::move(owner)) {}
  ~StagingBuffer() {
    if (buffer)
      provider->services.buffer_free(provider->services.context.get(), buffer);
  }
};

template <class Function> Function symbol(void *library, const char *name) {
  return reinterpret_cast<Function>(dlsym(library, name));
}

bool applicable(const Provider &provider, Operation operation, const Request &request, std::string &reason) {
  if (!provider.available(operation)) {
    reason = "optional coarse native capability unavailable";
    return false;
  }
  const auto *width = std::any_cast<std::uint32_t>(&request.attributes);
  if (!width || !*width || (*width & (*width - 1))) {
    reason = "width must be a nonzero power-of-two uint32_t";
    return false;
  }
  const bool unary = operation == Operation::Reduce;
  const auto minimumLevel = operation == Operation::Energy || operation == Operation::Variance ? 2 : unary ? 0 : 1;
  if (request.inputs.size() != (unary ? 1u : 2u)) {
    reason = "incorrect coarse operand count";
    return false;
  }
  FhnCkksBufferInfo first{};
  for (std::size_t i = 0; i < request.inputs.size(); ++i) {
    const FhnBuffer *buffer = nullptr;
    try {
      buffer = provider.native->buffer(request.inputs[i]);
    } catch (const Error &) {
      reason = "different native context or ciphertext format";
      return false;
    }
    FhnCkksBufferInfo metadata{};
    if (provider.services.buffer_info(provider.services.context.get(), buffer, &metadata) || metadata.components != 2 ||
        metadata.level < minimumLevel || !std::isfinite(metadata.scale) || metadata.scale <= 0 ||
        *width > metadata.logical_slots) {
      reason = "unsupported coarse ciphertext metadata or width";
      return false;
    }
    if (i && (metadata.level != first.level || metadata.scale != first.scale ||
              metadata.logical_slots != first.logical_slots)) {
      reason = "coarse operands require equal level, scale and logical slots";
      return false;
    }
    first = metadata;
  }
  return true;
}

Value execute(Runtime &runtime, const std::shared_ptr<const Provider> &provider, Operation operation,
              const Request &request) {
  const auto &services = provider->services;
  const auto bytes = services.level_bytes(services.context.get(), services.fresh_level(services.context.get()));
  if (!bytes || bytes > std::numeric_limits<std::size_t>::max())
    throw Error(ErrorCode::InvalidRequest, "Invalid coarse staging reservation");
  // Reserve the full native allocation before entering the allocation factory.
  auto stage =
    runtime.makeValue("contrib.stats.staging", ResourceKind::Value, static_cast<std::size_t>(bytes), [&]() -> std::any {
      auto owned = std::make_shared<StagingBuffer>(provider);
      owned->buffer = services.buffer_alloc(services.context.get());
      if (!owned->buffer)
        throw Error(ErrorCode::ExecutionFailed, "Coarse staging allocation failed");
      return owned;
    });
  const auto staging = std::any_cast<std::shared_ptr<StagingBuffer>>(stage.payload());
  const auto *a = provider->native->buffer(request.inputs[0]);
  const auto *b = request.inputs.size() == 2 ? provider->native->buffer(request.inputs[1]) : nullptr;
  const auto width = std::any_cast<std::uint32_t>(request.attributes);
  int status = -1;
  switch (operation) {
  case Operation::Energy:
    status = provider->energy(services.context.get(), staging->buffer, a, b, width);
    break;
  case Operation::Variance:
    status = provider->variance(services.context.get(), staging->buffer, a, b, width);
    break;
  case Operation::Dot:
    status = provider->dot(services.context.get(), staging->buffer, a, b, width);
    break;
  case Operation::Reduce:
    status = provider->reduce(services.context.get(), staging->buffer, a, width);
    break;
  }
  if (status)
    throw Error(ErrorCode::ExecutionFailed, "Coarse native execution failed");
  // Publication uses the public native adapter, so no private Value payload is
  // reproduced here. Both full buffers remain charged throughout this copy.
  return provider->native->importCopy(staging->buffer);
}
} // namespace

void registerStatsKernels(runtime::Runtime &runtime, runtime::NativeServices services,
                          const std::string &nativeLibraryPath) {
  void *library = dlopen(nativeLibraryPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!library)
    throw runtime::Error(runtime::ErrorCode::ExecutionFailed,
                         "Could not load coarse native library: " + std::string(dlerror()));
  auto retainedLibrary = std::shared_ptr<void>(library, [](void *handle) { dlclose(handle); });
  auto provider = std::make_shared<Provider>();
  provider->library = std::move(retainedLibrary);
  provider->services = std::move(services);
  provider->native = std::make_shared<runtime::NativeRuntime>(runtime, provider->services);
  const auto version = symbol<FhnCoarseApiVersionFn>(library, "ckks_fhn_coarse_api_version");
  if (version && version() == 1) {
    provider->energy = symbol<FhnCenteredEnergyFn>(library, "ckks_fhn_centered_energy");
    provider->variance = symbol<FhnCenteredVarianceFn>(library, "ckks_fhn_centered_variance");
    provider->dot = symbol<FhnPackedDotFn>(library, "ckks_fhn_packed_dot");
    provider->reduce = symbol<FhnPackedReduceFn>(library, "ckks_fhn_packed_reduce");
  }
  const auto add = [&](const char *name, Operation operation) {
    runtime.addRecipe({name, std::string(name) + ".native", 0,
                       [provider, operation](const runtime::Request &request, std::string &reason) {
                         return applicable(*provider, operation, request, reason);
                       },
                       [provider, operation](runtime::Runtime &rt, const runtime::Request &request) {
                         return runtime::Values{execute(rt, provider, operation, request)};
                       }});
  };
  add("contrib.stats.energy", Operation::Energy);
  add("contrib.stats.variance", Operation::Variance);
  add("contrib.packed.dot", Operation::Dot);
  add("contrib.packed.reduce", Operation::Reduce);
}
} // namespace fhenomenon::contrib
