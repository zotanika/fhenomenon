#include "FHN/contrib/MatvecKernels.h"
#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <limits>
#include <utility>

namespace fhenomenon::contrib {
namespace {
using namespace runtime;
enum class Operation { Prepare, Product, Dot, Reduce };
constexpr const char *artifactFormat = "contrib.matvec.preparation";
struct Provider {
  // Destruction releases native services before their extra library lease.
  std::shared_ptr<void> library;
  NativeServices services;
  std::shared_ptr<NativeRuntime> native;
  FhnMatvecPreparationRequiredBytesFn required = nullptr;
  FhnPrepareMatvecSourceFn prepare = nullptr;
  FhnMatvecPreparationBytesFn bytes = nullptr;
  FhnMatvecPreparationFreeFn free = nullptr;
  FhnMatvecProductFn product = nullptr;
  FhnMatvecDotFn dot = nullptr;
  FhnMatvecReduceFn reduce = nullptr;
  bool available() const { return required && prepare && bytes && free && product && dot && reduce; }
};
struct Prepared {
  std::shared_ptr<const Provider> provider;
  FhnMatvecPreparation *handle = nullptr;
  const uint64_t source_identity;
  const FhnCkksBufferInfo source_info;
  const uint32_t width;
  Prepared(std::shared_ptr<const Provider> owner, const Value &source, FhnCkksBufferInfo info, uint32_t w)
    : provider(std::move(owner)), source_identity(source.identity()), source_info(info), width(w) {}
  ~Prepared() {
    if (handle)
      provider->free(handle);
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
template <class T> T symbol(void *library, const char *name) { return reinterpret_cast<T>(dlsym(library, name)); }
bool powerOfTwo(uint32_t width) { return width && !(width & (width - 1)); }
bool metadata(const Provider &provider, const Value &value, FhnCkksBufferInfo &info) {
  try {
    const auto *buffer = provider.native->buffer(value);
    return provider.services.buffer_info(provider.services.context.get(), buffer, &info) == 0 && info.components == 2 &&
           info.level >= 0 && powerOfTwo(info.logical_slots) && std::isfinite(info.scale) && info.scale > 0;
  } catch (const Error &) {
    return false;
  }
}
std::shared_ptr<Prepared> preparation(const Provider &provider, const Value &value) {
  if (!value || value.kind() != ResourceKind::Artifact || value.format() != artifactFormat)
    return {};
  const auto *owned = std::any_cast<std::shared_ptr<Prepared>>(&value.payload());
  if (!owned || !*owned || (*owned)->provider.get() != &provider || !(*owned)->handle)
    return {};
  return *owned;
}
bool applicable(const Provider &provider, Operation operation, const Request &request, std::string &reason) {
  if (!provider.available()) {
    reason = "optional prepared matvec native capability unavailable";
    return false;
  }
  if (operation == Operation::Prepare) {
    const auto *options = std::any_cast<MatvecPrepareRequest>(&request.attributes);
    FhnCkksBufferInfo info{};
    if (!options || !powerOfTwo(options->width) || (options->flags & ~uint32_t{FHN_MATVEC_PREPARE_ALL}) ||
        request.inputs.size() != 1 || !metadata(provider, request.inputs[0], info) || info.level < 1 ||
        options->width > info.logical_slots) {
      reason = "invalid matvec source, width or preparation flags";
      return false;
    }
    return true;
  }
  const bool reduction = operation == Operation::Reduce;
  if (request.inputs.size() != (reduction ? 2u : 3u)) {
    reason = "incorrect prepared matvec operand count";
    return false;
  }
  const auto prepared = preparation(provider, request.inputs.back());
  if (!prepared) {
    reason = "different provider or invalid matvec preparation artifact";
    return false;
  }
  FhnCkksBufferInfo first{};
  if (!metadata(provider, request.inputs[0], first) || first.logical_slots != prepared->source_info.logical_slots) {
    reason = "different native context or incompatible matvec ciphertext";
    return false;
  }
  if (reduction) {
    const auto *width = std::any_cast<uint32_t>(&request.attributes);
    if (!width || !powerOfTwo(*width) || *width > prepared->width || first.level != prepared->source_info.level - 1) {
      reason = "reduction requires a matching product level and prepared width";
      return false;
    }
    return true; // Native preparation validates the expected product scale.
  }
  FhnCkksBufferInfo row{};
  if (request.inputs[0].identity() != prepared->source_identity || !metadata(provider, request.inputs[1], row) ||
      row.level != first.level || row.logical_slots != first.logical_slots ||
      std::abs(row.scale - first.scale) > 1e-12 * std::max(row.scale, first.scale)) {
    reason = "matvec source identity or row metadata does not match preparation";
    return false;
  }
  return true;
}
Value executePrepare(Runtime &runtime, const std::shared_ptr<const Provider> &provider, const Request &request) {
  const auto options = std::any_cast<MatvecPrepareRequest>(request.attributes);
  const FhnMatvecPrepareOptions nativeOptions{sizeof(FhnMatvecPrepareOptions), options.flags};
  auto *context = provider->services.context.get();
  const auto *source = provider->native->buffer(request.inputs[0]);
  const auto bytes = provider->required(context, source, options.width, &nativeOptions);
  if (!bytes || bytes > std::numeric_limits<size_t>::max())
    throw Error(ErrorCode::ExecutionFailed, "Invalid matvec preparation or unavailable evaluation keys");
  FhnCkksBufferInfo info{};
  if (provider->services.buffer_info(context, source, &info))
    throw Error(ErrorCode::ExecutionFailed, "Could not inspect matvec source");
  // makeValue reserves before this factory. The dependency pins the exact
  // immutable source Value; the native artifact independently snapshots bytes.
  return runtime.makeValue(artifactFormat, ResourceKind::Artifact, static_cast<size_t>(bytes),
                           [&]() -> std::any {
                             auto owned = std::make_shared<Prepared>(provider, request.inputs[0], info, options.width);
                             owned->handle = provider->prepare(context, source, options.width, &nativeOptions);
                             if (!owned->handle || provider->bytes(owned->handle) != bytes)
                               throw Error(ErrorCode::ExecutionFailed, "Matvec preparation failed or reserve changed");
                             return owned;
                           },
                           {request.inputs[0]});
}
Value execute(Runtime &runtime, const std::shared_ptr<const Provider> &provider, Operation operation,
              const Request &request) {
  if (operation == Operation::Prepare)
    return executePrepare(runtime, provider, request);
  const auto prepared = preparation(*provider, request.inputs.back());
  const auto &services = provider->services;
  const auto bytes = services.level_bytes(services.context.get(), services.fresh_level(services.context.get()));
  if (!bytes || bytes > std::numeric_limits<size_t>::max())
    throw Error(ErrorCode::InvalidRequest, "Invalid matvec staging reservation");
  auto stage =
    runtime.makeValue("contrib.matvec.staging", ResourceKind::Value, static_cast<size_t>(bytes), [&]() -> std::any {
      auto owned = std::make_shared<StagingBuffer>(provider);
      owned->buffer = services.buffer_alloc(services.context.get());
      if (!owned->buffer)
        throw Error(ErrorCode::ExecutionFailed, "Matvec staging allocation failed");
      return owned;
    });
  auto staging = std::any_cast<std::shared_ptr<StagingBuffer>>(stage.payload());
  const auto *operand = provider->native->buffer(request.inputs[operation == Operation::Reduce ? 0 : 1]);
  int status = -1;
  if (operation == Operation::Product)
    status = provider->product(services.context.get(), staging->buffer, prepared->handle, operand);
  else if (operation == Operation::Dot)
    status = provider->dot(services.context.get(), staging->buffer, prepared->handle, operand);
  else
    status = provider->reduce(services.context.get(), staging->buffer, prepared->handle, operand,
                              std::any_cast<uint32_t>(request.attributes));
  if (status)
    throw Error(ErrorCode::ExecutionFailed, "Prepared matvec execution failed");
  // Only the public adapter publishes the result. Both full allocations stay
  // charged until importCopy succeeds, and failure unwinds the temporary charge.
  return provider->native->importCopy(staging->buffer);
}
} // namespace
void registerMatvecKernels(runtime::Runtime &runtime, runtime::NativeServices services,
                           const std::string &nativeLibraryPath) {
  void *library = dlopen(nativeLibraryPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!library)
    throw runtime::Error(runtime::ErrorCode::ExecutionFailed,
                         "Could not load prepared matvec native library: " + std::string(dlerror()));
  auto retained = std::shared_ptr<void>(library, [](void *handle) { dlclose(handle); });
  auto provider = std::make_shared<Provider>();
  provider->library = std::move(retained);
  provider->services = std::move(services);
  provider->native = std::make_shared<runtime::NativeRuntime>(runtime, provider->services);
  const auto version = symbol<FhnMatvecApiVersionFn>(library, "ckks_fhn_matvec_api_version");
  if (version && version() == 1 &&
      symbol<FhnGetKernelsFn>(library, "ckks_fhn_get_kernels") == provider->services.get_kernels &&
      symbol<FhnCkksBufferInfoFn>(library, "ckks_fhn_buffer_info") == provider->services.buffer_info &&
      symbol<FhnFreshLevelFn>(library, "ckks_fhn_fresh_level") == provider->services.fresh_level) {
    provider->required =
      symbol<FhnMatvecPreparationRequiredBytesFn>(library, "ckks_fhn_matvec_preparation_required_bytes");
    provider->prepare = symbol<FhnPrepareMatvecSourceFn>(library, "ckks_fhn_prepare_matvec_source");
    provider->bytes = symbol<FhnMatvecPreparationBytesFn>(library, "ckks_fhn_matvec_preparation_bytes");
    provider->free = symbol<FhnMatvecPreparationFreeFn>(library, "ckks_fhn_matvec_preparation_free");
    provider->product = symbol<FhnMatvecProductFn>(library, "ckks_fhn_matvec_product");
    provider->dot = symbol<FhnMatvecDotFn>(library, "ckks_fhn_matvec_dot");
    provider->reduce = symbol<FhnMatvecReduceFn>(library, "ckks_fhn_matvec_reduce");
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
  add("contrib.matvec.prepare", Operation::Prepare);
  add("contrib.matvec.product", Operation::Product);
  add("contrib.matvec.dot", Operation::Dot);
  add("contrib.matvec.reduce", Operation::Reduce);
}
} // namespace fhenomenon::contrib
