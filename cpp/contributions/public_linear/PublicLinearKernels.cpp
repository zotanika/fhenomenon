#include "FHN/contrib/PublicLinearKernels.h"
#include "FHN/fhn_plaintext_api.h"
#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <limits>

namespace fhenomenon::contrib {
namespace {
using namespace runtime;
constexpr auto prepareOperation = "contrib.public_linear.prepare_row";
constexpr auto dotOperation = "contrib.public_linear.dot_row";
constexpr auto rowFormat = "contrib.public_linear.prepared_row";

struct Provider {
  NativeServices services;
  std::unique_ptr<NativeRuntime> native;
  std::shared_ptr<void> library;
  FhnPlaintextRequiredBytesFn required;
  FhnPlaintextEncodeFn encode;
  FhnPlaintextApplyFn multiply;
  FhnPlaintextBytesFn bytes;
  FhnPlaintextFreeFn release;
  template <class T> T symbol(const char *name) const {
    auto function = reinterpret_cast<T>(dlsym(library.get(), name));
    if (!function)
      throw Error(ErrorCode::Unsupported, std::string("native public plaintext extension missing ") + name);
    return function;
  }
  Provider(Runtime &runtime, NativeServices svc, const std::string &path)
    : services(std::move(svc)), library(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL), [](void *handle) {
        if (handle)
          dlclose(handle);
      }) {
    if (!library || symbol<FhnPlaintextApiVersionFn>("ckks_fhn_plaintext_api_version")() != 1)
      throw Error(ErrorCode::Unsupported, "native public plaintext extension version 1 required");
    if (symbol<FhnGetKernelsFn>("ckks_fhn_get_kernels") != services.get_kernels ||
        symbol<FhnCkksBufferInfoFn>("ckks_fhn_buffer_info") != services.buffer_info ||
        symbol<FhnFreshLevelFn>("ckks_fhn_fresh_level") != services.fresh_level)
      throw Error(ErrorCode::Unsupported, "public plaintext extension must match native services");
    required = symbol<FhnPlaintextRequiredBytesFn>("ckks_fhn_plaintext_required_bytes");
    encode = symbol<FhnPlaintextEncodeFn>("ckks_fhn_plaintext_encode");
    multiply = symbol<FhnPlaintextApplyFn>("ckks_fhn_plaintext_multiply");
    (void)symbol<FhnPlaintextApplyFn>("ckks_fhn_plaintext_add");
    bytes = symbol<FhnPlaintextBytesFn>("ckks_fhn_plaintext_bytes");
    release = symbol<FhnPlaintextFreeFn>("ckks_fhn_plaintext_free");
    native = std::make_unique<NativeRuntime>(runtime, services);
  }
};
struct Row {
  std::uint64_t identity = 0;
  std::shared_ptr<const Provider> provider;
  FhnCkksBufferInfo prototype{};
  std::unique_ptr<FhnPlaintext, FhnPlaintextFreeFn> plain;
  Row(std::shared_ptr<const Provider> owner, FhnCkksBufferInfo info, FhnPlaintext *data)
    : provider(std::move(owner)), prototype(info), plain(data, provider->release) {}
};
bool metadata(const Provider &provider, const Value &value, FhnCkksBufferInfo &info) {
  try {
    return value &&
           provider.services.buffer_info(provider.services.context.get(), provider.native->buffer(value), &info) == 0 &&
           info.components == 2 && info.level >= 1 && info.logical_slots &&
           !(info.logical_slots & (info.logical_slots - 1));
  } catch (const Error &) {
    return false;
  }
}
bool prepareApplicable(const Provider &provider, const Request &request, std::string &why) {
  const auto *weights = std::any_cast<std::vector<double>>(&request.attributes);
  FhnCkksBufferInfo info{};
  if (request.inputs.size() != 1 || !weights || !metadata(provider, request.inputs[0], info) ||
      weights->size() != info.logical_slots ||
      !std::all_of(weights->begin(), weights->end(), [](double v) { return std::isfinite(v); })) {
    why = "public row requires finite weights matching the native ciphertext slots and available level";
    return false;
  }
  return true;
}
const Row *prepared(const Provider &provider, const Value &value) {
  if (!value || value.kind() != ResourceKind::Artifact || value.format() != rowFormat)
    return nullptr;
  const auto *handle = std::any_cast<std::shared_ptr<const Row>>(&value.payload());
  if (!handle || !*handle || (*handle)->identity != value.identity() || (*handle)->provider.get() != &provider)
    return nullptr;
  return handle->get();
}
bool dotApplicable(const Provider &provider, const Request &request, std::string &why) {
  const auto *bias = std::any_cast<double>(&request.attributes);
  FhnCkksBufferInfo info{};
  const Row *row = request.inputs.size() == 2 ? prepared(provider, request.inputs[1]) : nullptr;
  if (!bias || !std::isfinite(*bias) || !row || !metadata(provider, request.inputs[0], info) ||
      info.logical_slots != row->prototype.logical_slots || info.level != row->prototype.level ||
      std::abs(info.scale - row->prototype.scale) > 1e-12 * std::max(info.scale, row->prototype.scale)) {
    why = "dot row requires matching native context/level/scale/slots, a prepared public row and finite bias";
    return false;
  }
  return true;
}
Value single(Values values) {
  if (values.size() != 1)
    throw Error(ErrorCode::ExecutionFailed, "public linear operation must produce one result");
  return std::move(values[0]);
}
Values prepare(Runtime &runtime, const std::shared_ptr<Provider> &provider, const Request &request) {
  const auto *source = provider->native->buffer(request.inputs[0]);
  const auto size = provider->required(provider->services.context.get(), source, FHN_PLAINTEXT_MULTIPLY);
  if (!size || size > std::numeric_limits<std::size_t>::max())
    throw Error(ErrorCode::InvalidRequest, "invalid public row reservation");
  const auto &weights = std::any_cast<const std::vector<double> &>(request.attributes);
  FhnCkksBufferInfo info{};
  if (!metadata(*provider, request.inputs[0], info))
    throw Error(ErrorCode::InvalidRequest, "invalid row prototype");
  std::shared_ptr<Row> owner;
  auto result = runtime.makeValue(rowFormat, ResourceKind::Artifact, static_cast<std::size_t>(size), [&]() -> std::any {
    FhnPlaintext *encoded = nullptr;
    if (provider->encode(provider->services.context.get(), source, weights.data(), weights.size(),
                         FHN_PLAINTEXT_MULTIPLY, &encoded) != 0 ||
        !encoded)
      throw Error(ErrorCode::ExecutionFailed, "public row encoding failed");
    std::unique_ptr<FhnPlaintext, FhnPlaintextFreeFn> guard(encoded, provider->release);
    if (provider->bytes(encoded) != size)
      throw Error(ErrorCode::ExecutionFailed, "public row reservation mismatch");
    owner = std::make_shared<Row>(provider, info, encoded);
    guard.release();
    return std::shared_ptr<const Row>(owner);
  });
  owner->identity = result.identity();
  return {std::move(result)};
}
Values dot(Runtime &runtime, const std::shared_ptr<Provider> &provider, const Request &request) {
  const auto *row = prepared(*provider, request.inputs[1]);
  if (!row)
    throw Error(ErrorCode::InvalidRequest, "invalid public row");
  const auto &services = provider->services;
  // Account the temporary publication buffer too. Native importCopy publishes
  // its own immutable buffer; the temporary is released after import.
  auto scratch = runtime.makeValue("contrib.public_linear.publication", ResourceKind::Artifact,
                                   static_cast<std::size_t>(row->prototype.reserved_bytes), [&]() -> std::any {
                                     auto *buffer = services.buffer_alloc(services.context.get());
                                     if (!buffer)
                                       throw Error(ErrorCode::ExecutionFailed, "public dot output allocation failed");
                                     return std::shared_ptr<FhnBuffer>(buffer, [provider](FhnBuffer *p) {
                                       provider->services.buffer_free(provider->services.context.get(), p);
                                     });
                                   });
  const auto &buffer = std::any_cast<const std::shared_ptr<FhnBuffer> &>(scratch.payload());
  if (provider->multiply(services.context.get(), buffer.get(), provider->native->buffer(request.inputs[0]),
                         row->plain.get()) != 0)
    throw Error(ErrorCode::ExecutionFailed, "public dot multiplication failed");
  auto value = provider->native->importCopy(buffer.get());
  scratch = {};
  for (std::uint32_t step = 1; step < row->prototype.logical_slots; step *= 2) {
    auto rotated = provider->native->rotate(value, step);
    value = provider->native->op(FHN_ADD_CC, {value, rotated});
  }
  const double bias = std::any_cast<double>(request.attributes);
  if (bias != 0)
    value = provider->native->op(FHN_ADD_CS, {value}, {}, {bias, 0});
  return {std::move(value)};
}
} // namespace

void registerPublicLinearKernels(runtime::Runtime &runtime, runtime::NativeServices services, const std::string &path) {
  auto provider = std::make_shared<Provider>(runtime, std::move(services), path);
  runtime.addRecipe({prepareOperation, "contrib.public_linear.prepare_row.native", 0,
                     [provider](const Request &r, std::string &why) { return prepareApplicable(*provider, r, why); },
                     [provider](Runtime &rt, const Request &r) { return prepare(rt, provider, r); }});
  runtime.addRecipe({dotOperation, "contrib.public_linear.dot_row.native", 0,
                     [provider](const Request &r, std::string &why) { return dotApplicable(*provider, r, why); },
                     [provider](Runtime &rt, const Request &r) { return dot(rt, provider, r); }});
}
runtime::Value PublicLinearRuntime::prepareRow(runtime::Value prototype, std::vector<double> weights) {
  return single(runtime_.submit({prepareOperation, {std::move(prototype)}, std::move(weights)}));
}
runtime::Value PublicLinearRuntime::dot(runtime::Value input, runtime::Value row, double bias) {
  return single(runtime_.submit({dotOperation, {std::move(input), std::move(row)}, bias}));
}
runtime::ScopeValue PublicLinearRuntime::dot(runtime::Scope &scope, runtime::ScopeValue input, runtime::ScopeValue row,
                                             double bias) {
  if (!scope.belongsTo(runtime_))
    throw Error(ErrorCode::InvalidRequest, "public linear scope belongs to another runtime");
  return scope.submit(dotOperation, {input, row}, bias)[0];
}
} // namespace fhenomenon::contrib
