#include "FHN/NativeRuntime.h"
#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <limits>
#include <utility>

namespace fhenomenon::runtime {
namespace {
using Services = std::shared_ptr<const NativeServices>;
constexpr const char *bufferFormat = "native.ckks.buffer";
constexpr const char *preparationFormat = "native.ckks.rotation-source";
constexpr const char *regionFormat = "native.ckks.region";
[[noreturn]] void invalid(const std::string &message) { throw Error(ErrorCode::InvalidRequest, message); }
[[noreturn]] void failed(const std::string &message) { throw Error(ErrorCode::ExecutionFailed, message); }
struct BufferLease {
  Services services;
  FhnBuffer *ptr;
  BufferLease(Services s, FhnBuffer *p) : services(std::move(s)), ptr(p) {}
  ~BufferLease() { services->buffer_free(services->context.get(), ptr); }
};
struct PreparationLease {
  Services services;
  FhnRotationPreparation *ptr;
  std::uint64_t source;
  PreparationLease(Services s, FhnRotationPreparation *p, std::uint64_t id)
    : services(std::move(s)), ptr(p), source(id) {}
  ~PreparationLease() { services->preparation_free(ptr); }
};
struct RegionLease {
  Services services;
  FhnExecutable *ptr;
  std::uint32_t outputs;
  RegionLease(Services s, FhnExecutable *p, std::uint32_t n) : services(std::move(s)), ptr(p), outputs(n) {}
  ~RegionLease() { services->executable_free(ptr); }
};
template <class T> std::shared_ptr<T> lease(const Value &value, const char *format, const Services &s) {
  if (!value || value.format() != format)
    invalid("Native resource format mismatch");
  auto payload = std::any_cast<std::shared_ptr<T>>(&value.payload());
  if (!payload || !*payload || (*payload)->services->context.get() != s->context.get())
    invalid("Native resource belongs to a different context");
  return *payload;
}
std::size_t checkedBytes(std::uint64_t bytes) {
  if (!bytes || bytes > std::numeric_limits<std::size_t>::max())
    invalid("Invalid native resource reservation");
  return static_cast<std::size_t>(bytes);
}
FhnCkksBufferInfo info(const Services &s, const FhnBuffer *buffer) {
  FhnCkksBufferInfo result{};
  if (!buffer || s->buffer_info(s->context.get(), buffer, &result))
    invalid("Invalid native ciphertext");
  return result;
}
Value allocate(Runtime &runtime, const Services &s, const std::function<void(FhnBuffer *)> &fill) {
  auto bytes = checkedBytes(s->level_bytes(s->context.get(), s->fresh_level(s->context.get())));
  return runtime.makeValue(bufferFormat, ResourceKind::Value, bytes, [&]() -> std::any {
    auto ptr = s->buffer_alloc(s->context.get());
    if (!ptr)
      failed("Native ciphertext allocation failed");
    std::shared_ptr<BufferLease> owned;
    try {
      owned = std::make_shared<BufferLease>(s, ptr);
    } catch (...) {
      s->buffer_free(s->context.get(), ptr);
      throw;
    }
    fill(ptr);
    return owned;
  });
}
FhnKernelFn kernel(const Services &s, FhnOpCode opcode) {
  auto table = s->get_kernels(s->context.get());
  if (table)
    for (std::uint32_t i = 0; i < table->num_kernels; ++i)
      if (table->kernels[i].opcode == opcode && table->kernels[i].fn)
        return table->kernels[i].fn;
  throw Error(ErrorCode::Unsupported, "Native physical opcode is unavailable");
}
std::size_t arity(FhnOpCode op) {
  switch (op) {
  case FHN_ADD_CC:
  case FHN_ADD_CP:
  case FHN_SUB_CC:
  case FHN_SUB_CP:
  case FHN_MULT_CC:
  case FHN_MULT_CP:
  case FHN_HMULT:
  case FHN_HROT_ADD:
  case FHN_HCONJ_ADD:
  case FHN_MAD:
  case FHN_AND:
  case FHN_OR:
  case FHN_XOR:
  case FHN_EQ:
  case FHN_LT:
  case FHN_LE:
    return 2;
  default:
    return 1;
  }
}
struct OpAttributes {
  FhnOpCode opcode;
  std::array<std::int64_t, 4> params;
  std::array<double, 2> fparams;
};
struct CompileAttributes {
  FhnCircuit circuit;
  RegionOptions options;
};
Value physical(Runtime &runtime, const Services &s, FhnOpCode op, const Values &values,
               const std::array<std::int64_t, 4> &params, const std::array<double, 2> &fparams) {
  if (values.size() != arity(op))
    invalid("Wrong native physical operand count");
  auto fn = kernel(s, op);
  std::vector<const FhnBuffer *> operands;
  for (const auto &value : values)
    operands.push_back(lease<BufferLease>(value, bufferFormat, s)->ptr);
  return allocate(runtime, s, [&](FhnBuffer *out) {
    if (fn(s->context.get(), out, operands.data(), params.data(), fparams.data()))
      failed("Native physical operation failed");
  });
}
template <class Fn> Fn symbol(void *library, const char *name, bool required = false) {
  auto fn = reinterpret_cast<Fn>(dlsym(library, name));
  if (required && !fn)
    failed(std::string("Missing native export: ") + name);
  return fn;
}
void registerRecipes(Runtime &runtime, const Services &s) {
  auto accepts = [s](const Request &request, std::string &reason) {
    for (const auto &value : request.inputs) {
      const auto *buffer = std::any_cast<std::shared_ptr<BufferLease>>(&value.payload());
      const auto *prepared = std::any_cast<std::shared_ptr<PreparationLease>>(&value.payload());
      const auto *region = std::any_cast<std::shared_ptr<RegionLease>>(&value.payload());
      FhnBackendCtx *context = buffer && *buffer       ? (*buffer)->services->context.get()
                               : prepared && *prepared ? (*prepared)->services->context.get()
                               : region && *region     ? (*region)->services->context.get()
                                                       : nullptr;
      if (context != s->context.get()) {
        reason = "different native context or format";
        return false;
      }
    }
    const auto completePreparation = s->preparation_required_bytes && s->prepare_rotation && s->preparation_bytes &&
                                     s->preparation_free && s->rotate_prepared;
    if (request.operation == "native.op") {
      const auto *attributes = std::any_cast<OpAttributes>(&request.attributes);
      if (attributes) {
        auto *table = s->get_kernels(s->context.get());
        bool available = false;
        if (table)
          for (std::uint32_t i = 0; i < table->num_kernels; ++i)
            available = available || (table->kernels[i].opcode == attributes->opcode && table->kernels[i].fn);
        if (!available) {
          reason = "native physical opcode unavailable";
          return false;
        }
      }
    } else if (request.operation == "native.prepare_rotation" || request.operation == "native.rotate") {
      if (!completePreparation) {
        reason = "native prepared rotation services unavailable";
        return false;
      }
    } else if (request.operation == "native.compile_region") {
      if (!s->compile || !s->run || !s->executable_free) {
        reason = "native region compiler services unavailable";
        return false;
      }
      const auto *attributes = std::any_cast<CompileAttributes>(&request.attributes);
      if (attributes &&
          (!attributes->options.max_nodes || attributes->circuit.num_nodes > attributes->options.max_nodes)) {
        reason = "region exceeds explicit node limit";
        return false;
      }
    } else if (request.operation == "native.run_region" && !s->run) {
      reason = "native compiled execution unavailable";
      return false;
    }
    return true;
  };
  runtime.addRecipe({"native.op", "native.physical", 0, accepts, [s](Runtime &rt, const Request &r) {
                       auto attr = std::any_cast<OpAttributes>(&r.attributes);
                       if (!attr)
                         invalid("Invalid physical operation attributes");
                       return Values{physical(rt, s, attr->opcode, r.inputs, attr->params, attr->fparams)};
                     }});
  runtime.addRecipe(
    {"native.prepare_rotation", "native.rotation-source", 0, accepts, [s](Runtime &rt, const Request &r) {
       if (r.inputs.size() != 1)
         invalid("Preparation requires one source");
       if (!s->preparation_required_bytes || !s->prepare_rotation || !s->preparation_bytes || !s->preparation_free ||
           !s->rotate_prepared)
         throw Error(ErrorCode::Unsupported, "Native prepared rotations unavailable");
       auto source = lease<BufferLease>(r.inputs[0], bufferFormat, s);
       auto bytes = checkedBytes(s->preparation_required_bytes(s->context.get(), source->ptr));
       return Values{rt.makeValue(
         preparationFormat, ResourceKind::Artifact, bytes,
         [&]() -> std::any {
           auto ptr = s->prepare_rotation(s->context.get(), source->ptr);
           if (!ptr)
             failed("Native rotation preparation failed");
           std::shared_ptr<PreparationLease> owned;
           try {
             owned = std::make_shared<PreparationLease>(s, ptr, r.inputs[0].identity());
           } catch (...) {
             s->preparation_free(ptr);
             throw;
           }
           if (s->preparation_bytes(ptr) != bytes)
             failed("Native preparation reservation changed");
           return owned;
         },
         r.inputs)};
     }});
  runtime.addRecipe({"native.rotate", "native.prepared-rotation", 0, accepts, [s](Runtime &rt, const Request &r) {
                       auto rotation = std::any_cast<std::int64_t>(&r.attributes);
                       if (!rotation || r.inputs.size() != 2)
                         invalid("Prepared rotation requires source and preparation");
                       lease<BufferLease>(r.inputs[0], bufferFormat, s);
                       auto prepared = lease<PreparationLease>(r.inputs[1], preparationFormat, s);
                       if (prepared->source != r.inputs[0].identity())
                         invalid("Preparation belongs to a different source identity");
                       return Values{allocate(rt, s, [&](FhnBuffer *out) {
                         if (s->rotate_prepared(s->context.get(), out, prepared->ptr, *rotation))
                           failed("Native prepared rotation failed");
                       })};
                     }});
  runtime.addRecipe({"native.compile_region", "native.explicit-region", 0, accepts, [s](Runtime &rt, const Request &r) {
                       auto attr = std::any_cast<CompileAttributes>(&r.attributes);
                       if (!attr)
                         invalid("Invalid region attributes");
                       const auto &circuit = attr->circuit;
                       if (!attr->options.max_nodes || circuit.num_nodes > attr->options.max_nodes)
                         throw Error(ErrorCode::Unsupported, "Region exceeds explicit node limit");
                       if (!attr->options.workspace_bytes)
                         invalid("Region workspace reservation must be finite and positive");
                       if (r.inputs.size() != circuit.num_inputs)
                         invalid("Region prototype count mismatch");
                       if (!s->compile || !s->run || !s->executable_free)
                         throw Error(ErrorCode::Unsupported, "Native region compiler unavailable");
                       std::vector<const FhnBuffer *> prototypes;
                       for (const auto &v : r.inputs)
                         prototypes.push_back(lease<BufferLease>(v, bufferFormat, s)->ptr);
                       return Values{rt.makeValue(
                         regionFormat, ResourceKind::Artifact, attr->options.workspace_bytes, [&]() -> std::any {
                           FhnCompileOptions options{sizeof(FhnCompileOptions), attr->options.workspace_bytes};
                           char error[512]{};
                           auto ptr =
                             s->compile(s->context.get(), &circuit, prototypes.data(),
                                        static_cast<std::uint32_t>(prototypes.size()), &options, error, sizeof(error));
                           if (!ptr)
                             failed(std::string("Native region compilation failed: ") + error);
                           std::shared_ptr<RegionLease> owned;
                           try {
                             owned = std::make_shared<RegionLease>(s, ptr, circuit.num_outputs);
                           } catch (...) {
                             s->executable_free(ptr);
                             throw;
                           }
                           return owned;
                         })};
                     }});
  runtime.addRecipe({"native.run_region", "native.compiled-execution", 0, accepts, [s](Runtime &rt, const Request &r) {
                       if (r.inputs.empty())
                         invalid("Region execution requires an executable");
                       auto region = lease<RegionLease>(r.inputs.front(), regionFormat, s);
                       std::vector<const FhnBuffer *> inputs;
                       for (std::size_t i = 1; i < r.inputs.size(); ++i)
                         inputs.push_back(lease<BufferLease>(r.inputs[i], bufferFormat, s)->ptr);
                       Values outputs;
                       std::vector<FhnBuffer *> destinations;
                       for (std::uint32_t i = 0; i < region->outputs; ++i) {
                         outputs.push_back(allocate(rt, s, [](FhnBuffer *) {}));
                         destinations.push_back(lease<BufferLease>(outputs.back(), bufferFormat, s)->ptr);
                       }
                       if (s->run(region->ptr, inputs.data(), static_cast<std::uint32_t>(inputs.size()),
                                  destinations.data(), static_cast<std::uint32_t>(destinations.size())))
                         failed("Native compiled region execution failed");
                       return outputs;
                     }});
}
} // namespace

NativeServices loadNativeServices(const std::string &path, const std::string &config) {
  void *raw = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!raw)
    failed("Could not load native compute library: " + std::string(dlerror()));
  auto library = std::shared_ptr<void>(raw, [](void *p) { dlclose(p); });
  auto abi = symbol<FhnGetAbiVersionFn>(raw, "ckks_fhn_get_abi_version", true);
  if (abi() != FHN_ABI_VERSION)
    invalid("Native ABI version mismatch");
  auto create = symbol<FhnCreateFn>(raw, "ckks_fhn_create", true);
  auto destroy = symbol<FhnDestroyFn>(raw, "ckks_fhn_destroy", true);
  NativeServices s;
  auto ctx = create(config.empty() ? nullptr : config.c_str());
  if (!ctx)
    failed("Could not create native compute context");
  s.context = std::shared_ptr<FhnBackendCtx>(ctx, [library, destroy](FhnBackendCtx *c) { destroy(c); });
  s.get_kernels = symbol<FhnGetKernelsFn>(raw, "ckks_fhn_get_kernels", true);
  s.buffer_alloc = symbol<FhnBufferAllocFn>(raw, "ckks_fhn_buffer_alloc", true);
  s.buffer_free = symbol<FhnBufferFreeFn>(raw, "ckks_fhn_buffer_free", true);
  s.fresh_level = symbol<FhnFreshLevelFn>(raw, "ckks_fhn_fresh_level", true);
  s.level_bytes = symbol<FhnLevelBytesFn>(raw, "ckks_fhn_level_bytes", true);
  s.buffer_info = symbol<FhnCkksBufferInfoFn>(raw, "ckks_fhn_buffer_info", true);
  s.preparation_required_bytes =
    symbol<FhnRotationPreparationRequiredBytesFn>(raw, "ckks_fhn_rotation_preparation_required_bytes");
  s.prepare_rotation = symbol<FhnPrepareRotationSourceFn>(raw, "ckks_fhn_prepare_rotation_source");
  s.preparation_bytes = symbol<FhnRotationPreparationBytesFn>(raw, "ckks_fhn_rotation_preparation_bytes");
  s.rotate_prepared = symbol<FhnRotatePreparedFn>(raw, "ckks_fhn_rotate_prepared");
  s.preparation_free = symbol<FhnRotationPreparationFreeFn>(raw, "ckks_fhn_rotation_preparation_free");
  s.compile = symbol<FhnCompileCircuitFn>(raw, "ckks_fhn_compile_circuit");
  s.run = symbol<FhnRunExecutableFn>(raw, "ckks_fhn_run_executable");
  s.executable_stats = symbol<FhnExecutableStatsFn>(raw, "ckks_fhn_executable_stats");
  s.executable_free = symbol<FhnExecutableFreeFn>(raw, "ckks_fhn_executable_free");
  return s;
}
NativeRuntime::NativeRuntime(Runtime &runtime, NativeServices services)
  : runtime_(runtime), services_(std::make_shared<NativeServices>(std::move(services))) {
  if (!services_->context || !services_->get_kernels || !services_->buffer_alloc || !services_->buffer_free ||
      !services_->fresh_level || !services_->level_bytes || !services_->buffer_info)
    invalid("Incomplete native compute services");
  registerRecipes(runtime_, services_);
}
FhnBackendCtx *NativeRuntime::context() const noexcept { return services_->context.get(); }
Value NativeRuntime::importCopy(const FhnBuffer *source) {
  info(services_, source);
  auto fn = kernel(services_, FHN_ADD_CS);
  return allocate(runtime_, services_, [&](FhnBuffer *out) {
    const FhnBuffer *inputs[]{source};
    std::array<std::int64_t, 4> ip{};
    std::array<double, 2> fp{};
    if (fn(context(), out, inputs, ip.data(), fp.data()))
      failed("Native source import failed");
  });
}
const FhnBuffer *NativeRuntime::buffer(const Value &v) const {
  return lease<BufferLease>(v, bufferFormat, services_)->ptr;
}
Value NativeRuntime::op(FhnOpCode opcode, Values inputs, std::array<std::int64_t, 4> params,
                        std::array<double, 2> fparams) {
  for (const auto &value : inputs)
    buffer(value);
  return runtime_.submit({"native.op", std::move(inputs), OpAttributes{opcode, params, fparams}}).at(0);
}
Value NativeRuntime::prepareRotation(Value source) {
  buffer(source);
  return runtime_.submit({"native.prepare_rotation", {std::move(source)}, {}}).at(0);
}
Value NativeRuntime::rotate(Value source, std::int64_t rotation, Value preparation) {
  buffer(source);
  if (preparation)
    lease<PreparationLease>(preparation, preparationFormat, services_);
  if (!preparation)
    return op(FHN_ROTATE, {std::move(source)}, {rotation, 0, 0, 0});
  return runtime_.submit({"native.rotate", {std::move(source), std::move(preparation)}, rotation}).at(0);
}
Value NativeRuntime::compileRegion(const FhnCircuit &circuit, Values inputs, RegionOptions options) {
  for (const auto &value : inputs)
    buffer(value);
  return runtime_.submit({"native.compile_region", std::move(inputs), CompileAttributes{circuit, options}}).at(0);
}
FhnExecutable *NativeRuntime::executable(const Value &v) const {
  return lease<RegionLease>(v, regionFormat, services_)->ptr;
}
FhnExecutableStats NativeRuntime::executableStats(const Value &v) const {
  auto ptr = executable(v);
  if (!services_->executable_stats)
    throw Error(ErrorCode::Unsupported, "Executable diagnostics unavailable");
  FhnExecutableStats stats{};
  stats.struct_size = sizeof(stats);
  if (services_->executable_stats(ptr, &stats))
    failed("Executable diagnostics failed");
  return stats;
}
Values NativeRuntime::runRegion(Value region, Values inputs) {
  executable(region);
  for (const auto &value : inputs)
    buffer(value);
  inputs.insert(inputs.begin(), std::move(region));
  return runtime_.submit({"native.run_region", std::move(inputs), {}});
}
Value NativeRuntime::semantic(FhnCircuitOp operation, Values inputs, std::int64_t rotation, double scalar) {
  const bool binary =
    operation == FHN_CIRCUIT_ADD || operation == FHN_CIRCUIT_SUBTRACT || operation == FHN_CIRCUIT_MULTIPLY;
  if (inputs.size() != (binary ? 2u : 1u))
    invalid("Wrong semantic operand count");
  // All result paths dispatch physical requests, which validate runtime membership.
  for (const auto &v : inputs)
    buffer(v);
  if (binary) {
    auto a = info(services_, buffer(inputs[0]));
    auto b = info(services_, buffer(inputs[1]));
    const auto level = std::min(a.level, b.level);
    if (a.level != level)
      inputs[0] = op(FHN_LEVEL_DOWN, {inputs[0]}, {level, 0, 0, 0});
    if (b.level != level)
      inputs[1] = op(FHN_LEVEL_DOWN, {inputs[1]}, {level, 0, 0, 0});
  }
  switch (operation) {
  case FHN_CIRCUIT_ADD:
    return op(FHN_ADD_CC, std::move(inputs));
  case FHN_CIRCUIT_SUBTRACT:
    return op(FHN_SUB_CC, std::move(inputs));
  case FHN_CIRCUIT_MULTIPLY:
    return op(FHN_HMULT, std::move(inputs));
  case FHN_CIRCUIT_ADD_CONSTANT:
    return op(FHN_ADD_CS, std::move(inputs), {}, {scalar, 0});
  case FHN_CIRCUIT_NEGATE:
    return op(FHN_NEGATE, std::move(inputs));
  case FHN_CIRCUIT_ROTATE:
    return rotate(inputs[0], rotation);
  case FHN_CIRCUIT_CONJUGATE:
    return op(FHN_CONJUGATE, std::move(inputs));
  case FHN_CIRCUIT_SCALE: {
    if (!std::isfinite(scalar))
      invalid("Nonfinite semantic scalar");
    if (std::trunc(scalar) != scalar)
      return op(FHN_MULT_CS, std::move(inputs), {}, {scalar, 0});
    if (std::abs(scalar) > 9007199254740991.0)
      invalid("Integer scalar exceeds exact double range");
    auto magnitude = static_cast<std::uint64_t>(std::abs(scalar));
    if (!magnitude)
      return op(FHN_SUB_CC, {inputs[0], inputs[0]});
    Value term = inputs[0], sum;
    while (magnitude) {
      if (magnitude & 1u)
        sum = sum ? op(FHN_ADD_CC, {sum, term}) : op(FHN_ADD_CS, {term}, {}, {0, 0});
      magnitude >>= 1u;
      if (magnitude)
        term = op(FHN_ADD_CC, {term, term});
    }
    return scalar < 0 ? op(FHN_NEGATE, {sum}) : sum;
  }
  default:
    invalid("Unknown semantic operation");
  }
}
} // namespace fhenomenon::runtime
