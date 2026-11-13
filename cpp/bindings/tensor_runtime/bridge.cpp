#include "bridge.h"
#include "FHN/Runtime.h"
#include "FHN/Scope.h"
#include "FHN/TensorKernels.h"
#include "backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
using namespace fhenomenon::runtime;
using Json = nlohmann::json;
constexpr const char *format = "fhenomenon.native.ckks";
thread_local std::string last_error, json_output;
void require(bool ok, const char *message) {
  if (!ok)
    throw std::invalid_argument(message);
}
void provider_check(int status) {
  if (status)
    throw Error(ErrorCode::ExecutionFailed, std::string("native CKKS: ") + fn_last_error());
}
Json provider_json(const char *text) {
  if (!text)
    throw Error(ErrorCode::ExecutionFailed, std::string("native CKKS metadata: ") + fn_last_error());
  return Json::parse(text);
}
template <class F> int guarded(F operation) {
  try {
    last_error.clear();
    operation();
    return 0;
  } catch (const std::exception &e) {
    last_error = e.what();
  } catch (...) {
    last_error = "unknown tensor Runtime failure";
  }
  return -1;
}
template <class T> T &handle(void *pointer) {
  require(pointer != nullptr, "closed or null native Runtime handle");
  return *static_cast<T *>(pointer);
}
const char *publish(Json value) {
  json_output = value.dump();
  return json_output.c_str();
}

struct Provider {
  void *evaluator;
  bool owns = false;
  Json profile;
  std::vector<double> scales;
  std::vector<std::uint64_t> moduli;
  std::size_t logical_slots = 0, physical_slots = 0, ciphertext_limb_bytes = 0;
  explicit Provider(void *value) : evaluator(value), profile(provider_json(fn_profile(value))) {
    require(profile.at("name") == "Native128" || profile.at("name") == "NativeTest",
            "tensor Runtime requires a supported native provider profile");
    scales = profile.at("evaluation_scales").get<std::vector<double>>();
    moduli = profile.at("evaluation_moduli").get<std::vector<std::uint64_t>>();
    require(scales.size() == profile.at("top_level").get<std::size_t>() + 1 && moduli.size() == scales.size(),
            "provider scale/modulus chain is incomplete");
    for (auto scale : scales)
      require(std::isfinite(scale) && scale > 0, "invalid provider evaluation scale");
    for (auto modulus : moduli)
      require(modulus > 1, "invalid provider evaluation modulus");
    logical_slots = profile.at("slots").get<std::size_t>();
    physical_slots = profile.at("physical_slots").get<std::size_t>();
    const auto degree = profile.at("ring_degree").get<std::size_t>();
    require(degree >= 2 && !(degree & (degree - 1)) && physical_slots == degree / 2 && logical_slots &&
              !(logical_slots & (logical_slots - 1)) && logical_slots <= physical_slots,
            "invalid provider ring/packing profile");
    require(degree <= std::numeric_limits<std::size_t>::max() / (2 * sizeof(std::uint64_t)),
            "provider ciphertext limb byte count overflow");
    ciphertext_limb_bytes = degree * 2 * sizeof(std::uint64_t);
  }
  ~Provider() {
    if (owns)
      fn_evaluator_free(evaluator);
  }
};
using Metadata = fhenomenon::tensor::CipherMetadata;
void validate_metadata(const Metadata &value, const Provider &provider) {
  require(value.bytes && value.count && value.moduli && value.period && !(value.period & (value.period - 1)) &&
            value.count <= value.period && value.period <= provider.logical_slots &&
            value.period <= provider.physical_slots,
          "ciphertext metadata violates the provider count/period contract");
  require(value.level >= 0 && static_cast<std::uint64_t>(value.level) < provider.scales.size() &&
            value.moduli == static_cast<std::uint64_t>(value.level) + 1,
          "ciphertext metadata violates the provider evaluation modulus chain");
  require(value.moduli <= std::numeric_limits<std::size_t>::max() / provider.ciphertext_limb_bytes &&
            value.bytes == value.moduli * provider.ciphertext_limb_bytes,
          "ciphertext metadata violates the provider polynomial byte layout");
  require(std::isfinite(value.scale) && value.scale > 0, "ciphertext metadata scale must be finite and positive");
  require(value.representation == "slots" || value.representation == "coefficients",
          "ciphertext metadata has an unsupported message representation");
  // A valid multiply can carry a noncanonical arithmetic scale. Profile
  // consistency constrains the chain/layout, not equality to its nominal scale.
}
Metadata metadata(void *ciphertext) {
  auto info = provider_json(fn_ciphertext_info(ciphertext));
  Metadata result{info.at("estimated_bytes"), info.at("count"), info.at("modulus_count"), info.at("period"),
                  info.at("level"),           info.at("scale"), info.at("representation")};
  require(result.bytes && result.count && result.moduli && result.period >= result.count &&
            !(result.period & (result.period - 1)) && result.bytes % result.moduli == 0 &&
            std::isfinite(result.scale) && result.scale > 0,
          "invalid ciphertext payload metadata");
  return result;
}
struct Lease {
  std::shared_ptr<Provider> provider;
  std::shared_ptr<void> ciphertext;
  Metadata info;
};
struct SemanticOrigin {
  std::string name, signature, implementation;
};
struct SemanticMeasurement {
  SemanticOrigin origin;
  std::size_t command_count = 0;
  double seconds = 0;
};
struct Diagnostics {
  std::size_t imports = 0, failed_calls = 0;
  std::size_t scope_calls = 0, completed_scopes = 0, failed_scopes = 0;
  std::string last_scope_status = "none";
  double last_mod_raise_public_bound = 1;
  std::map<std::string, std::size_t> recipe_calls;
  std::map<std::string, std::size_t> kernel_calls;
  std::map<std::string, double> kernel_seconds;
  Json last_kernel_plan = Json::object();
  std::map<std::tuple<std::string, std::string, std::string>, SemanticMeasurement> semantic_measurements;
};
struct Attributes {
  std::int64_t parameter = 0;
  std::vector<double> plain;
  double public_bound = 1;
  std::optional<SemanticOrigin> origin;
};
struct TensorAttributes {
  Json values;
  std::optional<SemanticOrigin> origin;
};
using Clock = std::chrono::steady_clock;
Clock::time_point semantic_start(const std::shared_ptr<Provider> &provider,
                                 const std::optional<SemanticOrigin> &origin) {
  if (!origin)
    return {};
  provider_check(fn_synchronize(provider->evaluator));
  return Clock::now();
}
void semantic_finish(const std::shared_ptr<Provider> &provider, Diagnostics &diagnostics,
                     const std::optional<SemanticOrigin> &origin, Clock::time_point started) {
  if (!origin)
    return;
  provider_check(fn_synchronize(provider->evaluator));
  const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
  auto &measurement = diagnostics.semantic_measurements[{origin->name, origin->signature, origin->implementation}];
  measurement.origin = *origin;
  ++measurement.command_count;
  measurement.seconds += seconds;
}

const Lease &lease(const Value &value, const std::shared_ptr<Provider> &provider = {}) {
  require(value && value.format() == format && value.kind() == ResourceKind::Value,
          "expected a native CKKS Runtime ciphertext Value");
  const auto *stored = std::any_cast<std::shared_ptr<const Lease>>(&value.payload());
  require(stored && *stored && (!provider || (*stored)->provider == provider),
          "ciphertext Value belongs to another provider context");
  return **stored;
}
void *raw(const Value &value, const std::shared_ptr<Provider> &provider = {}) {
  return lease(value, provider).ciphertext.get();
}
Metadata at_level(Metadata input, std::int64_t target) {
  require(input.level >= 0 && target >= 0, "arithmetic level requires slot representation");
  const auto base = static_cast<std::int64_t>(input.moduli) - input.level;
  require(base == 1 && target <= std::numeric_limits<std::int64_t>::max() - base,
          "native level L must contain exactly L+1 moduli");
  const auto width = input.bytes / input.moduli;
  const auto moduli = static_cast<std::uint64_t>(base + target);
  require(moduli <= std::numeric_limits<std::size_t>::max() / width, "ciphertext byte count overflow");
  input.bytes = width * moduli;
  input.moduli = moduli;
  input.level = target;
  return input;
}

bool aligned_scales(double a, double b) {
  return std::isfinite(a) && std::isfinite(b) && a > 0 && b > 0 && std::abs(a - b) <= 1e-12 * std::max(a, b);
}

Value produce(Runtime &runtime, const std::shared_ptr<Provider> &provider, const Metadata &expected,
              const std::function<int(void **)> &operation) {
  validate_metadata(expected, *provider);
  // Native profiles have one evaluation prime per level. Reserve the expected public payload
  // before calling the provider, then verify its actual metadata before publish.
  // These are payload estimates, not allocator capacity or key/workspace RAM.
  return runtime.makeValue(format, ResourceKind::Value, expected.bytes, [&]() -> std::any {
    void *result = nullptr;
    const int status = operation(&result);
    std::shared_ptr<void> owned(result, fn_ciphertext_free);
    provider_check(status);
    require(result != nullptr, "provider returned no ciphertext");
    auto actual = metadata(result);
    validate_metadata(actual, *provider);
    require(actual.bytes == expected.bytes && actual.level == expected.level && actual.moduli == expected.moduli &&
              actual.count == expected.count && actual.period == expected.period &&
              actual.representation == expected.representation && aligned_scales(actual.scale, expected.scale),
            "provider result violates the Runtime reservation/representation contract");
    return std::shared_ptr<const Lease>(std::make_shared<Lease>(Lease{provider, std::move(owned), actual}));
  });
}

// Metadata is public planning information. It carries the evaluator identity,
// never a ciphertext or client secret, and is checked by the generic Scope core.
struct DescriptorMetadata {
  std::shared_ptr<Provider> provider;
  Metadata info;
};
bool same_metadata(const Metadata &a, const Metadata &b) {
  return a.bytes == b.bytes && a.count == b.count && a.moduli == b.moduli && a.period == b.period &&
         a.level == b.level && a.representation == b.representation && aligned_scales(a.scale, b.scale);
}
ScopeDescriptor descriptor(const std::shared_ptr<Provider> &provider, const Metadata &info) {
  validate_metadata(info, *provider);
  return {format, ResourceKind::Value, info.bytes, DescriptorMetadata{provider, info}};
}
const Metadata &described(const ScopeDescriptor &value, const std::shared_ptr<Provider> &provider) {
  const auto *data = std::any_cast<DescriptorMetadata>(&value.metadata);
  require(value.format == format && value.kind == ResourceKind::Value && data && data->provider == provider &&
            value.bytes == data->info.bytes,
          "scope descriptor belongs to another provider or ciphertext format");
  validate_metadata(data->info, *provider);
  return data->info;
}
ScopeDescriptor describe(const Value &value, const std::shared_ptr<Provider> &provider) {
  return descriptor(provider, lease(value, provider).info);
}
ScopeDescriptor infer(const std::shared_ptr<Provider> &provider, const std::string &operation,
                      const std::vector<ScopeDescriptor> &inputs, const std::any &raw_attributes) {
  const bool binary =
    operation == "native.ckks.add" || operation == "native.ckks.sub" || operation == "native.ckks.mul";
  const bool plain = operation == "native.ckks.add_plain" || operation == "native.ckks.mul_plain";
  require(binary || plain || operation == "native.ckks.neg" || operation == "native.ckks.rotate" ||
            operation == "native.ckks.level_down" || operation == "native.ckks.mod_raise" ||
            operation == "native.ckks.slot_to_coeff" || operation == "native.ckks.coeff_to_slot",
          "unknown native scope operation");
  require(inputs.size() == (binary ? 2u : 1u), "invalid native scope operation arity");
  const auto *attributes = std::any_cast<Attributes>(&raw_attributes);
  require(attributes != nullptr, "invalid native scope operation attributes");
  require(std::isfinite(attributes->public_bound) && attributes->public_bound >= 1,
          "public bound must be finite and at least one");
  const auto &a = described(inputs.front(), provider);
  auto expected = a;
  require(a.level >= 0 && a.level <= provider->profile.at("top_level").get<std::int64_t>(),
          "ciphertext level is outside the provider evaluation chain");
  if (operation != "native.ckks.coeff_to_slot")
    require(a.representation == "slots", "slot arithmetic cannot consume coefficient messages");
  if (binary) {
    const auto &b = described(inputs[1], provider);
    require(b.representation == "slots", "slot arithmetic cannot consume coefficient messages");
    require((a.count == b.count && a.period == b.period) || a.period == 1 || b.period == 1,
            "ciphertext shape mismatch");
    require(a.bytes / a.moduli == b.bytes / b.moduli, "ciphertext polynomial layout mismatch");
    expected = at_level(expected, std::min(a.level, b.level));
    expected.count = std::max(a.count, b.count);
    expected.period = std::max(a.period, b.period);
    // Native levelDown normalizes only when the level actually changes.
    const double left_scale = a.level == expected.level ? a.scale : provider->scales.at(expected.level);
    const double right_scale = b.level == expected.level ? b.scale : provider->scales.at(expected.level);
    if (operation == "native.ckks.mul")
      expected.scale = left_scale * right_scale / static_cast<double>(provider->moduli.at(expected.level));
    else {
      require(aligned_scales(left_scale, right_scale), "add/sub requires arithmetically aligned scales");
      expected.scale = left_scale;
    }
  }
  if (plain) {
    require(attributes->plain.size() == 1 || attributes->plain.size() == a.count, "incompatible plaintext length");
    for (auto value : attributes->plain)
      require(std::isfinite(value), "nonfinite plaintext");
  } else
    require(attributes->plain.empty(), "unexpected plaintext attributes");
  if (operation == "native.ckks.mul" || operation == "native.ckks.mul_plain") {
    require(expected.level > 0, "multiplication requires ModRaise at level zero");
    expected = at_level(expected, expected.level - 1);
    if (operation == "native.ckks.mul_plain")
      expected.scale = provider->scales.at(expected.level);
  } else if (operation == "native.ckks.level_down" || operation == "native.ckks.mod_raise") {
    require(attributes->parameter >= 0 &&
              attributes->parameter <= provider->profile.at("top_level").get<std::int64_t>(),
            "requested level is outside the provider evaluation chain");
    if (operation == "native.ckks.level_down")
      require(attributes->parameter <= expected.level, "level-down cannot increase levels");
    else {
      const auto &targets = provider->profile.at("mod_raise_targets");
      require(std::find(targets.begin(), targets.end(), Json(attributes->parameter)) != targets.end(),
              "target refresh was not declared during client setup");
      require(attributes->public_bound <= provider->profile.at("refresh_maximum_public_bound").get<double>(),
              "public bound exceeds the prepared refresh range");
    }
    if (operation == "native.ckks.mod_raise" || attributes->parameter != expected.level)
      expected.scale = provider->scales.at(attributes->parameter);
    expected = at_level(expected, attributes->parameter);
  } else if (operation == "native.ckks.rotate") {
    require(attributes->parameter >= std::numeric_limits<int32_t>::min() &&
              attributes->parameter <= std::numeric_limits<int32_t>::max(),
            "rotation is outside int32 range");
    require(provider->profile.value("rotation_key_basis", "") == "all_positive_powers_of_two",
            "provider has not declared a complete rotation key basis");
    const auto slots = provider->profile.at("physical_slots").get<std::size_t>();
    require(slots && !(slots & (slots - 1)) && a.period <= slots,
            "rotation exceeds the provider slot layout capability");
  } else if (operation == "native.ckks.slot_to_coeff") {
    require(expected.level > 0, "slot-to-coefficient requires one remaining level");
    expected = at_level(expected, expected.level - 1);
    expected.representation = "coefficients";
    expected.scale = provider->scales.at(expected.level);
  } else if (operation == "native.ckks.coeff_to_slot") {
    require(a.representation == "coefficients", "coefficient-to-slot requires coefficient messages");
    require(expected.level > 0, "coefficient-to-slot requires one remaining level");
    expected = at_level(expected, expected.level - 1);
    expected.representation = "slots";
    expected.scale = provider->scales.at(expected.level);
  }
  require(std::isfinite(expected.scale) && expected.scale > 0, "inferred CKKS scale is not finite and positive");
  return descriptor(provider, expected);
}

struct CaptureState;
struct CipherState {
  Value value;
  ScopeValue symbol;
  std::weak_ptr<CaptureState> capture;
};
struct CipherHandle {
  std::shared_ptr<CipherState> state;
};
struct CaptureState {
  Scope *scope = nullptr;
  std::size_t max_exports = 0;
  bool failed = false;
  std::vector<std::shared_ptr<CipherState>> exports;
  ScopeValues export_symbols;
  std::map<std::tuple<std::string, std::string, std::string>, SemanticOrigin> completed_origins;
};
struct Context {
  std::shared_ptr<Provider> provider;
  std::shared_ptr<Diagnostics> diagnostics = std::make_shared<Diagnostics>();
  Runtime runtime;
  std::shared_ptr<const ScopeContract> contract;
  std::shared_ptr<CaptureState> active;
  std::optional<SemanticOrigin> semantic_origin;
  std::map<std::string, fhenomenon::tensor::KernelDefinition> tensor_kernels;
  std::size_t budget;
  std::int64_t evaluation_base_moduli = 0;
  Context(void *evaluator, std::size_t maximum)
    : provider(std::make_shared<Provider>(evaluator)), runtime({maximum, 96}), budget(maximum) {
    auto p = provider;
    for (auto definition : fhenomenon::tensor::tensorKernelDefinitions()) {
      const auto operation = definition.operation;
      tensor_kernels.emplace(operation, std::move(definition));
    }
    auto scope_contract = std::make_shared<ScopeContract>();
    scope_contract->describe = [p](const Value &value) { return describe(value, p); };
    scope_contract->infer = [this, p](const std::string &operation, const std::vector<ScopeDescriptor> &inputs,
                                      const std::any &attributes) {
      if (tensor_kernels.count(operation)) {
        auto plan = inferKernel(operation, inputs, attributes);
        std::vector<ScopeDescriptor> outputs;
        for (const auto &info : plan.outputs)
          outputs.push_back(descriptor(p, info));
        return outputs;
      }
      return std::vector<ScopeDescriptor>{infer(p, operation, inputs, attributes)};
    };
    scope_contract->validate = [p](const Value &value, const ScopeDescriptor &expected) {
      require(same_metadata(lease(value, p).info, described(expected, p)),
              "provider output violates the recorded scope descriptor");
    };
    contract = std::move(scope_contract);
    for (const auto &entry : tensor_kernels) {
      const auto operation = entry.first;
      runtime.addRecipe({operation, operation + ".native", 0,
                         [this, operation](const Request &request, std::string &reason) {
                           try {
                             std::vector<ScopeDescriptor> inputs;
                             for (const auto &value : request.inputs)
                               inputs.push_back(describe(value, provider));
                             (void)inferKernel(operation, inputs, request.attributes);
                             return true;
                           } catch (const std::exception &error) {
                             reason = error.what();
                             return false;
                           }
                         },
                         [this, operation](Runtime &rt, const Request &request) {
                           return executeKernel(rt, operation, request.inputs, request.attributes);
                         }});
      runtime.addScopeKernel(
        {operation + ".native", 0, 1,
         [operation](const ScopeView &view, std::string &) {
           return view.size() && view.command(0).operation == operation ? std::size_t{1} : std::size_t{0};
         },
         [this, operation](Runtime &rt, const ScopeView &view) {
           const auto &command = view.command(0);
           Values inputs;
           for (const auto symbol : command.inputs)
             inputs.push_back(view.value(symbol));
           auto outputs = executeKernel(rt, operation, inputs, command.attributes);
           require(outputs.size() == command.outputs.size(), "tensor kernel produced the wrong output count");
           Values live_outputs;
           for (const auto symbol : view.liveOuts(1)) {
             const auto found = std::find(command.outputs.begin(), command.outputs.end(), symbol);
             require(found != command.outputs.end(), "tensor kernel has an unexpected live-out");
             live_outputs.push_back(outputs.at(static_cast<std::size_t>(found - command.outputs.begin())));
           }
           return live_outputs;
         }});
    }
    for (const auto &name : {"add", "sub", "mul", "add_plain", "mul_plain", "neg", "rotate", "level_down", "mod_raise",
                             "slot_to_coeff", "coeff_to_slot"}) {
      const std::string operation = std::string("native.ckks.") + name;
      auto d = diagnostics;
      runtime.addRecipe(
        {operation, operation + ".native", 0,
         [p, operation](const Request &request, std::string &reason) {
           try {
             std::vector<ScopeDescriptor> inputs;
             for (const auto &value : request.inputs)
               inputs.push_back(describe(value, p));
             (void)infer(p, operation, inputs, request.attributes);
             return true;
           } catch (const std::exception &e) {
             reason = e.what();
             return false;
           }
         },
         [p, d, operation](Runtime &rt, const Request &request) {
           const auto &attributes = std::any_cast<const Attributes &>(request.attributes);
           const auto semantic_started = semantic_start(p, attributes.origin);
           std::vector<ScopeDescriptor> inputs;
           for (const auto &value : request.inputs)
             inputs.push_back(describe(value, p));
           const auto expected_descriptor = infer(p, operation, inputs, request.attributes);
           const auto expected = described(expected_descriptor, p);
           const auto &a = lease(request.inputs.at(0), p);
           auto value = produce(rt, p, expected, [&](void **out) {
             void *input = a.ciphertext.get();
             if (request.inputs.size() == 2) {
               const int opcode = operation == "native.ckks.add" ? 0 : operation == "native.ckks.sub" ? 1 : 2;
               return fn_binary(p->evaluator, opcode, input, raw(request.inputs[1], p), out);
             }
             if (operation == "native.ckks.add_plain" || operation == "native.ckks.mul_plain")
               return fn_plain(p->evaluator, operation == "native.ckks.add_plain" ? 0 : 2, input,
                               attributes.plain.data(), attributes.plain.size(), out);
             if (operation == "native.ckks.neg")
               return fn_neg(p->evaluator, input, out);
             if (operation == "native.ckks.rotate")
               return fn_rotate(p->evaluator, input, attributes.parameter, out);
             if (operation == "native.ckks.level_down")
               return fn_level_down(p->evaluator, input, attributes.parameter, out);
             if (operation == "native.ckks.mod_raise")
               return fn_mod_raise_bounded(p->evaluator, input, attributes.parameter, attributes.public_bound, out);
             if (operation == "native.ckks.slot_to_coeff")
               return fn_slot_to_coeff(p->evaluator, input, out);
             return fn_coeff_to_slot(p->evaluator, input, out);
           });
           ++d->recipe_calls[operation];
           if (operation == "native.ckks.mod_raise")
             d->last_mod_raise_public_bound = attributes.public_bound;
           semantic_finish(p, *d, attributes.origin, semantic_started);
           return Values{std::move(value)};
         }});
    }
  }
  fhenomenon::tensor::KernelPlan inferKernel(const std::string &operation, const std::vector<ScopeDescriptor> &inputs,
                                             const std::any &raw_attributes) const {
    const auto found = tensor_kernels.find(operation);
    require(found != tensor_kernels.end(), "unknown tensor kernel operation");
    const auto *attributes = std::any_cast<TensorAttributes>(&raw_attributes);
    require(attributes && attributes->values.is_object(), "tensor kernel attributes must be a JSON object");
    require(inputs.size() == found->second.contract.at("input_arity").get<std::size_t>(),
            "tensor kernel input arity violates its contract");
    std::vector<Metadata> metadata_inputs;
    for (const auto &value : inputs)
      metadata_inputs.push_back(described(value, provider));
    auto plan =
      found->second.infer({provider->profile, provider->scales, provider->moduli}, metadata_inputs, attributes->values);
    require(plan.outputs.size() == found->second.contract.at("output_arity").get<std::size_t>(),
            "tensor kernel inference violates its declared output arity");
    for (const auto &output : plan.outputs)
      validate_metadata(output, *provider);
    require(plan.details.is_object(), "tensor kernel planning details must be a JSON object");
    return plan;
  }
  Values executeKernel(Runtime &rt, const std::string &operation, const Values &inputs, const std::any &attributes) {
    const auto &tensor_attributes = std::any_cast<const TensorAttributes &>(attributes);
    const auto semantic_started = semantic_start(provider, tensor_attributes.origin);
    std::vector<ScopeDescriptor> descriptors;
    for (const auto &value : inputs)
      descriptors.push_back(describe(value, provider));
    const auto plan = inferKernel(operation, descriptors, attributes);
    // Charge intermediate ciphertext payload before the provider can allocate
    // it. This is not a claim to account for provider keys/internal scratch.
    auto workspace = rt.makeValue("tensor.kernel.temporary_ciphertext_payload", ResourceKind::Artifact,
                                  plan.temporary_ciphertext_bytes, [] { return std::any{}; });
    auto public_workspace = rt.makeValue("tensor.kernel.temporary_public_payload", ResourceKind::Artifact,
                                         plan.temporary_public_bytes, [] { return std::any{}; });
    fhenomenon::tensor::KernelExecution execution{
      rt, provider->evaluator, [this](const Value &value) { return raw(value, provider); },
      [this, &rt](const Metadata &expected, const std::function<int(void **)> &call) {
        return produce(rt, provider, expected, call);
      }};
    provider_check(fn_synchronize(provider->evaluator));
    const auto started = std::chrono::steady_clock::now();
    auto outputs = tensor_kernels.at(operation).execute(execution, inputs, tensor_attributes.values, plan);
    require(outputs.size() == plan.outputs.size(), "tensor kernel produced the wrong output count");
    for (std::size_t index = 0; index < outputs.size(); ++index) {
      require(rt.owns(outputs[index]) && same_metadata(lease(outputs[index], provider).info, plan.outputs[index]),
              "tensor kernel output violates its inferred contract");
    }
    provider_check(fn_synchronize(provider->evaluator));
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    ++diagnostics->kernel_calls[operation];
    diagnostics->kernel_seconds[operation] += seconds;
    diagnostics->last_kernel_plan = plan.details;
    diagnostics->last_kernel_plan["operation"] = operation;
    diagnostics->last_kernel_plan["temporary_ciphertext_payload_bytes"] = plan.temporary_ciphertext_bytes;
    diagnostics->last_kernel_plan["temporary_public_payload_bytes"] = plan.temporary_public_bytes;
    semantic_finish(provider, *diagnostics, tensor_attributes.origin, semantic_started);
    return outputs;
  }
};
std::shared_ptr<CipherState> state(void *pointer) {
  auto result = handle<CipherHandle>(pointer).state;
  require(bool(result), "closed or null native Runtime ciphertext state");
  return result;
}
const Value &concrete(const CipherState &state) {
  if (!state.value) {
    const auto capture = state.capture.lock();
    require(capture && capture->scope, "scope ciphertext was not exported or scope failed");
    throw std::invalid_argument("scope ciphertext is unresolved");
  }
  return state.value;
}
Value input(Context &context, const std::shared_ptr<CipherState> &state) {
  const auto &value = concrete(*state);
  require(context.runtime.owns(value), "ciphertext belongs to another Fhenomenon Runtime");
  (void)lease(value, context.provider);
  return value;
}
ScopeValue symbol(Context &context, const std::shared_ptr<CipherState> &state) {
  require(context.active && context.active->scope, "no active scope capture");
  if (state->value)
    return context.active->scope->input(input(context, state));
  require(state->capture.lock() == context.active, "scope ciphertext belongs to another scope or was not exported");
  (void)context.active->scope->descriptor(state->symbol);
  return state->symbol;
}
void submit(Context &context, const std::string &name, std::vector<std::shared_ptr<CipherState>> inputs,
            Attributes attributes, void **out) {
  require(out != nullptr, "missing output Value");
  *out = nullptr;
  attributes.origin = context.semantic_origin;
  auto result = std::make_shared<CipherState>();
  if (context.active) {
    require(context.active->scope != nullptr, "scope capture is no longer active");
    ScopeValues symbols;
    for (const auto &value : inputs)
      symbols.push_back(symbol(context, value));
    auto values = context.active->scope->submit("native.ckks." + name, std::move(symbols), std::move(attributes));
    require(values.size() == 1, "private scope operation produced the wrong symbol count");
    result->symbol = values.front();
    result->capture = context.active;
  } else {
    Values values;
    for (const auto &value : inputs)
      values.push_back(input(context, value));
    auto outputs = context.runtime.submit({"native.ckks." + name, std::move(values), std::move(attributes)});
    require(outputs.size() == 1, "tensor Runtime recipe produced the wrong result count");
    result->value = std::move(outputs.front());
  }
  auto output = std::make_unique<CipherHandle>(CipherHandle{std::move(result)});
  *out = output.release();
}
void submit_kernel(Context &context, const std::string &operation,
                   const std::vector<std::shared_ptr<CipherState>> &inputs, Json attributes, void **outputs,
                   std::size_t capacity, std::size_t *output_count) {
  const auto definition = context.tensor_kernels.find(operation);
  require(definition != context.tensor_kernels.end(), "unknown tensor kernel operation");
  const auto count = definition->second.contract.at("output_arity").get<std::size_t>();
  require(capacity >= count && (outputs || !count), "insufficient tensor kernel output capacity");
  std::vector<std::unique_ptr<CipherHandle>> handles;
  handles.reserve(count);
  for (std::size_t index = 0; index < count; ++index)
    handles.push_back(std::make_unique<CipherHandle>(CipherHandle{std::make_shared<CipherState>()}));
  TensorAttributes envelope{std::move(attributes), context.semantic_origin};
  if (context.active) {
    require(context.active->scope != nullptr, "scope capture is no longer active");
    ScopeValues symbols;
    for (const auto &value : inputs)
      symbols.push_back(symbol(context, value));
    auto result = context.active->scope->submit(operation, std::move(symbols), std::move(envelope), count);
    require(result.size() == count, "tensor kernel produced the wrong symbol count");
    for (std::size_t index = 0; index < count; ++index) {
      handles[index]->state->symbol = result[index];
      handles[index]->state->capture = context.active;
    }
  } else {
    Values values;
    for (const auto &value : inputs)
      values.push_back(input(context, value));
    auto result = context.runtime.submit({operation, std::move(values), std::move(envelope)});
    require(result.size() == count, "tensor kernel produced the wrong result count");
    for (std::size_t index = 0; index < count; ++index)
      handles[index]->state->value = std::move(result[index]);
  }
  for (std::size_t index = 0; index < count; ++index)
    outputs[index] = handles[index].release();
  *output_count = count;
}
template <class F> int context_call(void *pointer, F operation) {
  return guarded([&] {
    auto &context = handle<Context>(pointer);
    try {
      operation(context);
    } catch (...) {
      ++context.diagnostics->failed_calls;
      if (context.active)
        context.active->failed = true;
      throw;
    }
  });
}
} // namespace

extern "C" {
uint32_t fr_get_abi_version(void) { return FHN_TENSOR_RUNTIME_ABI_VERSION; }
const char *fr_last_error() { return last_error.c_str(); }
const void *fr_provider_library_identity() { return fn_library_identity(); }
int fr_create(void *evaluator, size_t budget, void **out) {
  return guarded([&] {
    require(fn_get_abi_version() == FHN_TENSOR_PROVIDER_ABI_VERSION, "incompatible tensor provider ABI");
    require(evaluator && out, "evaluator and Runtime output are required");
    auto context = std::make_unique<Context>(evaluator, budget);
    context->provider->owns = true;
    *out = context.release();
  });
}
void fr_runtime_free(void *pointer) { delete static_cast<Context *>(pointer); }
void fr_value_free(void *pointer) { delete static_cast<CipherHandle *>(pointer); }
int fr_import(void *pointer, void *ciphertext, void **out) {
  return context_call(pointer, [&](Context &context) {
    require(!context.active, "ciphertext import is forbidden during scope capture");
    require(out && ciphertext, "ciphertext and import output are required");
    *out = nullptr;
    auto info = metadata(ciphertext);
    require(info.level >= 0 && info.representation == "slots", "only client-encrypted slot messages may be imported");
    const auto base = static_cast<std::int64_t>(info.moduli) - info.level;
    require(base == 1 && (!context.evaluation_base_moduli || base == context.evaluation_base_moduli),
            "incompatible evaluation modulus chain");
    auto value = produce(context.runtime, context.provider, info, [&](void **result) {
      return fn_level_down(context.provider->evaluator, ciphertext, info.level, result);
    });
    auto owned = std::make_shared<CipherState>();
    owned->value = std::move(value);
    auto result = std::make_unique<CipherHandle>(CipherHandle{std::move(owned)});
    context.evaluation_base_moduli = base;
    ++context.diagnostics->imports;
    *out = result.release();
  });
}
void *fr_value_borrow(void *pointer) {
  void *result = nullptr;
  if (guarded([&] { result = raw(concrete(*state(pointer))); }))
    return nullptr;
  return result;
}
const char *fr_value_info(void *pointer) {
  const char *result = nullptr;
  if (guarded([&] {
        const auto owner = state(pointer);
        if (owner->value) {
          const auto &value = owner->value;
          auto info = provider_json(fn_ciphertext_info(raw(value)));
          info["runtime_identity"] = value.identity();
          info["runtime_bytes"] = value.bytes();
          info["runtime_format"] = value.format();
          info["symbolic"] = false;
          result = publish(std::move(info));
        } else {
          const auto capture = owner->capture.lock();
          require(capture && capture->scope, "scope ciphertext was not exported or scope failed");
          const auto &value = capture->scope->descriptor(owner->symbol);
          const auto &data = std::any_cast<const DescriptorMetadata &>(value.metadata);
          const auto &info = data.info;
          result = publish({{"count", info.count},
                            {"period", info.period},
                            {"level", info.level},
                            {"modulus_count", info.moduli},
                            {"scale", info.scale},
                            {"estimated_bytes", value.bytes},
                            {"representation", info.representation},
                            {"device", data.provider->profile.at("device")},
                            {"context_id", data.provider->profile.at("context_id")},
                            {"runtime_identity", nullptr},
                            {"runtime_bytes", value.bytes},
                            {"runtime_format", value.format},
                            {"symbolic", true},
                            {"scope_symbol", owner->symbol.id()}});
        }
      }))
    return nullptr;
  return result;
}
int fr_run_scope(void *pointer, int (*capture)(void *), void *userdata, size_t max_commands, size_t max_values,
                 size_t max_input_edges, size_t max_exports, size_t max_duration_ms) {
  return context_call(pointer, [&](Context &context) {
    require(capture != nullptr, "scope capture callback is required");
    require(!context.active, "nested scope capture is forbidden");
    require(context.tensor_kernels.size() <=
              std::numeric_limits<std::size_t>::max() / std::max(max_commands, std::size_t{1}),
            "tensor kernel matcher budget overflow");
    auto active = std::make_shared<CaptureState>();
    active->max_exports = max_exports;
    context.active = active;
    context.diagnostics->semantic_measurements.clear();
    ++context.diagnostics->scope_calls;
    context.diagnostics->last_scope_status = "capturing";
    ScopeOptions options;
    options.max_commands = max_commands;
    options.max_values = max_values;
    options.max_input_edges = max_input_edges;
    options.max_exports = max_exports;
    options.max_duration_ms = max_duration_ms;
    options.contract = context.contract;
    // Every semantic contribution matches at most one command. Reserve one
    // attempt per registered kernel per command, including primitive fallback.
    options.max_match_attempts = std::max(std::size_t{1}, max_commands * context.tensor_kernels.size());
    try {
      auto outputs = context.runtime.runScope(
        [&](Scope &scope) {
          active->scope = &scope;
          const int status = capture(userdata);
          active->scope = nullptr;
          require(status == 0 && !active->failed, "scope capture callback failed");
          context.diagnostics->last_scope_status = "executing";
          return active->export_symbols;
        },
        std::move(options));
      require(outputs.size() == active->exports.size(), "scope returned the wrong export count");
      for (const auto &entry : active->completed_origins) {
        auto &measurement = context.diagnostics->semantic_measurements[entry.first];
        measurement.origin = entry.second;
      }
      // No output is externally visible until the entire execution and contract
      // validation succeeds. Assigning Values and resetting weak refs cannot throw.
      for (std::size_t i = 0; i < outputs.size(); ++i) {
        active->exports[i]->value = std::move(outputs[i]);
        active->exports[i]->capture.reset();
      }
      ++context.diagnostics->completed_scopes;
      context.diagnostics->last_scope_status = "completed";
      context.active.reset();
    } catch (...) {
      active->scope = nullptr;
      ++context.diagnostics->failed_scopes;
      context.diagnostics->last_scope_status = "failed";
      context.active.reset();
      throw;
    }
  });
}
int fr_scope_export(void *pointer, void *value) {
  return context_call(pointer, [&](Context &context) {
    require(context.active && context.active->scope, "scope export requires active capture");
    require(context.active->exports.size() < context.active->max_exports, "scope export cap exceeded");
    auto output = state(value);
    const auto exported = symbol(context, output);
    context.active->exports.push_back(std::move(output));
    context.active->export_symbols.push_back(exported);
  });
}
int fr_binary(void *pointer, int opcode, void *a, void *b, void **out) {
  return context_call(pointer, [&](Context &context) {
    require(opcode >= 0 && opcode <= 2, "unknown binary operation");
    submit(context, opcode == 0 ? "add" : opcode == 1 ? "sub" : "mul", {state(a), state(b)}, {}, out);
  });
}
int fr_plain(void *pointer, int opcode, void *a, const double *plain, size_t count, void **out) {
  return context_call(pointer, [&](Context &context) {
    require(opcode == 0 || opcode == 2, "unknown plaintext operation");
    require(plain && count && count <= context.provider->profile.at("slots").get<std::size_t>(),
            "invalid plaintext vector");
    Attributes attributes;
    attributes.plain.assign(plain, plain + count);
    for (auto value : attributes.plain)
      require(std::isfinite(value), "nonfinite plaintext");
    submit(context, opcode == 0 ? "add_plain" : "mul_plain", {state(a)}, std::move(attributes), out);
  });
}
int fr_unary(void *pointer, int opcode, void *a, int64_t parameter, void **out) {
  return context_call(pointer, [&](Context &context) {
    const char *names[]{"neg", "rotate", "level_down", "mod_raise", "slot_to_coeff", "coeff_to_slot"};
    require(opcode >= 0 && opcode <= 5, "unknown unary operation");
    if (opcode == 1)
      require(parameter >= std::numeric_limits<int32_t>::min() && parameter <= std::numeric_limits<int32_t>::max(),
              "rotation is outside int32 range");
    Attributes attributes;
    attributes.parameter = parameter;
    submit(context, names[opcode], {state(a)}, std::move(attributes), out);
  });
}
int fr_mod_raise_bounded(void *pointer, void *a, uint32_t target, double public_bound, void **out) {
  return context_call(pointer, [&](Context &context) {
    require(std::isfinite(public_bound) && public_bound >= 1, "public bound must be finite and at least one");
    Attributes attributes;
    attributes.parameter = target;
    attributes.public_bound = public_bound;
    submit(context, "mod_raise", {state(a)}, std::move(attributes), out);
  });
}
const char *fr_kernel_catalog(void *pointer) {
  const char *result = nullptr;
  if (guarded([&] {
        const auto &context = handle<Context>(pointer);
        Json definitions = Json::array();
        for (const auto &entry : context.tensor_kernels) {
          auto contract = entry.second.contract;
          contract["operation"] = entry.first;
          contract["device"] = context.provider->profile.at("device");
          definitions.push_back(std::move(contract));
        }
        result = publish({{"schema_version", 1}, {"kernels", std::move(definitions)}});
      }))
    return nullptr;
  return result;
}
int fr_set_semantic_origin(void *pointer, const char *name, const char *signature, const char *implementation) {
  return context_call(pointer, [&](Context &context) {
    require(!context.active || context.active->scope, "cannot change semantic origin during scope execution");
    if (!name && !signature && !implementation) {
      context.semantic_origin.reset();
      return;
    }
    require(name && signature && implementation && *name && *signature && *implementation,
            "semantic origin requires nonempty name, signature and implementation");
    context.semantic_origin = SemanticOrigin{name, signature, implementation};
  });
}
int fr_complete_semantic_origin(void *pointer) {
  return context_call(pointer, [&](Context &context) {
    require(context.semantic_origin.has_value(), "no active semantic origin to complete");
    require(!context.active || context.active->scope, "cannot complete semantic origin during scope execution");
    const auto &origin = *context.semantic_origin;
    const auto key = std::make_tuple(origin.name, origin.signature, origin.implementation);
    if (context.active)
      context.active->completed_origins[key] = origin;
    else
      context.diagnostics->semantic_measurements[key].origin = origin;
  });
}
int fr_tensor_kernel(void *pointer, const char *operation, void *const *inputs, size_t input_count,
                     const char *attributes_json, void **outputs, size_t output_capacity, size_t *output_count) {
  if (output_count)
    *output_count = 0;
  if (outputs)
    for (std::size_t index = 0; index < output_capacity; ++index)
      outputs[index] = nullptr;
  return context_call(pointer, [&](Context &context) {
    require(operation && attributes_json && output_count && (inputs || !input_count),
            "tensor kernel operation, inputs, attributes and output count are required");
    const auto found = context.tensor_kernels.find(operation);
    require(found != context.tensor_kernels.end(), "unknown tensor kernel operation");
    require(input_count == found->second.contract.at("input_arity").get<std::size_t>(),
            "tensor kernel input arity violates its contract");
    auto attributes = Json::parse(attributes_json);
    require(attributes.is_object(), "tensor kernel attributes must be a JSON object");
    std::vector<std::shared_ptr<CipherState>> states;
    for (std::size_t index = 0; index < input_count; ++index)
      states.push_back(state(inputs[index]));
    submit_kernel(context, operation, states, std::move(attributes), outputs, output_capacity, output_count);
  });
}
const char *fr_profile(void *pointer) {
  const char *result = nullptr;
  if (guarded([&] {
        auto info = handle<Context>(pointer).provider->profile;
        info["execution_runtime"] = "Fhenomenon";
        info["runtime_value_format"] = format;
        info["evaluation_top_level"] = info.at("top_level");
        result = publish(std::move(info));
      }))
    return nullptr;
  return result;
}
const char *fr_provider_stats(void *pointer) {
  const char *result = nullptr;
  if (guarded([&] { result = publish(provider_json(fn_stats(handle<Context>(pointer).provider->evaluator))); }))
    return nullptr;
  return result;
}
const char *fr_runtime_stats(void *pointer) {
  const char *result = nullptr;
  if (guarded([&] {
        const auto &context = handle<Context>(pointer);
        const auto stats = context.runtime.stats();
        const auto scope = context.runtime.scopeStats();
        const auto &diagnostics = *context.diagnostics;
        Json trace = Json::array();
        for (const auto &event : context.runtime.trace())
          trace.push_back(
            {{"kind", static_cast<int>(event.kind)}, {"operation", event.operation}, {"recipe", event.recipe}});
        Json measurements = Json::array();
        for (const auto &entry : diagnostics.semantic_measurements) {
          const auto &value = entry.second;
          measurements.push_back({{"name", value.origin.name},
                                  {"signature", value.origin.signature},
                                  {"current_implementation", value.origin.implementation},
                                  {"seconds", value.seconds},
                                  {"command_count", value.command_count},
                                  {"timing_scope", "exclusive_synchronized_wall"}});
        }
        result = publish(
          {{"live_bytes", stats.live_bytes},
           {"peak_live_bytes", stats.peak_live_bytes},
           {"live_resources", stats.live_resources},
           {"max_live_bytes", context.budget},
           {"imports", diagnostics.imports},
           {"failed_calls", diagnostics.failed_calls},
           {"recipe_calls", diagnostics.recipe_calls},
           {"trace", trace},
           {"kernel_calls", diagnostics.kernel_calls},
           {"kernel_seconds", diagnostics.kernel_seconds},
           {"last_kernel_plan", diagnostics.last_kernel_plan},
           {"semantic_measurements", std::move(measurements)},
           {"kernel_timing_scope",
            "synchronized host/provider elapsed seconds; includes native launch and output validation overhead"},
           {"last_mod_raise_public_bound", diagnostics.last_mod_raise_public_bound},
           {"scope_calls", diagnostics.scope_calls},
           {"completed_scopes", diagnostics.completed_scopes},
           {"failed_scopes", diagnostics.failed_scopes},
           {"last_scope",
            {{"commands", scope.commands},
             {"values", scope.values},
             {"imports", scope.imports},
             {"input_edges", scope.input_edges},
             {"exports", scope.exports},
             {"executed_commands", scope.executed_commands},
             {"recipe_spans", scope.recipe_spans},
             {"kernel_spans", scope.kernel_spans},
             {"status", diagnostics.last_scope_status}}},
           {"bytes_scope",
            "provider estimated retained ciphertext payload plus semantic-kernel temporary ciphertext/public-buffer "
            "reservations; excludes allocator/container overhead, keys, provider internal scratch/cache, captured "
            "attributes, setup and client import temporaries"}});
      }))
    return nullptr;
  return result;
}
const char *fr_refresh_plan(void *pointer, uint32_t target) {
  const char *result = nullptr;
  if (guarded([&] {
        result = publish(provider_json(fn_refresh_plan(handle<Context>(pointer).provider->evaluator, target)));
      }))
    return nullptr;
  return result;
}
int fr_synchronize(void *pointer) {
  return context_call(pointer, [&](Context &context) {
    require(!context.active, "provider synchronization is forbidden during scope capture");
    provider_check(fn_synchronize(context.provider->evaluator));
  });
}
const char *fr_device_info(void *pointer) {
  const char *result = nullptr;
  if (guarded([&] { result = publish(provider_json(fn_device_info(handle<Context>(pointer).provider->evaluator))); }))
    return nullptr;
  return result;
}
} // extern C
