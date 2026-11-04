#include "FHN/contrib/SpectralKernels.h"
#include "FHN/contrib/MatvecKernels.h"
#include <algorithm>
#include <cmath>

namespace fhenomenon::contrib {
namespace {
using namespace runtime;
constexpr auto prepareOperation = "contrib.spectral.prepare-reference";
constexpr auto pairOperation = "contrib.spectral.pair";
constexpr auto preparationFormat = "contrib.spectral.reference-preparation";

struct Provider {
  NativeServices services;
  NativeRuntime native;
  bool use_prepared;

  Provider(Runtime &runtime, NativeServices nativeServices, bool usePrepared)
    : services(std::move(nativeServices)), native(runtime, services), use_prepared(usePrepared) {}
};

struct Preparation {
  std::uint64_t identity;
  std::shared_ptr<const Provider> provider;
  std::uint64_t source_identity;
  FhnCkksBufferInfo info;
};

bool metadata(const Provider &provider, const Value &value, FhnCkksBufferInfo &info) {
  if (!value || value.kind() != ResourceKind::Value || !provider.services.buffer_info)
    return false;
  try {
    return provider.services.buffer_info(provider.services.context.get(), provider.native.buffer(value), &info) == 0 &&
           info.components == 2 && info.level >= 1 && info.logical_slots > 0 && std::isfinite(info.scale) &&
           info.scale > 0;
  } catch (const Error &) {
    return false;
  }
}
bool compatible(const FhnCkksBufferInfo &a, const FhnCkksBufferInfo &b) {
  return a.logical_slots == b.logical_slots && a.level == b.level &&
         std::abs(a.scale - b.scale) <= 1e-12 * std::max(a.scale, b.scale);
}
bool prepareApplicable(const Provider &provider, const Request &request, std::string &reason) {
  FhnCkksBufferInfo info{};
  if (request.inputs.size() != 1 || request.attributes.has_value() || !metadata(provider, request.inputs[0], info)) {
    reason = "spectral reference requires one compatible native ciphertext";
    return false;
  }
  return true;
}
bool pairApplicable(const Provider &provider, const Request &request, std::string &reason) {
  if (request.inputs.size() != 2 || request.attributes.has_value()) {
    reason = "spectral pair requires a sensor and reference preparation";
    return false;
  }
  const auto &preparation = request.inputs[1];
  if (preparation.kind() != ResourceKind::Artifact || preparation.format() != preparationFormat) {
    reason = "invalid spectral reference preparation";
    return false;
  }
  const auto *owned = std::any_cast<std::shared_ptr<const Preparation>>(&preparation.payload());
  const auto &dependencies = preparation.dependencies();
  FhnCkksBufferInfo sensor{};
  if (!owned || !*owned || (*owned)->identity != preparation.identity() || (*owned)->provider.get() != &provider ||
      dependencies.empty() || dependencies[0].identity() != (*owned)->source_identity ||
      !metadata(provider, request.inputs[0], sensor) || !compatible((*owned)->info, sensor)) {
    reason = "spectral sensor metadata or reference preparation does not match this provider";
    return false;
  }
  return true;
}
Value singleResult(Values results) {
  if (results.size() != 1)
    throw Error(ErrorCode::ExecutionFailed, "Spectral operation expected a single result");
  return std::move(results[0]);
}
Values executePrepare(Runtime &runtime, const std::shared_ptr<Provider> &provider, const Request &request) {
  const auto &source = request.inputs[0];
  auto conjugate = provider->native.op(FHN_CONJUGATE, {source});
  FhnCkksBufferInfo original{}, info{};
  if (!metadata(*provider, source, original) || !metadata(*provider, conjugate, info) || !compatible(original, info))
    throw Error(ErrorCode::ExecutionFailed, "Conjugation returned incompatible spectral metadata");

  Values dependencies{conjugate};
  if (provider->use_prepared) {
    // A width-one product still covers every logical slot. It has no reduction
    // stages and requires no rotation keys.
    dependencies.push_back(singleResult(runtime.submit(
      {"contrib.matvec.prepare", {conjugate}, MatvecPrepareRequest{1, FHN_MATVEC_PREPARE_VECTOR | FHN_MATVEC_PREPARE_KEYS}})));
  }
  std::shared_ptr<Preparation> owned;
  auto preparation = runtime.makeValue(preparationFormat, ResourceKind::Artifact, sizeof(Preparation),
                                       [&]() -> std::any {
                                         owned = std::make_shared<Preparation>(
                                           Preparation{0, provider, conjugate.identity(), info});
                                         return std::shared_ptr<const Preparation>(owned);
                                       },
                                       std::move(dependencies));
  // Seal the payload to this Value before publication. Its public std::any
  // cannot be copied to another artifact to substitute the reference.
  owned->identity = preparation.identity();
  return {std::move(conjugate), std::move(preparation)};
}

std::size_t matchPairs(const Provider &provider, const ScopeView &view, std::size_t minimum, std::string &reason) {
  ScopeValue reference;
  std::size_t count = 0;
  for (; count < view.size(); ++count) {
    const auto &command = view.command(count);
    if (command.operation != pairOperation || command.inputs.size() != 2 || command.outputs.size() != 1)
      break;
    if (count == 0)
      reference = command.inputs[1];
    if (command.inputs[1] != reference)
      break;
    const auto *sensor = view.find(command.inputs[0]);
    const auto *preparation = view.find(reference);
    if (!sensor || !preparation ||
        !pairApplicable(provider, {command.operation, {*sensor, *preparation}, command.attributes}, reason))
      break;
  }
  if (count >= minimum)
    return count;
  reason = "scope requires compatible pairs with a shared reference preparation";
  return 0;
}
Values executePairs(Runtime &runtime, const ScopeView &view) {
  const auto liveouts = view.liveOuts(view.size());
  Values results(liveouts.size());
  for (std::size_t i = 0; i < view.size(); ++i) {
    const auto &command = view.command(i);
    auto result = singleResult(runtime.submit(
      {command.operation, {view.value(command.inputs[0]), view.value(command.inputs[1])}, command.attributes}));
    const auto found = std::find(liveouts.begin(), liveouts.end(), command.outputs[0]);
    if (found != liveouts.end())
      results[static_cast<std::size_t>(found - liveouts.begin())] = std::move(result);
  }
  return results;
}
void checkScope(const Runtime &runtime, const Scope &scope) {
  if (!scope.belongsTo(runtime))
    throw Error(ErrorCode::InvalidRequest, "Spectral scope belongs to a different runtime");
}
} // namespace

SpectralKeyRequirements nativeSpectralKeyRequirements() { return {true, true}; }

void registerSpectralKernels(Runtime &runtime, NativeServices services, const std::string &nativeLibraryPath) {
  const bool usePrepared = !nativeLibraryPath.empty();
  if (usePrepared)
    registerMatvecKernels(runtime, services, nativeLibraryPath);
  auto provider = std::make_shared<Provider>(runtime, std::move(services), usePrepared);
  runtime.addRecipe({prepareOperation, "contrib.spectral.prepare-reference.native", 0,
                     [provider](const Request &r, std::string &why) { return prepareApplicable(*provider, r, why); },
                     [provider](Runtime &rt, const Request &r) { return executePrepare(rt, provider, r); }});
  runtime.addRecipe(
    {pairOperation, "contrib.spectral.pair.prepared", 20,
     [provider](const Request &r, std::string &why) {
       if (!pairApplicable(*provider, r, why))
         return false;
       if (r.inputs[1].dependencies().size() == 2)
         return true;
       why = "reference has no prepared Matvec product";
       return false;
     },
     [](Runtime &rt, const Request &r) {
       const auto &deps = r.inputs[1].dependencies();
       return Values{singleResult(rt.submit({"contrib.matvec.product", {deps[0], r.inputs[0], deps[1]}, {}}))};
     }});
  runtime.addRecipe(
    {pairOperation, "contrib.spectral.pair.ordinary", 0,
     [provider](const Request &r, std::string &why) { return pairApplicable(*provider, r, why); },
     [provider](Runtime &, const Request &r) {
       return Values{provider->native.op(FHN_HMULT, {r.inputs[0], r.inputs[1].dependencies()[0]})};
     }});
  runtime.addScopeKernel({"contrib.spectral.scope.pair-tile", 20, 8,
                          [provider](const ScopeView &view, std::string &why) {
                            return matchPairs(*provider, view, 2, why);
                          },
                          executePairs});
  runtime.addScopeKernel({"contrib.spectral.scope.pair", 10, 1,
                          [provider](const ScopeView &view, std::string &why) {
                            return matchPairs(*provider, view, 1, why);
                          },
                          executePairs});
}

std::pair<Value, Value> SpectralRuntime::prepareReference(Value reference) {
  auto results = runtime_.submit({prepareOperation, {std::move(reference)}, {}});
  if (results.size() != 2)
    throw Error(ErrorCode::ExecutionFailed, "Spectral reference preparation expected two results");
  return {std::move(results[0]), std::move(results[1])};
}
std::pair<ScopeValue, ScopeValue> SpectralRuntime::prepareReference(Scope &scope, ScopeValue reference) {
  checkScope(runtime_, scope);
  auto outputs = scope.submit(prepareOperation, {reference}, {}, 2);
  return {outputs[0], outputs[1]};
}
Value SpectralRuntime::pair(Value sensor, Value preparation) {
  return singleResult(runtime_.submit({pairOperation, {std::move(sensor), std::move(preparation)}, {}}));
}
ScopeValue SpectralRuntime::pair(Scope &scope, ScopeValue sensor, ScopeValue preparation) {
  checkScope(runtime_, scope);
  return scope.submit(pairOperation, {sensor, preparation})[0];
}
} // namespace fhenomenon::contrib
