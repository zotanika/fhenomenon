#pragma once

#include "FHN/NativeRuntime.h"
#include <string>
#include <utility>

namespace fhenomenon::contrib {

struct SpectralKeyRequirements {
  bool multiplication;
  bool conjugation;
};
// Built-in kernels require multiplication and conjugation keys. Every logical
// slot is an independent frequency bin, so no rotation keys are required.
SpectralKeyRequirements nativeSpectralKeyRequirements();

// Registers contrib.spectral.prepare-reference ({reference}, {}) and
// contrib.spectral.pair ({sensor, preparation}, {}). Inputs use the same native
// context, level, scale and slot layout. A nonempty path requests Matvec product
// preparation from that library and propagates preparation failures. An empty
// path selects ordinary native HMULT.
// Scope kernels group up to eight adjacent pairs sharing one preparation and
// dispatch each through the same overrideable pair recipe.
void registerSpectralKernels(runtime::Runtime &, runtime::NativeServices, const std::string &nativeLibraryPath);

class SpectralRuntime {
  public:
  explicit SpectralRuntime(runtime::Runtime &runtime) : runtime_(runtime) {}

  // Returns {ordinary conjugated reference, sealed reusable preparation}.
  std::pair<runtime::Value, runtime::Value> prepareReference(runtime::Value reference);
  std::pair<runtime::ScopeValue, runtime::ScopeValue> prepareReference(runtime::Scope &, runtime::ScopeValue reference);

  // Returns sensor * conj(reference), independently in every slot.
  runtime::Value pair(runtime::Value sensor, runtime::Value preparation);
  runtime::ScopeValue pair(runtime::Scope &, runtime::ScopeValue sensor, runtime::ScopeValue preparation);

  private:
  runtime::Runtime &runtime_;
};
} // namespace fhenomenon::contrib
