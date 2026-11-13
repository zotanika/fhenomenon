#pragma once

#include "FHN/Runtime.h"
#include <cstdint>
#include <functional>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

// Public contribution interface. No ciphertext implementation or client key is
// exposed here. An adapter supplies public metadata and checked Value factories;
// only an individual kernel implementation needs a provider's arithmetic ABI.
namespace fhenomenon::tensor {
using Json = nlohmann::json;
using runtime::Runtime;
using runtime::Value;
using runtime::Values;

struct CipherMetadata {
  std::size_t bytes, count, moduli, period;
  std::int64_t level;
  double scale;
  std::string representation;
};

struct KernelEnvironment {
  const Json &profile;
  const std::vector<double> &scales;
  const std::vector<std::uint64_t> &moduli;
};

struct KernelPlan {
  std::vector<CipherMetadata> outputs;
  // Conservative simultaneously live native ciphertext payload, additional to
  // inputs and externally published outputs. The adapter reserves this before
  // executing any arithmetic. Provider keys, internal scratch and allocators are
  // separate accounting domains and must be stated in the catalog.
  std::size_t temporary_ciphertext_bytes = 0;
  // Kernel-owned public buffer payload (for example prepared dense diagonals).
  // Captured attributes and allocator/container bookkeeping are separate.
  std::size_t temporary_public_bytes = 0;
  Json details = Json::object();
};

struct KernelExecution {
  Runtime &runtime;
  void *provider;
  std::function<void *(const Value &)> borrow;
  std::function<Value(const CipherMetadata &, const std::function<int(void **)> &)> produce;
};

struct KernelDefinition {
  std::string operation;
  Json contract;
  // Pure: no provider operation, Runtime allocation, ciphertext inspection or
  // secret value access. The returned descriptors are validated after execute.
  std::function<KernelPlan(const KernelEnvironment &, const std::vector<CipherMetadata> &, const Json &)> infer;
  std::function<Values(KernelExecution &, const Values &, const Json &, const KernelPlan &)> execute;
};

// Register definitions at library initialization. Each Runtime takes an owning
// snapshot, so later contribution registration cannot mutate a captured graph.
// Duplicate operation names are rejected; implementation selection belongs in
// the frontend's registry, not a registration-order-dependent ABI switch.
bool registerTensorKernel(KernelDefinition definition);
std::vector<KernelDefinition> tensorKernelDefinitions();
} // namespace fhenomenon::tensor
