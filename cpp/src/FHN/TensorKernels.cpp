#include "FHN/TensorKernels.h"
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>

namespace fhenomenon::tensor {
namespace {
struct Registry {
  std::mutex mutex;
  std::map<std::string, KernelDefinition> definitions;
};
Registry &registry() {
  static Registry instance;
  return instance;
}
bool validArity(const Json &value) {
  if (!value.is_number_integer())
    return false;
  if (!value.is_number_unsigned() && value.get<std::int64_t>() < 0)
    return false;
  return value.get<std::uint64_t>() <= std::numeric_limits<std::size_t>::max();
}
} // namespace

bool registerTensorKernel(KernelDefinition definition) {
  if (definition.operation.empty() || !definition.infer || !definition.execute || !definition.contract.is_object() ||
      !definition.contract.contains("input_arity") || !definition.contract.contains("output_arity") ||
      !validArity(definition.contract.at("input_arity")) || !validArity(definition.contract.at("output_arity")))
    throw std::invalid_argument("tensor kernel requires an operation, contract, inference and execution");
  auto &stored = registry();
  std::lock_guard<std::mutex> lock(stored.mutex);
  const auto name = definition.operation;
  if (!stored.definitions.emplace(name, std::move(definition)).second)
    throw std::invalid_argument("tensor kernel operation is already registered: " + name);
  return true;
}

std::vector<KernelDefinition> tensorKernelDefinitions() {
  auto &stored = registry();
  std::lock_guard<std::mutex> lock(stored.mutex);
  std::vector<KernelDefinition> result;
  for (const auto &entry : stored.definitions)
    result.push_back(entry.second);
  return result;
}
} // namespace fhenomenon::tensor
