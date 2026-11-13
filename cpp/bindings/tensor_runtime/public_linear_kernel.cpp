#include "FHN/TensorKernels.h"
#include "backend.h"
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>

namespace fhenomenon::tensor {
namespace {
void require(bool condition, const char *message) {
  if (!condition)
    throw std::invalid_argument(message);
}
std::size_t dimension(const Json &attributes, const char *name) {
  const auto &value = attributes.at(name);
  require(value.is_number_unsigned() || value.is_number_integer(), "linear dimensions must be positive integers");
  if (value.is_number_integer())
    require(value.get<std::int64_t>() > 0, "linear dimensions must be positive integers");
  const auto result = value.get<std::uint64_t>();
  require(result && result <= std::numeric_limits<std::size_t>::max(), "linear dimension exceeds size_t");
  return static_cast<std::size_t>(result);
}

struct Geometry {
  std::size_t rows, in_features, out_features;
  const Json &weights;
};
Geometry geometry(const Json &attributes, const CipherMetadata &input) {
  require(attributes.is_object() && attributes.size() == 4 && attributes.contains("weights"),
          "linear attributes require rows, in_features, out_features and weights only");
  Geometry result{dimension(attributes, "rows"), dimension(attributes, "in_features"),
                  dimension(attributes, "out_features"), attributes.at("weights")};
  require(result.rows <= input.count / result.in_features && result.rows <= input.count / result.out_features,
          "linear input/output geometry exceeds the packed ciphertext count");
  require(result.in_features <= std::numeric_limits<std::size_t>::max() / result.out_features,
          "linear weight shape overflows size_t");
  const auto &weights = attributes.at("weights");
  require(weights.is_array() && weights.size() == result.in_features * result.out_features,
          "linear weights must be an in_features by out_features row-major matrix");
  for (const auto &weight : weights) {
    require(weight.is_number(), "linear weights must be finite numbers");
    const auto value = weight.get<double>();
    require(std::isfinite(value), "linear weights must be finite numbers");
  }
  return result;
}

using Diagonals = std::map<std::size_t, std::vector<double>>;
std::set<std::size_t> active_diagonals(const Geometry &shape, const CipherMetadata &input) {
  std::set<std::size_t> result;
  for (std::size_t row = 0; row < shape.rows; ++row)
    for (std::size_t column = 0; column < shape.out_features; ++column)
      for (std::size_t inner = 0; inner < shape.in_features; ++inner) {
        if (shape.weights[inner * shape.out_features + column].get<double>() == 0)
          continue;
        result.insert((row * shape.in_features + inner + input.period - (row * shape.out_features + column)) %
                      input.period);
      }
  if (result.empty())
    result.insert(0);
  return result;
}
Diagonals diagonals(const Geometry &shape, const CipherMetadata &input) {
  Diagonals result;
  for (std::size_t row = 0; row < shape.rows; ++row)
    for (std::size_t column = 0; column < shape.out_features; ++column)
      for (std::size_t inner = 0; inner < shape.in_features; ++inner) {
        const auto weight = shape.weights[inner * shape.out_features + column].get<double>();
        if (weight == 0)
          continue; // Public specialization only.
        const auto output_slot = row * shape.out_features + column;
        const auto input_slot = row * shape.in_features + inner;
        const auto shift = (input_slot + input.period - output_slot) % input.period;
        auto &diagonal = result[shift];
        if (diagonal.empty())
          diagonal.resize(input.count);
        diagonal[output_slot] = weight;
      }
  // A zero matrix still executes one multiply-by-zero, preserving the contract's
  // one-level consumption and producing encrypted zeros without a client key.
  if (result.empty())
    result.emplace(0, std::vector<double>(input.count));
  return result;
}

KernelPlan infer(const KernelEnvironment &environment, const std::vector<CipherMetadata> &inputs,
                 const Json &attributes) {
  require(inputs.size() == 1, "public linear requires exactly one ciphertext input");
  const auto &input = inputs.front();
  require(input.representation == "slots", "public linear requires slot representation");
  require(input.level > 0 && static_cast<std::size_t>(input.level) < environment.scales.size(),
          "public linear requires at least one remaining evaluation level");
  require(input.moduli == static_cast<std::size_t>(input.level) + 1 && input.moduli &&
            input.bytes % input.moduli == 0 && input.period && input.count && input.count <= input.period &&
            !(input.period & (input.period - 1)),
          "public linear requires the declared CKKS layout");
  require(std::isfinite(input.scale) && input.scale > 0, "public linear requires a finite positive scale");
  require(input.period <= environment.profile.at("physical_slots").get<std::size_t>() &&
            input.period <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()),
          "public linear exceeds provider rotation capacity");
  require(environment.profile.value("rotation_key_basis", "") == "all_positive_powers_of_two",
          "public linear requires a complete power-of-two rotation key basis");
  const auto shape = geometry(attributes, input);
  const auto shifts = active_diagonals(shape, input);
  auto output = input;
  --output.level;
  --output.moduli;
  output.bytes = input.bytes / input.moduli * output.moduli;
  output.scale = environment.scales.at(output.level);
  require(output.bytes <= (std::numeric_limits<std::size_t>::max() - input.bytes) / 2,
          "public linear temporary ciphertext accounting overflows size_t");
  KernelPlan result{{output}, input.bytes + 2 * output.bytes};
  require(input.count <= std::numeric_limits<std::size_t>::max() / sizeof(double) &&
            shifts.size() <= std::numeric_limits<std::size_t>::max() / (input.count * sizeof(double)),
          "public linear diagonal buffer accounting overflows size_t");
  const auto diagonal_bytes = shifts.size() * input.count * sizeof(double);
  result.temporary_public_bytes = diagonal_bytes;
  result.details = {{"diagonals", shifts.size()},
                    {"rotations", Json::array()},
                    {"rows", shape.rows},
                    {"in_features", shape.in_features},
                    {"out_features", shape.out_features},
                    {"level_consumption", 1},
                    {"padded_output_count", input.count},
                    {"period", input.period},
                    {"output_level", output.level},
                    {"output_scale", output.scale},
                    {"output_ciphertext_bytes", output.bytes}};
  for (const auto shift : shifts)
    if (shift)
      result.details["rotations"].push_back(shift);
  return result;
}

using Cipher = std::unique_ptr<void, decltype(&fn_ciphertext_free)>;
Cipher invoke(const std::function<int(void **)> &operation) {
  void *result = nullptr;
  const int status = operation(&result);
  Cipher owned(result, fn_ciphertext_free);
  if (status)
    throw runtime::Error(runtime::ErrorCode::ExecutionFailed,
                         std::string("public linear native arithmetic: ") + fn_last_error());
  require(result != nullptr, "public linear provider returned no ciphertext");
  return owned;
}

Values execute(KernelExecution &environment, const Values &inputs, const Json &attributes, const KernelPlan &plan) {
  const auto &output = plan.outputs.at(0);
  // Geometry depends only on count/period, which are unchanged by this kernel.
  const auto terms = diagonals(geometry(attributes, output), output);
  auto result = environment.produce(output, [&](void **out) {
    Cipher accumulator(nullptr, fn_ciphertext_free);
    for (const auto &entry : terms) {
      Cipher rotated(nullptr, fn_ciphertext_free);
      auto *source = environment.borrow(inputs.at(0));
      if (entry.first) {
        rotated = invoke([&](void **value) {
          return fn_rotate(environment.provider, source, static_cast<std::int32_t>(entry.first), value);
        });
        source = rotated.get();
      }
      auto term = invoke([&](void **value) {
        return fn_plain(environment.provider, 2, source, entry.second.data(), entry.second.size(), value);
      });
      rotated.reset();
      if (!accumulator)
        accumulator = std::move(term);
      else
        accumulator = invoke(
          [&](void **value) { return fn_binary(environment.provider, 0, accumulator.get(), term.get(), value); });
    }
    *out = accumulator.release();
    return 0;
  });
  return {std::move(result)};
}

[[maybe_unused]] const bool registered = registerTensorKernel(
  {"tensor.public_linear",
   {{"input_arity", std::size_t{1}},
    {"output_arity", std::size_t{1}},
    {"dtype", "CKKS approximate real; float64 public weights"},
    {"format", "fhenomenon.native.ckks"},
    {"representation", "slots"},
    {"layout", "dense row-major logical matrix in one slot-packed prefix"},
    {"packing", "input and output retain ciphertext count/period; unused output slots are zero"},
    {"geometry", "rows*in_features <= count; rows*out_features <= count"},
    {"level_consumption", 1},
    {"minimum_input_level", 1},
    {"prepared_key_requirements", {{"rotation_key_basis", "all_positive_powers_of_two"}}},
    {"workspace_accounting",
     {{"temporary_ciphertext_payload", "input.bytes + 2*output.bytes charged before arithmetic"},
      {"temporary_public_payload", "8*diagonal_count*input.count"}}},
    {"excluded_accounting",
     {"provider internal scratch and plaintext cache", "keys", "allocator/container overhead", "captured attributes"}},
    {"implementation", "native diagonal weighted rotations; composed arithmetic, not a fused GPU launch"},
    {"timing", "synchronized host/provider elapsed seconds; includes native launch and validation overhead"}},
   infer,
   execute});
} // namespace
} // namespace fhenomenon::tensor
