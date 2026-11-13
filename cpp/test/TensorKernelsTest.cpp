#include "FHN/TensorKernels.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>

namespace {
using namespace fhenomenon::tensor;
using fhenomenon::runtime::ResourceKind;

KernelDefinition integerContribution(std::string operation, int increment) {
  return {std::move(operation),
          {{"input_arity", 1}, {"output_arity", 1}},
          [](const KernelEnvironment &, const std::vector<CipherMetadata> &inputs, const Json &) {
            if (inputs.size() != 1)
              throw std::invalid_argument("one input required");
            return KernelPlan{{inputs.front()}};
          },
          [increment](KernelExecution &context, const Values &inputs, const Json &, const KernelPlan &) {
            const int result = std::any_cast<int>(inputs.at(0).payload()) + increment;
            return Values{context.runtime.makeValue("test.integer", ResourceKind::Value, sizeof(int),
                                                    [result] { return std::any(result); })};
          }};
}

const KernelDefinition &find(const std::vector<KernelDefinition> &definitions, const std::string &operation) {
  const auto found = std::find_if(definitions.begin(), definitions.end(),
                                  [&](const KernelDefinition &item) { return item.operation == operation; });
  if (found == definitions.end())
    throw std::runtime_error("registered operation is absent");
  return *found;
}

TEST(TensorKernelsTest, OwnsContributionMetadataAndExecutesOriginalCallback) {
  auto contribution = integerContribution("test.registry.integer", 7);
  ASSERT_TRUE(registerTensorKernel(contribution));
  contribution.contract["output_arity"] = 0;
  contribution.execute = {};
  const auto definitions = tensorKernelDefinitions();
  const auto &stored = find(definitions, "test.registry.integer");
  EXPECT_EQ(stored.contract.at("output_arity"), 1);

  const Json profile = Json::object();
  const std::vector<double> scales{1024};
  const std::vector<std::uint64_t> moduli{65537};
  const CipherMetadata metadata{sizeof(int), 1, 1, 1, 0, 1024, "slots"};
  const auto plan = stored.infer({profile, scales, moduli}, {metadata}, Json::object());
  ASSERT_EQ(plan.outputs.size(), 1u);
  EXPECT_EQ(plan.outputs.front().bytes, sizeof(int));
  EXPECT_EQ(plan.outputs.front().scale, 1024);

  Runtime runtime;
  auto input = runtime.makeValue("test.integer", ResourceKind::Value, sizeof(int), [] { return std::any(5); });
  KernelExecution execution{runtime, nullptr, {}, {}};
  auto outputs = stored.execute(execution, {input}, Json::object(), plan);
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(std::any_cast<int>(outputs.front().payload()), 12);
  EXPECT_EQ(std::any_cast<int>(input.payload()), 5);
  EXPECT_TRUE(runtime.owns(outputs.front()));
}

TEST(TensorKernelsTest, RejectsDuplicateWithoutReplacingOriginalDefinition) {
  ASSERT_TRUE(registerTensorKernel(integerContribution("test.registry.duplicate", 1)));
  EXPECT_THROW(registerTensorKernel(integerContribution("test.registry.duplicate", 99)), std::invalid_argument);
  auto definitions = tensorKernelDefinitions();
  auto &stored = find(definitions, "test.registry.duplicate");
  Runtime runtime;
  auto input = runtime.makeValue("test.integer", ResourceKind::Value, sizeof(int), [] { return std::any(4); });
  KernelExecution execution{runtime, nullptr, {}, {}};
  auto outputs = stored.execute(execution, {input}, Json::object(), {});
  EXPECT_EQ(std::any_cast<int>(outputs.at(0).payload()), 5);
}

TEST(TensorKernelsTest, RejectsIncompleteAndInvalidDefinitionsWithoutPublishingThem) {
  const auto before = tensorKernelDefinitions().size();
  for (const auto arity : {Json(-1), Json(0.5), Json(true), Json("1"), Json(nullptr)}) {
    auto invalid = integerContribution("test.registry.invalid", 1);
    invalid.contract["input_arity"] = arity;
    EXPECT_THROW(registerTensorKernel(invalid), std::invalid_argument);
    invalid = integerContribution("test.registry.invalid", 1);
    invalid.contract["output_arity"] = arity;
    EXPECT_THROW(registerTensorKernel(invalid), std::invalid_argument);
  }
  auto invalid = integerContribution("test.registry.invalid", 1);
  invalid.operation.clear();
  EXPECT_THROW(registerTensorKernel(invalid), std::invalid_argument);
  invalid = integerContribution("test.registry.invalid", 1);
  invalid.infer = {};
  EXPECT_THROW(registerTensorKernel(invalid), std::invalid_argument);
  invalid = integerContribution("test.registry.invalid", 1);
  invalid.execute = {};
  EXPECT_THROW(registerTensorKernel(invalid), std::invalid_argument);
  invalid = integerContribution("test.registry.invalid", 1);
  invalid.contract.erase("input_arity");
  EXPECT_THROW(registerTensorKernel(invalid), std::invalid_argument);
  invalid = integerContribution("test.registry.invalid", 1);
  invalid.contract = Json::array();
  EXPECT_THROW(registerTensorKernel(invalid), std::invalid_argument);
  EXPECT_EQ(tensorKernelDefinitions().size(), before);
}

TEST(TensorKernelsTest, SnapshotsDoNotChangeWhenNewContributionsAreRegistered) {
  const auto before = tensorKernelDefinitions();
  auto definition = integerContribution("test.registry.later", 1);
  definition.contract["input_arity"] = 0;
  definition.contract["output_arity"] = 0u;
  EXPECT_TRUE(registerTensorKernel(definition));
  const auto after = tensorKernelDefinitions();
  EXPECT_EQ(after.size(), before.size() + 1);
  EXPECT_THROW(find(before, "test.registry.later"), std::runtime_error);
  EXPECT_EQ(find(after, "test.registry.later").contract.at("output_arity"), 0);
}
} // namespace
