#include "FHN/Runtime.h"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <utility>

namespace {
using namespace fhenomenon::runtime;

Value integer(Runtime &runtime, int n, std::size_t bytes = 4, Values dependencies = {}) {
  return runtime.makeValue(
    "test.integer", ResourceKind::Value, bytes, [n] { return std::any(n); }, std::move(dependencies));
}

Recipe constant(std::string name, int priority, int number) {
  return {"test.constant", std::move(name), priority, [](const Request &, std::string &) { return true; },
          [number](Runtime &runtime, const Request &) { return Values{integer(runtime, number)}; }};
}

template <typename Action> void expectError(ErrorCode code, Action action) {
  try {
    action();
    FAIL() << "Expected runtime error";
  } catch (const Error &error) {
    EXPECT_EQ(error.code(), code) << error.what();
  }
}

TEST(RuntimeTest, ExecutesBeforeSubmitReturnsAndPassesAttributesAndInputs) {
  Runtime runtime;
  auto source = integer(runtime, 8);
  runtime.addRecipe({"test.add", "contributed", 0, [](const Request &, std::string &) { return true; },
                     [](Runtime &owner, const Request &request) {
                       const int n =
                         std::any_cast<int>(request.inputs.at(0).payload()) + std::any_cast<int>(request.attributes);
                       return Values{integer(owner, n)};
                     }});
  const auto outputs = runtime.submit({"test.add", {source}, 3});
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(std::any_cast<int>(outputs[0].payload()), 11);
  EXPECT_EQ(runtime.stats().live_bytes, 8u);
}

TEST(RuntimeTest, ChoosesHigherPriorityThenFirstRegisteredRecipe) {
  Runtime runtime;
  runtime.addRecipe(constant("low", -1, 1));
  runtime.addRecipe(constant("first", 5, 2));
  runtime.addRecipe(constant("second", 5, 3));
  auto outputs = runtime.submit({"test.constant", {}, {}});
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(std::any_cast<int>(outputs[0].payload()), 2);
}

TEST(RuntimeTest, RecordsExplicitDeclineAndUsesFallbackBeforeExecution) {
  Runtime runtime;
  auto declined = constant("decline", 10, 99);
  declined.accepts = [](const Request &, std::string &reason) {
    reason = "public format unsupported";
    return false;
  };
  runtime.addRecipe(std::move(declined));
  runtime.addRecipe(constant("fallback", 0, 7));
  auto outputs = runtime.submit({"test.constant", {}, {}});
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(std::any_cast<int>(outputs[0].payload()), 7);
  const auto events = runtime.trace();
  ASSERT_EQ(events.size(), 3u);
  EXPECT_EQ(events[0].kind, TraceKind::Declined);
  EXPECT_EQ(events[0].recipe, "decline");
  EXPECT_EQ(events[0].detail, "public format unsupported");
  EXPECT_EQ(events[1].kind, TraceKind::Selected);
  EXPECT_EQ(events[2].kind, TraceKind::Completed);
}

TEST(RuntimeTest, DistinguishesUnsupportedAndMalformedRequests) {
  Runtime runtime;
  expectError(ErrorCode::Unsupported, [&] { runtime.submit({"missing", {}, {}}); });
  expectError(ErrorCode::InvalidRequest, [&] { runtime.submit({"", {}, {}}); });
  expectError(ErrorCode::InvalidRequest, [&] { runtime.submit({"missing", {Value{}}, {}}); });
  expectError(ErrorCode::InvalidRequest, [&] { runtime.addRecipe({}); });
  auto decline = constant("decline", 0, 1);
  decline.accepts = [](const Request &, std::string &) { return false; };
  runtime.addRecipe(std::move(decline));
  expectError(ErrorCode::Unsupported, [&] { runtime.submit({"test.constant", {}, {}}); });
}

TEST(RuntimeTest, ExecutionFailureDoesNotRetryAndReleasesUnpublishedOutputs) {
  Runtime runtime;
  int fallback_calls = 0;
  runtime.addRecipe({"test.fail", "failing", 10, [](const Request &, std::string &) { return true; },
                     [](Runtime &owner, const Request &) -> Values {
                       Values unpublished{integer(owner, 1, 12), integer(owner, 2, 8)};
                       throw std::runtime_error("provider execution failed");
                     }});
  runtime.addRecipe({"test.fail", "fallback", 0, [](const Request &, std::string &) { return true; },
                     [&fallback_calls](Runtime &, const Request &) {
                       ++fallback_calls;
                       return Values{};
                     }});
  expectError(ErrorCode::ExecutionFailed, [&] { runtime.submit({"test.fail", {}, {}}); });
  EXPECT_EQ(fallback_calls, 0);
  EXPECT_EQ(runtime.stats().live_bytes, 0u);
  EXPECT_EQ(runtime.stats().live_resources, 0u);
  const auto events = runtime.trace();
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events.back().kind, TraceKind::Failed);
  EXPECT_NE(events.back().detail.find("provider execution failed"), std::string::npos);
  EXPECT_EQ(runtime.stats().peak_live_bytes, 20u);
}

TEST(RuntimeTest, LaterOutputBudgetFailureReleasesEarlierOutputAndKeepsInputs) {
  Runtime runtime({16, 256});
  const auto input = integer(runtime, 1, 4);
  runtime.addRecipe({"test.pair", "pair", 0, [](const Request &, std::string &) { return true; },
                     [](Runtime &owner, const Request &) {
                       Values outputs{integer(owner, 2, 8)};
                       outputs.push_back(integer(owner, 3, 8));
                       return outputs;
                     }});
  expectError(ErrorCode::BudgetExceeded, [&] { runtime.submit({"test.pair", {input}, {}}); });
  EXPECT_EQ(runtime.stats().live_bytes, 4u);
  EXPECT_EQ(runtime.stats().live_resources, 1u);
  EXPECT_EQ(std::any_cast<int>(input.payload()), 1);
  EXPECT_EQ(runtime.stats().peak_live_bytes, 12u);
}

TEST(RuntimeTest, ApplicabilityExceptionIsFailureNotDecline) {
  Runtime runtime;
  auto broken = constant("broken", 10, 1);
  broken.accepts = [](const Request &, std::string &) -> bool { throw std::runtime_error("bad applicability"); };
  runtime.addRecipe(std::move(broken));
  runtime.addRecipe(constant("fallback", 0, 2));
  expectError(ErrorCode::ExecutionFailed, [&] { runtime.submit({"test.constant", {}, {}}); });
  EXPECT_EQ(runtime.stats().live_bytes, 0u);
}

TEST(RuntimeTest, PreservesTypedProviderErrorsWithoutFallback) {
  Runtime runtime;
  auto recipe = constant("reject", 10, 1);
  recipe.execute = [](Runtime &, const Request &) -> Values {
    throw Error(ErrorCode::InvalidRequest, "source identity mismatch");
  };
  runtime.addRecipe(std::move(recipe));
  runtime.addRecipe(constant("fallback", 0, 2));
  expectError(ErrorCode::InvalidRequest, [&] { runtime.submit({"test.constant", {}, {}}); });
}

TEST(RuntimeTest, RetainsMultipleOutputsAndCountsHandleCopiesOnlyOnce) {
  Runtime runtime;
  runtime.addRecipe(
    {"test.pair", "pair", 0, [](const Request &, std::string &) { return true; },
     [](Runtime &owner, const Request &) { return Values{integer(owner, 2, 10), integer(owner, 3, 20)}; }});
  auto outputs = runtime.submit({"test.pair", {}, {}});
  ASSERT_EQ(outputs.size(), 2u);
  EXPECT_NE(outputs[0].identity(), outputs[1].identity());
  auto copy = outputs[0];
  EXPECT_EQ(copy.identity(), outputs[0].identity());
  EXPECT_EQ(runtime.stats().live_bytes, 30u);
  EXPECT_EQ(runtime.stats().live_resources, 2u);
  outputs.clear();
  EXPECT_EQ(runtime.stats().live_bytes, 10u);
  EXPECT_EQ(std::any_cast<int>(copy.payload()), 2);
  copy = {};
  EXPECT_EQ(runtime.stats().live_bytes, 0u);
  EXPECT_EQ(runtime.stats().peak_live_bytes, 30u);
}

TEST(RuntimeTest, ArtifactDependenciesRetainTheirSourceExactlyOnce) {
  Runtime runtime;
  auto source = integer(runtime, 4, 10);
  const auto source_id = source.identity();
  auto prepared = runtime.makeValue("test.prepared", ResourceKind::Artifact, 20,
                                    [] { return std::any(std::make_shared<int>(9)); }, {source, source});
  EXPECT_EQ(prepared.kind(), ResourceKind::Artifact);
  EXPECT_EQ(prepared.format(), "test.prepared");
  EXPECT_EQ(prepared.bytes(), 20u);
  ASSERT_EQ(prepared.dependencies().size(), 2u);
  EXPECT_EQ(prepared.dependencies()[0].identity(), source_id);
  source = {};
  EXPECT_EQ(runtime.stats().live_bytes, 30u);
  EXPECT_EQ(runtime.stats().live_resources, 2u);
  prepared = {};
  EXPECT_EQ(runtime.stats().live_bytes, 0u);
  EXPECT_EQ(runtime.stats().live_resources, 0u);
}

TEST(RuntimeTest, HandleAndNativeStyleSharedPayloadOutliveRuntime) {
  Value survivor;
  std::weak_ptr<int> weak;
  {
    Runtime runtime;
    survivor = runtime.makeValue("test.shared", ResourceKind::Value, 4, [&] {
      auto payload = std::make_shared<int>(17);
      weak = payload;
      return std::any(payload);
    });
  }
  EXPECT_FALSE(weak.expired());
  EXPECT_EQ(*std::any_cast<std::shared_ptr<int>>(survivor.payload()), 17);
  survivor = {};
  EXPECT_TRUE(weak.expired());
}

TEST(RuntimeTest, RejectsBudgetBeforeFactoryAndRollsBackFailedFactory) {
  Runtime runtime({10, 256});
  auto retained = integer(runtime, 1, 6);
  bool called = false;
  expectError(ErrorCode::BudgetExceeded, [&] {
    runtime.makeValue("oversize", ResourceKind::Value, 5, [&] {
      called = true;
      return std::any(0);
    });
  });
  EXPECT_FALSE(called);
  expectError(ErrorCode::ExecutionFailed, [&] {
    runtime.makeValue("failure", ResourceKind::Value, 4,
                      []() -> std::any { throw std::runtime_error("factory failed"); });
  });
  EXPECT_EQ(runtime.stats().live_bytes, 6u);
  EXPECT_EQ(runtime.stats().live_resources, 1u);
  auto boundary = integer(runtime, 2, 4);
  EXPECT_EQ(runtime.stats().live_bytes, 10u);
}

TEST(RuntimeTest, BudgetReservesOuterAllocationBeforeReentrantFactory) {
  Runtime runtime({10, 256});
  expectError(ErrorCode::BudgetExceeded, [&] {
    runtime.makeValue("outer", ResourceKind::Artifact, 8, [&] { return std::any(integer(runtime, 1, 3)); });
  });
  EXPECT_EQ(runtime.stats().live_bytes, 0u);
  EXPECT_EQ(runtime.stats().live_resources, 0u);
}

TEST(RuntimeTest, ZeroByteResourcesAreCountedAndByteOverflowRejected) {
  Runtime runtime;
  auto large = integer(runtime, 1, std::numeric_limits<std::size_t>::max());
  expectError(ErrorCode::BudgetExceeded, [&] { integer(runtime, 2, 1); });
  auto zero = integer(runtime, 3, 0);
  EXPECT_EQ(runtime.stats().live_resources, 2u);
  zero = {};
  large = {};
  EXPECT_EQ(runtime.stats().live_resources, 0u);
  EXPECT_EQ(runtime.stats().live_bytes, 0u);
}

TEST(RuntimeTest, RejectsForeignInputsDependenciesAndOutputs) {
  Runtime first;
  Runtime second;
  auto foreign = integer(first, 1);
  auto candidate = constant("must not run", 0, 2);
  bool inspected = false;
  candidate.accepts = [&](const Request &, std::string &) {
    inspected = true;
    return true;
  };
  second.addRecipe(std::move(candidate));
  expectError(ErrorCode::InvalidRequest, [&] { second.submit({"test.constant", {foreign}, {}}); });
  EXPECT_FALSE(inspected);
  bool called = false;
  expectError(ErrorCode::InvalidRequest, [&] {
    second.makeValue("foreign.dep", ResourceKind::Artifact, 1,
                     [&] {
                       called = true;
                       return std::any(0);
                     },
                     {foreign});
  });
  EXPECT_FALSE(called);
  second.addRecipe({"test.foreign", "invalid output", 0, [](const Request &, std::string &) { return true; },
                    [foreign](Runtime &, const Request &) { return Values{foreign}; }});
  expectError(ErrorCode::InvalidRequest, [&] { second.submit({"test.foreign", {}, {}}); });
  EXPECT_EQ(second.stats().live_bytes, 0u);
}

TEST(RuntimeTest, RejectsEmptyHandlesAndMissingFactories) {
  Runtime runtime;
  EXPECT_FALSE(Value{});
  expectError(ErrorCode::InvalidRequest, [] { Value{}.payload(); });
  expectError(ErrorCode::InvalidRequest, [&] { runtime.makeValue("test", ResourceKind::Value, 1, {}); });
  expectError(ErrorCode::InvalidRequest, [&] { integer(runtime, 1, 1, {Value{}}); });
  EXPECT_EQ(runtime.stats().live_resources, 0u);
}

TEST(RuntimeTest, DependenciesRemainAliveDuringProviderCleanup) {
  Runtime runtime;
  auto owned = std::make_shared<int>(7);
  std::weak_ptr<int> weak = owned;
  bool alive_during_cleanup = false;
  auto source = runtime.makeValue("source", ResourceKind::Value, sizeof(int), [&] { return std::any(owned); });
  owned.reset();
  auto artifact = runtime.makeValue("artifact", ResourceKind::Artifact, 1,
                                    [&] {
                                      return std::any(std::shared_ptr<int>(new int(0), [&](int *p) {
                                        alive_during_cleanup = !weak.expired();
                                        delete p;
                                      }));
                                    },
                                    {source});
  source = {};
  artifact = {};
  EXPECT_TRUE(alive_during_cleanup);
  EXPECT_TRUE(weak.expired());
  EXPECT_EQ(runtime.stats().live_resources, 0u);
}

TEST(RuntimeTest, BoundsTraceAndRetainsNewestEventsOrDisablesIt) {
  Runtime runtime({0, 3});
  runtime.addRecipe(constant("constant", 0, 1));
  runtime.submit({"test.constant", {}, {}});
  runtime.submit({"test.constant", {}, {}});
  const auto events = runtime.trace();
  ASSERT_EQ(events.size(), 3u);
  EXPECT_EQ(events.front().kind, TraceKind::Completed);
  EXPECT_EQ(events.back().kind, TraceKind::Completed);
  Runtime disabled({0, 0});
  disabled.addRecipe(constant("constant", 0, 1));
  disabled.submit({"test.constant", {}, {}});
  EXPECT_TRUE(disabled.trace().empty());
}

} // namespace
