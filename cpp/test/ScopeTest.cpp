#include "FHN/Scope.h"

#include <gtest/gtest.h>

#include <chrono>
#include <limits>
#include <memory>
#include <thread>
#include <type_traits>

namespace {
using namespace fhenomenon::runtime;
static_assert(!std::is_convertible<ScopeValue, Value>::value, "symbols are distinct from resources");
static_assert(!std::is_convertible<ScopeValue, bool>::value, "symbols cannot control encrypted branches");

Value number(Runtime &runtime, int value = 1, Values deps = {}) {
  return runtime.makeValue("scope.test", ResourceKind::Value, 4, [value] { return std::any(value); }, std::move(deps));
}
int read(const Value &value) { return std::any_cast<int>(value.payload()); }
template <typename Action> void errorIs(ErrorCode code, Action action) {
  try {
    action();
    FAIL() << "Expected runtime error";
  } catch (const Error &error) {
    EXPECT_EQ(error.code(), code) << error.what();
  }
}
void registerCopy(Runtime &runtime, int &calls) {
  runtime.addRecipe({"copy", "copy", 0, [](const Request &, std::string &) { return true; },
                     [&calls](Runtime &owner, const Request &request) {
                       ++calls;
                       return Values{number(owner, request.inputs.empty() ? 1 : read(request.inputs[0]))};
                     }});
}

TEST(ScopeTest, CapturesOnceBeforeExecutingAndExportsDuplicates) {
  Runtime runtime;
  int calls = 0, captures = 0;
  registerCopy(runtime, calls);
  auto input = number(runtime, 7);
  auto result = runtime.runScope([&](Scope &scope) {
    ++captures;
    auto first = scope.input(input);
    EXPECT_EQ(first, scope.input(input));
    auto output = scope.submit("copy", {first})[0];
    EXPECT_EQ(calls, 0);
    return ScopeValues{output, output, first};
  });
  ASSERT_EQ(result.size(), 3u);
  EXPECT_EQ(read(result[0]), 7);
  EXPECT_EQ(result[0].identity(), result[1].identity());
  EXPECT_EQ(result[2].identity(), input.identity());
  EXPECT_EQ(captures, 1);
  EXPECT_EQ(calls, 1);
}

TEST(ScopeTest, SpanSuppressesDispatchAndPreservesIntermediateLiveoutAndDownstreamUse) {
  Runtime runtime;
  int calls = 0;
  registerCopy(runtime, calls);
  ScopeValue intermediate, final;
  runtime.addScopeKernel({"pair", 3, 2,
                          [&](const ScopeView &view, std::string &) {
                            EXPECT_EQ(view.start(), 0u);
                            EXPECT_EQ(view.size(), 2u);
                            EXPECT_EQ(view.liveIns(2).size(), 1u);
                            EXPECT_EQ(view.liveOuts(2), (ScopeValues{intermediate, final}));
                            EXPECT_EQ(view.find(intermediate), nullptr);
                            errorIs(ErrorCode::InvalidRequest, [&] { view.value(intermediate); });
                            return 2u;
                          },
                          [&](Runtime &owner, const ScopeView &view) {
                            EXPECT_EQ(view.size(), 2u);
                            errorIs(ErrorCode::InvalidRequest, [&] { view.command(2); });
                            return Values{number(owner, 8), number(owner, 9)};
                          }});
  auto input = number(runtime, 1);
  auto outputs = runtime.runScope([&](Scope &scope) {
    intermediate = scope.submit("copy", {scope.input(input)})[0];
    final = scope.submit("copy", {intermediate})[0];
    return ScopeValues{final, intermediate};
  });
  EXPECT_EQ(calls, 0);
  ASSERT_EQ(outputs.size(), 2u);
  EXPECT_EQ(read(outputs[0]), 9);
  EXPECT_EQ(read(outputs[1]), 8);
  auto trace = runtime.trace();
  ASSERT_EQ(trace.size(), 2u);
  EXPECT_EQ(trace[0].operation, "scope.span");
  EXPECT_EQ(trace[0].recipe, "pair");
  EXPECT_NE(trace[0].detail.find("liveouts=2"), std::string::npos);
}

TEST(ScopeTest, DeclinesWhenIntermediateCannotBeProducedThenKeepsDownstreamInput) {
  Runtime runtime;
  int calls = 0, spans = 0;
  registerCopy(runtime, calls);
  runtime.addScopeKernel(
    {"only-final", 0, 2,
     [](const ScopeView &view, std::string &) { return view.size() == 2 && view.liveOuts(2).size() == 1 ? 2u : 0u; },
     [&](Runtime &owner, const ScopeView &) {
       ++spans;
       return Values{number(owner, 20)};
     }});
  auto result = runtime.runScope([&](Scope &scope) {
    auto first = scope.submit("copy", {})[0];
    auto second = scope.submit("copy", {first})[0];
    auto third = scope.submit("copy", {first})[0];
    return ScopeValues{second, third};
  });
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(read(result[0]), 1);
  EXPECT_EQ(read(result[1]), 1);
  EXPECT_EQ(calls, 3);
  EXPECT_EQ(spans, 0);
}

TEST(ScopeTest, ChoosesPriorityThenRegistrationAndClipsAcceptedPrefix) {
  Runtime runtime;
  int calls = 0, matches = 0;
  registerCopy(runtime, calls);
  auto kernel = [&](std::string name, int priority, int n) {
    return ScopeKernel{std::move(name), priority, 8,
                       [&](const ScopeView &view, std::string &) {
                         ++matches;
                         EXPECT_LE(view.size(), 2u);
                         return 1u;
                       },
                       [n](Runtime &owner, const ScopeView &view) {
                         EXPECT_EQ(view.size(), 1u);
                         return Values{number(owner, n)};
                       }};
  };
  runtime.addScopeKernel(kernel("low", 0, 1));
  runtime.addScopeKernel(kernel("first", 5, 2));
  runtime.addScopeKernel(kernel("second", 5, 3));
  ScopeOptions options;
  options.max_lookahead = 2;
  auto result = runtime.runScope([](Scope &scope) { return scope.submit("copy", {}); }, options);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(read(result[0]), 2);
  EXPECT_EQ(matches, 1);
  EXPECT_EQ(calls, 0);
}

TEST(ScopeTest, RegistrySnapshotStartsBeforeCaptureAndMutationsApplyNextRun) {
  Runtime runtime;
  int calls = 0, selected = 0;
  registerCopy(runtime, calls);
  ScopeKernel added{"later", 99, 1, [](const ScopeView &, std::string &) { return 1u; },
                    [&](Runtime &owner, const ScopeView &) {
                      ++selected;
                      return Values{number(owner, 4)};
                    }};
  runtime.runScope([&](Scope &scope) {
    runtime.addScopeKernel(added);
    return scope.submit("copy", {});
  });
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(selected, 0);
  runtime.runScope([](Scope &scope) { return scope.submit("copy", {}); });
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(selected, 1);
}

TEST(ScopeTest, RejectsImmediateAndNestedExecutionDuringCaptureAndRestoresRuntime) {
  Runtime runtime;
  int calls = 0;
  registerCopy(runtime, calls);
  errorIs(ErrorCode::InvalidRequest, [&] {
    runtime.runScope([&](Scope &scope) -> ScopeValues {
      scope.submit("copy", {});
      errorIs(ErrorCode::InvalidRequest, [&] { runtime.submit({"copy", {}, {}}); });
      errorIs(ErrorCode::InvalidRequest, [&] { number(runtime); });
      errorIs(ErrorCode::InvalidRequest, [&] { runtime.runScope([](Scope &) { return ScopeValues{}; }); });
      throw Error(ErrorCode::InvalidRequest, "capture rejected");
    });
  });
  EXPECT_EQ(calls, 0);
  runtime.submit({"copy", {}, {}});
  EXPECT_EQ(calls, 1);
}

TEST(ScopeTest, RejectsForeignEmptyAndStaleSymbolsWithoutExecutingCapture) {
  Runtime runtime, foreign;
  auto source = number(runtime), other = number(foreign);
  ScopeValue stale;
  runtime.runScope([&](Scope &scope) {
    stale = scope.input(source);
    return ScopeValues{};
  });
  errorIs(ErrorCode::InvalidRequest,
          [&] { runtime.runScope([&](Scope &scope) { return scope.submit("copy", {stale}); }); });
  errorIs(ErrorCode::InvalidRequest, [&] {
    runtime.runScope([&](Scope &scope) {
      scope.input(other);
      return ScopeValues{};
    });
  });
  errorIs(ErrorCode::InvalidRequest, [&] {
    runtime.runScope([&](Scope &scope) {
      scope.input(Value{});
      return ScopeValues{};
    });
  });
  errorIs(ErrorCode::InvalidRequest, [&] { runtime.runScope([](Scope &) { return ScopeValues{ScopeValue{}}; }); });
  EXPECT_TRUE(runtime.owns(source));
  EXPECT_FALSE(foreign.owns(source));
  EXPECT_FALSE(runtime.owns({}));
}

TEST(ScopeTest, EmptyAndInputOnlyScopesHaveNoDispatch) {
  Runtime runtime;
  auto input = number(runtime);
  EXPECT_TRUE(runtime.runScope([](Scope &) { return ScopeValues{}; }).empty());
  auto outputs = runtime.runScope([&](Scope &scope) { return ScopeValues{scope.input(input)}; });
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_EQ(outputs[0].identity(), input.identity());
  EXPECT_TRUE(runtime.trace().empty());
}

TEST(ScopeTest, BoundsEveryCaptureDimensionAndValidatesBeforeMutation) {
  Runtime runtime;
  int calls = 0;
  registerCopy(runtime, calls);
  auto input = number(runtime);
  ScopeOptions options;
  options.max_commands = 1;
  options.max_values = 2;
  options.max_input_edges = 1;
  options.max_exports = 1;
  runtime.runScope(
    [&](Scope &scope) {
      auto imported = scope.input(input);
      errorIs(ErrorCode::InvalidRequest, [&] { scope.submit("", {}); });
      errorIs(ErrorCode::InvalidRequest, [&] { scope.submit("copy", {ScopeValue{}}); });
      errorIs(ErrorCode::BudgetExceeded,
              [&] { scope.submit("copy", {imported}, {}, std::numeric_limits<std::size_t>::max()); });
      errorIs(ErrorCode::BudgetExceeded, [&] { scope.submit("copy", {imported, imported}); });
      auto output = scope.submit("copy", {imported});
      errorIs(ErrorCode::BudgetExceeded, [&] { scope.submit("copy", {}, {}, 0); });
      return output;
    },
    options);
  EXPECT_EQ(calls, 1);
  errorIs(ErrorCode::BudgetExceeded, [&] {
    runtime.runScope(
      [&](Scope &scope) {
        auto v = scope.input(input);
        return ScopeValues{v, v};
      },
      options);
  });
  for (int field = 0; field != 6; ++field) {
    ScopeOptions invalid;
    std::size_t *caps[] = {&invalid.max_commands, &invalid.max_values,    &invalid.max_input_edges,
                           &invalid.max_exports,  &invalid.max_lookahead, &invalid.max_match_attempts};
    *caps[field] = 0;
    errorIs(ErrorCode::InvalidRequest, [&] { runtime.runScope([](Scope &) { return ScopeValues{}; }, invalid); });
  }
}

TEST(ScopeTest, ChecksFallbackMultioutputAndPermitsZeroOutputs) {
  Runtime runtime;
  int calls = 0;
  runtime.addRecipe({"pair", "pair", 0, [](const Request &, std::string &) { return true; },
                     [](Runtime &owner, const Request &) { return Values{number(owner, 2), number(owner, 3)}; }});
  runtime.addRecipe({"effect", "effect", 0, [](const Request &, std::string &) { return true; },
                     [&](Runtime &, const Request &) {
                       ++calls;
                       return Values{};
                     }});
  auto result = runtime.runScope([](Scope &scope) {
    scope.submit("effect", {}, {}, 0);
    return scope.submit("pair", {}, {}, 2);
  });
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(read(result[0]), 2);
  EXPECT_EQ(read(result[1]), 3);
  EXPECT_EQ(calls, 1);
  errorIs(ErrorCode::InvalidRequest, [&] { runtime.runScope([](Scope &scope) { return scope.submit("pair", {}); }); });
  EXPECT_EQ(runtime.stats().live_resources, 2u);
}

TEST(ScopeTest, MatchAttemptBudgetReleasesEarlierResultsAndKeepsImports) {
  Runtime runtime;
  int calls = 0;
  registerCopy(runtime, calls);
  auto input = number(runtime);
  runtime.addScopeKernel({"decline", 0, 1, [](const ScopeView &, std::string &) { return 0u; },
                          [](Runtime &, const ScopeView &) { return Values{}; }});
  ScopeOptions options;
  options.max_match_attempts = 1;
  errorIs(ErrorCode::BudgetExceeded, [&] {
    runtime.runScope(
      [&](Scope &scope) {
        auto v = scope.submit("copy", {scope.input(input)})[0];
        return scope.submit("copy", {v});
      },
      options);
  });
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(runtime.stats().live_resources, 1u);
  EXPECT_EQ(read(input), 1);
}

TEST(ScopeTest, MatcherAndExecutorErrorsNeverRetryOrPublish) {
  for (int mode = 0; mode != 6; ++mode) {
    Runtime runtime, foreign;
    int calls = 0;
    registerCopy(runtime, calls);
    auto wrong = number(foreign);
    runtime.addScopeKernel({"broken", 0, 1,
                            [mode](const ScopeView &, std::string &) -> std::size_t {
                              if (mode == 0)
                                throw std::runtime_error("match failed");
                              return mode == 1 ? 2u : 1u;
                            },
                            [mode, wrong](Runtime &owner, const ScopeView &) -> Values {
                              auto unpublished = number(owner);
                              if (mode == 2)
                                throw std::runtime_error("execute failed");
                              if (mode == 3)
                                return {};
                              if (mode == 4)
                                return {unpublished, unpublished};
                              return {wrong};
                            }});
    errorIs(mode == 0 || mode == 2 ? ErrorCode::ExecutionFailed : ErrorCode::InvalidRequest,
            [&] { runtime.runScope([](Scope &scope) { return scope.submit("copy", {}); }); });
    EXPECT_EQ(calls, 0);
    EXPECT_EQ(runtime.stats().live_resources, 0u);
    ASSERT_FALSE(runtime.trace().empty());
    EXPECT_EQ(runtime.trace().back().kind, TraceKind::Failed);
  }
}

TEST(ScopeTest, ReleasesResultsAfterLastUseButRetainsDependencyResources) {
  Runtime runtime({12, 256});
  int calls = 0;
  registerCopy(runtime, calls);
  auto result = runtime.runScope([](Scope &scope) {
    auto v = scope.submit("copy", {})[0];
    for (int i = 0; i != 20; ++i)
      v = scope.submit("copy", {v})[0];
    return ScopeValues{v};
  });
  EXPECT_EQ(calls, 21);
  EXPECT_EQ(runtime.stats().peak_live_bytes, 8u);
  EXPECT_EQ(runtime.stats().live_resources, 1u);
  result.clear();
  runtime.addRecipe({"artifact", "artifact", 0, [](const Request &, std::string &) { return true; },
                     [](Runtime &owner, const Request &request) { return Values{number(owner, 5, request.inputs)}; }});
  result = runtime.runScope([](Scope &scope) {
    auto v = scope.submit("copy", {})[0];
    return scope.submit("artifact", {v});
  });
  EXPECT_EQ(runtime.stats().live_resources, 2u);
  EXPECT_EQ(result[0].dependencies().size(), 1u);
  result.clear();
  EXPECT_EQ(runtime.stats().live_resources, 0u);
}

TEST(ScopeTest, DisablingKernelsUsesImmediateRecipesAndTraceRemainsBounded) {
  Runtime runtime({0, 3});
  int calls = 0;
  registerCopy(runtime, calls);
  runtime.addScopeKernel({"must not inspect", 0, 1,
                          [](const ScopeView &, std::string &) -> std::size_t { throw std::runtime_error("disabled"); },
                          [](Runtime &, const ScopeView &) { return Values{}; }});
  ScopeOptions options;
  options.use_kernels = false;
  auto result = runtime.runScope([](Scope &scope) { return scope.submit("copy", {}); }, options);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(read(result[0]), 1);
  EXPECT_EQ(runtime.trace().size(), 3u);
  errorIs(ErrorCode::InvalidRequest, [&] { runtime.addScopeKernel({}); });
}

TEST(ScopeTest, MatchersCannotSubmitOrAllocateButExecutorsCan) {
  Runtime runtime;
  int calls = 0;
  registerCopy(runtime, calls);
  runtime.addScopeKernel({"read-only", 0, 1,
                          [&](const ScopeView &, std::string &) {
                            errorIs(ErrorCode::InvalidRequest, [&] { number(runtime); });
                            errorIs(ErrorCode::InvalidRequest, [&] { runtime.submit({"copy", {}, {}}); });
                            return 1u;
                          },
                          [](Runtime &owner, const ScopeView &) { return owner.submit({"copy", {}, {}}); }});
  auto result = runtime.runScope([](Scope &scope) { return scope.submit("copy", {}); });
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(read(result[0]), 1);
  EXPECT_EQ(calls, 1);
}

TEST(ScopeTest, SpanPublishesIntermediateNeededOnlyByDownstreamCommand) {
  Runtime runtime;
  int calls = 0;
  registerCopy(runtime, calls);
  ScopeValue first, second;
  runtime.addScopeKernel(
    {"pair", 0, 2,
     [&](const ScopeView &view, std::string &) {
       if (view.start())
         return 0u;
       EXPECT_EQ(view.liveIns(2).size(), 0u);
       EXPECT_EQ(view.liveOuts(1), (ScopeValues{first}));
       EXPECT_EQ(view.liveOuts(2), (ScopeValues{first, second}));
       errorIs(ErrorCode::InvalidRequest, [&] { view.liveIns(3); });
       errorIs(ErrorCode::InvalidRequest, [&] { view.liveOuts(3); });
       errorIs(ErrorCode::InvalidRequest, [&] { view.find({}); });
       return 2u;
     },
     [](Runtime &owner, const ScopeView &) { return Values{number(owner, 11), number(owner, 12)}; }});
  auto result = runtime.runScope([&](Scope &scope) {
    first = scope.submit("copy", {})[0];
    second = scope.submit("copy", {first})[0];
    auto third = scope.submit("copy", {first})[0];
    return ScopeValues{third, second};
  });
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(read(result[0]), 11);
  EXPECT_EQ(read(result[1]), 12);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(runtime.stats().live_resources, 2u);
}

TEST(ScopeTest, LaterResourceBudgetFailureReleasesOutputsAndKeepsImports) {
  Runtime runtime({8, 256});
  int calls = 0;
  registerCopy(runtime, calls);
  auto input = number(runtime, 9);
  errorIs(ErrorCode::BudgetExceeded, [&] {
    runtime.runScope([&](Scope &scope) {
      auto v = scope.submit("copy", {scope.input(input)})[0];
      return scope.submit("copy", {v});
    });
  });
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(runtime.stats().live_resources, 1u);
  EXPECT_EQ(read(input), 9);
}

TEST(ScopeTest, EveryFailureTraceReportsSpanAccounting) {
  Runtime runtime;
  runtime.addScopeKernel(
    {"bad-match", 0, 1, [](const ScopeView &, std::string &) -> std::size_t { throw std::runtime_error("bad-match"); },
     [](Runtime &, const ScopeView &) { return Values{}; }});
  errorIs(ErrorCode::ExecutionFailed, [&] { runtime.runScope([](Scope &scope) { return scope.submit("x", {}); }); });
  ASSERT_EQ(runtime.trace().size(), 1u);
  const auto detail = runtime.trace()[0].detail;
  EXPECT_NE(detail.find("start=0"), std::string::npos);
  EXPECT_NE(detail.find("count="), std::string::npos);
  EXPECT_NE(detail.find("liveins="), std::string::npos);
  EXPECT_NE(detail.find("liveouts="), std::string::npos);
}

TEST(ScopeTest, PriorityWinsOverLongerOverlapAndLookaheadBoundsMatchers) {
  Runtime runtime;
  int calls = 0, high = 0, low = 0;
  registerCopy(runtime, calls);
  runtime.addScopeKernel({"long", 0, 8, [](const ScopeView &view, std::string &) { return view.size(); },
                          [&](Runtime &owner, const ScopeView &view) {
                            ++low;
                            Values result;
                            for (auto symbol : view.liveOuts(view.size())) {
                              (void)symbol;
                              result.push_back(number(owner, 8));
                            }
                            return result;
                          }});
  runtime.addScopeKernel({"short", 9, 8,
                          [](const ScopeView &view, std::string &) {
                            EXPECT_EQ(view.size(), 2u);
                            return view.start() == 0 ? 1u : 0u;
                          },
                          [&](Runtime &owner, const ScopeView &view) {
                            ++high;
                            EXPECT_EQ(view.size(), 1u);
                            return Values{number(owner, 7)};
                          }});
  ScopeOptions options;
  options.max_lookahead = 2;
  auto result = runtime.runScope(
    [](Scope &scope) {
      auto first = scope.submit("copy", {})[0];
      auto second = scope.submit("copy", {first})[0];
      return scope.submit("copy", {second});
    },
    options);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(read(result[0]), 8);
  EXPECT_EQ(high, 1);
  EXPECT_EQ(low, 1);
  EXPECT_EQ(calls, 0);
}

TEST(ScopeTest, LiveInsDeduplicateAndExcludeDefinitionsInsideSpan) {
  Runtime runtime;
  int calls = 0;
  registerCopy(runtime, calls);
  auto source = number(runtime, 4);
  ScopeValue imported, earlier;
  runtime.addScopeKernel({"after-first", 0, 2,
                          [&](const ScopeView &view, std::string &) {
                            if (view.start() == 0)
                              return 0u;
                            EXPECT_EQ(view.liveIns(0), ScopeValues{});
                            EXPECT_EQ(view.liveIns(2), (ScopeValues{earlier, imported}));
                            EXPECT_EQ(read(view.value(earlier)), 4);
                            return 2u;
                          },
                          [](Runtime &owner, const ScopeView &) { return Values{number(owner, 13)}; }});
  auto result = runtime.runScope([&](Scope &scope) {
    EXPECT_TRUE(scope.belongsTo(runtime));
    Runtime foreign;
    EXPECT_FALSE(scope.belongsTo(foreign));
    imported = scope.input(source);
    earlier = scope.submit("copy", {imported})[0];
    auto internal = scope.submit("copy", {earlier, imported, earlier})[0];
    return scope.submit("copy", {internal, imported});
  });
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(read(result[0]), 13);
  EXPECT_EQ(calls, 1);
}

TEST(ScopeTest, AttributesAreOwnedSnapshotsAndImportCapDoesNotBreakDeduplication) {
  Runtime runtime;
  runtime.addRecipe({"attributes", "attributes", 0, [](const Request &, std::string &) { return true; },
                     [](Runtime &owner, const Request &request) {
                       return Values{number(owner, std::any_cast<int>(request.attributes))};
                     }});
  int attribute = 12;
  auto result = runtime.runScope([&](Scope &scope) {
    auto recorded = scope.submit("attributes", {}, attribute);
    attribute = 99;
    return recorded;
  });
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(read(result[0]), 12);
  auto first = number(runtime), second = number(runtime);
  ScopeOptions options;
  options.max_values = 1;
  auto imported = runtime.runScope(
    [&](Scope &scope) {
      auto symbol = scope.input(first);
      errorIs(ErrorCode::BudgetExceeded, [&] { scope.input(second); });
      EXPECT_EQ(symbol, scope.input(first));
      return ScopeValues{symbol};
    },
    options);
  ASSERT_EQ(imported.size(), 1u);
  EXPECT_EQ(imported[0].identity(), first.identity());
}

TEST(ScopeTest, ExecutionRegistrationMutationAppliesOnlyToFutureScopes) {
  Runtime runtime;
  int original = 0, replacement = 0;
  ScopeKernel later{"replacement", 99, 1, [](const ScopeView &, std::string &) { return 1u; },
                    [&](Runtime &owner, const ScopeView &) {
                      ++replacement;
                      return Values{number(owner, 17)};
                    }};
  runtime.addScopeKernel({"original", 0, 1, [](const ScopeView &, std::string &) { return 1u; },
                          [&](Runtime &owner, const ScopeView &) {
                            ++original;
                            owner.addScopeKernel(later);
                            return Values{number(owner, 15)};
                          }});
  auto capture = [](Scope &scope) {
    auto first = scope.submit("unknown", {})[0];
    return scope.submit("unknown", {first});
  };
  auto result = runtime.runScope(capture);
  EXPECT_EQ(original, 2);
  EXPECT_EQ(replacement, 0);
  EXPECT_EQ(read(result[0]), 15);
  result = runtime.runScope(capture);
  EXPECT_EQ(original, 2);
  EXPECT_EQ(replacement, 2);
  EXPECT_EQ(read(result[0]), 17);
}

TEST(ScopeTest, DeadResultsAndZeroOutputCommandsRemainExecutable) {
  Runtime runtime;
  int calls = 0;
  runtime.addScopeKernel({"side-effects", 0, 2, [](const ScopeView &view, std::string &) { return view.size(); },
                          [&](Runtime &, const ScopeView &view) {
                            ++calls;
                            EXPECT_TRUE(view.liveOuts(view.size()).empty());
                            return Values{};
                          }});
  auto result = runtime.runScope([](Scope &scope) {
    scope.submit("dead", {});
    scope.submit("effect", {}, {}, 0);
    return ScopeValues{};
  });
  EXPECT_TRUE(result.empty());
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(runtime.stats().live_resources, 0u);
}

TEST(ScopeTest, InspectionCapturesOnceWithoutMatchingSelectingOrExecuting) {
  Runtime runtime;
  int captures = 0, inspections = 0, accepts = 0, executes = 0, matches = 0;
  runtime.addRecipe({"copy", "copy", 0,
                     [&](const Request &, std::string &) {
                       ++accepts;
                       return true;
                     },
                     [&](Runtime &owner, const Request &) {
                       ++executes;
                       return Values{number(owner)};
                     }});
  runtime.addScopeKernel({"must-not-run", 0, 3,
                          [&](const ScopeView &, std::string &) {
                            ++matches;
                            return 1u;
                          },
                          [&](Runtime &owner, const ScopeView &) {
                            ++executes;
                            return Values{number(owner)};
                          }});
  auto input = number(runtime, 7);
  ScopeValue imported, output;
  ScopeOptions options;
  options.max_lookahead = 1;
  options.max_match_attempts = 1;
  runtime.inspectScope(
    [&](Scope &scope) {
      ++captures;
      imported = scope.input(input);
      auto copied = scope.submit("copy", {imported}, 19)[0];
      output = scope.submit("unsupported", {copied})[0];
      return ScopeValues{output, output, imported};
    },
    [&](const ScopeView &view) {
      ++inspections;
      EXPECT_EQ(view.start(), 0u);
      EXPECT_EQ(view.size(), 2u);
      EXPECT_EQ(view.command(0).operation, "copy");
      EXPECT_EQ(std::any_cast<int>(view.command(0).attributes), 19);
      EXPECT_EQ(view.liveIns(2), (ScopeValues{imported}));
      EXPECT_EQ(view.liveOuts(2), (ScopeValues{output}));
      EXPECT_EQ(view.value(imported).identity(), input.identity());
      EXPECT_EQ(view.find(output), nullptr);
      errorIs(ErrorCode::InvalidRequest, [&] { view.value(output); });
    },
    options);
  EXPECT_EQ(captures, 1);
  EXPECT_EQ(inspections, 1);
  EXPECT_EQ(accepts, 0);
  EXPECT_EQ(matches, 0);
  EXPECT_EQ(executes, 0);
  EXPECT_EQ(runtime.stats().live_resources, 1u);
  EXPECT_EQ(runtime.stats().peak_live_bytes, 4u);
  EXPECT_TRUE(runtime.trace().empty());
}

TEST(ScopeTest, InspectionSubviewsKeepOuterFanoutAndExportedCacheLive) {
  Runtime runtime;
  auto input = number(runtime);
  ScopeValue imported, root, inner, cache, final;
  runtime.inspectScope(
    [&](Scope &scope) {
      imported = scope.input(input);
      root = scope.submit("root", {imported})[0];
      auto pair = scope.submit("pair", {root, imported, root}, {}, 2);
      inner = pair[0];
      cache = pair[1];
      final = scope.submit("next", {inner})[0];
      scope.submit("outer-use", {root, cache, final}, {}, 0);
      return ScopeValues{final, cache, cache, imported};
    },
    [&](const ScopeView &view) {
      EXPECT_EQ(view.liveOuts(4), (ScopeValues{cache, final}));
      const auto first = view.subview(0, 2);
      EXPECT_EQ(first.liveIns(2), (ScopeValues{imported}));
      EXPECT_EQ(first.liveOuts(2), (ScopeValues{root, inner, cache}));
      EXPECT_EQ(first.subview(0, 1).liveOuts(1), (ScopeValues{root}));
      const auto middle = view.subview(1, 2);
      EXPECT_EQ(middle.start(), 1u);
      EXPECT_EQ(middle.size(), 2u);
      EXPECT_EQ(middle.liveIns(2), (ScopeValues{root, imported}));
      EXPECT_EQ(middle.liveOuts(2), (ScopeValues{cache, final}));
      const auto nested = middle.subview(0, 1);
      EXPECT_EQ(nested.start(), 1u);
      EXPECT_EQ(nested.liveOuts(1), (ScopeValues{inner, cache}));
      EXPECT_EQ(nested.command(0).outputs, (ScopeValues{inner, cache}));
      EXPECT_EQ(nested.find(root), nullptr);
    });
}

TEST(ScopeTest, InspectionSupportsEmptyAndInputOnlyExports) {
  Runtime runtime;
  int inspections = 0;
  auto input = number(runtime);
  runtime.inspectScope([](Scope &) { return ScopeValues{}; },
                       [&](const ScopeView &view) {
                         ++inspections;
                         EXPECT_EQ(view.size(), 0u);
                         EXPECT_TRUE(view.liveIns(0).empty());
                         EXPECT_TRUE(view.liveOuts(0).empty());
                       });
  ScopeValue imported;
  runtime.inspectScope(
    [&](Scope &scope) {
      imported = scope.input(input);
      return ScopeValues{imported, imported};
    },
    [&](const ScopeView &view) {
      ++inspections;
      EXPECT_EQ(view.size(), 0u);
      EXPECT_EQ(view.value(imported).identity(), input.identity());
      EXPECT_TRUE(view.liveOuts(0).empty());
    });
  EXPECT_EQ(inspections, 2);
  EXPECT_TRUE(runtime.trace().empty());
}

TEST(ScopeTest, InspectionRejectsInvalidSubviewsWithoutOverflow) {
  Runtime runtime;
  runtime.inspectScope(
    [](Scope &scope) { return scope.submit("unknown", {}); },
    [](const ScopeView &view) {
      const auto empty = view.subview(1, 0);
      EXPECT_EQ(empty.start(), 1u);
      EXPECT_EQ(empty.size(), 0u);
      EXPECT_TRUE(empty.liveIns(0).empty());
      EXPECT_TRUE(empty.liveOuts(0).empty());
      errorIs(ErrorCode::InvalidRequest, [&] { empty.command(0); });
      errorIs(ErrorCode::InvalidRequest, [&] { empty.subview(0, 1); });
      errorIs(ErrorCode::InvalidRequest, [&] { view.subview(0, 2); });
      errorIs(ErrorCode::InvalidRequest, [&] { view.subview(2, 0); });
      errorIs(ErrorCode::InvalidRequest, [&] { view.subview(std::numeric_limits<std::size_t>::max(), 1); });
      errorIs(ErrorCode::InvalidRequest, [&] { view.subview(1, std::numeric_limits<std::size_t>::max()); });
      errorIs(ErrorCode::InvalidRequest, [&] { view.liveIns(2); });
      errorIs(ErrorCode::InvalidRequest, [&] { view.liveOuts(2); });
    });
}

TEST(ScopeTest, InspectionValidatesForeignEmptyAndStaleSymbolsBeforeCallback) {
  Runtime runtime, foreign;
  auto input = number(runtime), other = number(foreign);
  ScopeValue stale;
  runtime.inspectScope(
    [&](Scope &scope) {
      stale = scope.input(input);
      return ScopeValues{stale};
    },
    [](const ScopeView &) {});
  int inspections = 0;
  const auto inspect = [&](const ScopeView &) { ++inspections; };
  errorIs(ErrorCode::InvalidRequest,
          [&] { runtime.inspectScope([&](Scope &) { return ScopeValues{stale}; }, inspect); });
  errorIs(ErrorCode::InvalidRequest,
          [&] { runtime.inspectScope([](Scope &) { return ScopeValues{ScopeValue{}}; }, inspect); });
  errorIs(ErrorCode::InvalidRequest,
          [&] { runtime.inspectScope([&](Scope &scope) { return ScopeValues{scope.input(other)}; }, inspect); });
  errorIs(ErrorCode::InvalidRequest,
          [&] { runtime.inspectScope([](Scope &scope) { return ScopeValues{scope.input(Value{})}; }, inspect); });
  errorIs(ErrorCode::InvalidRequest,
          [&] { runtime.inspectScope([&](Scope &scope) { return scope.submit("unknown", {stale}); }, inspect); });
  runtime.inspectScope([](Scope &) { return ScopeValues{}; },
                       [&](const ScopeView &view) { errorIs(ErrorCode::InvalidRequest, [&] { view.find(stale); }); });
  EXPECT_EQ(inspections, 0);
}

TEST(ScopeTest, InspectionEnforcesAllCaptureCapsAndCountsDuplicateExports) {
  Runtime runtime;
  auto input = number(runtime);
  ScopeOptions options;
  options.max_commands = 1;
  options.max_values = 2;
  options.max_input_edges = 1;
  options.max_exports = 1;
  int inspections = 0;
  const auto inspect = [&](const ScopeView &) { ++inspections; };
  runtime.inspectScope(
    [&](Scope &scope) {
      const auto imported = scope.input(input);
      errorIs(ErrorCode::BudgetExceeded,
              [&] { scope.submit("unknown", {}, {}, std::numeric_limits<std::size_t>::max()); });
      errorIs(ErrorCode::BudgetExceeded, [&] { scope.submit("unknown", {imported, imported}); });
      auto output = scope.submit("unknown", {imported});
      errorIs(ErrorCode::BudgetExceeded, [&] { scope.submit("effect", {}, {}, 0); });
      return output;
    },
    inspect, options);
  EXPECT_EQ(inspections, 1);
  errorIs(ErrorCode::BudgetExceeded, [&] {
    runtime.inspectScope(
      [&](Scope &scope) {
        const auto imported = scope.input(input);
        return ScopeValues{imported, imported};
      },
      inspect, options);
  });
  for (int field = 0; field != 6; ++field) {
    ScopeOptions invalid;
    std::size_t *caps[] = {&invalid.max_commands, &invalid.max_values,    &invalid.max_input_edges,
                           &invalid.max_exports,  &invalid.max_lookahead, &invalid.max_match_attempts};
    *caps[field] = 0;
    errorIs(ErrorCode::InvalidRequest,
            [&] { runtime.inspectScope([](Scope &) { return ScopeValues{}; }, inspect, invalid); });
  }
  EXPECT_EQ(inspections, 1);
}

TEST(ScopeTest, InspectionRejectsMutationAndReentrantExecution) {
  Runtime runtime;
  int calls = 0, nested = 0;
  registerCopy(runtime, calls);
  auto input = number(runtime);
  Scope *recorded = nullptr;
  const auto capture = [&](Scope &) {
    ++nested;
    return ScopeValues{};
  };
  const auto inspect = [&](const ScopeView &) { ++nested; };
  Recipe recipe{"later", "later", 0, [](const Request &, std::string &) { return true; },
                [](Runtime &, const Request &) { return Values{}; }};
  ScopeKernel kernel{"later", 0, 1, [](const ScopeView &, std::string &) { return 0u; },
                     [](Runtime &, const ScopeView &) { return Values{}; }};
  const auto rejectRuntimeMutation = [&] {
    errorIs(ErrorCode::InvalidRequest, [&] { runtime.submit({"copy", {}, {}}); });
    errorIs(ErrorCode::InvalidRequest, [&] { number(runtime); });
    errorIs(ErrorCode::InvalidRequest, [&] { runtime.runScope(capture); });
    errorIs(ErrorCode::InvalidRequest, [&] { runtime.inspectScope(capture, inspect); });
    errorIs(ErrorCode::InvalidRequest, [&] { runtime.addRecipe(recipe); });
    errorIs(ErrorCode::InvalidRequest, [&] { runtime.addScopeKernel(kernel); });
  };
  runtime.inspectScope(
    [&](Scope &scope) {
      recorded = &scope;
      rejectRuntimeMutation();
      return scope.submit("copy", {scope.input(input)});
    },
    [&](const ScopeView &view) {
      rejectRuntimeMutation();
      errorIs(ErrorCode::InvalidRequest, [&] { recorded->submit("copy", {}); });
      errorIs(ErrorCode::InvalidRequest, [&] { recorded->input(input); });
      EXPECT_EQ(view.size(), 1u);
      EXPECT_EQ(runtime.stats().live_resources, 1u);
    });
  EXPECT_EQ(nested, 0);
  EXPECT_EQ(calls, 0);
  runtime.addRecipe(recipe);
  runtime.addScopeKernel(kernel);
  auto result = runtime.runScope([](Scope &scope) { return scope.submit("copy", {}); });
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(calls, 1);
}

TEST(ScopeTest, InspectionRestoresGuardsAfterEveryCallbackException) {
  Runtime runtime;
  int calls = 0;
  registerCopy(runtime, calls);
  for (int phase = 0; phase != 2; ++phase) {
    for (int kind = 0; kind != 3; ++kind) {
      const auto fail = [&] {
        if (kind == 0)
          throw Error(ErrorCode::BudgetExceeded, "callback rejected");
        if (kind == 1)
          throw std::runtime_error("callback failed");
        throw 17;
      };
      errorIs(kind == 0 ? ErrorCode::BudgetExceeded : ErrorCode::ExecutionFailed, [&] {
        runtime.inspectScope(
          [&](Scope &scope) {
            auto output = scope.submit("copy", {});
            if (phase == 0)
              fail();
            return output;
          },
          [&](const ScopeView &) { fail(); });
      });
      runtime.inspectScope([](Scope &) { return ScopeValues{}; }, [](const ScopeView &) {});
      auto output = runtime.submit({"copy", {}, {}});
      ASSERT_EQ(output.size(), 1u);
      EXPECT_EQ(read(output[0]), 1);
    }
  }
  EXPECT_EQ(calls, 6);
  errorIs(ErrorCode::InvalidRequest, [&] { runtime.inspectScope({}, [](const ScopeView &) {}); });
  errorIs(ErrorCode::InvalidRequest, [&] { runtime.inspectScope([](Scope &) { return ScopeValues{}; }, {}); });
  EXPECT_TRUE(runtime.runScope([](Scope &) { return ScopeValues{}; }).empty());
}

// A provider contract carries public metadata; the core does not interpret int.
std::shared_ptr<ScopeContract> integerContract() {
  auto contract = std::make_shared<ScopeContract>();
  contract->describe = [](const Value &value) {
    return ScopeDescriptor{value.format(), value.kind(), value.bytes(), read(value)};
  };
  contract->infer = [](const std::string &operation, const std::vector<ScopeDescriptor> &inputs, const std::any &) {
    if (operation != "copy")
      throw Error(ErrorCode::Unsupported, "unknown operation in metadata preflight");
    return std::vector<ScopeDescriptor>{
      {"scope.test", ResourceKind::Value, 4, inputs.empty() ? 1 : std::any_cast<int>(inputs[0].metadata)}};
  };
  contract->validate = [](const Value &value, const ScopeDescriptor &descriptor) {
    if (read(value) != std::any_cast<int>(descriptor.metadata))
      throw Error(ErrorCode::InvalidRequest, "provider metadata mismatch");
  };
  return contract;
}

TEST(ScopeTest, ContractDescribesSymbolicOutputsBeforeExecutionAndCountsCompletedSpans) {
  Runtime runtime;
  int calls = 0, descriptions = 0, validations = 0;
  registerCopy(runtime, calls);
  auto input = number(runtime, 7);
  auto contract = integerContract();
  auto originalDescribe = contract->describe;
  contract->describe = [&](const Value &value) {
    ++descriptions;
    return originalDescribe(value);
  };
  auto originalValidate = contract->validate;
  contract->validate = [&](const Value &value, const ScopeDescriptor &descriptor) {
    ++validations;
    originalValidate(value, descriptor);
  };
  ScopeOptions options;
  options.contract = contract;
  runtime.addScopeKernel({"pair", 1, 2,
                          [&](const ScopeView &view, std::string &) {
                            EXPECT_EQ(std::any_cast<int>(view.descriptor(view.command(1).outputs[0]).metadata), 7);
                            EXPECT_EQ(calls, 0);
                            return 2u;
                          },
                          [&](Runtime &owner, const ScopeView &view) {
                            ++calls;
                            return Values{number(owner, read(view.value(view.command(0).inputs[0])))};
                          }});
  auto output = runtime.runScope(
    [&](Scope &scope) {
      auto first = scope.input(input);
      EXPECT_EQ(first, scope.input(input));
      auto a = scope.submit("copy", {first})[0];
      auto b = scope.submit("copy", {a})[0];
      EXPECT_EQ(scope.descriptor(b).format, "scope.test");
      EXPECT_EQ(std::any_cast<int>(scope.descriptor(b).metadata), 7);
      return ScopeValues{b, b};
    },
    options);
  ASSERT_EQ(output.size(), 2u);
  EXPECT_EQ(read(output[0]), 7);
  EXPECT_EQ(descriptions, 1);
  EXPECT_EQ(validations, 1);
  auto stats = runtime.scopeStats();
  EXPECT_EQ(stats.commands, 2u);
  EXPECT_EQ(stats.values, 3u);
  EXPECT_EQ(stats.imports, 1u);
  EXPECT_EQ(stats.input_edges, 2u);
  EXPECT_EQ(stats.exports, 2u);
  EXPECT_EQ(stats.executed_commands, 2u);
  EXPECT_EQ(stats.recipe_spans, 0u);
  EXPECT_EQ(stats.kernel_spans, 1u);
  runtime.inspectScope([&](Scope &scope) { return scope.submit("copy", {scope.input(input)}); },
                       [&](const ScopeView &view) {
                         EXPECT_EQ(std::any_cast<int>(view.descriptor(view.command(0).outputs[0]).metadata), 7);
                         EXPECT_EQ(view.find(view.command(0).outputs[0]), nullptr);
                       },
                       options);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(runtime.scopeStats().commands, 1u);
  EXPECT_EQ(runtime.scopeStats().executed_commands, 0u);
  EXPECT_EQ(runtime.scopeStats().kernel_spans, 0u);
}

TEST(ScopeTest, InvalidFinalMetadataCommandPreventsEveryExecutorAndPreservesCaptureProgress) {
  Runtime runtime;
  int calls = 0;
  registerCopy(runtime, calls);
  ScopeOptions options;
  options.contract = integerContract();
  errorIs(ErrorCode::Unsupported, [&] {
    runtime.runScope(
      [&](Scope &scope) {
        auto a = scope.submit("copy", {})[0];
        auto b = scope.submit("copy", {a})[0];
        errorIs(ErrorCode::InvalidRequest, [&] { runtime.runScope([](Scope &) { return ScopeValues{}; }); });
        EXPECT_EQ(runtime.scopeStats().commands, 2u);
        return scope.submit("not-supported", {b});
      },
      options);
  });
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(runtime.scopeStats().commands, 2u);
  EXPECT_EQ(runtime.scopeStats().values, 2u);
  EXPECT_EQ(runtime.scopeStats().input_edges, 1u);
  EXPECT_EQ(runtime.scopeStats().executed_commands, 0u);
  EXPECT_EQ(runtime.scopeStats().exports, 0u);
  runtime.runScope([](Scope &) { return ScopeValues{}; });
  EXPECT_EQ(runtime.scopeStats().commands, 0u);
}

TEST(ScopeTest, ContractRejectsIncompleteCallbacksAndInconsistentInputDescriptors) {
  Runtime runtime;
  auto input = number(runtime, 7);
  for (int missing = 0; missing != 3; ++missing) {
    auto contract = integerContract();
    if (missing == 0)
      contract->describe = {};
    if (missing == 1)
      contract->infer = {};
    if (missing == 2)
      contract->validate = {};
    ScopeOptions options;
    options.contract = contract;
    errorIs(ErrorCode::InvalidRequest, [&] { runtime.runScope([](Scope &) { return ScopeValues{}; }, options); });
  }
  for (int mismatch = 0; mismatch != 4; ++mismatch) {
    auto contract = integerContract();
    contract->describe = [mismatch](const Value &) {
      ScopeDescriptor descriptor{"scope.test", ResourceKind::Value, 4, 7};
      if (mismatch == 0)
        descriptor.format = "";
      if (mismatch == 1)
        descriptor.kind = ResourceKind::Artifact;
      if (mismatch == 2)
        descriptor.bytes = 5;
      if (mismatch == 3)
        descriptor.kind = static_cast<ResourceKind>(-1);
      return descriptor;
    };
    ScopeOptions options;
    options.contract = contract;
    errorIs(ErrorCode::InvalidRequest,
            [&] { runtime.runScope([&](Scope &scope) { return ScopeValues{scope.input(input)}; }, options); });
    EXPECT_EQ(runtime.scopeStats().values, 0u);
    EXPECT_EQ(runtime.scopeStats().imports, 0u);
  }
}

TEST(ScopeTest, BadInferredDescriptorsDoNotAppendCommandsOrSymbols) {
  Runtime runtime;
  for (int mismatch = 0; mismatch != 3; ++mismatch) {
    auto contract = integerContract();
    contract->infer = [mismatch](const std::string &, const std::vector<ScopeDescriptor> &, const std::any &) {
      ScopeDescriptor descriptor{"scope.test", ResourceKind::Value, 4, 1};
      if (mismatch == 0)
        return std::vector<ScopeDescriptor>{};
      if (mismatch == 1)
        descriptor.format.clear();
      if (mismatch == 2)
        descriptor.kind = static_cast<ResourceKind>(-1);
      return std::vector<ScopeDescriptor>{descriptor};
    };
    ScopeOptions options;
    options.contract = contract;
    errorIs(ErrorCode::InvalidRequest,
            [&] { runtime.runScope([](Scope &scope) { return scope.submit("copy", {}); }, options); });
    EXPECT_EQ(runtime.scopeStats().commands, 0u);
    EXPECT_EQ(runtime.scopeStats().values, 0u);
  }
}

TEST(ScopeTest, ContractValidatesFallbackAndKernelOutputsBeforePublishingAnySpan) {
  for (bool kernel : {false, true}) {
    for (int mismatch = 0; mismatch != 4; ++mismatch) {
      Runtime runtime;
      int calls = 0;
      auto produce = [&](Runtime &owner) {
        ++calls;
        if (calls == 1)
          return Values{number(owner, 1)};
        return Values{owner.makeValue(mismatch == 0 ? "wrong" : "scope.test",
                                      mismatch == 1 ? ResourceKind::Artifact : ResourceKind::Value,
                                      mismatch == 2 ? 5 : 4, [&] { return std::any(mismatch == 3 ? 99 : 1); })};
      };
      runtime.addRecipe({"copy", "copy", 0, [](const Request &, std::string &) { return true; },
                         [&](Runtime &owner, const Request &) { return produce(owner); }});
      if (kernel)
        runtime.addScopeKernel({"one", 0, 1, [](const ScopeView &, std::string &) { return 1u; },
                                [&](Runtime &owner, const ScopeView &) { return produce(owner); }});
      ScopeOptions options;
      options.contract = integerContract();
      errorIs(ErrorCode::InvalidRequest, [&] {
        runtime.runScope(
          [](Scope &scope) {
            auto a = scope.submit("copy", {})[0];
            auto b = scope.submit("copy", {a})[0];
            return scope.submit("copy", {b});
          },
          options);
      });
      EXPECT_EQ(calls, 2);
      EXPECT_EQ(runtime.scopeStats().executed_commands, 1u);
      EXPECT_EQ(runtime.scopeStats().recipe_spans, kernel ? 0u : 1u);
      EXPECT_EQ(runtime.scopeStats().kernel_spans, kernel ? 1u : 0u);
      EXPECT_EQ(runtime.stats().live_resources, 0u);
    }
  }
}

TEST(ScopeTest, StrictContractsFreezeRegistriesAndMetadataCallbacksCannotEvaluate) {
  Runtime runtime;
  int calls = 0, phases = 0;
  registerCopy(runtime, calls);
  auto input = number(runtime);
  const auto frozen = [&] {
    errorIs(ErrorCode::InvalidRequest, [&] { registerCopy(runtime, calls); });
    errorIs(ErrorCode::InvalidRequest, [&] {
      runtime.addScopeKernel({"new", 0, 1, [](const ScopeView &, std::string &) { return 0u; },
                              [](Runtime &, const ScopeView &) { return Values{}; }});
    });
    ++phases;
  };
  const auto metadataOnly = [&] {
    frozen();
    errorIs(ErrorCode::InvalidRequest, [&] { runtime.submit({"copy", {}, {}}); });
    errorIs(ErrorCode::InvalidRequest, [&] { number(runtime); });
  };
  auto contract = integerContract();
  const auto describe = contract->describe;
  const auto infer = contract->infer;
  const auto validateOutput = contract->validate;
  contract->describe = [&](const Value &value) {
    metadataOnly();
    return describe(value);
  };
  contract->infer = [&](const std::string &op, const std::vector<ScopeDescriptor> &inputs, const std::any &attrs) {
    metadataOnly();
    return infer(op, inputs, attrs);
  };
  contract->validate = [&](const Value &value, const ScopeDescriptor &desc) {
    metadataOnly();
    validateOutput(value, desc);
  };
  runtime.addScopeKernel({"one", 0, 1,
                          [&](const ScopeView &, std::string &) {
                            frozen();
                            return 1u;
                          },
                          [&](Runtime &owner, const ScopeView &) {
                            frozen();
                            return Values{number(owner)};
                          }});
  ScopeOptions options;
  options.contract = contract;
  auto outputs = runtime.runScope(
    [&](Scope &scope) {
      frozen();
      return scope.submit("copy", {scope.input(input)});
    },
    options);
  EXPECT_EQ(phases, 6);
  EXPECT_EQ(read(outputs[0]), 1);
  EXPECT_EQ(calls, 0);
  // A throwing metadata validation must restore allocation and registration.
  contract->validate = [](const Value &, const ScopeDescriptor &) { throw std::runtime_error("metadata error"); };
  errorIs(ErrorCode::ExecutionFailed,
          [&] { runtime.runScope([](Scope &scope) { return scope.submit("copy", {}); }, options); });
  registerCopy(runtime, calls);
  EXPECT_EQ(read(runtime.submit({"copy", {}, {}})[0]), 1);
  EXPECT_EQ(read(number(runtime)), 1);
}

TEST(ScopeTest, DescriptorAccessRejectsUncontractedForeignAndExpiredSymbols) {
  Runtime runtime;
  ScopeValue stale;
  ScopeOptions options;
  options.contract = integerContract();
  runtime.inspectScope(
    [&](Scope &scope) {
      stale = scope.submit("copy", {})[0];
      return ScopeValues{};
    },
    [](const ScopeView &) {}, options);
  runtime.inspectScope(
    [&](Scope &scope) {
      auto current = scope.submit("copy", {})[0];
      errorIs(ErrorCode::InvalidRequest, [&] { scope.descriptor(stale); });
      errorIs(ErrorCode::InvalidRequest, [&] { scope.descriptor(ScopeValue{}); });
      return ScopeValues{current};
    },
    [&](const ScopeView &view) { errorIs(ErrorCode::InvalidRequest, [&] { view.descriptor(stale); }); }, options);
  runtime.inspectScope(
    [&](Scope &scope) {
      auto current = scope.submit("copy", {})[0];
      errorIs(ErrorCode::InvalidRequest, [&] { scope.descriptor(current); });
      return ScopeValues{current};
    },
    [&](const ScopeView &view) {
      errorIs(ErrorCode::InvalidRequest, [&] { view.descriptor(view.command(0).outputs[0]); });
    });
}

TEST(ScopeTest, HundredThousandCommandsReleaseResourcesWithinTwoValueBudget) {
  constexpr std::size_t count = 100000;
  std::size_t created = 0, destroyed = 0, validated = 0;
  struct Tracked {
    std::size_t &destroyed;
    explicit Tracked(std::size_t &counter) : destroyed(counter) {}
    ~Tracked() { ++destroyed; }
  };
  Runtime runtime({8, 0});
  runtime.addRecipe({"step", "step", 0, [](const Request &, std::string &) { return true; },
                     [&](Runtime &owner, const Request &) {
                       return Values{owner.makeValue("tracked", ResourceKind::Value, 4, [&] {
                         ++created;
                         return std::any(std::make_shared<Tracked>(destroyed));
                       })};
                     }});
  ScopeOptions options;
  options.max_commands = count;
  options.max_values = count;
  options.max_input_edges = count;
  auto contract = std::make_shared<ScopeContract>();
  contract->describe = [](const Value &value) {
    return ScopeDescriptor{value.format(), value.kind(), value.bytes(), {}};
  };
  contract->infer = [](const std::string &, const std::vector<ScopeDescriptor> &, const std::any &) {
    return std::vector<ScopeDescriptor>{{"tracked", ResourceKind::Value, 4, {}}};
  };
  contract->validate = [&](const Value &, const ScopeDescriptor &) { ++validated; };
  options.contract = contract;
  auto result = runtime.runScope(
    [&](Scope &scope) {
      auto previous = scope.submit("step", {})[0];
      for (std::size_t i = 1; i < count; ++i)
        previous = scope.submit("step", {previous})[0];
      return ScopeValues{previous};
    },
    options);
  EXPECT_EQ(created, count);
  EXPECT_EQ(validated, count);
  EXPECT_EQ(destroyed, count - 1);
  EXPECT_EQ(runtime.stats().peak_live_bytes, 8u);
  EXPECT_EQ(runtime.stats().live_bytes, 4u);
  const auto stats = runtime.scopeStats();
  EXPECT_EQ(stats.commands, count);
  EXPECT_EQ(stats.values, count);
  EXPECT_EQ(stats.imports, 0u);
  EXPECT_EQ(stats.input_edges, count - 1);
  EXPECT_EQ(stats.exports, 1u);
  EXPECT_EQ(stats.executed_commands, count);
  EXPECT_EQ(stats.recipe_spans, count);
  EXPECT_EQ(stats.kernel_spans, 0u);
  result.clear();
  EXPECT_EQ(destroyed, count);
  EXPECT_EQ(runtime.stats().live_resources, 0u);
}

TEST(ScopeTest, DurationBudgetRejectsExpiredCaptureBeforeAnyExecution) {
  Runtime runtime;
  int calls = 0;
  registerCopy(runtime, calls);
  ScopeOptions options;
  options.max_duration_ms = 1;
  errorIs(ErrorCode::BudgetExceeded, [&] {
    runtime.runScope(
      [](Scope &scope) {
        auto output = scope.submit("copy", {});
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        return output;
      },
      options);
  });
  EXPECT_EQ(calls, 0);
  EXPECT_EQ(runtime.scopeStats().commands, 1u);
  EXPECT_EQ(runtime.scopeStats().executed_commands, 0u);
  errorIs(ErrorCode::BudgetExceeded, [&] {
    runtime.inspectScope([](Scope &) { return ScopeValues{}; },
                         [](const ScopeView &) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); }, options);
  });
  EXPECT_EQ(runtime.scopeStats().commands, 0u);
}

TEST(ScopeTest, DurationBudgetStopsAtExecutionBoundaryAndRetainsCompletedProgress) {
  for (std::size_t commands : {1u, 2u}) {
    Runtime runtime;
    int calls = 0;
    runtime.addRecipe({"slow", "slow", 0, [](const Request &, std::string &) { return true; },
                       [&](Runtime &owner, const Request &) {
                         ++calls;
                         std::this_thread::sleep_for(std::chrono::milliseconds(30));
                         return Values{number(owner)};
                       }});
    ScopeOptions options;
    options.max_duration_ms = 20;
    errorIs(ErrorCode::BudgetExceeded, [&] {
      runtime.runScope(
        [&](Scope &scope) {
          auto previous = scope.submit("slow", {})[0];
          if (commands == 2)
            previous = scope.submit("slow", {previous})[0];
          return ScopeValues{previous};
        },
        options);
    });
    EXPECT_EQ(calls, 1);
    EXPECT_EQ(runtime.scopeStats().commands, commands);
    EXPECT_EQ(runtime.scopeStats().executed_commands, 1u);
    EXPECT_EQ(runtime.scopeStats().recipe_spans, 1u);
    EXPECT_EQ(runtime.stats().live_resources, 0u);
    // Duration failure releases all guards for subsequent execution.
    EXPECT_TRUE(runtime.runScope([](Scope &) { return ScopeValues{}; }).empty());
  }
}

TEST(ScopeTest, ContractCallbacksCannotReenterRecordingAndAreSnapshotted) {
  Runtime runtime;
  int calls = 0;
  registerCopy(runtime, calls);
  auto input = number(runtime);
  Scope *capturing = nullptr;
  auto contract = integerContract();
  const auto infer = contract->infer;
  contract->infer = [&](const std::string &op, const std::vector<ScopeDescriptor> &inputs, const std::any &attrs) {
    errorIs(ErrorCode::InvalidRequest, [&] { capturing->submit("copy", {}); });
    errorIs(ErrorCode::InvalidRequest, [&] { capturing->input(input); });
    return infer(op, inputs, attrs);
  };
  ScopeOptions options;
  options.contract = contract;
  auto outputs = runtime.runScope(
    [&](Scope &scope) {
      capturing = &scope;
      contract->infer = {};
      contract->validate = {};
      return scope.submit("copy", {scope.input(input)});
    },
    options);
  EXPECT_EQ(read(outputs[0]), 1);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(runtime.scopeStats().commands, 1u);
  EXPECT_EQ(runtime.scopeStats().values, 2u);
}
} // namespace
