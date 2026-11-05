#include "FHN/Scope.h"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
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
} // namespace
