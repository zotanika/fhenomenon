#include "../examples/private_search/SearchWorkflow.h"
#include "FHN/NativeRuntime.h"
#include "Session/Session.h"
#include <gtest/gtest.h>

namespace {
using namespace fhenomenon;
using namespace fhenomenon::runtime;

TEST(NativeScope, SessionCapturesOnceAndRestoresPreviousSessionOnSuccessAndFailure) {
  Runtime rt;
  auto native = Session::create(rt);
  auto previous = Session::create(Backend::getInstance());
  auto input = rt.makeValue("test.int", ResourceKind::Value, sizeof(int), [] { return std::any(7); });
  int calls = 0;
  rt.addRecipe({"increment", "test.increment", 0, [](const Request &, std::string &) { return true; },
                [&](Runtime &runtime, const Request &r) {
                  ++calls;
                  EXPECT_TRUE(Session::isRecording());
                  return Values{runtime.makeValue("test.int", ResourceKind::Value, sizeof(int), [&] {
                    return std::any(std::any_cast<int>(r.inputs[0].payload()) + 1);
                  })};
                }});
  int captures = 0;
  auto result = native->run([&](Scope &scope) {
    ++captures;
    EXPECT_TRUE(Session::isRecording());
    auto next = scope.submit("increment", {scope.input(input)});
    EXPECT_EQ(calls, 0);
    return next;
  });
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(std::any_cast<int>(result[0].payload()), 8);
  EXPECT_EQ(captures, 1);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(Session::getSession(), previous);
  EXPECT_FALSE(Session::isRecording());
  EXPECT_THROW(native->run([&](Scope &scope) -> ScopeValues {
    scope.submit("increment", {scope.input(input)});
    throw std::runtime_error("capture failed");
  }),
               std::runtime_error);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(Session::getSession(), previous);
  EXPECT_THROW(native->run([](Scope &scope) { return scope.submit("missing.operation", {}); }), Error);
  EXPECT_EQ(Session::getSession(), previous);
  EXPECT_FALSE(native->isActive());
  EXPECT_EQ(native->run([&](Scope &scope) { return ScopeValues{scope.input(input)}; })[0].identity(), input.identity());
}

TEST(NativeScope, RejectsCallbackModeMismatchNestingAndMixedFhenonRecording) {
  Runtime rt;
  auto native = Session::create(rt);
  auto legacy = Session::create(Backend::getInstance());
  EXPECT_THROW(native->run([] {}), std::logic_error);
  EXPECT_THROW(legacy->run([](Scope &) { return ScopeValues{}; }), std::logic_error);
  auto profile = Profile::createProfile(ParameterGen::createCKKSParam(CKKSParamPreset::FGb));
  Fhenon<int> a = 2;
  a.belong(profile);
  native->run([&](Scope &) {
    EXPECT_THROW(Session::create(rt), std::logic_error);
    EXPECT_THROW(legacy->run([] {}), std::logic_error);
    EXPECT_THROW(native->run([](Scope &) { return ScopeValues{}; }), std::logic_error);
    return ScopeValues{};
  });
  int calls = 0;
  rt.addRecipe({"effect", "effect", 0, [](const Request &, std::string &) { return true; },
                [&](Runtime &, const Request &) {
                  ++calls;
                  return Values{};
                }});
  for (int mixed = 0; mixed < 4; ++mixed) {
    Fhenon<int> b = 5;
    b.belong(profile);
    EXPECT_THROW(native->run([&](Scope &scope) {
      scope.submit("effect", {}, {}, 0);
      if (mixed == 0)
        a = a + 1;
      else if (mixed == 1)
        a = std::move(b); // noexcept Fhenon assignment must not terminate.
      else if (mixed == 2) {
        Fhenon<int> moved(std::move(b));
      } else {
        a = a + b;
      }
      return ScopeValues{};
    }),
                 Error);
    EXPECT_EQ(calls, 0);
    EXPECT_FALSE(native->isActive());
  }
  EXPECT_FALSE(Session::isRecording());
  EXPECT_EQ(Session::getSession(), legacy);
  legacy->run([&] { a = a + 1; });
  EXPECT_EQ(a.decrypt(), 3);
}

TEST(NativeScope, RejectsMixedFrontendInsideContributedExecutorAndRestoresSession) {
  Runtime rt;
  auto native = Session::create(rt);
  auto previous = Session::create(Backend::getInstance());
  auto profile = Profile::createProfile(ParameterGen::createCKKSParam(CKKSParamPreset::FGb));
  Fhenon<int> a = 2, b = 5;
  a.belong(profile);
  b.belong(profile);
  rt.addRecipe({"mixed", "mixed", 0, [](const Request &, std::string &) { return true; },
                [&](Runtime &, const Request &) {
                  a = a + b;
                  return Values{};
                }});
  EXPECT_THROW(native->run([](Scope &scope) { return scope.submit("mixed", {}, {}, 0); }), std::logic_error);
  EXPECT_EQ(Session::getSession(), previous);
  EXPECT_FALSE(native->isActive());
  EXPECT_EQ(a.decrypt(), 2);
}

TEST(NativeScope, SessionHonorsRecordingLimitsWithoutExecutingPartialCapture) {
  Runtime rt;
  ScopeOptions options;
  options.max_commands = 1;
  auto session = Session::create(rt, options);
  int calls = 0;
  rt.addRecipe({"effect", "effect", 0, [](const Request &, std::string &) { return true; },
                [&](Runtime &, const Request &) {
                  ++calls;
                  return Values{};
                }});
  EXPECT_THROW(session->run([](Scope &scope) {
    scope.submit("effect", {}, {}, 0);
    return scope.submit("effect", {}, {}, 0);
  }),
               Error);
  EXPECT_EQ(calls, 0);
  EXPECT_FALSE(session->isActive());
}
TEST(NativeScope, GenuineCkksSharesPreparedStateAndIntermediateAcrossScopesWithoutCompiler) {
  Runtime rt({0, 1024});
  private_search::TrustedClient client(rt, FHN_NATIVE_TEST_LIBRARY);
  auto services = client.computeServices();
  ASSERT_EQ(services.compile, nullptr);
  NativeRuntime native(rt, services);
  auto session = Session::create(rt);
  std::vector<double> plain(private_search::slots);
  const double pattern[]{0.25, -1, 2, 0.5};
  for (size_t i = 0; i < plain.size(); ++i)
    plain[i] = pattern[i % 4];
  auto source = client.encryptVector(native, plain);
  const auto baseline = rt.stats().live_bytes;
  const auto trace_before = rt.trace().size();
  auto outputs = session->run([&](Scope &scope) {
    auto x = scope.input(source);
    auto prep = native.prepareRotation(scope, x);
    auto rotated = native.rotate(scope, x, 1, prep);
    auto sum = native.op(scope, FHN_ADD_CC, {x, rotated});
    EXPECT_EQ(rt.stats().live_bytes, baseline);
    EXPECT_EQ(rt.trace().size(), trace_before);
    return ScopeValues{prep, rotated, sum};
  });
  ASSERT_EQ(outputs.size(), 3u);
  ASSERT_EQ(outputs[0].dependencies().size(), 1u);
  EXPECT_EQ(outputs[0].dependencies()[0].identity(), source.identity());
  const auto first = client.decryptVector(native, outputs[1]);
  const auto sum = client.decryptVector(native, outputs[2]);
  for (size_t i = 0; i < plain.size(); ++i) {
    EXPECT_NEAR(first[i], plain[(i + 1) % plain.size()], 1e-5);
    EXPECT_NEAR(sum[i], plain[i] + plain[(i + 1) % plain.size()], 1e-5);
  }
  const auto next = session->run([&](Scope &scope) {
    const auto prep = scope.input(outputs[0]);
    const auto rotated = native.rotate(scope, scope.input(source), 2, prep);
    const auto combined = native.op(scope, FHN_ADD_CC, {rotated, scope.input(outputs[1])});
    return ScopeValues{prep, combined};
  });
  EXPECT_EQ(next[0].identity(), outputs[0].identity());
  const auto actual = client.decryptVector(native, next[1]);
  for (size_t i = 0; i < plain.size(); ++i)
    EXPECT_NEAR(actual[i], plain[(i + 2) % plain.size()] + plain[(i + 1) % plain.size()], 1e-5);
  size_t preparations = 0;
  for (const auto &event : rt.trace())
    if (event.kind == TraceKind::Selected && event.operation == "native.prepare_rotation")
      ++preparations;
  EXPECT_EQ(preparations, 1u);

  Runtime foreign;
  auto foreign_source = foreign.makeValue("dummy", ResourceKind::Value, 1, [] { return std::any(0); });
  foreign.runScope([&](Scope &scope) {
    auto x = scope.input(foreign_source);
    EXPECT_THROW(native.op(scope, FHN_NEGATE, {x}), Error);
    EXPECT_THROW(native.prepareRotation(scope, x), Error);
    EXPECT_THROW(native.rotate(scope, x, 1), Error);
    return ScopeValues{};
  });
}
} // namespace
