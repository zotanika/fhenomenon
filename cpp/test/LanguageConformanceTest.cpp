#include "FHN/Scope.h"
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <unordered_map>

#ifndef FHN_CONFORMANCE_FIXTURE
#error "FHN_CONFORMANCE_FIXTURE must point to contracts/runtime-v1.json"
#endif

namespace {
using namespace fhenomenon::runtime;
using Json = nlohmann::json;
Value integer(Runtime &rt, std::int64_t n) {
  return rt.makeValue("integer", ResourceKind::Value, 8, [n] { return n; });
}
std::int64_t number(const Value &v) { return std::any_cast<std::int64_t>(v.payload()); }
std::string codeName(ErrorCode code) {
  switch (code) {
  case ErrorCode::Unsupported:
    return "Unsupported";
  case ErrorCode::InvalidRequest:
    return "InvalidRequest";
  case ErrorCode::BudgetExceeded:
    return "BudgetExceeded";
  case ErrorCode::ExecutionFailed:
    return "ExecutionFailed";
  }
  return "unknown";
}
ScopeOptions options(const Json &test) {
  ScopeOptions result;
  auto contract = std::make_shared<ScopeContract>();
  contract->describe = [](const Value &v) {
    number(v);
    return ScopeDescriptor{v.format(), v.kind(), v.bytes(), {}};
  };
  contract->infer = [](const std::string &op, const std::vector<ScopeDescriptor> &inputs, const std::any &) {
    std::size_t arity = 1, outputs = 1;
    if (op == "add")
      arity = 2;
    else if (op == "split")
      outputs = 2;
    else if (op == "observe")
      outputs = 0;
    else if (op != "missing")
      throw Error(ErrorCode::InvalidRequest, "invalid integer operation");
    if (inputs.size() != arity)
      throw Error(ErrorCode::InvalidRequest, "incorrect integer arity");
    return std::vector<ScopeDescriptor>(outputs, ScopeDescriptor{"integer", ResourceKind::Value, 8, {}});
  };
  contract->validate = [](const Value &v, const ScopeDescriptor &) { number(v); };
  result.contract = contract;
  result.use_kernels = test.value("use_kernels", true);
  if (test.contains("caps")) {
    for (auto it = test.at("caps").begin(); it != test.at("caps").end(); ++it) {
      const auto n = it.value().get<std::size_t>();
      if (it.key() == "max_commands")
        result.max_commands = n;
      else if (it.key() == "max_values")
        result.max_values = n;
      else if (it.key() == "max_input_edges")
        result.max_input_edges = n;
      else if (it.key() == "max_exports")
        result.max_exports = n;
      else if (it.key() == "max_lookahead")
        result.max_lookahead = n;
      else if (it.key() == "max_match_attempts")
        result.max_match_attempts = n;
      else
        throw std::runtime_error("unknown fixture cap");
    }
  }
  return result;
}
void registerProvider(Runtime &rt, bool badOutput) {
  auto accepts = [](const Request &, std::string &) { return true; };
  rt.addRecipe({"add", "integer.add", 0, accepts, [badOutput](Runtime &runtime, const Request &r) {
                  const auto n = number(r.inputs.at(0)) + number(r.inputs.at(1));
                  if (badOutput)
                    return Values{runtime.makeValue("wrong", ResourceKind::Value, 8, [n] { return n; })};
                  return Values{integer(runtime, n)};
                }});
  rt.addRecipe({"split", "integer.split", 0, accepts, [](Runtime &runtime, const Request &r) {
                  const auto n = number(r.inputs.at(0));
                  return Values{integer(runtime, n), integer(runtime, -n)};
                }});
  rt.addRecipe({"observe", "integer.observe", 0, accepts, [](Runtime &, const Request &) { return Values{}; }});
}
void registerKernel(Runtime &rt, const std::string &kind) {
  if (kind.empty())
    return;
  rt.addScopeKernel({kind, 0, 2,
                     [kind](const ScopeView &view, std::string &reason) -> std::size_t {
                       if (kind == "two_adds" && view.size() == 2 && view.command(0).operation == "add" &&
                           view.command(1).operation == "add")
                         return 2;
                       reason = "requires two adds";
                       return 0;
                     },
                     [](Runtime &runtime, const ScopeView &view) {
                       std::unordered_map<std::size_t, std::int64_t> numbers;
                       for (std::size_t i = 0; i < view.size(); ++i) {
                         const auto &command = view.command(i);
                         auto get = [&](ScopeValue symbol) {
                           auto found = numbers.find(symbol.id());
                           return found == numbers.end() ? number(view.value(symbol)) : found->second;
                         };
                         numbers.emplace(command.outputs.at(0).id(),
                                         get(command.inputs.at(0)) + get(command.inputs.at(1)));
                       }
                       Values result;
                       for (const auto symbol : view.liveOuts(view.size()))
                         result.push_back(integer(runtime, numbers.at(symbol.id())));
                       return result;
                     }});
}

TEST(LanguageConformance, SharedRuntimeContractCases) {
  std::ifstream stream(FHN_CONFORMANCE_FIXTURE);
  ASSERT_TRUE(stream.good());
  const auto fixture = Json::parse(stream);
  ASSERT_EQ(fixture.at("version"), 1);
  for (const auto &test : fixture.at("cases")) {
    SCOPED_TRACE(test.at("name").get<std::string>());
    Runtime rt({test.value("runtime_bytes", std::size_t{0}), 256});
    registerProvider(rt, test.value("bad_output", false));
    registerKernel(rt, test.value("kernel", std::string{}));
    Values inputs;
    for (const auto &n : test.at("inputs"))
      inputs.push_back(integer(rt, n.get<std::int64_t>()));
    std::string error;
    Values output;
    try {
      output = rt.runScope(
        [&](Scope &scope) {
          ScopeValues symbols;
          for (const auto &v : inputs)
            symbols.push_back(scope.input(v));
          if (test.value("repeat_first_import", false))
            EXPECT_EQ(scope.input(inputs.at(0)), symbols.at(0));
          for (const auto &command : test.at("commands")) {
            ScopeValues arguments;
            for (const auto &index : command.at("inputs"))
              arguments.push_back(symbols.at(index.get<std::size_t>()));
            const auto produced = scope.submit(command.at("op").get<std::string>(), std::move(arguments), {},
                                               command.at("outputs").get<std::size_t>());
            symbols.insert(symbols.end(), produced.begin(), produced.end());
          }
          ScopeValues exports;
          for (const auto &index : test.at("exports"))
            exports.push_back(symbols.at(index.get<std::size_t>()));
          return exports;
        },
        options(test));
    } catch (const Error &e) {
      error = codeName(e.code());
    }
    if (test.contains("error"))
      EXPECT_EQ(error, test.at("error").get<std::string>());
    else {
      ASSERT_TRUE(error.empty()) << error;
      std::vector<std::int64_t> actual;
      for (const auto &v : output)
        actual.push_back(number(v));
      EXPECT_EQ(actual, test.at("expected").get<std::vector<std::int64_t>>());
      if (test.contains("aliases"))
        for (const auto &pair : test.at("aliases"))
          EXPECT_EQ(output.at(pair.at(0).get<std::size_t>()).identity(),
                    output.at(pair.at(1).get<std::size_t>()).identity());
    }
    const auto s = rt.scopeStats();
    const Json actual{{"commands", s.commands},         {"values", s.values},
                      {"imports", s.imports},           {"input_edges", s.input_edges},
                      {"exports", s.exports},           {"executed_commands", s.executed_commands},
                      {"recipe_spans", s.recipe_spans}, {"kernel_spans", s.kernel_spans}};
    EXPECT_EQ(actual, test.at("stats"));
    output.clear();
    EXPECT_EQ(rt.stats().live_resources, inputs.size());
  }
}
} // namespace
