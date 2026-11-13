// Build against the real provider and adapter; no mock arithmetic or scheduler.
#include "FHN/TensorKernels.h"
#include "backend.h"
#include "bridge.h"
#include <cmath>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
using Json = nlohmann::json;
using RunScope = int (*)(void *, int (*)(void *), void *, size_t, size_t, size_t, size_t, size_t);
using Export = int (*)(void *, void *);
void check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void runtime_ok(int status) {
  if (status)
    throw std::runtime_error(fr_last_error());
}
void native_ok(int status) {
  if (status)
    throw std::runtime_error(fn_last_error());
}
struct Capture {
  std::function<void()> body;
  std::string failure;
  static int call(void *pointer) {
    auto &self = *static_cast<Capture *>(pointer);
    try {
      self.body();
      return 0;
    } catch (const std::exception &error) {
      self.failure = error.what();
      return -1;
    }
  }
};
int main() {
  try {
    std::size_t malformed_executions = 0;
    // This contributor intentionally violates its public inference contract.
    // Rejection must happen at submission, before any symbolic output or
    // executor can be reached (including a valid first / invalid second output).
    fhenomenon::tensor::registerTensorKernel(
      {"tensor.test_invalid_metadata",
       {{"input_arity", 1}, {"output_arity", 2}},
       [](const fhenomenon::tensor::KernelEnvironment &environment,
          const std::vector<fhenomenon::tensor::CipherMetadata> &inputs, const Json &attributes) {
         auto invalid = inputs.at(0);
         const auto field = attributes.at("field").get<std::string>();
         if (field == "bytes_zero")
           invalid.bytes = 0;
         else if (field == "bytes_fractional_limb")
           ++invalid.bytes;
         else if (field == "bytes_wrong_width")
           invalid.bytes *= 2;
         else if (field == "count_zero")
           invalid.count = 0;
         else if (field == "count_above_period")
           invalid.count = invalid.period + 1;
         else if (field == "period_zero")
           invalid.period = 0;
         else if (field == "period_non_power_of_two")
           invalid.period = 3;
         else if (field == "period_above_logical_slots")
           invalid.period = environment.profile.at("slots").get<std::size_t>() * 2;
         else if (field == "period_above_physical_slots")
           invalid.period = environment.profile.at("physical_slots").get<std::size_t>() * 2;
         else if (field == "moduli_zero")
           invalid.moduli = 0;
         else if (field == "moduli_wrong_level")
           --invalid.moduli;
         else if (field == "level_negative")
           invalid.level = -1;
         else if (field == "level_above_chain")
           invalid.level = environment.profile.at("top_level").get<std::int64_t>() + 1;
         else if (field == "scale_zero")
           invalid.scale = 0;
         else if (field == "scale_negative")
           invalid.scale = -1;
         else if (field == "scale_nan")
           invalid.scale = std::numeric_limits<double>::quiet_NaN();
         else if (field == "scale_inf")
           invalid.scale = std::numeric_limits<double>::infinity();
         else if (field == "representation")
           invalid.representation = "unrecognized";
         fhenomenon::tensor::KernelPlan result{{inputs.at(0), invalid}};
         if (field == "details")
           result.details = Json::array();
         return result;
       },
       [&malformed_executions](fhenomenon::tensor::KernelExecution &, const fhenomenon::runtime::Values &, const Json &,
                               const fhenomenon::tensor::KernelPlan &) {
         ++malformed_executions;
         throw std::runtime_error("malformed metadata reached its executor");
         return fhenomenon::runtime::Values{};
       }});
    // A separate contribution exercises the generic two-output ABI. No bridge
    // switch, Python binding or primitive implementation changes are required.
    fhenomenon::tensor::registerTensorKernel(
      {"tensor.test_pair",
       {{"input_arity", 1}, {"output_arity", 2}},
       [](const fhenomenon::tensor::KernelEnvironment &, const std::vector<fhenomenon::tensor::CipherMetadata> &inputs,
          const Json &) { return fhenomenon::tensor::KernelPlan{{inputs.at(0), inputs.at(0)}}; },
       [](fhenomenon::tensor::KernelExecution &context, const fhenomenon::runtime::Values &inputs,
          const Json &attributes, const fhenomenon::tensor::KernelPlan &plan) {
         fhenomenon::runtime::Values outputs;
         for (std::size_t index = 0; index < plan.outputs.size(); ++index) {
           if (index == 1 && attributes.value("fail_second", false))
             throw std::runtime_error("test contribution second-output failure");
           outputs.push_back(context.produce(plan.outputs[index], [&](void **out) {
             return fn_level_down(context.provider, context.borrow(inputs.at(0)), plan.outputs[index].level, out);
           }));
         }
         return outputs;
       }});
    auto run = reinterpret_cast<RunScope>(dlsym(RTLD_DEFAULT, "fr_run_scope"));
    auto export_value = reinterpret_cast<Export>(dlsym(RTLD_DEFAULT, "fr_scope_export"));
    check(run && export_value, "wholegraph scope ABI missing");
    void *client = nullptr, *evaluator = nullptr, *runtime = nullptr, *source = nullptr, *x = nullptr;
    native_ok(fn_create_profile(1, nullptr, 0, &client, &evaluator));
    runtime_ok(fr_create(evaluator, 0, &runtime));
    const double data[]{0.25, -0.5};
    native_ok(fn_encrypt(client, data, 2, &source));
    runtime_ok(fr_import(runtime, source, &x));
    fn_ciphertext_free(source);
    for (const auto *field :
         {"bytes_zero", "bytes_fractional_limb", "bytes_wrong_width", "count_zero", "count_above_period", "period_zero",
          "period_non_power_of_two", "period_above_logical_slots", "period_above_physical_slots", "moduli_zero",
          "moduli_wrong_level", "level_negative", "level_above_chain", "scale_zero", "scale_negative", "scale_nan",
          "scale_inf", "representation", "details"}) {
      void *inputs[]{x};
      void *outputs[]{nullptr, nullptr};
      std::size_t count = 0;
      const auto attributes = Json{{"field", field}}.dump();
      const auto calls = Json::parse(fr_provider_stats(runtime)).at("arithmetic_calls");
      check(fr_tensor_kernel(runtime, "tensor.test_invalid_metadata", inputs, 1, attributes.c_str(), outputs, 2,
                             &count) != 0,
            "invalid inferred output metadata was accepted eagerly");
      check(malformed_executions == 0, "invalid inferred metadata reached eager execution");
      check(count == 0 && !outputs[0] && !outputs[1], "invalid inference published eager handles");
      bool rejected_during_capture = false;
      void *pending = nullptr;
      Capture invalid_descriptor{[&] {
                                   runtime_ok(fr_unary(runtime, 0, x, 0, &pending));
                                   runtime_ok(export_value(runtime, pending));
                                   rejected_during_capture =
                                     fr_tensor_kernel(runtime, "tensor.test_invalid_metadata", inputs, 1,
                                                      attributes.c_str(), outputs, 2, &count) != 0;
                                   if (!rejected_during_capture) {
                                     // Never feed unvalidated metadata into later arithmetic. This catches
                                     // the bad submission directly, avoiding architecture-dependent UB.
                                     fr_value_free(outputs[0]);
                                     fr_value_free(outputs[1]);
                                   }
                                 },
                                 {}};
      check(run(runtime, Capture::call, &invalid_descriptor, 100, 200, 300, 10, 0) != 0,
            "invalid inferred output metadata did not poison capture");
      check(rejected_during_capture, "invalid inferred metadata was published as a scope symbol");
      check(count == 0 && !outputs[0] && !outputs[1], "invalid inference published scoped handles");
      check(Json::parse(fr_provider_stats(runtime)).at("arithmetic_calls") == calls,
            "invalid inferred metadata allowed earlier graph arithmetic to execute");
      check(fr_value_info(pending) == nullptr, "invalid inferred metadata published a prior scope export");
      fr_value_free(pending);
      check(Json::parse(fr_runtime_stats(runtime)).at("live_resources") == 1,
            "invalid inferred metadata leaked resources");
    }
    {
      void *inputs[]{x};
      void *outputs[]{nullptr, nullptr};
      std::size_t count = 0;
      const auto calls = Json::parse(fr_provider_stats(runtime)).at("arithmetic_calls");
      check(fr_tensor_kernel(runtime, "tensor.test_pair", inputs, 1, "{}", outputs, 1, &count) != 0,
            "multi-output kernel accepted insufficient output capacity");
      check(count == 0 && !outputs[0] && !outputs[1], "capacity rejection published partial handles");
      check(Json::parse(fr_provider_stats(runtime)).at("arithmetic_calls") == calls,
            "capacity rejection executed provider arithmetic");
      runtime_ok(fr_tensor_kernel(runtime, "tensor.test_pair", inputs, 1, "{}", outputs, 2, &count));
      check(count == 2 && outputs[0] && outputs[1], "multi-output kernel did not publish both handles");
      fr_value_free(outputs[0]);
      fr_value_free(outputs[1]);
      check(fr_tensor_kernel(runtime, "tensor.test_pair", inputs, 1, "{\"fail_second\":true}", outputs, 2, &count) != 0,
            "failing multi-output kernel was accepted");
      check(count == 0 && !outputs[0] && !outputs[1], "failed kernel published its first output");
      check(Json::parse(fr_runtime_stats(runtime)).at("live_resources") == 1,
            "failed multi-output kernel leaked an unpublished resource");
      Capture pair{[&] {
                     runtime_ok(fr_tensor_kernel(runtime, "tensor.test_pair", inputs, 1, "{}", outputs, 2, &count));
                     runtime_ok(export_value(runtime, outputs[0]));
                     runtime_ok(export_value(runtime, outputs[1]));
                   },
                   {}};
      runtime_ok(run(runtime, Capture::call, &pair, 100, 200, 300, 10, 0));
      check(Json::parse(fr_runtime_stats(runtime)).at("last_scope").at("kernel_spans") == 1,
            "multi-output contribution did not execute as a ScopeKernel");
      double result[2];
      native_ok(fn_decrypt(client, fr_value_borrow(outputs[1]), result, 2));
      check(std::abs(result[0] - 0.25) < 2e-5 && std::abs(result[1] + 0.5) < 2e-5,
            "multi-output contribution changed its input value");
      fr_value_free(outputs[0]);
      fr_value_free(outputs[1]);
    }
    const auto before = Json::parse(fr_provider_stats(runtime)).at("arithmetic_calls");
    void *result = nullptr, *escaped = nullptr;
    Capture capture{[&] {
                      void *temporary = nullptr;
                      runtime_ok(fr_unary(runtime, 0, x, 0, &temporary));
                      check(fr_value_borrow(temporary) == nullptr,
                            "symbolic ciphertext leaked a concrete provider value");
                      const auto info = Json::parse(fr_value_info(temporary));
                      check(info.at("level") == 8 && info.at("count") == 2, "symbolic descriptor lost metadata");
                      check(info.at("scale") == Json::parse(fr_value_info(x)).at("scale"),
                            "symbolic negation descriptor failed to preserve CKKS scale");
                      const double half = 0.5;
                      runtime_ok(fr_plain(runtime, 2, temporary, &half, 1, &result));
                      runtime_ok(fr_unary(runtime, 1, result, -17, &escaped));
                      fr_value_free(temporary); // The graph must own symbolic dependencies independently of C handles.
                      check(Json::parse(fr_provider_stats(runtime)).at("arithmetic_calls") == before,
                            "provider arithmetic ran during graph capture");
                      runtime_ok(export_value(runtime, result));
                    },
                    {}};
    const int status = run(runtime, Capture::call, &capture, 100, 200, 300, 10, 0);
    if (status)
      throw std::runtime_error(capture.failure.empty() ? fr_last_error() : capture.failure);
    double actual[2];
    native_ok(fn_decrypt(client, fr_value_borrow(result), actual, 2));
    check(std::abs(actual[0] + 0.125) < 2e-5 && std::abs(actual[1] - 0.25) < 2e-5,
          "wholegraph result differs from hand-computed encrypted arithmetic");
    check(fr_value_info(escaped) == nullptr && fr_value_borrow(escaped) == nullptr,
          "nonexported symbolic ciphertext escaped its scope");
    const auto scope = Json::parse(fr_runtime_stats(runtime)).at("last_scope");
    check(scope.at("commands") == 3 && scope.at("exports") == 1 && scope.at("imports") == 1,
          "wholegraph capture statistics do not describe the recorded graph");
    fr_value_free(escaped);
    fr_value_free(result);

    // A later invalid operation must reject the entire graph before the earlier negation runs.
    const auto pre_failure = Json::parse(fr_provider_stats(runtime)).at("arithmetic_calls");
    void *failed_symbol = nullptr;
    Capture invalid{[&] {
                      runtime_ok(fr_unary(runtime, 0, x, 0, &failed_symbol));
                      runtime_ok(export_value(runtime, failed_symbol));
                      const double wrong_shape[]{1, 2, 3};
                      void *bad = nullptr;
                      runtime_ok(fr_plain(runtime, 0, failed_symbol, wrong_shape, 3, &bad));
                    },
                    {}};
    check(run(runtime, Capture::call, &invalid, 100, 200, 300, 10, 0) != 0,
          "invalid plaintext shape was accepted by scope contract");
    check(Json::parse(fr_provider_stats(runtime)).at("arithmetic_calls") == pre_failure,
          "an invalid wholegraph partially executed");
    check(fr_value_info(failed_symbol) == nullptr && fr_value_borrow(failed_symbol) == nullptr,
          "failed scope published an exported value");
    fr_value_free(failed_symbol);

    // C callers can ignore an arithmetic return code, but cannot execute a
    // silently truncated graph after an invalid refresh target or public bound.
    for (int failure = 0; failure < 2; ++failure) {
      void *pending = nullptr;
      Capture ignored{[&] {
                        runtime_ok(fr_unary(runtime, 0, x, 0, &pending));
                        runtime_ok(export_value(runtime, pending));
                        void *bad = nullptr;
                        const auto profile = Json::parse(fr_profile(runtime));
                        const double bound =
                          failure ? profile.at("refresh_maximum_public_bound").get<double>() * 2 : 1.;
                        check(fr_mod_raise_bounded(runtime, pending, failure ? 8 : 3, bound, &bad) != 0,
                              "capture accepted an unprepared refresh target or unsupported bound");
                        check(bad == nullptr, "invalid refresh published a result handle");
                      },
                      {}};
      check(run(runtime, Capture::call, &ignored, 100, 200, 300, 10, 0) != 0,
            "ignored arithmetic failure did not invalidate wholegraph capture");
      check(Json::parse(fr_provider_stats(runtime)).at("arithmetic_calls") == pre_failure,
            "poisoned wholegraph capture executed native arithmetic");
      check(fr_value_info(pending) == nullptr, "poisoned capture published an export");
      fr_value_free(pending);
    }

    // Export registration must retain shared state, never a pointer to a caller-freed handle.
    Capture released{[&] {
                       void *temporary = nullptr;
                       runtime_ok(fr_unary(runtime, 0, x, 0, &temporary));
                       runtime_ok(export_value(runtime, temporary));
                       fr_value_free(temporary);
                     },
                     {}};
    runtime_ok(run(runtime, Capture::call, &released, 100, 200, 300, 10, 0));
    fr_value_free(x);
    check(Json::parse(fr_runtime_stats(runtime)).at("live_resources") == 0,
          "scoped handle lifecycle leaked ciphertext resources");
    fr_runtime_free(runtime);
    fn_client_free(client);
    std::cout << "wholegraph native bridge tests passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
