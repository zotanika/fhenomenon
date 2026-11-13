#include "backend.h"
#include "engine.hpp"
#include "refresh.hpp"
#include "transforms.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace native = fhenomenon::tensor_ckks;
thread_local std::string error, output;
std::atomic<std::uint64_t> next_identity{1};
const unsigned char library_identity_token = 0;
void require(bool condition, const char *message) {
  if (!condition)
    throw std::invalid_argument(message);
}
template <class F> int guard(F operation) {
  try {
    error.clear();
    operation();
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
  } catch (...) {
    error = "unknown native CKKS failure";
  }
  return -1;
}
const char *text(std::string value) {
  output = std::move(value);
  return output.c_str();
}
template <class F> const char *json(F operation) {
  const char *result = nullptr;
  if (guard([&] { result = text(operation()); }))
    return nullptr;
  return result;
}
struct PublicState {
  std::shared_ptr<native::Engine> engine;
  std::recursive_mutex mutex;
  std::set<unsigned> targets;
  const std::uint64_t identity = next_identity.fetch_add(1);
  std::uint64_t refresh_calls = 0, mod_raise_calls = 0, stc_calls = 0, cts_calls = 0;
  unsigned last_target = 0, last_input_top = 0;
  std::int64_t last_gain = 0;
  double last_seconds = 0, last_bound = 1;
  bool last_targeted = false;
  explicit PublicState(int profile) {
    require(profile == 0 || profile == 1, "unknown native profile");
    native::Config config;
    config.log_degree = profile == 0 ? 17 : 10;
    config.test_only = profile == 1;
    config.evaluation_top = 8;
    config.logical_slots = 256;
    config.levels = config.evaluation_top + native::refreshDepth(config.log_degree) + 1;
    engine = std::make_shared<native::Engine>(config);
    targets.insert(config.evaluation_top);
  }
};
enum class Kind { Client, Evaluator, Ciphertext };
struct Header {
  Kind kind;
  const void *library_identity = &library_identity_token;
  explicit Header(Kind value) : kind(value) {}
};
struct ClientHandle : Header {
  static constexpr Kind tag = Kind::Client;
  std::shared_ptr<PublicState> state;
  std::unique_ptr<native::Client> client;
  double last_egress_bound = 1;
  ClientHandle() : Header(tag) {}
};
struct EvaluatorHandle : Header {
  static constexpr Kind tag = Kind::Evaluator;
  std::shared_ptr<PublicState> state;
  explicit EvaluatorHandle(std::shared_ptr<PublicState> value) : Header(tag), state(std::move(value)) {}
};
struct CipherHandle : Header {
  static constexpr Kind tag = Kind::Ciphertext;
  std::shared_ptr<PublicState> state;
  native::Value value;
  CipherHandle(std::shared_ptr<PublicState> owner, native::Value data)
    : Header(tag), state(std::move(owner)), value(std::move(data)) {}
};
template <class T> T &handle(void *pointer) {
  require(pointer != nullptr, "closed or null native handle");
  require(static_cast<Header *>(pointer)->library_identity == &library_identity_token,
          "native handle belongs to another provider library");
  require(static_cast<Header *>(pointer)->kind == T::tag, "wrong native handle type");
  return *static_cast<T *>(pointer);
}
std::shared_ptr<PublicState> checked(void *evaluator, const CipherHandle &a, bool slots = true) {
  auto state = handle<EvaluatorHandle>(evaluator).state;
  require(state == a.state, "ciphertext belongs to another key context");
  if (slots)
    require(state->engine->representation(a.value) == "slots", "slot arithmetic requires slot representation");
  return state;
}
std::vector<std::complex<double>> plaintext(const double *values, std::size_t count) {
  require(values && count, "plaintext must contain at least one value");
  std::vector<std::complex<double>> result;
  result.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    require(std::isfinite(values[i]), "nonfinite plaintext");
    result.emplace_back(values[i], 0.);
  }
  return result;
}
template <class F> void publish_value(void **result, const std::shared_ptr<PublicState> &state, F operation) {
  require(result != nullptr, "missing ciphertext output");
  *result = nullptr;
  auto value = operation();
  require(bool(value), "native engine returned no ciphertext");
  auto owned = std::make_unique<CipherHandle>(state, std::move(value));
  *result = owned.release();
}
std::string plan_json(PublicState &s, unsigned target) {
  const auto &config = s.engine->config();
  require(target <= config.evaluation_top, "refresh target exceeds evaluation chain");
  const auto depth = native::refreshDepth(config.log_degree);
  const auto top = target + depth;
  std::uint64_t sum = 0;
  std::ostringstream o;
  o << "{\"target_level\":" << target << ",\"input_top_level\":" << top << ",\"refresh_depth\":" << depth
    << ",\"strategy\":\"target_specific_rns_prefix\","
       "\"full_refresh_then_drop\":false,\"active_modulus_counts_by_rescale_level\":[";
  for (unsigned i = target; i <= top; ++i) {
    if (i != target)
      o << ',';
    o << i + 1;
    sum += i + 1;
  }
  o << "],\"scheduled_modulus_sum\":" << sum << ",\"unit_bound_public_gain\":" << native::refreshGain(*s.engine, 1.)
    << ",\"gain_policy\":\"largest exact power of two satisfying G*public_bound*S0/q0 <= 1/16; depends only on public "
       "parameters\""
    << ",\"work_metric\":\"sum_of_active_modulus_counts_at_rescale_levels_not_kernel_flops\"}";
  return o.str();
}
} // namespace

extern "C" {
uint32_t fn_get_abi_version(void) { return FHN_TENSOR_PROVIDER_ABI_VERSION; }
const char *fn_last_error() { return error.c_str(); }
const void *fn_library_identity() { return &library_identity_token; }
int fn_create(const int32_t *steps, size_t count, void **client, void **evaluator) {
  return fn_create_profile(0, steps, count, client, evaluator);
}
int fn_create_profile(int profile, const int32_t *steps, size_t count, void **client, void **evaluator) {
  return guard([&] {
    require(client && evaluator && client != evaluator, "separate client and evaluator outputs are required");
    *client = nullptr;
    *evaluator = nullptr;
    require(steps || !count, "rotation steps are missing");
    std::vector<std::int32_t> rotations;
    if (count)
      rotations.assign(steps, steps + count);
    auto c = std::make_unique<ClientHandle>();
    c->state = std::make_shared<PublicState>(profile);
    c->client = std::make_unique<native::Client>(*c->state->engine, rotations);
    auto e = std::make_unique<EvaluatorHandle>(c->state);
    *client = c.release();
    *evaluator = e.release();
  });
}
void fn_client_free(void *pointer) {
  if (pointer)
    guard([&] { delete &handle<ClientHandle>(pointer); });
}
void fn_evaluator_free(void *pointer) {
  if (pointer)
    guard([&] { delete &handle<EvaluatorHandle>(pointer); });
}
void fn_ciphertext_free(void *pointer) {
  if (pointer)
    guard([&] { delete &handle<CipherHandle>(pointer); });
}
int fn_select_device(void *pointer, int use_cuda) {
  return guard([&] {
    auto &c = handle<ClientHandle>(pointer);
    std::lock_guard<std::recursive_mutex> lock(c.state->mutex);
    require(use_cuda == 0 || use_cuda == 1, "device selector must be zero or one");
    require(c.client->encryptCalls() == 0, "select device before encryption");
    if (use_cuda)
      c.state->engine->enableCuda();
    else
      require(!c.state->engine->isCuda(), "cannot switch a CUDA context back to CPU");
  });
}
int fn_synchronize(void *pointer) {
  return guard([&] {
    auto s = handle<EvaluatorHandle>(pointer).state;
    std::lock_guard<std::recursive_mutex> lock(s->mutex);
    s->engine->synchronize();
  });
}
const char *fn_device_info(void *pointer) {
  return json([&] {
    auto s = handle<EvaluatorHandle>(pointer).state;
    std::lock_guard<std::recursive_mutex> lock(s->mutex);
    std::ostringstream o;
    o << "{\"device\":\"" << (s->engine->isCuda() ? "cuda" : "cpu")
      << "\",\"device_id\":0,\"host_fallback_allowed\":false}";
    return o.str();
  });
}
int fn_prepare_mod_raise(void *pointer, uint32_t target) {
  return guard([&] {
    auto &c = handle<ClientHandle>(pointer);
    auto s = c.state;
    std::lock_guard<std::recursive_mutex> lock(s->mutex);
    require(c.client->encryptCalls() == 0, "declare refresh targets during client setup");
    require(target <= s->engine->config().evaluation_top, "ModRaise target exceeds evaluation chain");
    // The native engine generates shared public evaluation keys during setup.
    // Refresh selects the target's actual active prefix; no per-target secret is retained.
    s->targets.insert(target);
  });
}
const char *fn_refresh_plan(void *pointer, uint32_t target) {
  return json([&] {
    auto s = handle<EvaluatorHandle>(pointer).state;
    std::lock_guard<std::recursive_mutex> lock(s->mutex);
    return plan_json(*s, target);
  });
}
int fn_encrypt(void *pointer, const double *values, size_t count, void **result) {
  return guard([&] {
    auto &c = handle<ClientHandle>(pointer);
    std::lock_guard<std::recursive_mutex> lock(c.state->mutex);
    require(count <= c.state->engine->config().logical_slots, "plaintext exceeds logical slot capacity");
    publish_value(result, c.state, [&] { return c.client->encrypt(plaintext(values, count)); });
  });
}
int fn_decrypt(void *client, void *ciphertext, double *values, size_t count) {
  return fn_decrypt_bounded(client, ciphertext, values, count, 1.);
}
int fn_decrypt_bounded(void *pointer, void *ciphertext, double *values, size_t count, double bound) {
  return guard([&] {
    auto &c = handle<ClientHandle>(pointer);
    auto &a = handle<CipherHandle>(ciphertext);
    require(c.state == a.state, "wrong client secret key");
    std::lock_guard<std::recursive_mutex> lock(c.state->mutex);
    require(values && count == c.state->engine->count(a.value), "wrong decrypt output length");
    auto data = c.client->decrypt(a.value, bound);
    for (std::size_t i = 0; i < count; ++i)
      values[i] = data[i].real();
    c.last_egress_bound = bound;
  });
}
int fn_binary(void *pointer, int op, void *a_pointer, void *b_pointer, void **result) {
  return guard([&] {
    auto &a = handle<CipherHandle>(a_pointer);
    auto &b = handle<CipherHandle>(b_pointer);
    auto s = checked(pointer, a);
    checked(pointer, b);
    std::lock_guard<std::recursive_mutex> lock(s->mutex);
    require(op >= 0 && op <= 2, "unknown binary operation");
    publish_value(result, s, [&] {
      return op == 0   ? s->engine->add(a.value, b.value)
             : op == 1 ? s->engine->sub(a.value, b.value)
                       : s->engine->mul(a.value, b.value);
    });
  });
}
int fn_plain(void *pointer, int op, void *a_pointer, const double *values, size_t count, void **result) {
  return guard([&] {
    auto &a = handle<CipherHandle>(a_pointer);
    auto s = checked(pointer, a);
    std::lock_guard<std::recursive_mutex> lock(s->mutex);
    require(op == 0 || op == 2, "unknown plaintext operation");
    require(count == 1 || count == s->engine->count(a.value), "incompatible plaintext length");
    auto data = plaintext(values, count);
    if (count != 1)
      data.resize(s->engine->period(a.value));
    publish_value(result, s, [&] { return s->engine->plain(a.value, data, op == 2); });
  });
}
int fn_neg(void *pointer, void *a_pointer, void **result) {
  return guard([&] {
    auto &a = handle<CipherHandle>(a_pointer);
    auto s = checked(pointer, a);
    std::lock_guard<std::recursive_mutex> lock(s->mutex);
    publish_value(result, s, [&] { return s->engine->neg(a.value); });
  });
}
int fn_rotate(void *pointer, void *a_pointer, int32_t step, void **result) {
  return guard([&] {
    auto &a = handle<CipherHandle>(a_pointer);
    auto s = checked(pointer, a);
    std::lock_guard<std::recursive_mutex> lock(s->mutex);
    publish_value(result, s, [&] { return s->engine->rotate(a.value, step); });
  });
}
int fn_level_down(void *pointer, void *a_pointer, uint32_t level, void **result) {
  return guard([&] {
    auto &a = handle<CipherHandle>(a_pointer);
    auto s = checked(pointer, a);
    std::lock_guard<std::recursive_mutex> lock(s->mutex);
    publish_value(result, s, [&] { return s->engine->levelDown(a.value, level); });
  });
}
int fn_mod_raise(void *pointer, void *a_pointer, uint32_t target, void **result) {
  return fn_mod_raise_bounded(pointer, a_pointer, target, 1., result);
}
int fn_mod_raise_bounded(void *pointer, void *a_pointer, uint32_t target, double bound, void **result) {
  return guard([&] {
    auto &a = handle<CipherHandle>(a_pointer);
    auto s = checked(pointer, a);
    std::lock_guard<std::recursive_mutex> lock(s->mutex);
    require(s->targets.count(target), "target refresh was not declared during client setup");
    const auto gain = native::refreshGain(*s->engine, bound);
    s->engine->synchronize();
    const auto start = std::chrono::steady_clock::now();
    publish_value(result, s, [&] {
      auto refreshed = native::refresh(*s->engine, a.value, target, bound);
      s->engine->synchronize();
      return refreshed;
    });
    ++s->refresh_calls;
    ++s->mod_raise_calls;
    s->last_target = target;
    s->last_input_top = target + native::refreshDepth(s->engine->config().log_degree);
    s->last_bound = bound;
    s->last_targeted = true;
    s->last_gain = gain;
    s->last_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  });
}
int fn_bootstrap(void *pointer, void *a_pointer, void **result) {
  return guard([&] {
    auto &a = handle<CipherHandle>(a_pointer);
    auto s = checked(pointer, a);
    std::lock_guard<std::recursive_mutex> lock(s->mutex);
    const auto target = s->engine->config().evaluation_top;
    const auto gain = native::refreshGain(*s->engine, 1.);
    s->engine->synchronize();
    const auto start = std::chrono::steady_clock::now();
    publish_value(result, s, [&] {
      auto refreshed = native::refresh(*s->engine, a.value, target, 1.);
      s->engine->synchronize();
      return refreshed;
    });
    ++s->refresh_calls;
    s->last_target = target;
    s->last_input_top = target + native::refreshDepth(s->engine->config().log_degree);
    s->last_bound = 1.;
    s->last_targeted = false;
    s->last_gain = gain;
    s->last_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  });
}
int fn_slot_to_coeff(void *pointer, void *a_pointer, void **result) {
  return guard([&] {
    auto &a = handle<CipherHandle>(a_pointer);
    auto s = checked(pointer, a);
    std::lock_guard<std::recursive_mutex> lock(s->mutex);
    require(s->engine->level(a.value) > 0, "slot-to-coefficient requires one remaining level");
    publish_value(result, s, [&] {
      return s->engine->withRepresentation(native::slotToCoefficient(*s->engine, a.value), "coefficients");
    });
    ++s->stc_calls;
  });
}
int fn_coeff_to_slot(void *pointer, void *a_pointer, void **result) {
  return guard([&] {
    auto &a = handle<CipherHandle>(a_pointer);
    auto s = checked(pointer, a, false);
    std::lock_guard<std::recursive_mutex> lock(s->mutex);
    require(s->engine->representation(a.value) == "coefficients", "expected coefficient representation");
    require(s->engine->level(a.value) > 0, "coefficient-to-slot requires one remaining level");
    publish_value(result, s, [&] {
      auto internal = s->engine->withRepresentation(a.value, "slots");
      return s->engine->withRepresentation(native::coefficientToSlot(*s->engine, internal), "slots");
    });
    ++s->cts_calls;
  });
}
const char *fn_profile(void *pointer) {
  return json([&] {
    auto s = handle<EvaluatorHandle>(pointer).state;
    std::lock_guard<std::recursive_mutex> lock(s->mutex);
    const auto &c = s->engine->config();
    std::ostringstream o;
    o << std::setprecision(17);
    o << "{\"name\":\"" << (c.test_only ? "NativeTest" : "Native128")
      << "\",\"backend\":\"Fhenomenon native CKKS\",\"device\":\"" << (s->engine->isCuda() ? "cuda" : "cpu")
      << "\",\"ring_degree\":" << (1U << c.log_degree) << ",\"slots\":"
      << c.logical_slots
      // Client setup always prepares this complete rotation basis (engine.cpp).
      << ",\"rotation_key_basis\":\"all_positive_powers_of_two\""
      << ",\"physical_slots\":" << s->engine->physicalSlots() << ",\"top_level\":" << c.evaluation_top
      << ",\"full_chain_modulus_count\":" << c.levels << ",\"evaluation_scale_bits\":" << c.log_scale
      << ",\"refresh_maximum_public_bound\":" << double(s->engine->bottomModulus()) / (16. * s->engine->valueScale(0))
      << ",\"refresh_public_gain_policy\":\"exact power-of-two amplification from declared public bound; no private "
         "range inspection\""
      << ",\"test_only\":" << (c.test_only ? "true" : "false")
      << ",\"production_security_claim\":false,\"security_policy_bits\":" << (c.test_only ? 0 : 128)
      << ",\"security_scope\":\"modulus-budget policy only; independent security certification not performed\""
         ",\"secret_distribution\":\"uniform_ternary\",\"toy_profile\":"
      << (c.test_only ? "true" : "false")
      << ",\"egress_minimum_level\":0,\"egress_minimum_modulus_count\":1"
         ",\"coefficient_message_order\":\"natural_invariant_subring\""
         ",\"slot_to_coeff_level_cost\":1,\"coeff_to_slot_level_cost\":1"
         ",\"maximum_chain_evaluation_keys_retained\":true,\"mod_raise_targets\":[";
    bool first = true;
    for (auto target : s->targets) {
      if (!first)
        o << ',';
      first = false;
      o << target;
    }
    o << "],\"evaluation_scales\":[";
    for (unsigned level = 0; level <= c.evaluation_top; ++level) {
      if (level)
        o << ',';
      o << s->engine->valueScale(level);
    }
    o << "],\"evaluation_moduli\":[";
    for (unsigned level = 0; level <= c.evaluation_top; ++level) {
      if (level)
        o << ',';
      o << s->engine->evaluationModulus(level);
    }
    o << "],\"mod_raise_strategy\":\"target_specific_rns_prefix\",\"context_id\":" << s->identity << '}';
    return o.str();
  });
}
const char *fn_stats(void *pointer) {
  return json([&] {
    auto s = handle<EvaluatorHandle>(pointer).state;
    std::lock_guard<std::recursive_mutex> lock(s->mutex);
    std::ostringstream o;
    o << std::setprecision(17);
    o << "{\"arithmetic_calls\":" << s->engine->arithmeticCalls() << ",\"bootstrap_calls\":" << s->refresh_calls
      << ",\"mod_raise_calls\":" << s->mod_raise_calls << ",\"physical_refresh_top_level\":" << s->last_input_top
      << ",\"last_refresh_target\":" << s->last_target << ",\"last_refresh_seconds\":" << s->last_seconds
      << ",\"last_refresh_public_bound\":" << s->last_bound
      << ",\"last_refresh_was_targeted\":" << (s->last_targeted ? "true" : "false")
      << ",\"last_refresh_public_gain\":" << s->last_gain << ",\"slot_to_coeff_calls\":" << s->stc_calls
      << ",\"coeff_to_slot_calls\":" << s->cts_calls
      << ",\"evaluator_decrypt_calls\":0,\"host_workspace_bytes\":" << s->engine->workspaceBytes()
      << ",\"evaluation_material_bytes_with_gpu_workspace_and_tables\":" << s->engine->keyBytes()
      << ",\"plaintext_cache_bytes\":" << s->engine->plaintextCacheBytes()
      << ",\"plaintext_cache_hits\":" << s->engine->plaintextCacheHits()
      << ",\"memory_accounting_scope\":\"host workspace and evaluation material are separate; host tables and "
         "allocator reserve are not accounted; ciphertext payloads are reported by Runtime\"}";
    return o.str();
  });
}
const char *fn_client_stats(void *pointer) {
  return json([&] {
    auto &c = handle<ClientHandle>(pointer);
    std::lock_guard<std::recursive_mutex> lock(c.state->mutex);
    std::ostringstream o;
    o << "{\"encrypt_calls\":" << c.client->encryptCalls() << ",\"decrypt_calls\":" << c.client->decryptCalls()
      << ",\"last_egress_level\":" << (c.client->lastEgressModuli() ? "0" : "null")
      << ",\"last_egress_modulus_count\":" << c.client->lastEgressModuli()
      << ",\"last_egress_public_bound\":" << c.last_egress_bound << '}';
    return o.str();
  });
}
const char *fn_ciphertext_info(void *pointer) {
  return json([&] {
    auto &a = handle<CipherHandle>(pointer);
    std::lock_guard<std::recursive_mutex> lock(a.state->mutex);
    auto &e = *a.state->engine;
    std::ostringstream o;
    o << std::setprecision(17);
    o << "{\"count\":" << e.count(a.value) << ",\"period\":" << e.period(a.value) << ",\"level\":" << e.level(a.value)
      << ",\"modulus_count\":" << e.level(a.value) + 1 << ",\"scale\":" << e.scale(a.value)
      << ",\"estimated_bytes\":" << e.bytes(a.value) << ",\"representation\":\"" << e.representation(a.value)
      << "\",\"device\":\"" << (e.isCuda() ? "cuda" : "cpu") << "\",\"context_id\":" << a.state->identity << '}';
    return o.str();
  });
}
} // extern C
