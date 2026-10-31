// Diagnostic only: mirror the existing coarse-dot arithmetic, with phase clocks.
// This target deliberately depends on backend internals and is not a public API.
#include "../src/CKKS/backend/BackendInternal.h"
#include "CKKS/Backend.h"
#include "CKKS/CoarseKernels.h"
#include "CKKS/ModArith.h"
#include "CKKS/detail/Region.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fhenomenon::ckks;
namespace {
using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
constexpr uint32_t width = 8;
constexpr unsigned warmup = 2, repetitions = 7;
volatile uint64_t consumed = 0;
double milliseconds(Clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}
void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
uint64_t checksum(const FhnBuffer *buffer) {
  uint64_t value = 0;
  const auto view = buffer->view();
  for (uint32_t p = 0; p < view.layout.num_polys; ++p)
    for (size_t i = 0; i < view.layout.poly.wordCount(); ++i)
      value = (value * 31) ^ view.poly(p).data[i];
  return value;
}
Json summary(const std::vector<double> &samples, unsigned operations = 1) {
  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  return {{"samples_ms", samples}, {"median_ms", sorted[sorted.size() / 2]},
          {"minimum_ms", sorted.front()}, {"operations_per_sample", operations}};
}
template <class Work, class Consume>
Json measure(Work work, Consume consume, unsigned operations = 1) {
  std::vector<double> samples;
  for (unsigned rep = 0; rep < warmup + repetitions; ++rep) {
    const auto start = Clock::now();
    work();
    const auto elapsed = milliseconds(start);
    consume(); // Outputs remain observable; checksumming is outside the timed scope.
    if (rep >= warmup)
      samples.push_back(elapsed);
  }
  return summary(samples, operations);
}
struct Buffers {
  FhnBackendCtx *ctx;
  std::vector<FhnBuffer *> values;
  explicit Buffers(FhnBackendCtx *c) : ctx(c) {}
  ~Buffers() { for (auto *b : values) ckks_fhn_buffer_free(ctx, b); }
  FhnBuffer *make() {
    auto *b = ckks_fhn_buffer_alloc(ctx);
    require(b != nullptr, "buffer allocation failed");
    values.push_back(b);
    return b;
  }
};
struct Phases {
  double validation = 0, multiply = 0, copy = 0;
  std::array<double, 3> rotate{};
};
bool validateInput(FhnBackendCtx *ctx, const FhnBuffer *value) {
  return value && value->owner == ctx && value->certified && value->logical_slots &&
         !(value->logical_slots & (value->logical_slots - 1)) &&
         value->logical_slots <= ctx->tables->params().numSlots() && region::validateInput(value->view(), *ctx->tables);
}
CiphertextView staging(Arena &arena, CiphertextLayout layout) {
  const auto words = layout.poly.wordCount();
  return {arena.allocate<uint64_t>(2 * words), 2 * words, words, layout};
}
// Same checks, key leases, arena layout, arithmetic, stage order, and publication
// as CoarseKernels.cpp's Dot path. Clocks diagnose boundaries, not accounting.
void profiledDot(FhnBackendCtx *ctx, FhnBuffer *out, const FhnBuffer *a, const FhnBuffer *b, Phases &phases) {
  auto start = Clock::now();
  require(ctx && out && out->owner == ctx && validateInput(ctx, a) && width <= a->logical_slots,
          "invalid left input");
  const auto &tables = *ctx->tables;
  const auto layout = a->layout;
  require(validateInput(ctx, b) && a->logical_slots == b->logical_slots && layout.poly == b->layout.poly &&
              std::abs(layout.scale - b->layout.scale) <= 1e-12 * std::max(layout.scale, b->layout.scale) &&
              layout.poly.basis.main_end >= 2, "invalid right input");
  auto destination = out->view();
  destination.layout = layout;
  require(destination.wellFormed(), "invalid destination");
  auto relin = ctx->relin.lock();
  std::array<std::shared_ptr<const EvaluationKey>, NttTables::kMaxLogDegree> rotations{};
  std::array<uint64_t, NttTables::kMaxLogDegree> galois{};
  require(relin && region::validateKey(relin->view(), SwitchKeyKind::Relinearization, 1, tables), "invalid relin key");
  size_t stages = 0;
  for (uint32_t distance = width / 2; distance; distance /= 2) {
    const auto g = eval::galoisElement(distance, tables.params().logDegree());
    const auto found = ctx->rotations.find(g);
    require(found != ctx->rotations.end() && (rotations[stages] = found->second.lock()) &&
                region::validateKey(rotations[stages]->view(), SwitchKeyKind::Galois, g, tables), "invalid rotation key");
    galois[stages++] = g;
  }
  phases.validation += milliseconds(start);
  Arena::Scope scope(ctx->scratch);
  auto first = staging(ctx->scratch, layout);
  auto second = staging(ctx->scratch, layout);
  require(first.data && second.data && ctx->scratch.remaining() >= region::scratchBytes(tables.params()),
          "insufficient scratch");
  start = Clock::now();
  require(region::multiply(first, a->view(), b->view(), relin->view(), tables, ctx->scratch), "region multiply failed");
  phases.multiply += milliseconds(start);
  auto accumulator = asConst(first);
  auto *next = &second;
  for (size_t stage = 0; stage < stages; ++stage) {
    const region::LinearTerm term{accumulator, 1};
    start = Clock::now();
    require(region::rotateAdd(*next, accumulator, galois[stage], rotations[stage]->view(), &term, 1, tables,
                              ctx->scratch), "region rotateAdd failed");
    phases.rotate[stage] += milliseconds(start);
    accumulator = asConst(*next);
    next = next == &first ? &second : &first;
  }
  start = Clock::now();
  const auto words = accumulator.layout.poly.wordCount();
  for (uint32_t component = 0; component < 2; ++component)
    std::copy_n(accumulator.poly(component).data, words, out->slab.data() + component * ctx->stride);
  out->layout = accumulator.layout;
  out->logical_slots = a->logical_slots;
  out->certified = true;
  phases.copy += milliseconds(start);
}
void exact(const FhnBuffer *a, const FhnBuffer *b) {
  require(a->layout.poly == b->layout.poly && a->layout.scale == b->layout.scale &&
              a->layout.num_polys == b->layout.num_polys && a->logical_slots == b->logical_slots &&
              a->certified == b->certified, "layout mismatch");
  for (uint32_t p = 0; p < a->layout.num_polys; ++p)
    require(std::equal(a->view().poly(p).data, a->view().poly(p).data + a->layout.poly.wordCount(),
                       b->view().poly(p).data), "residue mismatch");
}
FhnKernelFn kernel(FhnBackendCtx *ctx, FhnOpCode code) {
  const auto *table = ckks_fhn_get_kernels(ctx);
  for (uint32_t i = 0; i < table->num_kernels; ++i)
    if (table->kernels[i].opcode == code)
      return table->kernels[i].fn;
  throw std::runtime_error("missing kernel");
}
void call(FhnKernelFn fn, FhnBackendCtx *ctx, FhnBuffer *out, const FhnBuffer *a, const FhnBuffer *b = nullptr,
          int64_t distance = 0) {
  const FhnBuffer *args[]{a, b};
  const int64_t ip[]{distance, 0, 0, 0};
  const double fp[]{0, 0};
  require(fn(ctx, out, args, ip, fp) == 0, "native kernel failed");
}
struct Product { uint64_t y, w, factor, q; };
// Keep the loop as an externally observable function call on runtime key data;
// every produced word is checked, and a checksum is consumed outside timing.
__attribute__((noinline)) void generalProducts(const std::vector<Product> &products, std::vector<uint64_t> &out) {
  for (size_t i = 0; i < products.size(); ++i)
    out[i] = modarith::mulMod(products[i].y, products[i].w, products[i].q);
}
__attribute__((noinline)) void shoupProducts(const std::vector<Product> &products, std::vector<uint64_t> &out) {
  for (size_t i = 0; i < products.size(); ++i) {
    const auto &p = products[i];
    const uint64_t lazy = modarith::mulModShoupLazy(p.y, p.w, p.factor, p.q);
    out[i] = lazy >= p.q ? lazy - p.q : lazy;
  }
}
uint64_t checksum(const std::vector<uint64_t> &values) {
  uint64_t value = 0;
  for (auto x : values) value = value * 31 ^ x;
  return value;
}
Json microprobe(FhnBackendCtx *ctx, const FhnBuffer *input) {
  const auto key = ctx->relin.lock();
  require(bool(key), "missing relin key");
  const auto view = key->view();
  const auto &params = ctx->tables->params();
  std::vector<uint64_t> moduli = params.mainPrimes();
  moduli.insert(moduli.end(), params.auxPrimes().begin(), params.auxPrimes().end());
  std::vector<Product> products;
  for (uint32_t d = 0; d < view.digit_count; ++d)
    for (uint32_t p = 0; p < 2; ++p)
      for (uint32_t l = 0; l < moduli.size(); ++l) {
        const auto q = moduli[l];
        const auto *key_words = view.digits[d].poly(p).limb(l);
        const auto *input_words = input->view().poly(p).limb(l % params.levelCount());
        for (size_t i = 0; i < params.degree(); ++i) {
          require(key_words[i] < q, "noncanonical key in microprobe");
          products.push_back({input_words[i] % q, key_words[i], modarith::shoupFactor(key_words[i], q), q});
        }
      }
  std::vector<uint64_t> general(products.size()), shoup(products.size());
  generalProducts(products, general);
  shoupProducts(products, shoup);
  require(general == shoup, "Shoup microprobe residue mismatch");
  const auto general_sum = checksum(general), shoup_sum = checksum(shoup);
  auto general_stats = measure([&] { generalProducts(products, general); }, [&] { consumed ^= checksum(general); });
  auto shoup_stats = measure([&] { shoupProducts(products, shoup); }, [&] { consumed ^= checksum(shoup); });
  require(general == shoup, "Shoup microprobe changed after timing");
  return {{"scope", "isolated modular products using actual immutable relin-key coefficients and live input residues; no key switching"},
          {"excluded", "factor precomputation, vector allocation, checksum and exactness checking"},
          {"products_per_sample", products.size()}, {"factor_bytes", products.size() * sizeof(uint64_t)},
          {"exact_residues", true}, {"general_checksum", general_sum}, {"shoup_checksum", shoup_sum},
          {"general_mulMod", general_stats}, {"fixed_coefficient_shoup", shoup_stats}};
}
} // namespace

int main() {
  try {
    const auto process_start = Clock::now();
    std::unique_ptr<FhnBackendCtx, decltype(&ckks_fhn_destroy)> ctx(ckks_fhn_create(nullptr), ckks_fhn_destroy);
    require(bool(ctx), "context creation failed");
    std::unique_ptr<FhnClientCtx, decltype(&ckks_fhn_client_destroy)> client(ckks_fhn_client_create(ctx.get()), ckks_fhn_client_destroy);
    require(bool(client), "client creation failed");
    std::array<FhnInstruction, 4> instructions{};
    instructions[0].opcode = FHN_HMULT;
    for (unsigned s = 0; s < 3; ++s) {
      instructions[s + 1].opcode = FHN_ROTATE;
      instructions[s + 1].params[0] = 4 >> s;
    }
    FhnProgram program{};
    program.version = FHN_ABI_VERSION;
    program.instructions = instructions.data();
    program.num_instructions = instructions.size();
    require(ckks_fhn_client_prepare(client.get(), &program) == 0, "key preparation failed");
    Buffers buffers(ctx.get());
    auto *x = buffers.make();
    const std::array<double, 8> x_values{1, -2, 3, 0, -1, 2, 1, -3};
    require(ckks_fhn_client_encrypt_vec_f64(client.get(), x, x_values.data(), width) == 0, "vector encryption failed");
    std::array<FhnBuffer *, 8> rows{}, baseline{}, profiled{}, ordinary{};
    std::array<double, 8> oracle{};
    const auto hmult = kernel(ctx.get(), FHN_HMULT), hrotadd = kernel(ctx.get(), FHN_HROT_ADD);
    const auto addzero = kernel(ctx.get(), FHN_ADD_CS);
    double max_error = 0;
    for (unsigned j = 0; j < width; ++j) {
      rows[j] = buffers.make(); baseline[j] = buffers.make(); profiled[j] = buffers.make(); ordinary[j] = buffers.make();
      std::array<double, 8> values{};
      for (unsigned k = 0; k < width; ++k) {
        values[k] = static_cast<int>((j * 3 + k * 2) % 5) - 2;
        oracle[j] += values[k] * x_values[k];
      }
      require(ckks_fhn_client_encrypt_vec_f64(client.get(), rows[j], values.data(), width) == 0, "row encryption failed");
      require(ckks_fhn_packed_dot(ctx.get(), baseline[j], rows[j], x, width) == 0, "baseline dot failed");
      Phases discarded;
      profiledDot(ctx.get(), profiled[j], rows[j], x, discarded);
      exact(baseline[j], profiled[j]);
      std::array<double, 8> decoded{};
      require(ckks_fhn_client_decrypt_vec_f64(client.get(), profiled[j], decoded.data(), width) == 0, "decryption failed");
      for (double value : decoded) {
        const auto error = std::abs(value - oracle[j]);
        max_error = std::max(max_error, error);
        require(std::isfinite(value) && error <= 1e-4 + 1e-6 * std::abs(oracle[j]), "logical-slot oracle mismatch");
      }
    }
    auto consume = [&](const auto &outputs) { for (auto *b : outputs) consumed ^= checksum(b); };
    auto run_ordinary = [&] {
      for (unsigned j = 0; j < width; ++j) {
        call(hmult, ctx.get(), ordinary[j], rows[j], x);
        for (int64_t distance : {4, 2, 1}) call(hrotadd, ctx.get(), ordinary[j], ordinary[j], ordinary[j], distance);
      }
    };
    run_ordinary();
    for (unsigned j = 0; j < width; ++j) exact(baseline[j], ordinary[j]);
    Json report{{"benchmark", "fixed matvec-fused 8x8 phase diagnostic"}, {"warmup", warmup}, {"repetitions", repetitions},
                {"x", x_values}, {"row_formula", "rows[j][k] = (j*3+k*2)%5-2"}, {"reduction_distances", {4, 2, 1}},
                {"oracle", oracle}, {"exact_layout_and_residues", true}, {"checked_logical_slots", 64},
                {"max_absolute_oracle_error", max_error}, {"exclusions", {"context/table creation", "key generation", "encryption", "decryption", "checksum", "correctness checks", "JSON formatting"}},
                {"phase_warning", "Phase clocks are diagnostic boundaries, not precise accounting; outer total includes staging, clock overhead and lease/scope destruction. Separate probe medians cannot be subtracted from total."}};
    const auto &params = ctx->tables->params();
    report["parameters"] = {{"log_degree", params.logDegree()}, {"main_primes", params.mainPrimes()},
                            {"aux_primes", params.auxPrimes()}, {"dnum", params.dnum()}, {"log_default_scale", params.logDefaultScale()}};
    report["public_coarse_c_function_total"] = measure([&] {
      for (unsigned j = 0; j < width; ++j)
        require(ckks_fhn_packed_dot(ctx.get(), baseline[j], rows[j], x, width) == 0, "timed C baseline failed");
    }, [&] { consume(baseline); }, 8);
    report["ordinary_native_kernels_total"] = measure(run_ordinary, [&] { consume(ordinary); }, 32);
    std::vector<double> total_samples, validation_samples, multiply_samples, copy_samples;
    std::array<std::vector<double>, 3> rotation_samples;
    for (unsigned rep = 0; rep < warmup + repetitions; ++rep) {
      Phases phases;
      const auto start = Clock::now();
      for (unsigned j = 0; j < width; ++j) profiledDot(ctx.get(), profiled[j], rows[j], x, phases);
      const auto elapsed = milliseconds(start);
      consume(profiled);
      if (rep >= warmup) {
        total_samples.push_back(elapsed); validation_samples.push_back(phases.validation);
        multiply_samples.push_back(phases.multiply); copy_samples.push_back(phases.copy);
        for (unsigned s = 0; s < 3; ++s) rotation_samples[s].push_back(phases.rotate[s]);
      }
    }
    report["profiled_total"] = summary(total_samples, 8);
    report["phases"] = {{"input_and_four_key_validation", summary(validation_samples, 8)},
                        {"region_multiply", summary(multiply_samples, 8)}, {"output_copy", summary(copy_samples, 8)}};
    for (unsigned s = 0; s < 3; ++s) report["phases"]["region_rotateAdd_" + std::to_string(4 >> s)] = summary(rotation_samples[s], 8);
    unsigned valid = 0;
    report["probes"]["same_live_shared_vector_input_validation"] = measure([&] {
      valid = 0;
      for (unsigned j = 0; j < 8; ++j) valid += validateInput(ctx.get(), x);
    }, [&] { require(valid == 8, "input validation probe failed"); consumed ^= valid; }, 8);
    report["probes"]["four_key_canonical_scans"] = measure([&] {
      valid = 0;
      for (unsigned j = 0; j < 8; ++j) {
        auto key = ctx->relin.lock();
        valid += key && region::validateKey(key->view(), SwitchKeyKind::Relinearization, 1, *ctx->tables);
        for (int64_t distance : {4, 2, 1}) {
          auto g = eval::galoisElement(distance, params.logDegree());
          key = ctx->rotations.at(g).lock();
          valid += key && region::validateKey(key->view(), SwitchKeyKind::Galois, g, *ctx->tables);
        }
      }
    }, [&] { require(valid == 32, "key validation probe failed"); consumed ^= valid; }, 32);
    uint64_t allocated_words = 0;
    report["probes"]["FhnBuffer_allocate_free"] = measure([&] {
      allocated_words = 0;
      for (unsigned j = 0; j < 32; ++j) {
        auto *b = ckks_fhn_buffer_alloc(ctx.get()); require(b != nullptr, "allocation probe failed");
        allocated_words += b->slab.capacityWords(); ckks_fhn_buffer_free(ctx.get(), b);
      }
    }, [&] { consumed ^= allocated_words; }, 32);
    report["probes"]["native_ADD_CS_zero_publication"] = measure([&] {
      for (unsigned j = 0; j < 8; ++j) call(addzero, ctx.get(), profiled[j], baseline[j]);
    }, [&] { consume(profiled); }, 8);
    report["probes"]["native_ADD_CS_zero_publication"]["scope"] = "checked native ADD_CS(0): validation, staging, scalar-zero copy and publication, not a pure memcpy";
    for (unsigned j = 0; j < width; ++j) exact(baseline[j], profiled[j]);
    report["modular_product_microprobe"] = microprobe(ctx.get(), x);
    report["consumed_checksum"] = consumed;
    report["process_elapsed_ms_including_setup"] = milliseconds(process_start);
    std::cout << report.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "fhn-matvec-profile: %s\n", error.what());
    return 1;
  }
}
