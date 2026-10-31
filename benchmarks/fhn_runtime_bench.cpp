#include "FHN/NativeRuntime.h"
#include "FHN/contrib/MatvecKernels.h"
#include "FHN/contrib/StatsKernels.h"
#include "corpus_circuit.h"
#include "corpus_oracle.h"
#include "corpus_shapes.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <dlfcn.h>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>

using namespace fhenomenon;
using namespace fhenomenon::runtime;
using namespace fhenomenon::corpus;
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
template <class T> T symbol(void *library, const char *name) {
  auto fn = reinterpret_cast<T>(dlsym(library, name));
  if (!fn)
    throw std::runtime_error(std::string("missing trusted client export: ") + name);
  return fn;
}
bool binary(FhnCircuitOp op) {
  return op == FHN_CIRCUIT_ADD || op == FHN_CIRCUIT_SUBTRACT || op == FHN_CIRCUIT_MULTIPLY;
}

// A benchmark harness, not a runtime graph planner. It walks the unchanged
// corpus semantics in source order and drops handles after their last use.
Values immediate(NativeRuntime &native, const Circuit &circuit, const Values &inputs,
                 const std::map<uint32_t, Value> &preparations) {
  std::map<uint32_t, Value> values;
  std::map<uint32_t, size_t> remaining;
  for (size_t i = 0; i < inputs.size(); ++i)
    values.emplace(circuit.inputs[i], inputs[i]);
  for (const auto &node : circuit.nodes) {
    ++remaining[node.operands[0]];
    if (binary(node.op))
      ++remaining[node.operands[1]];
  }
  for (auto id : circuit.outputs)
    ++remaining[id];
  for (const auto &node : circuit.nodes) {
    Values args{values.at(node.operands[0])};
    if (binary(node.op))
      args.push_back(values.at(node.operands[1]));
    const auto prepared = preparations.find(node.operands[0]);
    auto result = node.op == FHN_CIRCUIT_ROTATE && prepared != preparations.end()
                    ? native.rotate(args[0], node.rotation, prepared->second)
                    : native.semantic(node.op, args, node.rotation, node.scalar);
    values.emplace(node.result_id, std::move(result));
    if (!--remaining[node.operands[0]])
      values.erase(node.operands[0]);
    if (binary(node.op) && !--remaining[node.operands[1]])
      values.erase(node.operands[1]);
  }
  Values outputs;
  for (auto id : circuit.outputs)
    outputs.push_back(values.at(id));
  return outputs;
}

struct Fixture {
  // Order is intentional: client and raw inputs die before context/library.
  std::shared_ptr<void> library;
  std::string library_path;
  NativeServices services;
  std::unique_ptr<FhnClientCtx, FhnClientDestroyFn> client{nullptr, nullptr};
  FhnClientEncryptVectorDoubleFn encrypt = nullptr;
  FhnClientDecryptVectorDoubleFn decrypt = nullptr;
  FhnClientPrepareExecutableFn prepare_executable = nullptr;
  std::vector<std::shared_ptr<FhnBuffer>> inputs;
  double setup_ms = 0;
  Fixture(const std::string &path, const Shape &shape) {
    library_path = path;
    const auto start = Clock::now();
    library = std::shared_ptr<void>(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL), [](void *p) {
      if (p)
        dlclose(p);
    });
    if (!library)
      throw std::runtime_error("cannot load trusted client library");
    services = loadNativeServices(path);
    auto create = symbol<FhnClientCreateFn>(library.get(), "ckks_fhn_client_create");
    auto destroy = symbol<FhnClientDestroyFn>(library.get(), "ckks_fhn_client_destroy");
    client = {create(services.context.get()), destroy};
    if (!client)
      throw std::runtime_error("client creation failed");
    encrypt = symbol<FhnClientEncryptVectorDoubleFn>(library.get(), "ckks_fhn_client_encrypt_vec_f64");
    decrypt = symbol<FhnClientDecryptVectorDoubleFn>(library.get(), "ckks_fhn_client_decrypt_vec_f64");
    prepare_executable = symbol<FhnClientPrepareExecutableFn>(library.get(), "ckks_fhn_client_prepare_executable");
    auto prepare = symbol<FhnClientPrepareFn>(library.get(), "ckks_fhn_client_prepare");
    // This harness imports mathematical multiplication, so request its HMULT key.
    std::vector<FhnInstruction> instructions(shape.program->instructions,
                                             shape.program->instructions + shape.program->num_instructions);
    for (auto &instruction : instructions)
      if (instruction.opcode == FHN_MULT_CC)
        instruction.opcode = FHN_HMULT;
    auto program = *shape.program;
    program.instructions = instructions.data();
    if (prepare(client.get(), &program))
      throw std::runtime_error("key preparation failed");
    for (uint32_t i = 0; i < program.num_inputs; ++i) {
      auto ptr = services.buffer_alloc(services.context.get());
      if (!ptr)
        throw std::runtime_error("input allocation failed");
      auto s = services;
      inputs.emplace_back(ptr, [s](FhnBuffer *b) { s.buffer_free(s.context.get(), b); });
      const auto &source = shape.inputs.at(program.input_ids[i]);
      std::vector<double> slots(source.begin(), source.end());
      if (encrypt(client.get(), ptr, slots.data(), shape.slot_count))
        throw std::runtime_error("encryption failed");
    }
    setup_ms = elapsed(start);
  }
};

struct Run {
  Clock::time_point setup_start = Clock::now();
  Runtime runtime{{0, 256}};
  NativeServices services;
  NativeRuntime native;
  std::string mode, shape_name;
  std::uint32_t width;
  Values inputs;
  std::map<uint32_t, Value> preparations;
  Value region, matvec_preparation;
  std::uint32_t matvec_flags = 0;
  double registration_ms = 0, input_import_ms = 0, preparation_ms = 0, max_error = 0;
  size_t preparation_bytes = 0;
  Run(Fixture &fixture, const Circuit &circuit, const std::string &mode, const Shape &shape)
    : services([&] {
        auto services = fixture.services;
        if (mode != "region") { // Prove the ordinary path has no compiler to call.
          services.compile = nullptr;
          services.run = nullptr;
          services.executable_stats = nullptr;
          services.executable_free = nullptr;
        }
        return services;
      }()),
      native(runtime, services), mode(mode), shape_name(shape.name), width(shape.slot_count) {
    // Every comparison has the same registered providers. Compiler services
    // remain absent from all modes except the explicit region reference.
    contrib::registerStatsKernels(runtime, services, fixture.library_path);
    contrib::registerMatvecKernels(runtime, services, fixture.library_path);
    registration_ms = elapsed(setup_start);
    const auto import_start = Clock::now();
    for (const auto &input : fixture.inputs)
      inputs.push_back(native.importCopy(input.get()));
    input_import_ms = elapsed(import_start);
    auto start = Clock::now();
    if (mode.rfind("matvec-", 0) == 0) {
      matvec_flags = mode == "matvec-bound"    ? 0
                     : mode == "matvec-maps"   ? FHN_MATVEC_PREPARE_MAPS
                     : mode == "matvec-keys"   ? FHN_MATVEC_PREPARE_KEYS
                     : mode == "matvec-vector" ? FHN_MATVEC_PREPARE_VECTOR
                                               : FHN_MATVEC_PREPARE_ALL;
      matvec_preparation =
        runtime.submit({"contrib.matvec.prepare", {inputs[0]}, contrib::MatvecPrepareRequest{width, matvec_flags}})
          .at(0);
      preparation_bytes = matvec_preparation.bytes();
    } else if (mode == "prepared") {
      std::map<uint32_t, size_t> uses;
      for (const auto &node : circuit.nodes)
        if (node.op == FHN_CIRCUIT_ROTATE)
          ++uses[node.operands[0]];
      for (size_t i = 0; i < inputs.size(); ++i)
        if (uses[circuit.inputs[i]] > 1) {
          auto prepared = native.prepareRotation(inputs[i]);
          preparation_bytes += prepared.bytes();
          preparations.emplace(circuit.inputs[i], std::move(prepared));
        }
    } else if (mode == "region") {
      region = native.compileRegion(circuit.view(), inputs, {1024, 32 * 1024 * 1024});
      if (fixture.prepare_executable(fixture.client.get(), native.executable(region)))
        throw std::runtime_error("region key preparation failed");
    }
    preparation_ms = elapsed(start);
  }
  Value reduceOrdinary(Value value, std::uint32_t count) {
    for (std::uint32_t distance = count / 2; distance; distance /= 2) {
      auto rotated = native.rotate(value, distance);
      value = native.semantic(FHN_CIRCUIT_ADD, {value, rotated});
    }
    return value;
  }
  Value contributed(const char *operation, Values args, std::uint32_t count) {
    return runtime.submit({operation, std::move(args), count}).at(0);
  }
  Value finishSplit(Value energy) {
    // Expose the canonical intermediate to existing, separate commands before
    // passing their output back to the contributed partial reduction.
    auto rotated = native.rotate(energy, width / 2);
    auto halfway = native.semantic(FHN_CIRCUIT_ADD, {energy, rotated});
    return contributed("contrib.packed.reduce", {halfway}, width / 2);
  }
  Values executeContribution() {
    Values outputs;
    if (shape_name == "stats") {
      for (const auto &x : inputs) {
        // The ordinary sum is both a public app output and an input to the
        // contribution. Its full cost is inside every timed application run.
        auto sum = reduceOrdinary(x, width);
        Value variance;
        if (mode == "contributed") {
          variance = contributed("contrib.stats.variance", {x, sum}, width);
        } else {
          auto energy = contributed("contrib.stats.energy", {x, sum}, width);
          // Same stable expression, fully ordinary reduction: separates the
          // algebraic improvement from the benefit of the larger native kernel.
          variance = mode == "energy-only" ? reduceOrdinary(energy, width) : finishSplit(energy);
        }
        outputs.push_back(sum);
        outputs.push_back(variance);
      }
    } else if (shape_name == "matvec-fused") {
      for (size_t row = 1; row < inputs.size(); ++row) {
        // All row kernels share the original vector handle; the split path
        // produces its product with the existing normalized multiply command.
        outputs.push_back(mode == "contributed"
                            ? contributed("contrib.packed.dot", {inputs[0], inputs[row]}, width)
                            : finishSplit(native.semantic(FHN_CIRCUIT_MULTIPLY, {inputs[0], inputs[row]})));
      }
    } else {
      throw std::runtime_error("no contributed composition for workload");
    }
    return outputs;
  }
  Values execute(const Circuit &circuit) {
    if (matvec_preparation) {
      Values outputs;
      for (size_t row = 1; row < inputs.size(); ++row) {
        if (mode == "matvec-split") {
          auto product =
            runtime.submit({"contrib.matvec.product", {inputs[0], inputs[row], matvec_preparation}, {}}).at(0);
          auto rotated = native.rotate(product, width / 2);
          auto partial = native.semantic(FHN_CIRCUIT_ADD, {product, rotated});
          outputs.push_back(contributed("contrib.matvec.reduce", {partial, matvec_preparation}, width / 2));
        } else if (mode == "matvec-ordinary-product") {
          auto product = native.semantic(FHN_CIRCUIT_MULTIPLY, {inputs[0], inputs[row]});
          outputs.push_back(contributed("contrib.matvec.reduce", {product, matvec_preparation}, width));
        } else {
          outputs.push_back(
            runtime.submit({"contrib.matvec.dot", {inputs[0], inputs[row], matvec_preparation}, {}}).at(0));
        }
      }
      return outputs;
    }
    if (mode == "contributed" || mode == "split" || mode == "energy-only")
      return executeContribution();
    return region ? native.runRegion(region, inputs) : immediate(native, circuit, inputs, preparations);
  }
  void verify(Fixture &fixture, const Shape &shape, const Values &outputs) {
    auto expected = evaluate(*shape.program, shape.inputs);
    if (!expected || outputs.size() != shape.output_ids.size())
      throw std::runtime_error("oracle output mismatch");
    for (size_t i = 0; i < outputs.size(); ++i) {
      std::vector<double> actual(shape.slot_count);
      if (fixture.decrypt(fixture.client.get(), native.buffer(outputs[i]), actual.data(), shape.slot_count))
        throw std::runtime_error("decryption failed");
      for (size_t j = 0; j < actual.size(); ++j) {
        const auto want = static_cast<double>(expected->at(shape.output_ids[i])[j]);
        const auto error = std::abs(actual[j] - want);
        max_error = std::max(max_error, error);
        if (!std::isfinite(actual[j]) || error > 1e-4 + 1e-6 * std::abs(want))
          throw std::runtime_error("all-slot oracle mismatch for " + shape.name + " output " + std::to_string(i) +
                                   " slot " + std::to_string(j));
      }
    }
  }
};
double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values.size() % 2 ? values[values.size() / 2]
                           : (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 4 || (argc == 4 && std::string(argv[3]) != "matvec-reuse"))
      throw std::runtime_error("usage: fhn-bench-runtime NATIVE_LIBRARY [REPS] [matvec-reuse]");
    const unsigned reps = argc >= 3 ? static_cast<unsigned>(std::stoul(argv[2])) : 7;
    const bool matvec_reuse = argc == 4;
    if (!reps || reps > 10000)
      throw std::runtime_error("REPS must be 1..10000");
    // Keep one image lease across shape fixtures, including on sanitizer builds;
    // fresh contexts and clients remain scoped to each independent workload.
    const auto load_start = Clock::now();
    auto benchmark_library = std::shared_ptr<void>(dlopen(argv[1], RTLD_NOW | RTLD_LOCAL), [](void *p) {
      if (p)
        dlclose(p);
    });
    if (!benchmark_library)
      throw std::runtime_error("cannot load benchmark backend library");
    const auto image_load_ms = elapsed(load_start);
    nlohmann::json report = {
      {"schema", 2},
      {"suite", matvec_reuse ? "matvec-reuse" : "runtime"},
      {"repetitions", reps},
      {"backend_image_load_ms", image_load_ms},
      {"timing_scope", "warm dispatch, allocation, evaluation, contribution staging/publication and temporary release; "
                       "excludes initial input import, setup, decryption and final output release"},
      {"memory_scope", "runtime retained ciphertext/artifact slabs and region reservation; excludes keys, context "
                       "scratch/tables, metadata and allocator overhead"},
      {"workloads", nlohmann::json::array()}};
    for (const auto &shape : allShapes()) {
      if (matvec_reuse && shape.name != "matvec-fused")
        continue;
      if (shape.name != "stats" && shape.name != "matvec-fused" && shape.name != "matvec")
        continue;
      const auto graph_start = Clock::now();
      auto circuit = importCircuit(*shape.program);
      if (!circuit)
        throw std::runtime_error("corpus import failed");
      const auto graph_import_ms = elapsed(graph_start);
      Fixture fixture(argv[1], shape);
      std::vector<std::string> modes{"immediate", "prepared", "region"};
      if (shape.name == "stats" || shape.name == "matvec-fused") {
        modes.push_back("contributed");
        modes.push_back("split");
      }
      if (shape.name == "stats")
        modes.push_back("energy-only");
      if (matvec_reuse) {
        for (const auto *mode : {"matvec-bound", "matvec-maps", "matvec-keys", "matvec-vector", "matvec-full",
                                 "matvec-split", "matvec-ordinary-product"})
          modes.emplace_back(mode);
      }
      std::vector<std::unique_ptr<Run>> runs;
      for (const auto &mode : modes) {
        runs.push_back(std::make_unique<Run>(fixture, *circuit, mode, shape));
        for (unsigned warmup = 0; warmup < 2; ++warmup)
          runs.back()->verify(fixture, shape, runs.back()->execute(*circuit));
      }
      std::vector<std::vector<double>> samples(modes.size());
      // Rotate measurement order to reduce systematic order bias.
      for (unsigned rep = 0; rep < reps; ++rep)
        for (size_t step = 0; step < modes.size(); ++step) {
          const auto i = (step + rep) % modes.size();
          auto start = Clock::now();
          auto outputs = runs[i]->execute(*circuit);
          samples[i].push_back(elapsed(start));
          runs[i]->verify(fixture, shape, outputs);
        }
      nlohmann::json row = {{"shape", shape.name},
                            {"source_nodes", circuit->nodes.size()},
                            {"outputs", shape.output_ids.size()},
                            {"slots_per_output", shape.slot_count},
                            {"graph_import_ms", graph_import_ms},
                            {"context_keys_encryption_ms", fixture.setup_ms},
                            {"modes", nlohmann::json::array()}};
      for (size_t i = 0; i < modes.size(); ++i) {
        auto &run = *runs[i];
        nlohmann::json mode = {{"name", modes[i]},
                               {"median_ms", median(samples[i])},
                               {"samples_ms", samples[i]},
                               {"registration_ms", run.registration_ms},
                               {"input_import_ms", run.input_import_ms},
                               {"preparation_ms", run.preparation_ms},
                               {"prepared_sources", run.preparations.size() + (run.matvec_preparation ? 1 : 0)},
                               {"preparation_bytes", run.preparation_bytes},
                               {"peak_tracked_bytes", run.runtime.stats().peak_live_bytes},
                               {"retained_tracked_bytes", run.runtime.stats().live_bytes},
                               {"max_absolute_error", run.max_error}};
        if (run.matvec_preparation) {
          mode["preparation_flags"] = run.matvec_flags;
          mode["key_flag_scope"] = "retained immutable-key Shoup companions plus per-call fixed-scalar Shoup hoisting";
          mode["compiler_available"] = false;
        }
        if (run.region) {
          const auto stats = run.native.executableStats(run.region);
          mode["region_actual_workspace_bytes"] = stats.storage_bytes + stats.scratch_bytes;
          mode["region_reserved_bytes"] = run.region.bytes();
        }
        row["modes"].push_back(std::move(mode));
      }
      report["workloads"].push_back(std::move(row));
    }
    if (report["workloads"].size() != (matvec_reuse ? 1 : 3))
      throw std::runtime_error("missing required corpus workload");
    std::cout << report.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "runtime benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
