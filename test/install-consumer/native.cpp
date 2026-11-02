#include "FHN/NativeRuntime.h"
#include "FHN/contrib/MatvecKernels.h"
#include "FHN/contrib/SearchKernels.h"
#include "FHN/contrib/StatsKernels.h"
#include "FHN/fhn_backend_api.h"
#include "FHN/fhn_circuit_api.h"
#include <cmath>
#include <dlfcn.h>
#include <memory>

template <class T> T symbol(void *library, const char *name) { return reinterpret_cast<T>(dlsym(library, name)); }
void registerRotationPair(fhenomenon::runtime::Runtime &);
int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  std::unique_ptr<void, decltype(&dlclose)> library(dlopen(argv[1], RTLD_NOW), dlclose);
  if (!library)
    return 2;
  const auto version = symbol<FhnGetAbiVersionFn>(library.get(), "ckks_fhn_get_abi_version");
  const auto create = symbol<FhnCreateFn>(library.get(), "ckks_fhn_create");
  const auto destroy = symbol<FhnDestroyFn>(library.get(), "ckks_fhn_destroy");
  const auto kernels = symbol<FhnGetKernelsFn>(library.get(), "ckks_fhn_get_kernels");
  const auto allocate = symbol<FhnBufferAllocFn>(library.get(), "ckks_fhn_buffer_alloc");
  const auto release = symbol<FhnBufferFreeFn>(library.get(), "ckks_fhn_buffer_free");
  const auto create_client = symbol<FhnClientCreateFn>(library.get(), "ckks_fhn_client_create");
  const auto destroy_client = symbol<FhnClientDestroyFn>(library.get(), "ckks_fhn_client_destroy");
  const auto prepare = symbol<FhnClientPrepareFn>(library.get(), "ckks_fhn_client_prepare");
  const auto encrypt = symbol<FhnClientEncryptVectorDoubleFn>(library.get(), "ckks_fhn_client_encrypt_vec_f64");
  const auto decrypt = symbol<FhnClientDecryptVectorDoubleFn>(library.get(), "ckks_fhn_client_decrypt_vec_f64");
  if (!version || version() != FHN_ABI_VERSION || !create || !destroy || !kernels || !allocate || !release ||
      !create_client || !destroy_client || !prepare || !encrypt || !decrypt)
    return 3;
  std::unique_ptr<FhnBackendCtx, FhnDestroyFn> ctx(create(nullptr), destroy);
  if (!ctx)
    return 4;
  std::unique_ptr<FhnClientCtx, FhnClientDestroyFn> client(create_client(ctx.get()), destroy_client);
  if (!client)
    return 5;
  auto free_buffer = [&](FhnBuffer *b) { release(ctx.get(), b); };
  std::unique_ptr<FhnBuffer, decltype(free_buffer)> input(allocate(ctx.get()), free_buffer);
  std::unique_ptr<FhnBuffer, decltype(free_buffer)> output(allocate(ctx.get()), free_buffer);
  const double values[]{0.25, -1.5, 2.125, 0.75};
  double actual[4]{};
  if (!input || !output || encrypt(client.get(), input.get(), values, 4) != 0)
    return 6;
  FhnInstruction instruction{};
  instruction.opcode = FHN_HMULT;
  FhnProgram program{FHN_ABI_VERSION, 1, &instruction, 0, nullptr, 0, nullptr};
  if (prepare(client.get(), &program) != 0)
    return 7;
  const auto *table = kernels(ctx.get());
  FhnKernelFn multiply = nullptr;
  for (uint32_t i = 0; table && i < table->num_kernels; ++i)
    if (table->kernels[i].opcode == FHN_HMULT)
      multiply = table->kernels[i].fn;
  const FhnBuffer *args[]{input.get(), input.get()};
  if (!multiply || multiply(ctx.get(), output.get(), args, nullptr, nullptr) != 0 ||
      decrypt(client.get(), output.get(), actual, 4) != 0)
    return 8;
  for (unsigned i = 0; i < 4; ++i)
    if (!std::isfinite(actual[i]) || std::abs(actual[i] - values[i] * values[i]) > 1e-5)
      return 9;
  const auto compile = symbol<FhnCompileCircuitFn>(library.get(), "ckks_fhn_compile_circuit");
  const auto prepare_executable =
    symbol<FhnClientPrepareExecutableFn>(library.get(), "ckks_fhn_client_prepare_executable");
  const auto run = symbol<FhnRunExecutableFn>(library.get(), "ckks_fhn_run_executable");
  const auto get_stats = symbol<FhnExecutableStatsFn>(library.get(), "ckks_fhn_executable_stats");
  const auto free_executable = symbol<FhnExecutableFreeFn>(library.get(), "ckks_fhn_executable_free");
  if (!compile || !prepare_executable || !run || !get_stats || !free_executable)
    return 10;
  FhnCircuitNode nodes[]{{FHN_CIRCUIT_MULTIPLY, 2, {1, 1}, 0, 0}, {FHN_CIRCUIT_SCALE, 3, {2, 0}, 0, 3}};
  uint32_t input_ids[]{1}, output_ids[]{3};
  FhnCircuit circuit{FHN_CIRCUIT_VERSION, 2, nodes, 1, input_ids, 1, output_ids};
  const FhnBuffer *bound_inputs[]{input.get()};
  FhnBuffer *bound_outputs[]{output.get()};
  char error[256]{};
  std::unique_ptr<FhnExecutable, FhnExecutableFreeFn> executable(
    compile(ctx.get(), &circuit, bound_inputs, 1, nullptr, error, sizeof(error)), free_executable);
  if (!executable || prepare_executable(client.get(), executable.get()) != 0)
    return 11;
  FhnExecutableStats stats{};
  stats.struct_size = sizeof(stats);
  if (get_stats(executable.get(), &stats) != 0 || stats.regions == 0 || stats.storage_bytes == 0)
    return 12;
  if (run(executable.get(), bound_inputs, 1, bound_outputs, 1) != 0 ||
      decrypt(client.get(), output.get(), actual, 4) != 0)
    return 13;
  for (unsigned i = 0; i < 4; ++i)
    if (!std::isfinite(actual[i]) || std::abs(actual[i] - 3 * values[i] * values[i]) > 1e-5)
      return 14;
  using namespace fhenomenon::runtime;
  NativeServices services;
  // Borrow this fixture's existing context; it outlives all resources below.
  services.context = std::shared_ptr<FhnBackendCtx>(ctx.get(), [](FhnBackendCtx *) {});
  services.get_kernels = kernels;
  services.buffer_alloc = allocate;
  services.buffer_free = release;
  services.fresh_level = symbol<FhnFreshLevelFn>(library.get(), "ckks_fhn_fresh_level");
  services.level_bytes = symbol<FhnLevelBytesFn>(library.get(), "ckks_fhn_level_bytes");
  services.buffer_info = symbol<FhnCkksBufferInfoFn>(library.get(), "ckks_fhn_buffer_info");
  services.preparation_required_bytes =
    symbol<FhnRotationPreparationRequiredBytesFn>(library.get(), "ckks_fhn_rotation_preparation_required_bytes");
  services.prepare_rotation = symbol<FhnPrepareRotationSourceFn>(library.get(), "ckks_fhn_prepare_rotation_source");
  services.preparation_bytes =
    symbol<FhnRotationPreparationBytesFn>(library.get(), "ckks_fhn_rotation_preparation_bytes");
  services.rotate_prepared = symbol<FhnRotatePreparedFn>(library.get(), "ckks_fhn_rotate_prepared");
  services.preparation_free = symbol<FhnRotationPreparationFreeFn>(library.get(), "ckks_fhn_rotation_preparation_free");
  services.compile = compile;
  services.run = run;
  services.executable_stats = get_stats;
  services.executable_free = free_executable;
  Runtime runtime;
  NativeRuntime native(runtime, services);
  auto source = native.importCopy(input.get());
  FhnInstruction rotations[3]{};
  rotations[0].opcode = rotations[1].opcode = rotations[2].opcode = FHN_ROTATE;
  rotations[0].params[0] = 1;
  rotations[1].params[0] = -1;
  rotations[2].params[0] = 2;
  FhnProgram rotation_program{FHN_ABI_VERSION, 3, rotations, 0, nullptr, 0, nullptr};
  if (prepare(client.get(), &rotation_program))
    return 15;
  auto prepared = native.prepareRotation(source);
  registerRotationPair(runtime);
  const auto pair = runtime.submit({"example.rotation-pair", {source, prepared}, std::array<std::int64_t, 2>{1, -1}});
  if (pair.size() != 2 || prepared.dependencies().at(0).identity() != source.identity())
    return 16;
  for (size_t which = 0; which < 2; ++which) {
    if (decrypt(client.get(), native.buffer(pair[which]), actual, 4))
      return 17;
    for (unsigned i = 0; i < 4; ++i)
      if (!std::isfinite(actual[i]) || std::abs(actual[i] - values[(i + (which ? 3 : 1)) % 4]) > 1e-5)
        return 18;
  }
  // The optional compiler consumes an immediate result at an explicit boundary.
  auto shifted = native.semantic(FHN_CIRCUIT_ADD_CONSTANT, {source}, 0, 1);
  auto region = native.compileRegion(circuit, {shifted});
  if (prepare_executable(client.get(), native.executable(region)))
    return 19;
  auto result = native.runRegion(region, {shifted});
  if (decrypt(client.get(), native.buffer(result.at(0)), actual, 4))
    return 20;
  for (unsigned i = 0; i < 4; ++i)
    if (!std::isfinite(actual[i]) || std::abs(actual[i] - 3 * (values[i] + 1) * (values[i] + 1)) > 1e-5)
      return 21;
  // This contribution links as its own installed target. Ordinary commands
  // produce its shared sum; an exposed energy value can return to ordinary
  // rotate/add before the contributed partial reduction completes the app.
  fhenomenon::contrib::registerStatsKernels(runtime, services, argv[1]);
  auto sum = native.semantic(FHN_CIRCUIT_ADD, {source, native.rotate(source, 2)});
  sum = native.semantic(FHN_CIRCUIT_ADD, {sum, native.rotate(sum, 1)});
  auto full_variance = runtime.submit({"contrib.stats.variance", {source, sum}, std::uint32_t{4}}).at(0);
  auto energy = runtime.submit({"contrib.stats.energy", {source, sum}, std::uint32_t{4}}).at(0);
  auto partial = native.semantic(FHN_CIRCUIT_ADD, {energy, native.rotate(energy, 2)});
  auto split_variance = runtime.submit({"contrib.packed.reduce", {partial}, std::uint32_t{2}}).at(0);
  double plain_sum = 0, plain_squares = 0;
  for (double value : values) {
    plain_sum += value;
    plain_squares += value * value;
  }
  const auto expected_variance = 4 * plain_squares - plain_sum * plain_sum;
  for (const auto &variance : {full_variance, split_variance}) {
    if (decrypt(client.get(), native.buffer(variance), actual, 4))
      return 22;
    for (double value : actual)
      if (!std::isfinite(value) || std::abs(value - expected_variance) > 1e-5)
        return 23;
  }
  // One prepared source is reused across complete dots and a product that
  // crosses ordinary rotate/add commands before resuming a partial reduction.
  fhenomenon::contrib::registerMatvecKernels(runtime, services, argv[1]);
  auto matvec_state =
    runtime.submit({"contrib.matvec.prepare", {source}, fhenomenon::contrib::MatvecPrepareRequest{4}}).at(0);
  auto dot = runtime.submit({"contrib.matvec.dot", {source, source, matvec_state}, {}}).at(0);
  auto product = runtime.submit({"contrib.matvec.product", {source, source, matvec_state}, {}}).at(0);
  auto halfway = native.semantic(FHN_CIRCUIT_ADD, {product, native.rotate(product, 2)});
  auto split_dot = runtime.submit({"contrib.matvec.reduce", {halfway, matvec_state}, std::uint32_t{2}}).at(0);
  for (const auto &value : {dot, split_dot}) {
    if (decrypt(client.get(), native.buffer(value), actual, 4))
      return 24;
    for (double slot : actual)
      if (!std::isfinite(slot) || std::abs(slot - plain_squares) > 1e-5)
        return 25;
  }
  fhenomenon::contrib::registerSearchKernels(runtime, services, argv[1]);
  auto index = runtime.submit({"contrib.search.index", {source}, fhenomenon::contrib::SearchShape{4, 1, 1}}).at(0);
  auto query_state = runtime.submit({"contrib.search.prepare-query", {source}, std::uint32_t{4}}).at(0);
  auto scores = runtime.submit({"contrib.search.score", {source, index, query_state}, {}});
  if (scores.size() != 1 || decrypt(client.get(), native.buffer(scores[0]), actual, 4))
    return 26;
  if (!std::isfinite(actual[0]) || std::abs(actual[0] - plain_squares) > 1e-5)
    return 27;
  return 0;
}
