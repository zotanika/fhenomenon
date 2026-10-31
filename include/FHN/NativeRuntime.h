#pragma once

#include "FHN/Runtime.h"
#include "FHN/fhn_circuit_api.h"
#include "FHN/fhn_native_api.h"
#include "FHN/fhn_prepared_api.h"
#include <array>

namespace fhenomenon::runtime {

// Compute services only. Context ownership must also retain the loaded library.
// Injection permits compute providers without a dynamically loaded library.
struct NativeServices {
  std::shared_ptr<FhnBackendCtx> context;
  FhnGetKernelsFn get_kernels = nullptr;
  FhnBufferAllocFn buffer_alloc = nullptr;
  FhnBufferFreeFn buffer_free = nullptr;
  FhnFreshLevelFn fresh_level = nullptr;
  FhnLevelBytesFn level_bytes = nullptr;
  FhnCkksBufferInfoFn buffer_info = nullptr;
  FhnRotationPreparationRequiredBytesFn preparation_required_bytes = nullptr;
  FhnPrepareRotationSourceFn prepare_rotation = nullptr;
  FhnRotationPreparationBytesFn preparation_bytes = nullptr;
  FhnRotatePreparedFn rotate_prepared = nullptr;
  FhnRotationPreparationFreeFn preparation_free = nullptr;
  FhnCompileCircuitFn compile = nullptr;
  FhnRunExecutableFn run = nullptr;
  FhnExecutableStatsFn executable_stats = nullptr;
  FhnExecutableFreeFn executable_free = nullptr;
};
NativeServices loadNativeServices(const std::string &library_path, const std::string &config = {});

struct RegionOptions {
  std::uint32_t max_nodes = 256;
  std::size_t workspace_bytes = 8 * 1024 * 1024;
};

// Registers native.op, native.prepare_rotation, native.rotate,
// native.compile_region and native.run_region. Registered recipes own their
// services and remain valid after this adapter dies. No recipe owns a client.
class NativeRuntime {
  public:
  NativeRuntime(Runtime &, NativeServices);
  FhnBackendCtx *context() const noexcept;
  Value importCopy(const FhnBuffer *);
  const FhnBuffer *buffer(const Value &) const;
  Value op(FhnOpCode, Values, std::array<std::int64_t, 4> params = {}, std::array<double, 2> fparams = {});
  // Explicit mathematical adapter; raw op() always preserves physical semantics.
  Value semantic(FhnCircuitOp, Values, std::int64_t rotation = 0, double scalar = 0);
  Value prepareRotation(Value);
  Value rotate(Value, std::int64_t rotation, Value preparation = {});
  // Reserves the finite workspace cap before invoking the optional compiler.
  // Prototypes are metadata only and are not retained as dependencies.
  Value compileRegion(const FhnCircuit &, Values prototypes, RegionOptions = {});
  FhnExecutable *executable(const Value &) const; // trusted host key preparation
  FhnExecutableStats executableStats(const Value &) const;
  Values runRegion(Value region, Values inputs);

  private:
  Runtime &runtime_;
  std::shared_ptr<const NativeServices> services_;
};
} // namespace fhenomenon::runtime
