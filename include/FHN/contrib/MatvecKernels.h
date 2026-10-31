#pragma once
#include "FHN/NativeRuntime.h"
#include "FHN/fhn_matvec_api.h"
#include <string>
namespace fhenomenon::contrib {
struct MatvecPrepareRequest {
  std::uint32_t width;
  std::uint32_t flags = FHN_MATVEC_PREPARE_ALL;
};
// Registers contrib.matvec.prepare({source}, MatvecPrepareRequest),
// contrib.matvec.product/dot({source,row,preparation}), and
// contrib.matvec.reduce({product_or_intermediate,preparation}, uint32_t width).
// Artifacts retain the source Value identity and cannot bind a fresh same-shaped
// Value. They own context/library services, but no client or evaluation keys.
// Native result staging and publication both reserve full buffer capacity.
// Optional exports must come from the same plugin as services. get_kernels,
// buffer_info and fresh_level must retain that plugin's original function pointers;
// buffer_alloc/free may be instrumented. Missing or mismatched exports
// decline the recipe, allowing caller-provided fallbacks. No compiler is used.
void registerMatvecKernels(runtime::Runtime &, runtime::NativeServices, const std::string &nativeLibraryPath);
} // namespace fhenomenon::contrib
