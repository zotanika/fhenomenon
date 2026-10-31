#include "CKKS/Backend.h"
#include "CKKS/Circuit.h"
#include "CKKS/CoarseKernels.h"
#include "CKKS/PreparedMatvec.h"
#include "CKKS/PreparedRotation.h"
// Referencing the adapter pulls its one object into this shared library.
extern "C" uint32_t ckks_fhn_get_abi_version() {
  // Keep optional adapter translation units in the shared plugin. Freeing null is
  // deliberately harmless and requires no initialized compute context.
  ckks_fhn_executable_free(nullptr);
  ckks_fhn_rotation_preparation_free(nullptr);
  ckks_fhn_matvec_preparation_free(nullptr);
  return ckks_fhn_get_info() && ckks_fhn_coarse_api_version() == 1 ? FHN_ABI_VERSION : 0u;
}
