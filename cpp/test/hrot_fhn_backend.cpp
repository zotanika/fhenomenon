// Real Toy primitives exposed through an alternative valid capability table:
// HROT is native, ROTATE and HROT_ADD are absent, and scalar I/O is absent.
#include "FHN/ToyFheKernels.h"
#include <vector>

extern "C" {
uint32_t alt_fhn_get_abi_version() { return toyfhe_fhn_get_abi_version(); }
FhnBackendCtx *alt_fhn_create(const char *config) { return toyfhe_fhn_create(config); }
void alt_fhn_destroy(FhnBackendCtx *ctx) { toyfhe_fhn_destroy(ctx); }
FhnKernelTable *alt_fhn_get_kernels(FhnBackendCtx *ctx) {
  static std::vector<FhnKernelEntry> entries = [ctx] {
    std::vector<FhnKernelEntry> result;
    auto *source = toyfhe_fhn_get_kernels(ctx);
    for (uint32_t i = 0; i < source->num_kernels; ++i) {
      auto entry = source->kernels[i];
      if (entry.opcode == FHN_HROT_ADD)
        continue;
      if (entry.opcode == FHN_ROTATE)
        entry.opcode = FHN_HROT;
      result.push_back(entry);
    }
    return result;
  }();
  static FhnKernelTable table{static_cast<uint32_t>(entries.size()), entries.data()};
  return &table;
}
FhnBuffer *alt_fhn_buffer_alloc(FhnBackendCtx *ctx) { return toyfhe_fhn_buffer_alloc(ctx); }
void alt_fhn_buffer_free(FhnBackendCtx *ctx, FhnBuffer *p) { toyfhe_fhn_buffer_free(ctx, p); }
int alt_fhn_encrypt_vec_f64(FhnBackendCtx *ctx, FhnBuffer *out, const double *v, uint32_t n) {
  return toyfhe_fhn_encrypt_vec_f64(ctx, out, v, n);
}
int alt_fhn_decrypt_vec_f64(FhnBackendCtx *ctx, const FhnBuffer *in, double *v, uint32_t n) {
  return toyfhe_fhn_decrypt_vec_f64(ctx, in, v, n);
}
}
