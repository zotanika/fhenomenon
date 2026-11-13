#include "FHN/fhn_backend_api.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

// A deliberately minimal FHN backend: valid handshake, empty kernel
// table, new/delete-backed buffers, identity "encryption" of one int64 —
// and, the point of the fixture, a PARTIAL level-model trio (only
// fhn_fresh_level), which conformant loaders must ignore whole.
namespace {
struct PartialCtx {
  int unused;
};
FhnKernelTable g_table{0, nullptr};
} // namespace

extern "C" uint32_t ptl_fhn_get_abi_version(void) { return FHN_ABI_VERSION; }
extern "C" FhnBackendInfo *ptl_fhn_get_info(void) {
  static FhnBackendInfo info{"partial-fixture", "0", FHN_DEVICE_CPU, 0};
  return &info;
}
extern "C" FhnBackendCtx *ptl_fhn_create(const char *config) {
  if (config && std::strcmp(config, "{\"fixture\":true}") != 0)
    return nullptr;
  return reinterpret_cast<FhnBackendCtx *>(new PartialCtx{0});
}
extern "C" void ptl_fhn_destroy(FhnBackendCtx *ctx) { delete reinterpret_cast<PartialCtx *>(ctx); }
extern "C" FhnKernelTable *ptl_fhn_get_kernels(FhnBackendCtx *) { return &g_table; }
extern "C" FhnBuffer *ptl_fhn_buffer_alloc(FhnBackendCtx *) { return reinterpret_cast<FhnBuffer *>(new int64_t(0)); }
extern "C" void ptl_fhn_buffer_free(FhnBackendCtx *, FhnBuffer *buffer) { delete reinterpret_cast<int64_t *>(buffer); }
extern "C" int ptl_fhn_encrypt_i64(FhnBackendCtx *, FhnBuffer *out, int64_t value) {
  *reinterpret_cast<int64_t *>(out) = value;
  return 0;
}
extern "C" int ptl_fhn_decrypt_i64(FhnBackendCtx *, const FhnBuffer *in, int64_t *value_out) {
  *value_out = *reinterpret_cast<const int64_t *>(in);
  return 0;
}
// The partial trio: fresh_level WITHOUT level_bytes/opcode_level_effect.
extern "C" int64_t ptl_fhn_fresh_level(FhnBackendCtx *) { return 3; }

// Second prefix deliberately has packed f64 I/O and no integer exports.
// This pins the loader contract needed by CKKS-only backends such as Cheddar.
extern "C" uint32_t vonly_fhn_get_abi_version() { return FHN_ABI_VERSION; }
extern "C" FhnBackendCtx *vonly_fhn_create(const char *config) { return ptl_fhn_create(config); }
extern "C" void vonly_fhn_destroy(FhnBackendCtx *ctx) { ptl_fhn_destroy(ctx); }
extern "C" FhnKernelTable *vonly_fhn_get_kernels(FhnBackendCtx *) { return &g_table; }
extern "C" FhnBuffer *vonly_fhn_buffer_alloc(FhnBackendCtx *) {
  return reinterpret_cast<FhnBuffer *>(new std::vector<double>());
}
extern "C" void vonly_fhn_buffer_free(FhnBackendCtx *, FhnBuffer *buffer) {
  delete reinterpret_cast<std::vector<double> *>(buffer);
}
extern "C" int vonly_fhn_encrypt_vec_f64(FhnBackendCtx *, FhnBuffer *buffer, const double *values, uint32_t count) {
  if (!buffer || !values || !count)
    return -1;
  reinterpret_cast<std::vector<double> *>(buffer)->assign(values, values + count);
  return 0;
}
extern "C" int vonly_fhn_decrypt_vec_f64(FhnBackendCtx *, const FhnBuffer *buffer, double *values, uint32_t count) {
  if (!buffer || !values || !count)
    return -1;
  const auto &stored = *reinterpret_cast<const std::vector<double> *>(buffer);
  if (stored.size() != count)
    return -1;
  std::copy(stored.begin(), stored.end(), values);
  return 0;
}
