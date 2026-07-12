#pragma once

#include "FHN/fhn_backend_api.h"

#include <optional>
#include <string>

namespace fhenomenon {
namespace corpus {

// A dlopen'd FHN backend, consumed purely through the public C ABI —
// deliberately independent of the C++ Backend hierarchy so this file
// doubles as a reference ABI consumer for backend integrators.
class CorpusBackend {
  public:
  CorpusBackend(const CorpusBackend &) = delete;
  CorpusBackend &operator=(const CorpusBackend &) = delete;
  CorpusBackend(CorpusBackend &&other) noexcept;
  CorpusBackend &operator=(CorpusBackend &&other) noexcept;
  ~CorpusBackend();

  // nullopt on failure with a human-readable reason in *error.
  // Requires: ABI handshake, create/destroy/get_kernels, buffer plane,
  // and encrypt/decrypt i64 (the corpus cannot verify without them).
  static std::optional<CorpusBackend> load(const std::string &library_path, const std::string &symbol_prefix,
                                           std::string *error);

  FhnBackendCtx *ctx() const { return ctx_; }
  FhnKernelTable *kernels() const { return kernels_; }
  FhnBufferAllocFn bufferAlloc() const { return buffer_alloc_; }
  FhnBufferFreeFn bufferFree() const { return buffer_free_; }
  FhnEncryptInt64Fn encryptI64() const { return encrypt_i64_; }
  FhnDecryptInt64Fn decryptI64() const { return decrypt_i64_; }
  FhnBufferPrefetchFn prefetch() const { return prefetch_; } // nullable
  FhnBufferEvictFn evict() const { return evict_; }          // nullable

  private:
  CorpusBackend() = default;

  void *dl_ = nullptr;
  FhnBackendCtx *ctx_ = nullptr;
  FhnKernelTable *kernels_ = nullptr;
  FhnDestroyFn destroy_ = nullptr;
  FhnBufferAllocFn buffer_alloc_ = nullptr;
  FhnBufferFreeFn buffer_free_ = nullptr;
  FhnEncryptInt64Fn encrypt_i64_ = nullptr;
  FhnDecryptInt64Fn decrypt_i64_ = nullptr;
  FhnBufferPrefetchFn prefetch_ = nullptr;
  FhnBufferEvictFn evict_ = nullptr;
};

} // namespace corpus
} // namespace fhenomenon
