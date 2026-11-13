#pragma once

#include "FHN/fhn_backend_api.h"
#include "FHN/fhn_circuit_api.h"

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
  // Requires ABI handshake, create/destroy/get_kernels and buffer allocation.
  // Data-plane pairs are optional; the runner reports unexecutable shapes.
  static std::optional<CorpusBackend> load(const std::string &library_path, const std::string &symbol_prefix,
                                           std::string *error, const char *config_json = nullptr);

  FhnBackendCtx *ctx() const { return ctx_; }
  FhnKernelTable *kernels() const { return kernels_; }
  // Prefer the isolated client data plane when a backend exports it.
  bool hasVector() const { return client_ || encrypt_vector_; }
  int prepare(const FhnProgram *program) const;
  int encryptPacked(FhnBuffer *out, const double *values, uint32_t count) const;
  int decryptPacked(const FhnBuffer *in, double *values, uint32_t count) const;
  bool hasCircuit() const { return compile_circuit_ != nullptr; }
  FhnExecutable *compileCircuit(const FhnCircuit *circuit, const FhnBuffer *const *inputs, uint32_t count,
                                const FhnCompileOptions *options, char *error, uint32_t error_capacity) const;
  int prepareExecutable(FhnExecutable *) const;
  // Executables must be freed before this loader/context is destroyed.
  FhnRunExecutableFn runExecutable() const { return run_executable_; }
  FhnExecutableStatsFn executableStats() const { return executable_stats_; }
  FhnExecutableFreeFn executableFree() const { return executable_free_; }
  FhnBufferAllocFn bufferAlloc() const { return buffer_alloc_; }
  FhnBufferFreeFn bufferFree() const { return buffer_free_; }
  FhnEncryptInt64Fn encryptI64() const { return client_ ? nullptr : encrypt_i64_; }
  FhnDecryptInt64Fn decryptI64() const { return client_ ? nullptr : decrypt_i64_; }
  FhnEncryptVectorDoubleFn encryptVector() const { return encrypt_vector_; }
  FhnDecryptVectorDoubleFn decryptVector() const { return decrypt_vector_; }
  FhnEncryptDoubleFn encryptF64() const { return client_ ? nullptr : encrypt_f64_; }
  FhnDecryptDoubleFn decryptF64() const { return client_ ? nullptr : decrypt_f64_; }
  FhnBufferPrefetchFn prefetch() const { return prefetch_; } // nullable
  FhnBufferEvictFn evict() const { return evict_; }          // nullable
  // Optional level model trio, resolved all-or-nothing (see load()).
  FhnFreshLevelFn freshLevel() const { return fresh_level_; }                       // nullable
  FhnLevelBytesFn levelBytes() const { return level_bytes_; }                       // nullable
  FhnOpcodeLevelEffectFn opcodeLevelEffect() const { return opcode_level_effect_; } // nullable

  private:
  CorpusBackend() = default;

  void *dl_ = nullptr;
  FhnBackendCtx *ctx_ = nullptr;
  FhnClientCtx *client_ = nullptr;
  FhnClientDestroyFn client_destroy_ = nullptr;
  FhnClientPrepareFn client_prepare_ = nullptr;
  FhnClientEncryptVectorDoubleFn client_encrypt_ = nullptr;
  FhnClientDecryptVectorDoubleFn client_decrypt_ = nullptr;
  FhnCompileCircuitFn compile_circuit_ = nullptr;
  FhnClientPrepareExecutableFn prepare_executable_ = nullptr;
  FhnRunExecutableFn run_executable_ = nullptr;
  FhnExecutableStatsFn executable_stats_ = nullptr;
  FhnExecutableFreeFn executable_free_ = nullptr;
  FhnKernelTable *kernels_ = nullptr;
  FhnDestroyFn destroy_ = nullptr;
  FhnBufferAllocFn buffer_alloc_ = nullptr;
  FhnBufferFreeFn buffer_free_ = nullptr;
  FhnEncryptInt64Fn encrypt_i64_ = nullptr;
  FhnDecryptInt64Fn decrypt_i64_ = nullptr;
  FhnEncryptVectorDoubleFn encrypt_vector_ = nullptr;
  FhnDecryptVectorDoubleFn decrypt_vector_ = nullptr;
  FhnEncryptDoubleFn encrypt_f64_ = nullptr;
  FhnDecryptDoubleFn decrypt_f64_ = nullptr;
  FhnBufferPrefetchFn prefetch_ = nullptr;
  FhnBufferEvictFn evict_ = nullptr;
  FhnFreshLevelFn fresh_level_ = nullptr;
  FhnLevelBytesFn level_bytes_ = nullptr;
  FhnOpcodeLevelEffectFn opcode_level_effect_ = nullptr;
};

} // namespace corpus
} // namespace fhenomenon
