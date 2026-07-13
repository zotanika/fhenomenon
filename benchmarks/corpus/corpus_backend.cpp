#include "corpus_backend.h"

#include <dlfcn.h>

namespace fhenomenon {
namespace corpus {

CorpusBackend::CorpusBackend(CorpusBackend &&other) noexcept
  : dl_(other.dl_), ctx_(other.ctx_), kernels_(other.kernels_), destroy_(other.destroy_),
    buffer_alloc_(other.buffer_alloc_), buffer_free_(other.buffer_free_), encrypt_i64_(other.encrypt_i64_),
    decrypt_i64_(other.decrypt_i64_), prefetch_(other.prefetch_), evict_(other.evict_) {
  other.dl_ = nullptr;
  other.ctx_ = nullptr;
  other.kernels_ = nullptr;
  other.destroy_ = nullptr;
  other.buffer_alloc_ = nullptr;
  other.buffer_free_ = nullptr;
  other.encrypt_i64_ = nullptr;
  other.decrypt_i64_ = nullptr;
  other.prefetch_ = nullptr;
  other.evict_ = nullptr;
}

CorpusBackend &CorpusBackend::operator=(CorpusBackend &&other) noexcept {
  if (this == &other) {
    return *this;
  }

  // Tear down whatever this instance currently owns before adopting other's
  // state, mirroring the destructor's ctx-then-library ordering.
  if (ctx_ && destroy_) {
    destroy_(ctx_);
  }
  if (dl_) {
    dlclose(dl_);
  }

  dl_ = other.dl_;
  ctx_ = other.ctx_;
  kernels_ = other.kernels_;
  destroy_ = other.destroy_;
  buffer_alloc_ = other.buffer_alloc_;
  buffer_free_ = other.buffer_free_;
  encrypt_i64_ = other.encrypt_i64_;
  decrypt_i64_ = other.decrypt_i64_;
  prefetch_ = other.prefetch_;
  evict_ = other.evict_;

  other.dl_ = nullptr;
  other.ctx_ = nullptr;
  other.kernels_ = nullptr;
  other.destroy_ = nullptr;
  other.buffer_alloc_ = nullptr;
  other.buffer_free_ = nullptr;
  other.encrypt_i64_ = nullptr;
  other.decrypt_i64_ = nullptr;
  other.prefetch_ = nullptr;
  other.evict_ = nullptr;

  return *this;
}

CorpusBackend::~CorpusBackend() {
  // Moved-from instances have dl_ == nullptr and ctx_ == nullptr, so both
  // guards below are no-ops for them.
  if (ctx_ && destroy_) {
    destroy_(ctx_);
  }
  if (dl_) {
    dlclose(dl_);
  }
}

std::optional<CorpusBackend> CorpusBackend::load(const std::string &library_path, const std::string &symbol_prefix,
                                                 std::string *error) {
  void *dl = dlopen(library_path.c_str(), RTLD_LAZY);
  if (!dl) {
    if (error) {
      *error = dlerror();
    }
    return std::nullopt;
  }

  auto sym = [&](const char *base) { return symbol_prefix + base; };

  // 1. ABI handshake before anything else: same rationale as ExternalBackend
  //    — a library built against a different header revision may have
  //    renumbered opcodes or changed struct layouts.
  auto get_abi_version = reinterpret_cast<FhnGetAbiVersionFn>(dlsym(dl, sym("fhn_get_abi_version").c_str()));
  if (!get_abi_version) {
    if (error) {
      *error = "CorpusBackend: library missing fhn_get_abi_version (pre-versioning binary?)";
    }
    dlclose(dl);
    return std::nullopt;
  }
  const uint32_t abi = get_abi_version();
  if (abi != FHN_ABI_VERSION) {
    if (error) {
      *error = "CorpusBackend: ABI version mismatch (library " + std::to_string(abi) + ", host " +
               std::to_string(FHN_ABI_VERSION) + ")";
    }
    dlclose(dl);
    return std::nullopt;
  }

  // 2. Resolve the required symbols: create/destroy/get_kernels, the buffer
  //    plane, and encrypt/decrypt i64 (the corpus cannot verify without
  //    them).
  auto create = reinterpret_cast<FhnCreateFn>(dlsym(dl, sym("fhn_create").c_str()));
  auto destroy = reinterpret_cast<FhnDestroyFn>(dlsym(dl, sym("fhn_destroy").c_str()));
  auto get_kernels = reinterpret_cast<FhnGetKernelsFn>(dlsym(dl, sym("fhn_get_kernels").c_str()));
  auto buffer_alloc = reinterpret_cast<FhnBufferAllocFn>(dlsym(dl, sym("fhn_buffer_alloc").c_str()));
  auto buffer_free = reinterpret_cast<FhnBufferFreeFn>(dlsym(dl, sym("fhn_buffer_free").c_str()));
  auto encrypt_i64 = reinterpret_cast<FhnEncryptInt64Fn>(dlsym(dl, sym("fhn_encrypt_i64").c_str()));
  auto decrypt_i64 = reinterpret_cast<FhnDecryptInt64Fn>(dlsym(dl, sym("fhn_decrypt_i64").c_str()));
  if (!create || !destroy || !get_kernels || !buffer_alloc || !buffer_free || !encrypt_i64 || !decrypt_i64) {
    if (error) {
      *error = "CorpusBackend: library missing required fhn_* symbols";
    }
    dlclose(dl);
    return std::nullopt;
  }

  // 3. Optional movement hooks. A half-pair is worse than neither (see
  //    ExternalBackend): a lone evict could demote a buffer with nothing
  //    able to restore it, silently corrupting the next use.
  auto prefetch = reinterpret_cast<FhnBufferPrefetchFn>(dlsym(dl, sym("fhn_buffer_prefetch").c_str()));
  auto evict = reinterpret_cast<FhnBufferEvictFn>(dlsym(dl, sym("fhn_buffer_evict").c_str()));
  if ((prefetch != nullptr) != (evict != nullptr)) {
    prefetch = nullptr;
    evict = nullptr;
  }

  // 4. Create the backend context and resolve its kernel table.
  FhnBackendCtx *ctx = create(nullptr);
  if (!ctx) {
    if (error) {
      *error = "CorpusBackend: fhn_create returned null";
    }
    dlclose(dl);
    return std::nullopt;
  }

  FhnKernelTable *kernels = get_kernels(ctx);
  if (!kernels) {
    destroy(ctx);
    if (error) {
      *error = "CorpusBackend: fhn_get_kernels returned null";
    }
    dlclose(dl);
    return std::nullopt;
  }

  CorpusBackend backend;
  backend.dl_ = dl;
  backend.ctx_ = ctx;
  backend.kernels_ = kernels;
  backend.destroy_ = destroy;
  backend.buffer_alloc_ = buffer_alloc;
  backend.buffer_free_ = buffer_free;
  backend.encrypt_i64_ = encrypt_i64;
  backend.decrypt_i64_ = decrypt_i64;
  backend.prefetch_ = prefetch;
  backend.evict_ = evict;
  return backend;
}

} // namespace corpus
} // namespace fhenomenon
