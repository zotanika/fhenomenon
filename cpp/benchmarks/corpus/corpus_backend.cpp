#include "corpus_backend.h"

#include <dlfcn.h>

namespace fhenomenon {
namespace corpus {

CorpusBackend::CorpusBackend(CorpusBackend &&other) noexcept
  : dl_(other.dl_), ctx_(other.ctx_), client_(other.client_), client_destroy_(other.client_destroy_),
    client_prepare_(other.client_prepare_), client_encrypt_(other.client_encrypt_),
    client_decrypt_(other.client_decrypt_), compile_circuit_(other.compile_circuit_),
    prepare_executable_(other.prepare_executable_), run_executable_(other.run_executable_),
    executable_stats_(other.executable_stats_), executable_free_(other.executable_free_), kernels_(other.kernels_),
    destroy_(other.destroy_), buffer_alloc_(other.buffer_alloc_), buffer_free_(other.buffer_free_),
    encrypt_i64_(other.encrypt_i64_), decrypt_i64_(other.decrypt_i64_), encrypt_vector_(other.encrypt_vector_),
    decrypt_vector_(other.decrypt_vector_), encrypt_f64_(other.encrypt_f64_), decrypt_f64_(other.decrypt_f64_),
    prefetch_(other.prefetch_), evict_(other.evict_), fresh_level_(other.fresh_level_),
    level_bytes_(other.level_bytes_), opcode_level_effect_(other.opcode_level_effect_) {
  other.dl_ = nullptr;
  other.ctx_ = nullptr;
  other.client_ = nullptr;
  other.client_destroy_ = nullptr;
  other.client_prepare_ = nullptr;
  other.client_encrypt_ = nullptr;
  other.client_decrypt_ = nullptr;
  other.compile_circuit_ = nullptr;
  other.prepare_executable_ = nullptr;
  other.run_executable_ = nullptr;
  other.executable_stats_ = nullptr;
  other.executable_free_ = nullptr;
  other.kernels_ = nullptr;
  other.destroy_ = nullptr;
  other.buffer_alloc_ = nullptr;
  other.buffer_free_ = nullptr;
  other.encrypt_i64_ = nullptr;
  other.decrypt_i64_ = nullptr;
  other.encrypt_vector_ = nullptr;
  other.decrypt_vector_ = nullptr;
  other.encrypt_f64_ = nullptr;
  other.decrypt_f64_ = nullptr;
  other.prefetch_ = nullptr;
  other.evict_ = nullptr;
  other.fresh_level_ = nullptr;
  other.level_bytes_ = nullptr;
  other.opcode_level_effect_ = nullptr;
}

CorpusBackend &CorpusBackend::operator=(CorpusBackend &&other) noexcept {
  if (this == &other) {
    return *this;
  }

  // Tear down whatever this instance currently owns before adopting other's
  // state, mirroring the destructor's ctx-then-library ordering.
  if (client_ && client_destroy_) {
    client_destroy_(client_);
  }
  if (ctx_ && destroy_) {
    destroy_(ctx_);
  }
  if (dl_) {
    dlclose(dl_);
  }

  dl_ = other.dl_;
  ctx_ = other.ctx_;
  client_ = other.client_;
  client_destroy_ = other.client_destroy_;
  client_prepare_ = other.client_prepare_;
  client_encrypt_ = other.client_encrypt_;
  client_decrypt_ = other.client_decrypt_;
  compile_circuit_ = other.compile_circuit_;
  prepare_executable_ = other.prepare_executable_;
  run_executable_ = other.run_executable_;
  executable_stats_ = other.executable_stats_;
  executable_free_ = other.executable_free_;
  kernels_ = other.kernels_;
  destroy_ = other.destroy_;
  buffer_alloc_ = other.buffer_alloc_;
  buffer_free_ = other.buffer_free_;
  encrypt_i64_ = other.encrypt_i64_;
  decrypt_i64_ = other.decrypt_i64_;
  encrypt_vector_ = other.encrypt_vector_;
  decrypt_vector_ = other.decrypt_vector_;
  encrypt_f64_ = other.encrypt_f64_;
  decrypt_f64_ = other.decrypt_f64_;
  prefetch_ = other.prefetch_;
  evict_ = other.evict_;
  fresh_level_ = other.fresh_level_;
  level_bytes_ = other.level_bytes_;
  opcode_level_effect_ = other.opcode_level_effect_;

  other.dl_ = nullptr;
  other.ctx_ = nullptr;
  other.client_ = nullptr;
  other.client_destroy_ = nullptr;
  other.client_prepare_ = nullptr;
  other.client_encrypt_ = nullptr;
  other.client_decrypt_ = nullptr;
  other.compile_circuit_ = nullptr;
  other.prepare_executable_ = nullptr;
  other.run_executable_ = nullptr;
  other.executable_stats_ = nullptr;
  other.executable_free_ = nullptr;
  other.kernels_ = nullptr;
  other.destroy_ = nullptr;
  other.buffer_alloc_ = nullptr;
  other.buffer_free_ = nullptr;
  other.encrypt_i64_ = nullptr;
  other.decrypt_i64_ = nullptr;
  other.encrypt_vector_ = nullptr;
  other.decrypt_vector_ = nullptr;
  other.encrypt_f64_ = nullptr;
  other.decrypt_f64_ = nullptr;
  other.prefetch_ = nullptr;
  other.evict_ = nullptr;
  other.fresh_level_ = nullptr;
  other.level_bytes_ = nullptr;
  other.opcode_level_effect_ = nullptr;

  return *this;
}

CorpusBackend::~CorpusBackend() {
  // Moved-from instances have dl_ == nullptr and ctx_ == nullptr, so both
  // guards below are no-ops for them.
  if (client_ && client_destroy_) {
    client_destroy_(client_);
  }
  if (ctx_ && destroy_) {
    destroy_(ctx_);
  }
  if (dl_) {
    dlclose(dl_);
  }
}

std::optional<CorpusBackend> CorpusBackend::load(const std::string &library_path, const std::string &symbol_prefix,
                                                 std::string *error, const char *config_json) {
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
  //    plane. Encryption/decryption pairs are optional.
  auto create = reinterpret_cast<FhnCreateFn>(dlsym(dl, sym("fhn_create").c_str()));
  auto destroy = reinterpret_cast<FhnDestroyFn>(dlsym(dl, sym("fhn_destroy").c_str()));
  auto get_kernels = reinterpret_cast<FhnGetKernelsFn>(dlsym(dl, sym("fhn_get_kernels").c_str()));
  auto buffer_alloc = reinterpret_cast<FhnBufferAllocFn>(dlsym(dl, sym("fhn_buffer_alloc").c_str()));
  auto buffer_free = reinterpret_cast<FhnBufferFreeFn>(dlsym(dl, sym("fhn_buffer_free").c_str()));
  auto encrypt_i64 = reinterpret_cast<FhnEncryptInt64Fn>(dlsym(dl, sym("fhn_encrypt_i64").c_str()));
  auto decrypt_i64 = reinterpret_cast<FhnDecryptInt64Fn>(dlsym(dl, sym("fhn_decrypt_i64").c_str()));
  auto encrypt_vector = reinterpret_cast<FhnEncryptVectorDoubleFn>(dlsym(dl, sym("fhn_encrypt_vec_f64").c_str()));
  auto decrypt_vector = reinterpret_cast<FhnDecryptVectorDoubleFn>(dlsym(dl, sym("fhn_decrypt_vec_f64").c_str()));
  auto encrypt_f64 = reinterpret_cast<FhnEncryptDoubleFn>(dlsym(dl, sym("fhn_encrypt_f64").c_str()));
  auto decrypt_f64 = reinterpret_cast<FhnDecryptDoubleFn>(dlsym(dl, sym("fhn_decrypt_f64").c_str()));
  if (!encrypt_i64 || !decrypt_i64) {
    encrypt_i64 = nullptr;
    decrypt_i64 = nullptr;
  }
  if (!encrypt_vector || !decrypt_vector) {
    encrypt_vector = nullptr;
    decrypt_vector = nullptr;
  }
  if (!encrypt_f64 || !decrypt_f64) {
    encrypt_f64 = nullptr;
    decrypt_f64 = nullptr;
  }
  if (!create || !destroy || !get_kernels || !buffer_alloc || !buffer_free) {
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

  // 4. Optional level model trio: fhn_fresh_level/fhn_level_bytes/
  //    fhn_opcode_level_effect. All three or none (see ExternalBackend) — a
  //    partial subset can't size a fresh buffer or classify an opcode's
  //    effect on the buffers it already has.
  auto fresh_level = reinterpret_cast<FhnFreshLevelFn>(dlsym(dl, sym("fhn_fresh_level").c_str()));
  auto level_bytes = reinterpret_cast<FhnLevelBytesFn>(dlsym(dl, sym("fhn_level_bytes").c_str()));
  auto opcode_level_effect =
    reinterpret_cast<FhnOpcodeLevelEffectFn>(dlsym(dl, sym("fhn_opcode_level_effect").c_str()));
  const int level_model_count = (fresh_level != nullptr) + (level_bytes != nullptr) + (opcode_level_effect != nullptr);
  if (level_model_count != 0 && level_model_count != 3) {
    fresh_level = nullptr;
    level_bytes = nullptr;
    opcode_level_effect = nullptr;
  }

  // Optional isolated trusted-client group. A partial implementation is an
  // error: do not accidentally fall back to keys in a compute context.
  auto client_create = reinterpret_cast<FhnClientCreateFn>(dlsym(dl, sym("fhn_client_create").c_str()));
  auto client_destroy = reinterpret_cast<FhnClientDestroyFn>(dlsym(dl, sym("fhn_client_destroy").c_str()));
  auto client_prepare = reinterpret_cast<FhnClientPrepareFn>(dlsym(dl, sym("fhn_client_prepare").c_str()));
  auto client_encrypt =
    reinterpret_cast<FhnClientEncryptVectorDoubleFn>(dlsym(dl, sym("fhn_client_encrypt_vec_f64").c_str()));
  auto client_decrypt =
    reinterpret_cast<FhnClientDecryptVectorDoubleFn>(dlsym(dl, sym("fhn_client_decrypt_vec_f64").c_str()));
  const int client_exports = (client_create != nullptr) + (client_destroy != nullptr) + (client_prepare != nullptr) +
                             (client_encrypt != nullptr) + (client_decrypt != nullptr);
  if (client_exports != 0 && client_exports != 5) {
    if (error)
      *error = "CorpusBackend: incomplete trusted-client export group";
    dlclose(dl);
    return std::nullopt;
  }

  // Semantic circuits are optional, but partial groups must not masquerade
  // as working compilation. They use the isolated client key-preparation path.
  auto compile_circuit = reinterpret_cast<FhnCompileCircuitFn>(dlsym(dl, sym("fhn_compile_circuit").c_str()));
  auto prepare_executable =
    reinterpret_cast<FhnClientPrepareExecutableFn>(dlsym(dl, sym("fhn_client_prepare_executable").c_str()));
  auto run_executable = reinterpret_cast<FhnRunExecutableFn>(dlsym(dl, sym("fhn_run_executable").c_str()));
  auto executable_stats = reinterpret_cast<FhnExecutableStatsFn>(dlsym(dl, sym("fhn_executable_stats").c_str()));
  auto executable_free = reinterpret_cast<FhnExecutableFreeFn>(dlsym(dl, sym("fhn_executable_free").c_str()));
  const int circuit_exports = (compile_circuit != nullptr) + (prepare_executable != nullptr) +
                              (run_executable != nullptr) + (executable_stats != nullptr) +
                              (executable_free != nullptr);
  if (circuit_exports != 0 && (circuit_exports != 5 || !client_create)) {
    if (error)
      *error = "CorpusBackend: incomplete circuit/trusted-client export group";
    dlclose(dl);
    return std::nullopt;
  }

  // 5. Create the backend context and resolve its kernel table.
  FhnBackendCtx *ctx = create(config_json);
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
  backend.compile_circuit_ = compile_circuit;
  backend.prepare_executable_ = prepare_executable;
  backend.run_executable_ = run_executable;
  backend.executable_stats_ = executable_stats;
  backend.executable_free_ = executable_free;
  backend.destroy_ = destroy;
  backend.buffer_alloc_ = buffer_alloc;
  backend.buffer_free_ = buffer_free;
  backend.encrypt_i64_ = encrypt_i64;
  backend.decrypt_i64_ = decrypt_i64;
  backend.encrypt_vector_ = encrypt_vector;
  backend.decrypt_vector_ = decrypt_vector;
  backend.encrypt_f64_ = encrypt_f64;
  backend.decrypt_f64_ = decrypt_f64;
  backend.prefetch_ = prefetch;
  backend.evict_ = evict;
  backend.fresh_level_ = fresh_level;
  backend.level_bytes_ = level_bytes;
  backend.opcode_level_effect_ = opcode_level_effect;
  if (client_create) {
    backend.client_ = client_create(ctx);
    if (!backend.client_) {
      if (error)
        *error = "CorpusBackend: fhn_client_create failed";
      return std::nullopt; // backend destructor tears down ctx before dlclose.
    }
    backend.client_destroy_ = client_destroy;
    backend.client_prepare_ = client_prepare;
    backend.client_encrypt_ = client_encrypt;
    backend.client_decrypt_ = client_decrypt;
  }
  return backend;
}

int CorpusBackend::prepare(const FhnProgram *program) const { return client_ ? client_prepare_(client_, program) : 0; }
int CorpusBackend::encryptPacked(FhnBuffer *out, const double *values, uint32_t count) const {
  if (client_)
    return client_encrypt_(client_, out, values, count);
  return encrypt_vector_ ? encrypt_vector_(ctx_, out, values, count) : -1;
}
int CorpusBackend::decryptPacked(const FhnBuffer *in, double *values, uint32_t count) const {
  if (client_)
    return client_decrypt_(client_, in, values, count);
  return decrypt_vector_ ? decrypt_vector_(ctx_, in, values, count) : -1;
}

} // namespace corpus
} // namespace fhenomenon

namespace fhenomenon::corpus {
FhnExecutable *CorpusBackend::compileCircuit(const FhnCircuit *circuit, const FhnBuffer *const *inputs, uint32_t count,
                                             const FhnCompileOptions *options, char *error,
                                             uint32_t error_capacity) const {
  if (!compile_circuit_)
    return nullptr;
  return compile_circuit_(ctx_, circuit, inputs, count, options, error, error_capacity);
}
int CorpusBackend::prepareExecutable(FhnExecutable *executable) const {
  return client_ && prepare_executable_ ? prepare_executable_(client_, executable) : -1;
}
} // namespace fhenomenon::corpus
