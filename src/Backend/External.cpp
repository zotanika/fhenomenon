#include "Backend/External.h"
#include "Fhenon.h"

#include <dlfcn.h>
#include <iostream>
#include <stdexcept>

namespace fhenomenon {

ExternalBackend::ExternalBackend(const std::string &libraryPath, const char *config_json,
                                 const std::string &symbolPrefix) {
  // 1. dlopen the shared library
  dl_handle_ = dlopen(libraryPath.c_str(), RTLD_LAZY);
  if (!dl_handle_) {
    throw std::runtime_error(std::string("ExternalBackend: dlopen failed: ") + dlerror());
  }

  // Build symbol names with optional prefix
  auto sym = [&](const char *base) { return symbolPrefix + base; };

  // 2. ABI handshake before anything else: a library built against a
  //    different header revision may have renumbered opcodes or changed
  //    struct layouts, and would otherwise load and dispatch silently.
  vtable_.get_abi_version = reinterpret_cast<FhnGetAbiVersionFn>(dlsym(dl_handle_, sym("fhn_get_abi_version").c_str()));
  if (!vtable_.get_abi_version) {
    dlclose(dl_handle_);
    dl_handle_ = nullptr;
    throw std::runtime_error("ExternalBackend: library missing fhn_get_abi_version (pre-versioning binary?)");
  }
  const uint32_t abi = vtable_.get_abi_version();
  if (abi != FHN_ABI_VERSION) {
    dlclose(dl_handle_);
    dl_handle_ = nullptr;
    throw std::runtime_error("ExternalBackend: ABI version mismatch (library " + std::to_string(abi) + ", host " +
                             std::to_string(FHN_ABI_VERSION) + ")");
  }

  // 3. Resolve the 4 required symbols
  vtable_.get_info = reinterpret_cast<FhnGetInfoFn>(dlsym(dl_handle_, sym("fhn_get_info").c_str()));
  vtable_.create = reinterpret_cast<FhnCreateFn>(dlsym(dl_handle_, sym("fhn_create").c_str()));
  vtable_.destroy = reinterpret_cast<FhnDestroyFn>(dlsym(dl_handle_, sym("fhn_destroy").c_str()));
  vtable_.get_kernels = reinterpret_cast<FhnGetKernelsFn>(dlsym(dl_handle_, sym("fhn_get_kernels").c_str()));

  if (!vtable_.get_info || !vtable_.create || !vtable_.destroy || !vtable_.get_kernels) {
    dlclose(dl_handle_);
    dl_handle_ = nullptr;
    throw std::runtime_error("ExternalBackend: library missing required fhn_* symbols");
  }

  // 4. Resolve the host-side data plane. Buffers are required (a generic
  //    host cannot feed kernels without them); encrypt/decrypt are optional
  //    (present only when the backend holds key material).
  vtable_.buffer_alloc = reinterpret_cast<FhnBufferAllocFn>(dlsym(dl_handle_, sym("fhn_buffer_alloc").c_str()));
  vtable_.buffer_free = reinterpret_cast<FhnBufferFreeFn>(dlsym(dl_handle_, sym("fhn_buffer_free").c_str()));
  if (!vtable_.buffer_alloc || !vtable_.buffer_free) {
    dlclose(dl_handle_);
    dl_handle_ = nullptr;
    throw std::runtime_error("ExternalBackend: library missing required fhn_buffer_alloc/fhn_buffer_free");
  }
  vtable_.encrypt_i64 = reinterpret_cast<FhnEncryptInt64Fn>(dlsym(dl_handle_, sym("fhn_encrypt_i64").c_str()));
  vtable_.encrypt_f64 = reinterpret_cast<FhnEncryptDoubleFn>(dlsym(dl_handle_, sym("fhn_encrypt_f64").c_str()));
  vtable_.decrypt_i64 = reinterpret_cast<FhnDecryptInt64Fn>(dlsym(dl_handle_, sym("fhn_decrypt_i64").c_str()));
  vtable_.decrypt_f64 = reinterpret_cast<FhnDecryptDoubleFn>(dlsym(dl_handle_, sym("fhn_decrypt_f64").c_str()));

  // 5. Resolve optional advanced symbols (NULL if absent)
  vtable_.submit = reinterpret_cast<FhnSubmitFn>(dlsym(dl_handle_, sym("fhn_submit").c_str()));
  vtable_.poll = reinterpret_cast<FhnPollFn>(dlsym(dl_handle_, sym("fhn_poll").c_str()));
  vtable_.wait = reinterpret_cast<FhnWaitFn>(dlsym(dl_handle_, sym("fhn_wait").c_str()));
  vtable_.get_outputs = reinterpret_cast<FhnGetOutputsFn>(dlsym(dl_handle_, sym("fhn_get_outputs").c_str()));
  vtable_.exec_free = reinterpret_cast<FhnExecFreeFn>(dlsym(dl_handle_, sym("fhn_exec_free").c_str()));

  // 6. Get backend info
  info_ = vtable_.get_info();

  // 7. Create backend context
  fhn_ctx_ = vtable_.create(config_json);
  if (!fhn_ctx_) {
    dlclose(dl_handle_);
    dl_handle_ = nullptr;
    throw std::runtime_error("ExternalBackend: fhn_create returned null");
  }

  // 8. Get kernel table and create executor
  fhn_table_ = vtable_.get_kernels(fhn_ctx_);
  if (!fhn_table_) {
    vtable_.destroy(fhn_ctx_);
    dlclose(dl_handle_);
    dl_handle_ = nullptr;
    throw std::runtime_error("ExternalBackend: fhn_get_kernels returned null");
  }

  executor_ = std::make_unique<FhnDefaultExecutor>(fhn_table_);

  std::cout << "ExternalBackend loaded: " << info_->name << " v" << info_->version
            << " (device_type=" << static_cast<int>(info_->device_type) << ")" << std::endl;
}

ExternalBackend::~ExternalBackend() {
  executor_.reset();
  if (fhn_ctx_ && vtable_.destroy) {
    vtable_.destroy(fhn_ctx_);
    fhn_ctx_ = nullptr;
  }
  if (dl_handle_) {
    dlclose(dl_handle_);
    dl_handle_ = nullptr;
  }
}

std::shared_ptr<FhnBuffer> ExternalBackend::makeBuffer() const {
  FhnBuffer *raw = vtable_.buffer_alloc(fhn_ctx_);
  if (!raw) {
    throw std::runtime_error("ExternalBackend: fhn_buffer_alloc failed");
  }
  // The deleter captures the vtable and context by value; the Backend
  // singleton outlives every Fhenon that holds one of these buffers.
  auto free_fn = vtable_.buffer_free;
  auto *ctx = fhn_ctx_;
  return std::shared_ptr<FhnBuffer>(raw, [free_fn, ctx](FhnBuffer *buf) { free_fn(ctx, buf); });
}

void ExternalBackend::transform(FhenonBase &entity, const Parameter & /*params*/) const {
  // Host-side data plane: encryption never goes through the kernel table.
  if (entity.type() == typeid(int)) {
    if (!vtable_.encrypt_i64) {
      throw std::runtime_error("ExternalBackend: backend does not export fhn_encrypt_i64 (no key material)");
    }
    auto &derived = dynamic_cast<Fhenon<int> &>(entity);
    auto buffer = makeBuffer();
    if (vtable_.encrypt_i64(fhn_ctx_, buffer.get(), derived.getValue()) != 0) {
      throw std::runtime_error("ExternalBackend: fhn_encrypt_i64 failed");
    }
    entity.ciphertext_ = buffer;
    entity.isEncrypted_ = true;
    return;
  }
  if (entity.type() == typeid(double)) {
    if (!vtable_.encrypt_f64) {
      throw std::runtime_error("ExternalBackend: backend does not export fhn_encrypt_f64 (no key material)");
    }
    auto &derived = dynamic_cast<Fhenon<double> &>(entity);
    auto buffer = makeBuffer();
    if (vtable_.encrypt_f64(fhn_ctx_, buffer.get(), derived.getValue()) != 0) {
      throw std::runtime_error("ExternalBackend: fhn_encrypt_f64 failed");
    }
    entity.ciphertext_ = buffer;
    entity.isEncrypted_ = true;
    return;
  }
  throw std::runtime_error("ExternalBackend: unsupported type for encryption");
}

std::shared_ptr<FhenonBase> ExternalBackend::add(const FhenonBase & /*a*/, const FhenonBase & /*b*/) const {
  // Direct Backend::add() is legacy interface.
  // External backends use FHN executor pipeline.
  return nullptr;
}

std::shared_ptr<FhenonBase> ExternalBackend::multiply(const FhenonBase & /*a*/, const FhenonBase & /*b*/) const {
  return nullptr;
}

std::any ExternalBackend::decrypt(const FhenonBase &entity) const {
  if (!entity.isEncrypted_ || !entity.ciphertext_.has_value()) {
    // Unencrypted: fall back to the plaintext value.
    if (entity.type() == typeid(int)) {
      return dynamic_cast<const Fhenon<int> &>(entity).getValue();
    }
    if (entity.type() == typeid(double)) {
      return dynamic_cast<const Fhenon<double> &>(entity).getValue();
    }
    return {};
  }

  auto buffer = std::any_cast<std::shared_ptr<FhnBuffer>>(entity.ciphertext_);
  if (entity.type() == typeid(int)) {
    if (!vtable_.decrypt_i64) {
      throw std::runtime_error("ExternalBackend: backend does not export fhn_decrypt_i64 (no key material)");
    }
    int64_t value = 0;
    if (vtable_.decrypt_i64(fhn_ctx_, buffer.get(), &value) != 0) {
      throw std::runtime_error("ExternalBackend: fhn_decrypt_i64 failed");
    }
    return static_cast<int>(value);
  }
  if (entity.type() == typeid(double)) {
    if (!vtable_.decrypt_f64) {
      throw std::runtime_error("ExternalBackend: backend does not export fhn_decrypt_f64 (no key material)");
    }
    double value = 0.0;
    if (vtable_.decrypt_f64(fhn_ctx_, buffer.get(), &value) != 0) {
      throw std::runtime_error("ExternalBackend: fhn_decrypt_f64 failed");
    }
    return value;
  }
  throw std::runtime_error("ExternalBackend: unsupported type for decryption");
}

} // namespace fhenomenon
