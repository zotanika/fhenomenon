#pragma once

#include "Configuration.h"
#include "FHN/fhn_backend_api.h"
#include "KeyManager.h"
#include "Parameter/Parameter.h"

#include <any>
#include <memory>

namespace fhenomenon {

class Backend;
class FhenonBase;
class FhnDefaultExecutor;

template <typename T> class Fhenon;

// Ciphertext handle stored in FhenonBase::ciphertext_ by FHN-backed
// backends. The owner tag restores the type firewall that distinct std::any
// payload types used to provide: the bytes behind an FhnBuffer are
// backend-specific, so a buffer created by one backend must never reach
// another backend's kernels.
struct FhnCiphertext {
  std::shared_ptr<FhnBuffer> buffer;
  const Backend *owner = nullptr;
};

// Everything the session needs to run an FhnProgram against a backend:
// a context, an executor over its kernel table, and the buffer half of the
// host-side data plane. Backends that expose this run the FHN path; the
// legacy per-operation path remains for those that don't.
struct FhnRuntime {
  FhnBackendCtx *ctx = nullptr;
  FhnDefaultExecutor *executor = nullptr;
  FhnBufferAllocFn buffer_alloc = nullptr;
  FhnBufferFreeFn buffer_free = nullptr;
  // Optional movement hooks; null = single memory space, movement skipped.
  FhnBufferPrefetchFn prefetch = nullptr;
  FhnBufferEvictFn evict = nullptr;
  // Keeps the backend context (and, for dlopened backends, the library
  // itself) alive for as long as any buffer allocated through this runtime
  // exists. Buffer deleters must capture it, or a Fhenon outliving its
  // backend would free through a destroyed context / unloaded library.
  std::shared_ptr<void> keepalive;
};

enum class BackendType {
  BuiltinBackend,
  // Add backends
  ExternalBackend
};

class Backend {
  private:
  // Backend(const Backend&) = delete;
  // void operator=(const Backend&) = delete;

  // static std::unique_ptr<Backend> instance;
  // static std::unique_ptr<KeyManager> keyManager;
  // template<typename T>
  // static scheduler::Scheduler<T> scheduler;
  // static std::unique_ptr<BackendInterface> scheme_;

  static std::unique_ptr<Backend> createBuiltinBackend();
  static std::unique_ptr<Backend> createExternalBackend(std::string_view libPath);

  public:
  virtual ~Backend() = default;

  virtual BackendType getBackendType() const = 0;

  // Non-null when the backend can execute FhnPrograms (context + executor +
  // buffer data plane). The pointer remains owned by the backend.
  virtual const FhnRuntime *fhnRuntime() const { return nullptr; }

  virtual void transform(FhenonBase &entity, const Parameter &params) const = 0;
  virtual std::shared_ptr<FhenonBase> add(const FhenonBase &a, const FhenonBase &b) const = 0;
  virtual std::shared_ptr<FhenonBase> multiply(const FhenonBase &a, const FhenonBase &b) const = 0;
  virtual std::any decrypt(const FhenonBase &entity) const = 0;

  virtual std::shared_ptr<FhenonBase> bitAnd(const FhenonBase &a, const FhenonBase &b) const = 0;
  virtual std::shared_ptr<FhenonBase> bitOr(const FhenonBase &a, const FhenonBase &b) const = 0;
  virtual std::shared_ptr<FhenonBase> bitXor(const FhenonBase &a, const FhenonBase &b) const = 0;
  virtual std::shared_ptr<FhenonBase> compareEq(const FhenonBase &a, const FhenonBase &b) const = 0;
  virtual std::shared_ptr<FhenonBase> compareLt(const FhenonBase &a, const FhenonBase &b) const = 0;
  virtual std::shared_ptr<FhenonBase> compareLe(const FhenonBase &a, const FhenonBase &b) const = 0;

  /*
  template<typename T>
  void transform(Fhenon<T>& entity, const Parameter& params);
  template<typename T>
  Fhenon<T> add(const Fhenon<T>& a, const Fhenon<T>& b);
  template<typename T>
  Fhenon<T> multiply(const Fhenon<T>& a, const Fhenon<T>& b);
  template<typename T>
  T decrypt(const Fhenon<T>& entity) const;
  */

  // static Backend& getBackend() {
  //     static Backend backend;
  //     return backend;
  // }
  static Backend &getInstance(std::string_view libPath = "", std::string_view configPath = "");

  static KeyManager &getKeyManager() {
    static KeyManager keyManager;
    return keyManager;
  }
};

} // namespace fhenomenon
