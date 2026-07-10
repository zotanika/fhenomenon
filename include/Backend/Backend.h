#pragma once

#include "Configuration.h"
#include "KeyManager.h"
#include "Parameter/Parameter.h"

#include <any>

namespace fhenomenon {

class FhenonBase;

template <typename T> class Fhenon;

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
