#pragma once

#include "Configuration.h"
#include "KeyManager.h"
#include "Parameter/Parameter.h"

#include <any>

namespace fhenomenon {

class CompuonBase;

template <typename T> class Compuon;

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
  virtual void transform(CompuonBase &entity, const Parameter &params) const = 0;
  virtual std::shared_ptr<CompuonBase> add(const CompuonBase &a, const CompuonBase &b) const = 0;
  virtual std::shared_ptr<CompuonBase> multiply(const CompuonBase &a, const CompuonBase &b) const = 0;
  virtual std::any decrypt(const CompuonBase &entity) const = 0;

  /*
  template<typename T>
  void transform(Compuon<T>& entity, const Parameter& params);
  template<typename T>
  Compuon<T> add(const Compuon<T>& a, const Compuon<T>& b);
  template<typename T>
  Compuon<T> multiply(const Compuon<T>& a, const Compuon<T>& b);
  template<typename T>
  T decrypt(const Compuon<T>& entity) const;
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
