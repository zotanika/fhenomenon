#pragma once

#include "Backend/Backend.h"
#include "FHN/FhnDefaultExecutor.h"
#include "FHN/fhn_backend_api.h"

#include <any>
#include <memory>
#include <string>

namespace fhenomenon {

class ExternalBackend : public Backend {
  public:
  BackendType getBackendType() const override { return BackendType::ExternalBackend; }

  // symbolPrefix: override for symbol names (e.g., "toyfhe_" -> "toyfhe_fhn_get_info")
  // Default "" expects standard "fhn_get_info", "fhn_create", etc.
  explicit ExternalBackend(const std::string &libraryPath, const char *config_json = nullptr,
                           const std::string &symbolPrefix = "");
  ~ExternalBackend() override;

  // Prevent copy
  ExternalBackend(const ExternalBackend &) = delete;
  ExternalBackend &operator=(const ExternalBackend &) = delete;

  // FHN accessors
  const FhnRuntime *fhnRuntime() const override { return &runtime_; }
  FhnDefaultExecutor *getFhnExecutor() const { return executor_.get(); }
  FhnBackendCtx *getFhnCtx() const { return fhn_ctx_; }
  const FhnBackendInfo *getInfo() const { return info_; }
  const FhnBackendVTable &getVTable() const { return vtable_; }

  // Backend interface (delegating to FHN kernel table)
  void transform(FhenonBase &entity, const Parameter &params) const override;
  std::shared_ptr<FhenonBase> add(const FhenonBase &a, const FhenonBase &b) const override;
  std::shared_ptr<FhenonBase> multiply(const FhenonBase &a, const FhenonBase &b) const override;
  std::any decrypt(const FhenonBase &entity) const override;

  std::shared_ptr<FhenonBase> bitAnd(const FhenonBase &, const FhenonBase &) const override { return nullptr; }
  std::shared_ptr<FhenonBase> bitOr(const FhenonBase &, const FhenonBase &) const override { return nullptr; }
  std::shared_ptr<FhenonBase> bitXor(const FhenonBase &, const FhenonBase &) const override { return nullptr; }
  std::shared_ptr<FhenonBase> compareEq(const FhenonBase &, const FhenonBase &) const override { return nullptr; }
  std::shared_ptr<FhenonBase> compareLt(const FhenonBase &, const FhenonBase &) const override { return nullptr; }
  std::shared_ptr<FhenonBase> compareLe(const FhenonBase &, const FhenonBase &) const override { return nullptr; }

  private:
  // Owns the dlopened library and backend context. Buffer deleters share it,
  // so both stay alive until the backend AND every buffer allocated through
  // it are gone — a deleter must never call into an unloaded library.
  struct LibCore {
    void *dl_handle = nullptr;
    FhnDestroyFn destroy = nullptr;
    FhnBackendCtx *ctx = nullptr;
    ~LibCore();
  };

  std::shared_ptr<FhnBuffer> makeBuffer() const;
  std::shared_ptr<FhnBuffer> bufferOf(const FhenonBase &entity, const char *opName) const;

  std::shared_ptr<LibCore> core_;
  void *dl_handle_ = nullptr;
  FhnBackendVTable vtable_{};
  FhnBackendInfo *info_ = nullptr;
  FhnBackendCtx *fhn_ctx_ = nullptr;
  FhnKernelTable *fhn_table_ = nullptr;
  std::unique_ptr<FhnDefaultExecutor> executor_;
  FhnRuntime runtime_{};
};

} // namespace fhenomenon
