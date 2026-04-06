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
  explicit ExternalBackend(const std::string &libraryPath,
                           const char *config_json = nullptr,
                           const std::string &symbolPrefix = "");
  ~ExternalBackend() override;

  // Prevent copy
  ExternalBackend(const ExternalBackend &) = delete;
  ExternalBackend &operator=(const ExternalBackend &) = delete;

  // FHN accessors
  FhnDefaultExecutor *getFhnExecutor() const { return executor_.get(); }
  FhnBackendCtx *getFhnCtx() const { return fhn_ctx_; }
  const FhnBackendInfo *getInfo() const { return info_; }
  const FhnBackendVTable &getVTable() const { return vtable_; }

  // Backend interface (delegating to FHN kernel table)
  void transform(CompuonBase &entity, const Parameter &params) const override;
  std::shared_ptr<CompuonBase> add(const CompuonBase &a, const CompuonBase &b) const override;
  std::shared_ptr<CompuonBase> multiply(const CompuonBase &a, const CompuonBase &b) const override;
  std::any decrypt(const CompuonBase &entity) const override;

  std::shared_ptr<CompuonBase> bitAnd(const CompuonBase &, const CompuonBase &) const override { return nullptr; }
  std::shared_ptr<CompuonBase> bitOr(const CompuonBase &, const CompuonBase &) const override { return nullptr; }
  std::shared_ptr<CompuonBase> bitXor(const CompuonBase &, const CompuonBase &) const override { return nullptr; }
  std::shared_ptr<CompuonBase> compareEq(const CompuonBase &, const CompuonBase &) const override { return nullptr; }
  std::shared_ptr<CompuonBase> compareLt(const CompuonBase &, const CompuonBase &) const override { return nullptr; }
  std::shared_ptr<CompuonBase> compareLe(const CompuonBase &, const CompuonBase &) const override { return nullptr; }

private:
  void *dl_handle_ = nullptr;
  FhnBackendVTable vtable_{};
  FhnBackendInfo *info_ = nullptr;
  FhnBackendCtx *fhn_ctx_ = nullptr;
  FhnKernelTable *fhn_table_ = nullptr;
  std::unique_ptr<FhnDefaultExecutor> executor_;
};

} // namespace fhenomenon
