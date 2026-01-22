#pragma once

#include "Backend/Backend.h"

#include <any>
#include <memory>
#include <string>

namespace fhenomenon {

class ExternalBackend : public Backend {
  public:
  BackendType getBackendType() const override { return BackendType::ExternalBackend; }

  ExternalBackend([[maybe_unused]] const std::string &libraryPath) {
    // Load the external library and initialize the backend
  }

  void transform([[maybe_unused]] CompuonBase &entity, [[maybe_unused]] const Parameter &params) const override {
    // Transform the entity using the external backend
  }

  std::shared_ptr<CompuonBase> add([[maybe_unused]] const CompuonBase &a,
                                   [[maybe_unused]] const CompuonBase &b) const override {
    // Add entities using the external backend
    // return Compuon<int>(0);  // Placeholder
    return nullptr;
  }

  std::shared_ptr<CompuonBase> multiply([[maybe_unused]] const CompuonBase &a,
                                        [[maybe_unused]] const CompuonBase &b) const override {
    // Multiply entities using the external backend
    // return Compuon<int>(0);  // Placeholder
    return nullptr;
  }

  std::any decrypt([[maybe_unused]] const CompuonBase &entity) const override {
    // Decrypt entity using the external backend
    return 0; // Placeholder
  }

  std::shared_ptr<CompuonBase> bitAnd(const CompuonBase &, const CompuonBase &) const override { return nullptr; }
  std::shared_ptr<CompuonBase> bitOr(const CompuonBase &, const CompuonBase &) const override { return nullptr; }
  std::shared_ptr<CompuonBase> bitXor(const CompuonBase &, const CompuonBase &) const override { return nullptr; }
  std::shared_ptr<CompuonBase> compareEq(const CompuonBase &, const CompuonBase &) const override { return nullptr; }
  std::shared_ptr<CompuonBase> compareLt(const CompuonBase &, const CompuonBase &) const override { return nullptr; }
  std::shared_ptr<CompuonBase> compareLe(const CompuonBase &, const CompuonBase &) const override { return nullptr; }
};

} // namespace fhenomenon
