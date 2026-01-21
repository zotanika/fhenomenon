#pragma once

#include "Backend/Backend.h"
#include <any>

namespace fhenomenon {

// Forward declaration of the opaque Rust context handle if needed,
// but for the header we just need the class definition.
// We'll store the context as a void* (or custom struct pointer) in the Private implementation or just void* here.

class TfheBackend final : public Backend {
  private:
  void *context_ = nullptr; // Pointer to Rust TfheContext
  bool initialized_ = false;

  void ensureReady() const;

  public:
  TfheBackend();
  ~TfheBackend();

  BackendType getBackendType() const override { return BackendType::ExternalBackend; } // Or a new type if we add one

  // Initialize TFHE context
  void initialize();

  // Backend interface implementation
  void transform(CompuonBase &entity, const Parameter &params) const override;
  std::shared_ptr<CompuonBase> add(const CompuonBase &a, const CompuonBase &b) const override;
  std::shared_ptr<CompuonBase> multiply(const CompuonBase &a, const CompuonBase &b) const override;
  std::any decrypt(const CompuonBase &entity) const override;

  std::shared_ptr<CompuonBase> bitAnd(const CompuonBase &a, const CompuonBase &b) const override;
  std::shared_ptr<CompuonBase> bitOr(const CompuonBase &a, const CompuonBase &b) const override;
  std::shared_ptr<CompuonBase> bitXor(const CompuonBase &a, const CompuonBase &b) const override;
  std::shared_ptr<CompuonBase> compareEq(const CompuonBase &a, const CompuonBase &b) const override;
  std::shared_ptr<CompuonBase> compareLt(const CompuonBase &a, const CompuonBase &b) const override;
  std::shared_ptr<CompuonBase> compareLe(const CompuonBase &a, const CompuonBase &b) const override;
};

} // namespace fhenomenon
