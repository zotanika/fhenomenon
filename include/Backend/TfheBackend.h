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
  void transform(FhenonBase &entity, const Parameter &params) const override;
  std::shared_ptr<FhenonBase> add(const FhenonBase &a, const FhenonBase &b) const override;
  std::shared_ptr<FhenonBase> multiply(const FhenonBase &a, const FhenonBase &b) const override;
  std::any decrypt(const FhenonBase &entity) const override;

  std::shared_ptr<FhenonBase> bitAnd(const FhenonBase &a, const FhenonBase &b) const override;
  std::shared_ptr<FhenonBase> bitOr(const FhenonBase &a, const FhenonBase &b) const override;
  std::shared_ptr<FhenonBase> bitXor(const FhenonBase &a, const FhenonBase &b) const override;
  std::shared_ptr<FhenonBase> compareEq(const FhenonBase &a, const FhenonBase &b) const override;
  std::shared_ptr<FhenonBase> compareLt(const FhenonBase &a, const FhenonBase &b) const override;
  std::shared_ptr<FhenonBase> compareLe(const FhenonBase &a, const FhenonBase &b) const override;
};

} // namespace fhenomenon
