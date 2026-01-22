#pragma once

#include "Backend/Backend.h"
#include "Compuon.h"
#include "Crypto/ToyFHE.h"

#include <memory>
#include <string>
#include <vector>

namespace fhenomenon {

extern template class Compuon<int>;

class BuiltinBackend final : public Backend {
  private:
  mutable toyfhe::Engine engine_;
  toyfhe::Parameters params_;
  mutable bool initialized_;
  void *context_ = nullptr; // For TFHE backend

  void ensureReady() const;

  public:
  BuiltinBackend();
  ~BuiltinBackend();

  BackendType getBackendType() const override { return BackendType::BuiltinBackend; }

  // Initialize backend (ToyFHE or TFHE)
  void initialize();

  // Key management
  void generateKeys();
  void loadKeys(const std::string &publicKeyPath, const std::string &secretKeyPath);
  void saveKeys(const std::string &publicKeyPath, const std::string &secretKeyPath);

  // Backend interface implementation
  void transform(CompuonBase &entity, const Parameter &params) const override;
  std::shared_ptr<CompuonBase> add(const CompuonBase &a, const CompuonBase &b) const override;
  std::shared_ptr<CompuonBase> multiply(const CompuonBase &a, const CompuonBase &b) const override;
  std::shared_ptr<CompuonBase> addPlain(const CompuonBase &a, double scalar);
  std::shared_ptr<CompuonBase> multiplyPlain(const CompuonBase &a, double scalar);
  std::any decrypt(const CompuonBase &entity) const override;

  std::shared_ptr<CompuonBase> bitAnd(const CompuonBase &a, const CompuonBase &b) const override;
  std::shared_ptr<CompuonBase> bitOr(const CompuonBase &a, const CompuonBase &b) const override;
  std::shared_ptr<CompuonBase> bitXor(const CompuonBase &a, const CompuonBase &b) const override;
  std::shared_ptr<CompuonBase> compareEq(const CompuonBase &a, const CompuonBase &b) const override;
  std::shared_ptr<CompuonBase> compareLt(const CompuonBase &a, const CompuonBase &b) const override;
  std::shared_ptr<CompuonBase> compareLe(const CompuonBase &a, const CompuonBase &b) const override;
};

} // namespace fhenomenon
