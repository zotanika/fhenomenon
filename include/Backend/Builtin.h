#pragma once

#include "Backend/Backend.h"
#include "Backend/SEALBackend.h"
#include "Compuon.h"

#include <memory>

namespace fhenomenon {

extern template class Compuon<int>;

class BuiltinBackend final : public Backend {
  private:
  std::unique_ptr<SEALBackend> sealBackend_;

  public:
  BuiltinBackend();
  ~BuiltinBackend() = default;

  BackendType getBackendType() const override { return BackendType::BuiltinBackend; }
  
  // Initialize with parameters
  void initialize(const Parameter &params);
  
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

  // Accessors
  SEALBackend &getSEALBackend() { return *sealBackend_; }
  const SEALBackend &getSEALBackend() const { return *sealBackend_; }
};

} // namespace fhenomenon
