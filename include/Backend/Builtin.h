#pragma once

#include "Backend/Backend.h"
#include "Fhenon.h"
#include "Crypto/ToyFHE.h"
#include "FHN/FhnDefaultExecutor.h"
#include "FHN/ToyFheKernels.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Forward declarations for TFHE FFI types (at global scope for C linkage)
#ifdef FHENOMENON_USE_TFHE
extern "C" {
struct TfheContext;
struct CiphertextHandle;
}
using TfheBinaryOp = CiphertextHandle *(*)(TfheContext *, CiphertextHandle *, CiphertextHandle *);
#endif

namespace fhenomenon {

extern template class Fhenon<int>;

class BuiltinBackend final : public Backend {
  private:
  mutable toyfhe::Engine engine_;
  toyfhe::Parameters params_;
  mutable std::once_flag initFlag_;
  mutable std::once_flag keyGenFlag_;
#ifdef FHENOMENON_USE_TFHE
  TfheContext *context_ = nullptr;

  // Helper for binary TFHE operations to reduce code duplication
  std::shared_ptr<FhenonBase> executeBinaryTfheOp(const FhenonBase &a, const FhenonBase &b, TfheBinaryOp op,
                                                   const char *opName) const;
#endif

  void ensureReady() const;

  // FHN executor infrastructure
  FhnBackendCtx *fhn_ctx_ = nullptr;
  FhnKernelTable *fhn_table_ = nullptr;
  std::unique_ptr<FhnDefaultExecutor> fhn_executor_;

  public:
  BuiltinBackend();
  ~BuiltinBackend();

  BackendType getBackendType() const override { return BackendType::BuiltinBackend; }

  FhnDefaultExecutor *getFhnExecutor() const { return fhn_executor_.get(); }
  FhnBackendCtx *getFhnCtx() const { return fhn_ctx_; }

  // Initialize backend (ToyFHE or TFHE)
  void initialize();

  // Key management
  void generateKeys();
  void loadKeys(const std::string &publicKeyPath, const std::string &secretKeyPath);
  void saveKeys(const std::string &publicKeyPath, const std::string &secretKeyPath);

  // Backend interface implementation
  void transform(FhenonBase &entity, const Parameter &params) const override;
  std::shared_ptr<FhenonBase> add(const FhenonBase &a, const FhenonBase &b) const override;
  std::shared_ptr<FhenonBase> multiply(const FhenonBase &a, const FhenonBase &b) const override;
  std::shared_ptr<FhenonBase> addPlain(const FhenonBase &a, double scalar);
  std::shared_ptr<FhenonBase> multiplyPlain(const FhenonBase &a, double scalar);
  std::any decrypt(const FhenonBase &entity) const override;

  std::shared_ptr<FhenonBase> bitAnd(const FhenonBase &a, const FhenonBase &b) const override;
  std::shared_ptr<FhenonBase> bitOr(const FhenonBase &a, const FhenonBase &b) const override;
  std::shared_ptr<FhenonBase> bitXor(const FhenonBase &a, const FhenonBase &b) const override;
  std::shared_ptr<FhenonBase> compareEq(const FhenonBase &a, const FhenonBase &b) const override;
  std::shared_ptr<FhenonBase> compareLt(const FhenonBase &a, const FhenonBase &b) const override;
  std::shared_ptr<FhenonBase> compareLe(const FhenonBase &a, const FhenonBase &b) const override;
};

} // namespace fhenomenon
