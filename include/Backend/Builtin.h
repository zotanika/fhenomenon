#pragma once

#include "Backend/Backend.h"
#include "FHN/FhnDefaultExecutor.h"
#include "FHN/ToyFheKernels.h"
#include "Fhenon.h"

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
  mutable std::once_flag initFlag_;
#ifdef FHENOMENON_USE_TFHE
  TfheContext *context_ = nullptr;

  // Helper for binary TFHE operations to reduce code duplication
  std::shared_ptr<FhenonBase> executeBinaryTfheOp(const FhenonBase &a, const FhenonBase &b, TfheBinaryOp op,
                                                  const char *opName) const;
#endif

  void ensureReady() const;

  // FHN executor infrastructure. ctx_core_ owns the context: buffer
  // deleters share it, so the context survives until the last outstanding
  // buffer is freed even if the backend is destroyed first.
  std::shared_ptr<FhnBackendCtx> ctx_core_;
  FhnBackendCtx *fhn_ctx_ = nullptr;
  FhnKernelTable *fhn_table_ = nullptr;
  std::unique_ptr<FhnDefaultExecutor> fhn_executor_;

  // Extract the FHN buffer from an entity, verifying this backend owns it.
  std::shared_ptr<FhnBuffer> bufferOf(const FhenonBase &entity, const char *opName) const;
  FhnRuntime runtime_{};

  // Wrap a fresh data-plane buffer in a shared_ptr that frees through the
  // data plane.
  std::shared_ptr<FhnBuffer> makeBuffer() const;

  // Run one compute instruction through the FHN executor:
  // result = op(a[, b]), with fparams[0] = fparam.
  std::shared_ptr<FhnBuffer> runSingleOp(FhnOpCode op, FhnBuffer *a, FhnBuffer *b, double fparam) const;

  public:
  BuiltinBackend();
  ~BuiltinBackend();

  BackendType getBackendType() const override { return BackendType::BuiltinBackend; }

  const FhnRuntime *fhnRuntime() const override { return runtime_.executor ? &runtime_ : nullptr; }
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
