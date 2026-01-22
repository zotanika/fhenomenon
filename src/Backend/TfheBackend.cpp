#include "Backend/TfheBackend.h"
#include "Compuon.h"
#include "Utils/log.h"

#include <iostream>
#include <stdexcept>

// FFI declarations matching src/Backend/rust/src/lib.rs
extern "C" {
struct TfheContext;
struct CiphertextHandle;

TfheContext *tfhe_context_create();
void tfhe_context_destroy(TfheContext *ctx);
void tfhe_ciphertext_destroy(CiphertextHandle *handle);

CiphertextHandle *tfhe_encrypt_int32(TfheContext *ctx, int32_t value);
int32_t tfhe_decrypt_int32(TfheContext *ctx, CiphertextHandle *handle);
CiphertextHandle *tfhe_add_int32(TfheContext *ctx, CiphertextHandle *a, CiphertextHandle *b);
CiphertextHandle *tfhe_mul_int32(TfheContext *ctx, CiphertextHandle *a, CiphertextHandle *b);

CiphertextHandle *tfhe_bitand_int32(TfheContext *ctx, CiphertextHandle *a, CiphertextHandle *b);
CiphertextHandle *tfhe_bitor_int32(TfheContext *ctx, CiphertextHandle *a, CiphertextHandle *b);
CiphertextHandle *tfhe_bitxor_int32(TfheContext *ctx, CiphertextHandle *a, CiphertextHandle *b);
CiphertextHandle *tfhe_eq_int32(TfheContext *ctx, CiphertextHandle *a, CiphertextHandle *b);
CiphertextHandle *tfhe_lt_int32(TfheContext *ctx, CiphertextHandle *a, CiphertextHandle *b);
CiphertextHandle *tfhe_le_int32(TfheContext *ctx, CiphertextHandle *a, CiphertextHandle *b);
}

namespace fhenomenon {

// Custom deleter for CiphertextHandle shared_ptr
struct CiphertextHandleDeleter {
  void operator()(CiphertextHandle *handle) const { tfhe_ciphertext_destroy(handle); }
};

using SharedCiphertextHandle = std::shared_ptr<CiphertextHandle>;

TfheBackend::TfheBackend() { initialize(); }

TfheBackend::~TfheBackend() {
  if (context_) {
    tfhe_context_destroy(static_cast<TfheContext *>(context_));
    context_ = nullptr;
  }
}

void TfheBackend::initialize() {
  if (!initialized_) {
    context_ = tfhe_context_create();
    initialized_ = true;
    LOG_MESSAGE("TfheBackend: Initialized TFHE context.");
  }
}

void TfheBackend::ensureReady() const {
  if (!initialized_ || !context_) {
    // Should be initialized in constructor, but just in case
    throw std::runtime_error("TfheBackend: Context not initialized");
  }
}

void TfheBackend::transform(CompuonBase &entity, [[maybe_unused]] const Parameter &params) const {
  ensureReady();
  auto type = entity.type();

  if (type == typeid(int)) {
    auto &derivedEntity = dynamic_cast<Compuon<int> &>(entity);
    int32_t value = derivedEntity.getValue();

    CiphertextHandle *raw_handle = tfhe_encrypt_int32(static_cast<TfheContext *>(context_), value);

    // Store as shared_ptr with custom deleter to manage Rust memory
    SharedCiphertextHandle handle(raw_handle, CiphertextHandleDeleter());

    entity.ciphertext_ = handle;
    entity.isEncrypted_ = true;
    LOG_MESSAGE("TfheBackend: Encrypted int value " << value);
  } else {
    throw std::runtime_error("TfheBackend: Unsupported type for encryption (only int supported currently)");
  }
}

std::any TfheBackend::decrypt(const CompuonBase &entity) const {
  ensureReady();

  if (!entity.isEncrypted_ || !entity.ciphertext_.has_value()) {
    // Fallback to plain value if not encrypted
    // Logic duplicated from BuiltinBackend, maybe refactor later
    auto type = entity.type();
    if (type == typeid(int)) {
      return dynamic_cast<const Compuon<int> &>(entity).getValue();
    }
    return {};
  }

  auto type = entity.type();
  if (type == typeid(int)) {
    // Safe because we only put SharedCiphertextHandle in ciphertext_ for this backend
    // TODO: ideally check backend type stored in Compuon if we mix backends
    try {
      auto handle = std::any_cast<SharedCiphertextHandle>(entity.ciphertext_);
      int32_t value = tfhe_decrypt_int32(static_cast<TfheContext *>(context_), handle.get());
      LOG_MESSAGE("TfheBackend: Decrypted int value " << value);
      return value;
    } catch (const std::bad_any_cast &e) {
      throw std::runtime_error("TfheBackend: Failed to cast ciphertext handle. Wrong backend?");
    }
  }

  throw std::runtime_error("TfheBackend: Unsupported type for decryption");
}

namespace {
template <typename T>
std::shared_ptr<CompuonBase> makeResultCompuon(const Compuon<T> &reference, SharedCiphertextHandle ciphertext) {
  auto resultCompuon = std::make_shared<Compuon<T>>(0);
  resultCompuon->ciphertext_ = ciphertext;
  resultCompuon->isEncrypted_ = true;
  resultCompuon->setProfile(reference.getProfile());
  return resultCompuon;
}
} // namespace

std::shared_ptr<CompuonBase> TfheBackend::add(const CompuonBase &a, const CompuonBase &b) const {
  ensureReady();

  // Simplification: only supporting int + int
  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);

    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);

    CiphertextHandle *result_raw = tfhe_add_int32(static_cast<TfheContext *>(context_), ctA.get(), ctB.get());
    SharedCiphertextHandle result(result_raw, CiphertextHandleDeleter());

    LOG_MESSAGE("TfheBackend: Performed addition");
    return makeResultCompuon(derivedA, result);
  }

  throw std::runtime_error("TfheBackend: Unsupported types for add");
}

std::shared_ptr<CompuonBase> TfheBackend::multiply(const CompuonBase &a, const CompuonBase &b) const {
  ensureReady();

  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);

    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);

    CiphertextHandle *result_raw = tfhe_mul_int32(static_cast<TfheContext *>(context_), ctA.get(), ctB.get());
    SharedCiphertextHandle result(result_raw, CiphertextHandleDeleter());

    LOG_MESSAGE("TfheBackend: Performed multiplication");
    return makeResultCompuon(derivedA, result);
  }

  throw std::runtime_error("TfheBackend: Unsupported types for multiply");
}

std::shared_ptr<CompuonBase> TfheBackend::bitAnd(const CompuonBase &a, const CompuonBase &b) const {
  ensureReady();
  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);
    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);
    CiphertextHandle *result_raw = tfhe_bitand_int32(static_cast<TfheContext *>(context_), ctA.get(), ctB.get());
    SharedCiphertextHandle result(result_raw, CiphertextHandleDeleter());
    return makeResultCompuon(derivedA, result);
  }
  throw std::runtime_error("TfheBackend: Unsupported types for bitAnd");
}

std::shared_ptr<CompuonBase> TfheBackend::bitOr(const CompuonBase &a, const CompuonBase &b) const {
  ensureReady();
  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);
    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);
    CiphertextHandle *result_raw = tfhe_bitor_int32(static_cast<TfheContext *>(context_), ctA.get(), ctB.get());
    SharedCiphertextHandle result(result_raw, CiphertextHandleDeleter());
    return makeResultCompuon(derivedA, result);
  }
  throw std::runtime_error("TfheBackend: Unsupported types for bitOr");
}

std::shared_ptr<CompuonBase> TfheBackend::bitXor(const CompuonBase &a, const CompuonBase &b) const {
  ensureReady();
  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);
    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);
    CiphertextHandle *result_raw = tfhe_bitxor_int32(static_cast<TfheContext *>(context_), ctA.get(), ctB.get());
    SharedCiphertextHandle result(result_raw, CiphertextHandleDeleter());
    return makeResultCompuon(derivedA, result);
  }
  throw std::runtime_error("TfheBackend: Unsupported types for bitXor");
}

std::shared_ptr<CompuonBase> TfheBackend::compareEq(const CompuonBase &a, const CompuonBase &b) const {
  ensureReady();
  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);
    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);
    CiphertextHandle *result_raw = tfhe_eq_int32(static_cast<TfheContext *>(context_), ctA.get(), ctB.get());
    SharedCiphertextHandle result(result_raw, CiphertextHandleDeleter());
    return makeResultCompuon(derivedA, result);
  }
  throw std::runtime_error("TfheBackend: Unsupported types for compareEq");
}

std::shared_ptr<CompuonBase> TfheBackend::compareLt(const CompuonBase &a, const CompuonBase &b) const {
  ensureReady();
  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);
    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);
    CiphertextHandle *result_raw = tfhe_lt_int32(static_cast<TfheContext *>(context_), ctA.get(), ctB.get());
    SharedCiphertextHandle result(result_raw, CiphertextHandleDeleter());
    return makeResultCompuon(derivedA, result);
  }
  throw std::runtime_error("TfheBackend: Unsupported types for compareLt");
}

std::shared_ptr<CompuonBase> TfheBackend::compareLe(const CompuonBase &a, const CompuonBase &b) const {
  ensureReady();
  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);
    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);
    CiphertextHandle *result_raw = tfhe_le_int32(static_cast<TfheContext *>(context_), ctA.get(), ctB.get());
    SharedCiphertextHandle result(result_raw, CiphertextHandleDeleter());
    return makeResultCompuon(derivedA, result);
  }
  throw std::runtime_error("TfheBackend: Unsupported types for compareLe");
}

} // namespace fhenomenon
