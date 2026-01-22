#include "Backend/Builtin.h"
#include "Utils/log.h"

#include <iostream>
#include <memory>
#include <stdexcept>

// --- TFHE-rs FFI Declarations ---
#ifdef FHENOMENON_USE_TFHE
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
#endif
// ------------------------------

namespace fhenomenon {

#ifdef FHENOMENON_USE_TFHE
// Custom deleter for CiphertextHandle shared_ptr
struct CiphertextHandleDeleter {
  void operator()(CiphertextHandle *handle) const { tfhe_ciphertext_destroy(handle); }
};
using SharedCiphertextHandle = std::shared_ptr<CiphertextHandle>;
#endif

BuiltinBackend::BuiltinBackend() : initialized_(false) {
#ifndef FHENOMENON_USE_TFHE
  engine_.initialize(params_);
  engine_.generateKeys();
#endif
  initialize();
#ifndef FHENOMENON_USE_TFHE
  generateKeys();
#endif
}

BuiltinBackend::~BuiltinBackend() {
#ifdef FHENOMENON_USE_TFHE
  if (context_) {
    tfhe_context_destroy(static_cast<TfheContext *>(context_));
    context_ = nullptr;
  }
#endif
}

void BuiltinBackend::ensureReady() const {
  if (!initialized_) {
    const_cast<BuiltinBackend *>(this)->initialize();
  }
#ifndef FHENOMENON_USE_TFHE
  if (!engine_.areKeysGenerated()) {
    const_cast<BuiltinBackend *>(this)->generateKeys();
  }
#endif
}

void BuiltinBackend::initialize() {
#ifdef FHENOMENON_USE_TFHE
  if (!context_) {
    context_ = tfhe_context_create();
    LOG_MESSAGE("BuiltinBackend: Initialized TFHE context.");
  }
#else
  engine_.initialize(params_);
#endif
  initialized_ = true;
}

void BuiltinBackend::generateKeys() {
#ifndef FHENOMENON_USE_TFHE
  engine_.generateKeys();
#else
  // TFHE keys are managed internally by the context/Rust side for now in this bindings setup
  // or generated lazily.
#endif
}

void BuiltinBackend::loadKeys([[maybe_unused]] const std::string &publicKeyPath,
                              [[maybe_unused]] const std::string &secretKeyPath) {
  LOG_MESSAGE("BuiltinBackend: loadKeys is a no-op.");
}

void BuiltinBackend::saveKeys([[maybe_unused]] const std::string &publicKeyPath,
                              [[maybe_unused]] const std::string &secretKeyPath) {
  LOG_MESSAGE("BuiltinBackend: saveKeys is a no-op.");
}

void BuiltinBackend::transform(CompuonBase &entity, [[maybe_unused]] const Parameter &params) const {
  ensureReady();
  auto type = entity.type();

#ifdef FHENOMENON_USE_TFHE
  if (type == typeid(int)) {
    auto &derivedEntity = dynamic_cast<Compuon<int> &>(entity);
    int32_t value = derivedEntity.getValue();

    CiphertextHandle *raw_handle = tfhe_encrypt_int32(static_cast<TfheContext *>(context_), value);
    SharedCiphertextHandle handle(raw_handle, CiphertextHandleDeleter());

    entity.ciphertext_ = handle;
    entity.isEncrypted_ = true;
    LOG_MESSAGE("BuiltinBackend: Encrypted int value " << value << " (TFHE)");
  } else {
    throw std::runtime_error("BuiltinBackend: Unsupported type for encryption (only int supported with TFHE)");
  }
#else
  if (type == typeid(int)) {
    auto &derivedEntity = dynamic_cast<Compuon<int> &>(entity);
    const int value = derivedEntity.getValue();
    auto ciphertext = std::make_shared<toyfhe::Ciphertext>(engine_.encryptInt(value));
    entity.ciphertext_ = ciphertext;
    entity.isEncrypted_ = true;
    LOG_MESSAGE("BuiltinBackend: Encrypted int value " << value);
  } else if (type == typeid(double)) {
    auto &derivedEntity = dynamic_cast<Compuon<double> &>(entity);
    const double value = derivedEntity.getValue();
    auto ciphertext = std::make_shared<toyfhe::Ciphertext>(engine_.encryptDouble(value));
    entity.ciphertext_ = ciphertext;
    entity.isEncrypted_ = true;
    LOG_MESSAGE("BuiltinBackend: Encrypted double value " << value);
  } else if (type == typeid(float)) {
    auto &derivedEntity = dynamic_cast<Compuon<float> &>(entity);
    const double value = static_cast<double>(derivedEntity.getValue());
    auto ciphertext = std::make_shared<toyfhe::Ciphertext>(engine_.encryptDouble(value));
    entity.ciphertext_ = ciphertext;
    entity.isEncrypted_ = true;
    LOG_MESSAGE("BuiltinBackend: Encrypted float value " << value);
  } else {
    throw std::runtime_error("BuiltinBackend: Unsupported type for encryption");
  }
#endif
}

namespace {
template <typename T>
#ifdef FHENOMENON_USE_TFHE
std::shared_ptr<CompuonBase> makeResultCompuon(const Compuon<T> &reference, SharedCiphertextHandle ciphertext) {
  auto resultCompuon = std::make_shared<Compuon<T>>(0);
  resultCompuon->ciphertext_ = ciphertext;
  resultCompuon->isEncrypted_ = true;
  resultCompuon->setProfile(reference.getProfile());
  return resultCompuon;
}
#else
std::shared_ptr<CompuonBase> makeResultCompuon(const Compuon<T> &reference, const toyfhe::Ciphertext &ciphertext) {
  auto resultCompuon = std::make_shared<Compuon<T>>(0);
  resultCompuon->ciphertext_ = std::make_shared<toyfhe::Ciphertext>(ciphertext);
  resultCompuon->isEncrypted_ = true;
  resultCompuon->setProfile(reference.getProfile());
  return resultCompuon;
}
#endif
} // namespace

std::shared_ptr<CompuonBase> BuiltinBackend::add(const CompuonBase &a, const CompuonBase &b) const {
  ensureReady();

  if (!a.isEncrypted_ || !b.isEncrypted_ || !a.ciphertext_.has_value() || !b.ciphertext_.has_value()) {
    throw std::runtime_error("BuiltinBackend: Cannot add unencrypted Compuon values");
  }

#ifdef FHENOMENON_USE_TFHE
  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);
    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);

    CiphertextHandle *result_raw = tfhe_add_int32(static_cast<TfheContext *>(context_), ctA.get(), ctB.get());
    SharedCiphertextHandle result(result_raw, CiphertextHandleDeleter());

    LOG_MESSAGE("BuiltinBackend: Performed addition (TFHE)");
    return makeResultCompuon(derivedA, result);
  }
  throw std::runtime_error("BuiltinBackend: Unsupported types for add (TFHE only supports int)");
#else
  auto type = a.type();
  if (type == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);
    auto ctA = std::any_cast<std::shared_ptr<toyfhe::Ciphertext>>(a.ciphertext_);
    auto ctB = std::any_cast<std::shared_ptr<toyfhe::Ciphertext>>(b.ciphertext_);
    const auto result = engine_.add(*ctA, *ctB);
    LOG_MESSAGE("BuiltinBackend: Performed ToyFHE addition (int)");
    return makeResultCompuon(derivedA, result);
  } else if (type == typeid(double)) {
    const auto &derivedA = dynamic_cast<const Compuon<double> &>(a);
    auto ctA = std::any_cast<std::shared_ptr<toyfhe::Ciphertext>>(a.ciphertext_);
    auto ctB = std::any_cast<std::shared_ptr<toyfhe::Ciphertext>>(b.ciphertext_);
    const auto result = engine_.add(*ctA, *ctB);
    LOG_MESSAGE("BuiltinBackend: Performed ToyFHE addition (double)");
    return makeResultCompuon(derivedA, result);
  } else if (type == typeid(float)) {
    const auto &derivedA = dynamic_cast<const Compuon<float> &>(a);
    auto ctA = std::any_cast<std::shared_ptr<toyfhe::Ciphertext>>(a.ciphertext_);
    auto ctB = std::any_cast<std::shared_ptr<toyfhe::Ciphertext>>(b.ciphertext_);
    const auto result = engine_.add(*ctA, *ctB);
    LOG_MESSAGE("BuiltinBackend: Performed ToyFHE addition (float)");
    return makeResultCompuon(derivedA, result);
  }
  throw std::runtime_error("BuiltinBackend: Unsupported type for add");
#endif
}

std::shared_ptr<CompuonBase> BuiltinBackend::multiply(const CompuonBase &a, const CompuonBase &b) const {
  ensureReady();
  if (!a.isEncrypted_ || !b.isEncrypted_ || !a.ciphertext_.has_value() || !b.ciphertext_.has_value()) {
    throw std::runtime_error("BuiltinBackend: Cannot multiply unencrypted Compuon values");
  }

#ifdef FHENOMENON_USE_TFHE
  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);
    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);

    CiphertextHandle *result_raw = tfhe_mul_int32(static_cast<TfheContext *>(context_), ctA.get(), ctB.get());
    SharedCiphertextHandle result(result_raw, CiphertextHandleDeleter());

    LOG_MESSAGE("BuiltinBackend: Performed multiplication (TFHE)");
    return makeResultCompuon(derivedA, result);
  }
  throw std::runtime_error("BuiltinBackend: Unsupported types for multiply (TFHE only supports int)");
#else
  if (a.type() != b.type()) {
    throw std::runtime_error("BuiltinBackend: Cannot multiply Compuon values of different types");
  }

  auto type = a.type();

  if (type == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);
    auto ctA = std::any_cast<std::shared_ptr<toyfhe::Ciphertext>>(a.ciphertext_);
    auto ctB = std::any_cast<std::shared_ptr<toyfhe::Ciphertext>>(b.ciphertext_);
    const auto result = engine_.multiply(*ctA, *ctB);
    LOG_MESSAGE("BuiltinBackend: Performed ToyFHE multiplication (int)");
    return makeResultCompuon(derivedA, result);
  } else if (type == typeid(double)) {
    const auto &derivedA = dynamic_cast<const Compuon<double> &>(a);
    auto ctA = std::any_cast<std::shared_ptr<toyfhe::Ciphertext>>(a.ciphertext_);
    auto ctB = std::any_cast<std::shared_ptr<toyfhe::Ciphertext>>(b.ciphertext_);
    const auto result = engine_.multiply(*ctA, *ctB);
    LOG_MESSAGE("BuiltinBackend: Performed ToyFHE multiplication (double)");
    return makeResultCompuon(derivedA, result);
  } else if (type == typeid(float)) {
    const auto &derivedA = dynamic_cast<const Compuon<float> &>(a);
    auto ctA = std::any_cast<std::shared_ptr<toyfhe::Ciphertext>>(a.ciphertext_);
    auto ctB = std::any_cast<std::shared_ptr<toyfhe::Ciphertext>>(b.ciphertext_);
    const auto result = engine_.multiply(*ctA, *ctB);
    LOG_MESSAGE("BuiltinBackend: Performed ToyFHE multiplication (float)");
    return makeResultCompuon(derivedA, result);
  }
  throw std::runtime_error("BuiltinBackend: Unsupported type for multiply");
#endif
}

std::shared_ptr<CompuonBase> BuiltinBackend::addPlain([[maybe_unused]] const CompuonBase &a,
                                                      [[maybe_unused]] double scalar) {
  ensureReady();
#ifdef FHENOMENON_USE_TFHE
  throw std::runtime_error("BuiltinBackend: addPlain not implemented for TFHE");
#else
  if (!a.isEncrypted_ || !a.ciphertext_.has_value()) {
    throw std::runtime_error("BuiltinBackend: Cannot add plain to unencrypted Compuon value");
  }
  auto type = a.type();
  if (type == typeid(double)) {
    const auto &derivedA = dynamic_cast<const Compuon<double> &>(a);
    auto ctA = std::any_cast<std::shared_ptr<toyfhe::Ciphertext>>(a.ciphertext_);
    const auto result = engine_.addPlain(*ctA, scalar);
    LOG_MESSAGE("BuiltinBackend: ToyFHE addPlain(double) with scalar " << scalar);
    return makeResultCompuon(derivedA, result);
  }
  throw std::runtime_error("BuiltinBackend: addPlain currently supports double types only");
#endif
}

std::shared_ptr<CompuonBase> BuiltinBackend::multiplyPlain([[maybe_unused]] const CompuonBase &a,
                                                           [[maybe_unused]] double scalar) {
  ensureReady();
#ifdef FHENOMENON_USE_TFHE
  throw std::runtime_error("BuiltinBackend: multiplyPlain not implemented for TFHE");
#else
  if (!a.isEncrypted_ || !a.ciphertext_.has_value()) {
    throw std::runtime_error("BuiltinBackend: Cannot multiply plain with unencrypted Compuon value");
  }
  auto type = a.type();
  if (type == typeid(double)) {
    const auto &derivedA = dynamic_cast<const Compuon<double> &>(a);
    auto ctA = std::any_cast<std::shared_ptr<toyfhe::Ciphertext>>(a.ciphertext_);
    const auto result = engine_.multiplyPlain(*ctA, scalar);
    LOG_MESSAGE("BuiltinBackend: ToyFHE multiplyPlain(double) with scalar " << scalar);
    return makeResultCompuon(derivedA, result);
  }
  throw std::runtime_error("BuiltinBackend: multiplyPlain currently supports double types only");
#endif
}

std::any BuiltinBackend::decrypt(const CompuonBase &entity) const {
  ensureReady();

#ifdef FHENOMENON_USE_TFHE
  if (!entity.isEncrypted_ || !entity.ciphertext_.has_value()) {
    // fallback
    auto type = entity.type();
    if (type == typeid(int))
      return dynamic_cast<const Compuon<int> &>(entity).getValue();
    return {};
  }
  auto type = entity.type();
  if (type == typeid(int)) {
    try {
      auto handle = std::any_cast<SharedCiphertextHandle>(entity.ciphertext_);
      int32_t value = tfhe_decrypt_int32(static_cast<TfheContext *>(context_), handle.get());
      LOG_MESSAGE("BuiltinBackend: Decrypted int value " << value << " (TFHE)");
      return value;
    } catch (const std::bad_any_cast &) {
      throw std::runtime_error("BuiltinBackend: Failed to cast ciphertext handle");
    }
  }
  throw std::runtime_error("BuiltinBackend: Unsupported type for decryption (TFHE only supports int)");
#else
  if (!entity.isEncrypted_ || !entity.ciphertext_.has_value()) {
    auto type = entity.type();
    if (type == typeid(int)) {
      const auto &derivedEntity = dynamic_cast<const Compuon<int> &>(entity);
      return derivedEntity.getValue();
    } else if (type == typeid(double)) {
      const auto &derivedEntity = dynamic_cast<const Compuon<double> &>(entity);
      return derivedEntity.getValue();
    } else if (type == typeid(float)) {
      const auto &derivedEntity = dynamic_cast<const Compuon<float> &>(entity);
      return derivedEntity.getValue();
    }
    return {};
  }

  auto type = entity.type();
  if (type == typeid(int)) {
    auto ct = std::any_cast<std::shared_ptr<toyfhe::Ciphertext>>(entity.ciphertext_);
    const auto value = engine_.decryptInt(*ct);
    LOG_MESSAGE("BuiltinBackend: Decrypted int value " << value);
    return static_cast<int>(value);
  } else if (type == typeid(double)) {
    auto ct = std::any_cast<std::shared_ptr<toyfhe::Ciphertext>>(entity.ciphertext_);
    const auto value = engine_.decryptDouble(*ct);
    LOG_MESSAGE("BuiltinBackend: Decrypted double value " << value);
    return value;
  } else if (type == typeid(float)) {
    auto ct = std::any_cast<std::shared_ptr<toyfhe::Ciphertext>>(entity.ciphertext_);
    const auto value = engine_.decryptDouble(*ct);
    LOG_MESSAGE("BuiltinBackend: Decrypted float value " << static_cast<float>(value));
    return static_cast<float>(value);
  }
  return {};
#endif
}

std::shared_ptr<CompuonBase> BuiltinBackend::bitAnd([[maybe_unused]] const CompuonBase &a,
                                                    [[maybe_unused]] const CompuonBase &b) const {
#ifdef FHENOMENON_USE_TFHE
  ensureReady();
  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);
    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);
    CiphertextHandle *result_raw = tfhe_bitand_int32(static_cast<TfheContext *>(context_), ctA.get(), ctB.get());
    SharedCiphertextHandle result(result_raw, CiphertextHandleDeleter());
    return makeResultCompuon(derivedA, result);
  }
#endif
  throw std::runtime_error("Bitwise AND not supported by BuiltinBackend");
}

std::shared_ptr<CompuonBase> BuiltinBackend::bitOr([[maybe_unused]] const CompuonBase &a,
                                                   [[maybe_unused]] const CompuonBase &b) const {
#ifdef FHENOMENON_USE_TFHE
  ensureReady();
  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);
    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);
    CiphertextHandle *result_raw = tfhe_bitor_int32(static_cast<TfheContext *>(context_), ctA.get(), ctB.get());
    SharedCiphertextHandle result(result_raw, CiphertextHandleDeleter());
    return makeResultCompuon(derivedA, result);
  }
#endif
  throw std::runtime_error("Bitwise OR not supported by BuiltinBackend");
}

std::shared_ptr<CompuonBase> BuiltinBackend::bitXor([[maybe_unused]] const CompuonBase &a,
                                                    [[maybe_unused]] const CompuonBase &b) const {
#ifdef FHENOMENON_USE_TFHE
  ensureReady();
  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);
    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);
    CiphertextHandle *result_raw = tfhe_bitxor_int32(static_cast<TfheContext *>(context_), ctA.get(), ctB.get());
    SharedCiphertextHandle result(result_raw, CiphertextHandleDeleter());
    return makeResultCompuon(derivedA, result);
  }
#endif
  throw std::runtime_error("Bitwise XOR not supported by BuiltinBackend");
}

std::shared_ptr<CompuonBase> BuiltinBackend::compareEq([[maybe_unused]] const CompuonBase &a,
                                                       [[maybe_unused]] const CompuonBase &b) const {
#ifdef FHENOMENON_USE_TFHE
  ensureReady();
  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);
    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);
    CiphertextHandle *result_raw = tfhe_eq_int32(static_cast<TfheContext *>(context_), ctA.get(), ctB.get());
    SharedCiphertextHandle result(result_raw, CiphertextHandleDeleter());
    return makeResultCompuon(derivedA, result);
  }
#endif
  throw std::runtime_error("Equality comparison not supported by BuiltinBackend");
}

std::shared_ptr<CompuonBase> BuiltinBackend::compareLt([[maybe_unused]] const CompuonBase &a,
                                                       [[maybe_unused]] const CompuonBase &b) const {
#ifdef FHENOMENON_USE_TFHE
  ensureReady();
  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);
    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);
    CiphertextHandle *result_raw = tfhe_lt_int32(static_cast<TfheContext *>(context_), ctA.get(), ctB.get());
    SharedCiphertextHandle result(result_raw, CiphertextHandleDeleter());
    return makeResultCompuon(derivedA, result);
  }
#endif
  throw std::runtime_error("Less than comparison not supported by BuiltinBackend");
}

std::shared_ptr<CompuonBase> BuiltinBackend::compareLe([[maybe_unused]] const CompuonBase &a,
                                                       [[maybe_unused]] const CompuonBase &b) const {
#ifdef FHENOMENON_USE_TFHE
  ensureReady();
  if (a.type() == typeid(int) && b.type() == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);
    auto ctA = std::any_cast<SharedCiphertextHandle>(a.ciphertext_);
    auto ctB = std::any_cast<SharedCiphertextHandle>(b.ciphertext_);
    CiphertextHandle *result_raw = tfhe_le_int32(static_cast<TfheContext *>(context_), ctA.get(), ctB.get());
    SharedCiphertextHandle result(result_raw, CiphertextHandleDeleter());
    return makeResultCompuon(derivedA, result);
  }
#endif
  throw std::runtime_error("Less equal comparison not supported by BuiltinBackend");
}

} // namespace fhenomenon
