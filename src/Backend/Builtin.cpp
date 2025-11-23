#include "Backend/Builtin.h"
#include "Utils/log.h"

#include <stdexcept>

namespace fhenomenon {

BuiltinBackend::BuiltinBackend() : initialized_(false) {
  engine_.initialize(params_);
  engine_.generateKeys();
  initialize();
  generateKeys();
}

void BuiltinBackend::ensureReady() const {
  if (!initialized_) {
    const_cast<BuiltinBackend *>(this)->initialize();
  }

  if (!engine_.areKeysGenerated()) {
    const_cast<BuiltinBackend *>(this)->generateKeys();
  }
}

void BuiltinBackend::initialize() {
  engine_.initialize(params_);
  initialized_ = true;
}

void BuiltinBackend::generateKeys() { engine_.generateKeys(); }

void BuiltinBackend::loadKeys([[maybe_unused]] const std::string &publicKeyPath,
                              [[maybe_unused]] const std::string &secretKeyPath) {
  LOG_MESSAGE("BuiltinBackend: ToyFHE backend currently generates keys at runtime; loadKeys is a no-op.");
}

void BuiltinBackend::saveKeys([[maybe_unused]] const std::string &publicKeyPath,
                              [[maybe_unused]] const std::string &secretKeyPath) {
  LOG_MESSAGE("BuiltinBackend: ToyFHE backend currently does not persist keys; saveKeys is a no-op.");
}

void BuiltinBackend::transform(CompuonBase &entity, [[maybe_unused]] const Parameter &params) const {
  ensureReady();

  auto type = entity.type();

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
}

namespace {
template <typename T>
std::shared_ptr<CompuonBase> makeResultCompuon(const Compuon<T> &reference, const toyfhe::Ciphertext &ciphertext) {
  auto resultCompuon = std::make_shared<Compuon<T>>(0);
  resultCompuon->ciphertext_ = std::make_shared<toyfhe::Ciphertext>(ciphertext);
  resultCompuon->isEncrypted_ = true;
  resultCompuon->setProfile(reference.getProfile());
  return resultCompuon;
}
} // namespace

std::shared_ptr<CompuonBase> BuiltinBackend::add(const CompuonBase &a, const CompuonBase &b) const {
  ensureReady();

  if (!a.isEncrypted_ || !b.isEncrypted_ || !a.ciphertext_ || !b.ciphertext_) {
    throw std::runtime_error("BuiltinBackend: Cannot add unencrypted Compuon values");
  }

  auto type = a.type();

  if (type == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);
    [[maybe_unused]] const auto &derivedB = dynamic_cast<const Compuon<int> &>(b);
    const auto result = engine_.add(*a.ciphertext_, *b.ciphertext_);
    LOG_MESSAGE("BuiltinBackend: Performed ToyFHE addition (int)");
    return makeResultCompuon(derivedA, result);
  } else if (type == typeid(double)) {
    const auto &derivedA = dynamic_cast<const Compuon<double> &>(a);
    [[maybe_unused]] const auto &derivedB = dynamic_cast<const Compuon<double> &>(b);
    const auto result = engine_.add(*a.ciphertext_, *b.ciphertext_);
    LOG_MESSAGE("BuiltinBackend: Performed ToyFHE addition (double)");
    return makeResultCompuon(derivedA, result);
  } else if (type == typeid(float)) {
    const auto &derivedA = dynamic_cast<const Compuon<float> &>(a);
    [[maybe_unused]] const auto &derivedB = dynamic_cast<const Compuon<float> &>(b);
    const auto result = engine_.add(*a.ciphertext_, *b.ciphertext_);
    LOG_MESSAGE("BuiltinBackend: Performed ToyFHE addition (float)");
    return makeResultCompuon(derivedA, result);
  }

  throw std::runtime_error("BuiltinBackend: Unsupported type for add");
}

std::shared_ptr<CompuonBase> BuiltinBackend::multiply(const CompuonBase &a, const CompuonBase &b) const {
  ensureReady();

  if (!a.isEncrypted_ || !b.isEncrypted_ || !a.ciphertext_ || !b.ciphertext_) {
    throw std::runtime_error("BuiltinBackend: Cannot multiply unencrypted Compuon values");
  }

  if (a.type() != b.type()) {
    throw std::runtime_error("BuiltinBackend: Cannot multiply Compuon values of different types");
  }

  auto type = a.type();

  if (type == typeid(int)) {
    const auto &derivedA = dynamic_cast<const Compuon<int> &>(a);
    const auto result = engine_.multiply(*a.ciphertext_, *b.ciphertext_);
    LOG_MESSAGE("BuiltinBackend: Performed ToyFHE multiplication (int)");
    return makeResultCompuon(derivedA, result);
  } else if (type == typeid(double)) {
    const auto &derivedA = dynamic_cast<const Compuon<double> &>(a);
    const auto result = engine_.multiply(*a.ciphertext_, *b.ciphertext_);
    LOG_MESSAGE("BuiltinBackend: Performed ToyFHE multiplication (double)");
    return makeResultCompuon(derivedA, result);
  } else if (type == typeid(float)) {
    const auto &derivedA = dynamic_cast<const Compuon<float> &>(a);
    const auto result = engine_.multiply(*a.ciphertext_, *b.ciphertext_);
    LOG_MESSAGE("BuiltinBackend: Performed ToyFHE multiplication (float)");
    return makeResultCompuon(derivedA, result);
  }

  throw std::runtime_error("BuiltinBackend: Unsupported type for multiply");
}

std::shared_ptr<CompuonBase> BuiltinBackend::addPlain(const CompuonBase &a, double scalar) {
  ensureReady();

  if (!a.isEncrypted_ || !a.ciphertext_) {
    throw std::runtime_error("BuiltinBackend: Cannot add plain to unencrypted Compuon value");
  }

  auto type = a.type();

  if (type == typeid(double)) {
    const auto &derivedA = dynamic_cast<const Compuon<double> &>(a);
    const auto result = engine_.addPlain(*a.ciphertext_, scalar);
    LOG_MESSAGE("BuiltinBackend: ToyFHE addPlain(double) with scalar " << scalar);
    return makeResultCompuon(derivedA, result);
  }

  throw std::runtime_error("BuiltinBackend: addPlain currently supports double types only");
}

std::shared_ptr<CompuonBase> BuiltinBackend::multiplyPlain(const CompuonBase &a, double scalar) {
  ensureReady();

  if (!a.isEncrypted_ || !a.ciphertext_) {
    throw std::runtime_error("BuiltinBackend: Cannot multiply plain with unencrypted Compuon value");
  }

  auto type = a.type();

  if (type == typeid(double)) {
    const auto &derivedA = dynamic_cast<const Compuon<double> &>(a);
    const auto result = engine_.multiplyPlain(*a.ciphertext_, scalar);
    LOG_MESSAGE("BuiltinBackend: ToyFHE multiplyPlain(double) with scalar " << scalar);
    return makeResultCompuon(derivedA, result);
  }

  throw std::runtime_error("BuiltinBackend: multiplyPlain currently supports double types only");
}

std::any BuiltinBackend::decrypt(const CompuonBase &entity) const {
  ensureReady();

  if (!entity.isEncrypted_ || !entity.ciphertext_) {
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
    const auto value = engine_.decryptInt(*entity.ciphertext_);
    LOG_MESSAGE("BuiltinBackend: Decrypted int value " << value);
    return static_cast<int>(value);
  } else if (type == typeid(double)) {
    const auto value = engine_.decryptDouble(*entity.ciphertext_);
    LOG_MESSAGE("BuiltinBackend: Decrypted double value " << value);
    return value;
  } else if (type == typeid(float)) {
    const auto value = engine_.decryptDouble(*entity.ciphertext_);
    LOG_MESSAGE("BuiltinBackend: Decrypted float value " << static_cast<float>(value));
    return static_cast<float>(value);
  }

  return {};
}

} // namespace fhenomenon
