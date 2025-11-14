#include "Backend/Builtin.h"
#include "Utils/log.h"

#include <stdexcept>

using namespace fhenomenon;

BuiltinBackend::BuiltinBackend() : sealBackend_(std::make_unique<SEALBackend>()) {
}

void BuiltinBackend::initialize(const Parameter &params) {
  sealBackend_->initialize(params);
}

void BuiltinBackend::generateKeys() {
  sealBackend_->generateKeys();
}

void BuiltinBackend::loadKeys(const std::string &publicKeyPath, const std::string &secretKeyPath) {
  sealBackend_->loadKeys(publicKeyPath, secretKeyPath);
}

void BuiltinBackend::saveKeys(const std::string &publicKeyPath, const std::string &secretKeyPath) {
  sealBackend_->saveKeys(publicKeyPath, secretKeyPath);
}

void BuiltinBackend::transform(CompuonBase &entity, const Parameter &params) const {
  if (!sealBackend_->isInitialized()) {
    sealBackend_->initialize(params);
  }
  
  if (!sealBackend_->areKeysGenerated()) {
    sealBackend_->generateKeys();
  }

  auto type = entity.type();

  if (type == typeid(int)) {
    Compuon<int> &derivedEntity = dynamic_cast<Compuon<int> &>(entity);
    int value = derivedEntity.getValue();
    
    // Encrypt using SEAL
    seal::Ciphertext ciphertext = sealBackend_->encryptInt(value);
    entity.ciphertext_ = std::make_shared<seal::Ciphertext>(std::move(ciphertext));
    entity.isEncrypted_ = true;
    
    LOG_MESSAGE("BuiltinBackend: Encrypted int value " << value);
  } else if (type == typeid(double)) {
    Compuon<double> &derivedEntity = dynamic_cast<Compuon<double> &>(entity);
    double value = derivedEntity.getValue();
    
    // Encrypt using SEAL CKKS
    seal::Ciphertext ciphertext = sealBackend_->encryptDouble(value);
    entity.ciphertext_ = std::make_shared<seal::Ciphertext>(std::move(ciphertext));
    entity.isEncrypted_ = true;
    
    LOG_MESSAGE("BuiltinBackend: Encrypted double value " << value);
  } else if (type == typeid(float)) {
    Compuon<float> &derivedEntity = dynamic_cast<Compuon<float> &>(entity);
    float value = derivedEntity.getValue();
    
    // Encrypt using SEAL CKKS (treat float as double)
    seal::Ciphertext ciphertext = sealBackend_->encryptDouble(static_cast<double>(value));
    entity.ciphertext_ = std::make_shared<seal::Ciphertext>(std::move(ciphertext));
    entity.isEncrypted_ = true;
    
    LOG_MESSAGE("BuiltinBackend: Encrypted float value " << value);
  } else {
    throw std::runtime_error("BuiltinBackend: Unsupported type for encryption");
  }
}

std::shared_ptr<CompuonBase> BuiltinBackend::add(const CompuonBase &a, const CompuonBase &b) const {
  if (!a.isEncrypted_ || !b.isEncrypted_ || !a.ciphertext_ || !b.ciphertext_) {
    throw std::runtime_error("BuiltinBackend: Cannot add unencrypted Compuon values");
  }

  auto type = a.type();
  
  if (type == typeid(int)) {
    const Compuon<int> &derivedA = dynamic_cast<const Compuon<int> &>(a);
    [[maybe_unused]] const Compuon<int> &derivedB = dynamic_cast<const Compuon<int> &>(b);
    
    // Perform homomorphic addition
    seal::Ciphertext result = sealBackend_->add(*a.ciphertext_, *b.ciphertext_);
    
    // Create result Compuon
    auto resultCompuon = std::make_shared<Compuon<int>>(0);
    resultCompuon->ciphertext_ = std::make_shared<seal::Ciphertext>(std::move(result));
    resultCompuon->isEncrypted_ = true;
    resultCompuon->setProfile(derivedA.getProfile());
    
    LOG_MESSAGE("BuiltinBackend: Performed homomorphic addition");
    return resultCompuon;
  } else if (type == typeid(double)) {
    const Compuon<double> &derivedA = dynamic_cast<const Compuon<double> &>(a);
    [[maybe_unused]] const Compuon<double> &derivedB = dynamic_cast<const Compuon<double> &>(b);
    
    // Perform homomorphic addition
    seal::Ciphertext result = sealBackend_->add(*a.ciphertext_, *b.ciphertext_);
    
    // Create result Compuon
    auto resultCompuon = std::make_shared<Compuon<double>>(0.0);
    resultCompuon->ciphertext_ = std::make_shared<seal::Ciphertext>(std::move(result));
    resultCompuon->isEncrypted_ = true;
    resultCompuon->setProfile(derivedA.getProfile());
    
    LOG_MESSAGE("BuiltinBackend: Performed homomorphic addition (double)");
    return resultCompuon;
  } else if (type == typeid(float)) {
    const Compuon<float> &derivedA = dynamic_cast<const Compuon<float> &>(a);
    [[maybe_unused]] const Compuon<float> &derivedB = dynamic_cast<const Compuon<float> &>(b);
    
    // Perform homomorphic addition
    seal::Ciphertext result = sealBackend_->add(*a.ciphertext_, *b.ciphertext_);
    
    // Create result Compuon
    auto resultCompuon = std::make_shared<Compuon<float>>(0.0f);
    resultCompuon->ciphertext_ = std::make_shared<seal::Ciphertext>(std::move(result));
    resultCompuon->isEncrypted_ = true;
    resultCompuon->setProfile(derivedA.getProfile());
    
    LOG_MESSAGE("BuiltinBackend: Performed homomorphic addition (float)");
    return resultCompuon;
  }
  
  return nullptr;
}

std::shared_ptr<CompuonBase> BuiltinBackend::multiply(const CompuonBase &a, const CompuonBase &b) const {
  if (!a.isEncrypted_ || !b.isEncrypted_ || !a.ciphertext_ || !b.ciphertext_) {
    throw std::runtime_error("BuiltinBackend: Cannot multiply unencrypted Compuon values");
  }

  if (a.type() != b.type()) {
    throw std::runtime_error("BuiltinBackend: Cannot multiply Compuon values of different types");
  }

  auto type = a.type();
  
  if (type == typeid(int)) {
    const Compuon<int> &derivedA = dynamic_cast<const Compuon<int> &>(a);
    [[maybe_unused]] const Compuon<int> &derivedB = dynamic_cast<const Compuon<int> &>(b);
    
    // Perform homomorphic multiplication
    seal::Ciphertext result = sealBackend_->multiply(*a.ciphertext_, *b.ciphertext_);
    
    // Create result Compuon
    auto resultCompuon = std::make_shared<Compuon<int>>(0);
    resultCompuon->ciphertext_ = std::make_shared<seal::Ciphertext>(std::move(result));
    resultCompuon->isEncrypted_ = true;
    resultCompuon->setProfile(derivedA.getProfile());
    
    LOG_MESSAGE("BuiltinBackend: Performed homomorphic multiplication");
    return resultCompuon;
  } else if (type == typeid(double)) {
    const Compuon<double> &derivedA = dynamic_cast<const Compuon<double> &>(a);
    [[maybe_unused]] const Compuon<double> &derivedB = dynamic_cast<const Compuon<double> &>(b);
    
    // Perform homomorphic multiplication
    seal::Ciphertext result = sealBackend_->multiply(*a.ciphertext_, *b.ciphertext_);
    
    // Create result Compuon
    auto resultCompuon = std::make_shared<Compuon<double>>(0.0);
    resultCompuon->ciphertext_ = std::make_shared<seal::Ciphertext>(std::move(result));
    resultCompuon->isEncrypted_ = true;
    resultCompuon->setProfile(derivedA.getProfile());
    
    LOG_MESSAGE("BuiltinBackend: Performed homomorphic multiplication (double)");
    return resultCompuon;
  } else if (type == typeid(float)) {
    const Compuon<float> &derivedA = dynamic_cast<const Compuon<float> &>(a);
    [[maybe_unused]] const Compuon<float> &derivedB = dynamic_cast<const Compuon<float> &>(b);
    
    // Perform homomorphic multiplication
    seal::Ciphertext result = sealBackend_->multiply(*a.ciphertext_, *b.ciphertext_);
    
    // Create result Compuon
    auto resultCompuon = std::make_shared<Compuon<float>>(0.0f);
    resultCompuon->ciphertext_ = std::make_shared<seal::Ciphertext>(std::move(result));
    resultCompuon->isEncrypted_ = true;
    resultCompuon->setProfile(derivedA.getProfile());
    
    LOG_MESSAGE("BuiltinBackend: Performed homomorphic multiplication (float)");
    return resultCompuon;
  }
  
  return nullptr;
}

std::shared_ptr<CompuonBase> BuiltinBackend::addPlain(const CompuonBase &a, double scalar) {
  if (!a.isEncrypted_ || !a.ciphertext_) {
    throw std::runtime_error("BuiltinBackend: Cannot add plain to unencrypted Compuon value");
  }

  auto type = a.type();
  
  if (type == typeid(double)) {
    const Compuon<double> &derivedA = dynamic_cast<const Compuon<double> &>(a);
    
    // Perform homomorphic addition with plaintext
    seal::Ciphertext result = sealBackend_->addPlain(*a.ciphertext_, scalar);
    
    // Create result Compuon
    auto resultCompuon = std::make_shared<Compuon<double>>(0.0);
    resultCompuon->ciphertext_ = std::make_shared<seal::Ciphertext>(std::move(result));
    resultCompuon->isEncrypted_ = true;
    resultCompuon->setProfile(derivedA.getProfile());
    
    LOG_MESSAGE("BuiltinBackend: Performed homomorphic addition with plain " << scalar);
    return resultCompuon;
  }
  
  return nullptr;
}

std::shared_ptr<CompuonBase> BuiltinBackend::multiplyPlain(const CompuonBase &a, double scalar) {
  if (!a.isEncrypted_ || !a.ciphertext_) {
    throw std::runtime_error("BuiltinBackend: Cannot multiply plain with unencrypted Compuon value");
  }

  auto type = a.type();
  
  if (type == typeid(double)) {
    const Compuon<double> &derivedA = dynamic_cast<const Compuon<double> &>(a);
    
    // Perform homomorphic multiplication with plaintext
    seal::Ciphertext result = sealBackend_->multiplyPlain(*a.ciphertext_, scalar);
    
    // Create result Compuon
    auto resultCompuon = std::make_shared<Compuon<double>>(0.0);
    resultCompuon->ciphertext_ = std::make_shared<seal::Ciphertext>(std::move(result));
    resultCompuon->isEncrypted_ = true;
    resultCompuon->setProfile(derivedA.getProfile());
    
    LOG_MESSAGE("BuiltinBackend: Performed homomorphic multiplication with plain " << scalar);
    return resultCompuon;
  }
  
  return nullptr;
}

std::any BuiltinBackend::decrypt(const CompuonBase &entity) const {
  if (!entity.isEncrypted_ || !entity.ciphertext_) {
    // Return plain value if not encrypted
    auto type = entity.type();
    if (type == typeid(int)) {
      const Compuon<int> &derivedEntity = dynamic_cast<const Compuon<int> &>(entity);
      return derivedEntity.getValue();
    } else if (type == typeid(double)) {
      const Compuon<double> &derivedEntity = dynamic_cast<const Compuon<double> &>(entity);
      return derivedEntity.getValue();
    } else if (type == typeid(float)) {
      const Compuon<float> &derivedEntity = dynamic_cast<const Compuon<float> &>(entity);
      return derivedEntity.getValue();
    }
    return nullptr;
  }

  auto type = entity.type();

  if (type == typeid(int)) {
    int64_t decrypted = sealBackend_->decryptInt(*entity.ciphertext_);
    LOG_MESSAGE("BuiltinBackend: Decrypted int value " << decrypted);
    return static_cast<int>(decrypted);
  } else if (type == typeid(double)) {
    double decrypted = sealBackend_->decryptDouble(*entity.ciphertext_);
    LOG_MESSAGE("BuiltinBackend: Decrypted double value " << decrypted);
    return decrypted;
  } else if (type == typeid(float)) {
    double decrypted = sealBackend_->decryptDouble(*entity.ciphertext_);
    LOG_MESSAGE("BuiltinBackend: Decrypted float value " << static_cast<float>(decrypted));
    return static_cast<float>(decrypted);
  }

  return nullptr;
}
