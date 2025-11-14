#pragma once

#include "Configuration.h"

// Suppress warnings from SEAL library headers
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wduplicated-branches"
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif

#include "seal/seal.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <fstream>
#include <memory>

namespace fhenomenon {

class KeyManager {
  private:
  void loadKeys(const Configuration &config) {
    if (!context_) {
      throw std::runtime_error("KeyManager: Context not set. Cannot load keys.");
    }

    std::ifstream pkFile(config.getPublicKeyPath().data(), std::ios::binary);
    if (!pkFile.is_open()) {
      throw std::runtime_error(std::string("KeyManager: Cannot open public key file: ") + std::string(config.getPublicKeyPath()));
    }
    publicKey_.load(*context_, pkFile);
    pkFile.close();

    std::ifstream skFile(config.getSecretKeyPath().data(), std::ios::binary);
    if (!skFile.is_open()) {
      throw std::runtime_error(std::string("KeyManager: Cannot open secret key file: ") + std::string(config.getSecretKeyPath()));
    }
    secretKey_.load(*context_, skFile);
    skFile.close();

    keysLoaded_ = true;
  }

  void setCtxt(const std::shared_ptr<seal::SEALContext> &ctx) {
    context_ = ctx;
  }

  std::shared_ptr<seal::SEALContext> context_;
  seal::PublicKey publicKey_;
  seal::SecretKey secretKey_;
  bool keysLoaded_;

  public:
  KeyManager() : keysLoaded_(false) {}
  KeyManager(const std::shared_ptr<seal::SEALContext> &ctx, const Configuration &config) 
    : keysLoaded_(false) {
    setCtxt(ctx);
    loadKeys(config);
  }

  void setContext(const std::shared_ptr<seal::SEALContext> &ctx) {
    setCtxt(ctx);
  }

  const seal::PublicKey &getPublicKey() const { return publicKey_; }
  const seal::SecretKey &getSecretKey() const { return secretKey_; }

  bool areKeysLoaded() const { return keysLoaded_; }
  std::shared_ptr<seal::SEALContext> getContext() const { return context_; }
};

} // namespace fhenomenon
