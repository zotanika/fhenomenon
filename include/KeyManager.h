#pragma once

#include "Configuration.h"

// (FIXME) using Microsoft SEAL
namespace seal {

class PublicKey {};
class SecretKey {};
class SEALContext {};

} // namespace seal

namespace fhenomenon {

class KeyManager {
  private:
  void loadKeys(const Configuration &config) {
    std::ifstream pkFile(config.getPublicKeyPath().data(), std::ios::binary);
    // publicKey.load(*context, pkFile);

    std::ifstream skFile(config.getSecretKeyPath().data(), std::ios::binary);
    // secretKey.load(*context, skFile);

    keysLoaded_ = true;
  }

  void setCtxt([[maybe_unused]] const std::shared_ptr<seal::SEALContext> &ctx) {
    std::cout << "(TODO) KeyManager::setCtxt" << std::endl;
  }

  seal::PublicKey publicKey_;
  seal::SecretKey secretKey_;
  bool keysLoaded_;

  public:
  KeyManager() {}
  KeyManager(const std::shared_ptr<seal::SEALContext> &ctx, const Configuration &config) : keysLoaded_(false) {
    setCtxt(ctx);
    loadKeys(config);
  }

  const seal::PublicKey &getPublicKey() const { return publicKey_; }
  const seal::SecretKey &getSecretKey() const { return secretKey_; }

  bool areKeysLoaded() const { return keysLoaded_; }
};

} // namespace fhenomenon
