#pragma once

#include "Configuration.h"

#include <string>

namespace fhenomenon {

class KeyManager {
  public:
  KeyManager() = default;
  explicit KeyManager(const Configuration &config) { load(config); }

  void load(const Configuration &config) {
    publicKeyPath_ = config.getPublicKeyPath();
    secretKeyPath_ = config.getSecretKeyPath();
    keysLoaded_ = true;
  }

  bool areKeysLoaded() const { return keysLoaded_; }
  const std::string &getPublicKeyPath() const { return publicKeyPath_; }
  const std::string &getSecretKeyPath() const { return secretKeyPath_; }

  private:
  bool keysLoaded_ = false;
  std::string publicKeyPath_;
  std::string secretKeyPath_;
};

} // namespace fhenomenon
