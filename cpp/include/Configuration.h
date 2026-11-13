#pragma once

#include "Common.h"

#include "nlohmann/json.hpp"

#include <fstream>
#include <string>

namespace fhenomenon {

class Configuration {
  private:
  std::string publicKeyPath;
  std::string secretKeyPath;

  public:
  Configuration(std::string_view configPath) {
    nlohmann::json config;
    std::ifstream configFile(configPath.data());
    configFile >> config;

    publicKeyPath = config["public_key"];
    secretKeyPath = config["secret_key"];
  }

  std::string_view getPublicKeyPath() const { return publicKeyPath; }
  std::string_view getSecretKeyPath() const { return secretKeyPath; }
};

} // namespace fhenomenon
