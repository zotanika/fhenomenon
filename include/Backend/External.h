#pragma once

#include "Backend/Backend.h"

// (FIXME) using Microsoft SEAL
namespace seal {

class Ciphertext {};

} // namespace seal

namespace fhenomenon {

class ExternalBackend : public Backend {
  public:
  BackendType getBackendType() const override { return BackendType::ExternalBackend; }

  ExternalBackend([[maybe_unused]] const std::string &libraryPath) {
    // Load the external library and initialize the backend
  }

  void transform([[maybe_unused]] CompuonBase &entity, [[maybe_unused]] const Parameter &params) override {
    // Transform the entity using the external backend
  }

  std::shared_ptr<CompuonBase> add([[maybe_unused]] const CompuonBase &a, [[maybe_unused]] const CompuonBase &b) override {
    // Add entities using the external backend
    // return Compuon<int>(0);  // Placeholder
    return nullptr;
  }

  std::shared_ptr<CompuonBase> multiply([[maybe_unused]] const CompuonBase &a,
                                       [[maybe_unused]] const CompuonBase &b) override {
    // Multiply entities using the external backend
    // return Compuon<int>(0);  // Placeholder
    return nullptr;
  }

  std::any decrypt([[maybe_unused]] const CompuonBase &entity) const override {
    // Decrypt entity using the external backend
    return 0; // Placeholder
  }

  /*
  Compuon<seal::Ciphertext> add(const Compuon<seal::Ciphertext>& a, const
  Compuon<seal::Ciphertext>& b) {
      // Add ciphertexts using the external backend
      return Compuon<seal::Ciphertext>(*a.encrypted_data);  // Placeholder
  }

  Compuon<seal::Ciphertext> multiply(const Compuon<seal::Ciphertext>& a, const
  Compuon<seal::Ciphertext>& b) {
      // Multiply ciphertexts using the external backend
      return Compuon<seal::Ciphertext>(*a.encrypted_data);  // Placeholder
  }
  */
};

} // namespace fhenomenon
