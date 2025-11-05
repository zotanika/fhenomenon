#pragma once

#include "Backend/Backend.h"
#include "Parameter/Parameter.h"
#include "seal/seal.h"

#include <memory>
#include <unordered_map>

namespace fhenomenon {

// Forward declarations
class CompuonBase;

/**
 * SEALBackend provides SEAL library integration for FHE operations.
 * This backend implements encryption, decryption, and homomorphic operations
 * using Microsoft SEAL with CKKS and BFV schemes.
 */
class SEALBackend {
public:
  SEALBackend();
  ~SEALBackend() = default;

  // Initialize with parameters
  void initialize(const Parameter &params);
  
  // Key management
  void generateKeys();
  void loadKeys(const std::string &publicKeyPath, const std::string &secretKeyPath);
  void saveKeys(const std::string &publicKeyPath, const std::string &secretKeyPath);

  // Encryption/Decryption
  seal::Ciphertext encryptDouble(double value);
  seal::Ciphertext encryptInt(int64_t value);
  double decryptDouble(const seal::Ciphertext &ciphertext) const;
  int64_t decryptInt(const seal::Ciphertext &ciphertext) const;

  // Homomorphic operations
  seal::Ciphertext add(const seal::Ciphertext &a, const seal::Ciphertext &b);
  seal::Ciphertext addPlain(const seal::Ciphertext &a, double value);
  seal::Ciphertext multiply(const seal::Ciphertext &a, const seal::Ciphertext &b);
  seal::Ciphertext multiplyPlain(const seal::Ciphertext &a, double value);
  seal::Ciphertext square(const seal::Ciphertext &a);

  // Accessors
  std::shared_ptr<seal::SEALContext> getContext() const { return context_; }
  const seal::PublicKey &getPublicKey() const { return publicKey_; }
  const seal::SecretKey &getSecretKey() const { return secretKey_; }
  const seal::Encryptor &getEncryptor() const { return *encryptor_; }
  const seal::Decryptor &getDecryptor() const { return *decryptor_; }
  const seal::Evaluator &getEvaluator() const { return *evaluator_; }

  bool isInitialized() const { return context_ != nullptr; }
  bool areKeysGenerated() const { return keysGenerated_; }

private:
  void setupCKKS(std::size_t polyModulusDegree, std::vector<int> coeffModulusBits);
  void setupBFV(std::size_t polyModulusDegree, std::vector<int> coeffModulusBits, int plainModulusBits);

  std::shared_ptr<seal::SEALContext> context_;
  seal::PublicKey publicKey_;
  seal::SecretKey secretKey_;
  seal::RelinKeys relinKeys_;
  
  std::unique_ptr<seal::Encryptor> encryptor_;
  std::unique_ptr<seal::Decryptor> decryptor_;
  std::unique_ptr<seal::Evaluator> evaluator_;
  std::unique_ptr<seal::CKKSEncoder> ckksEncoder_;
  std::unique_ptr<seal::BatchEncoder> bfvEncoder_;

  bool keysGenerated_;
  bool useCKKS_;
  double scale_;
};

} // namespace fhenomenon

