#include "Backend/SEALBackend.h"
#include "Parameter/CKKSParameter.h"
#include "Utils/log.h"

#include <fstream>
#include <stdexcept>

namespace fhenomenon {

SEALBackend::SEALBackend() 
  : keysGenerated_(false), useCKKS_(true), scale_(pow(2.0, 40)) {
}

void SEALBackend::initialize(const Parameter &params) {
  // Try to cast to CKKSParameter to get SEAL-specific parameters
  const auto *ckksParams = dynamic_cast<const CKKSParameter *>(&params);
  
  if (ckksParams) {
    // Use CKKS scheme
    useCKKS_ = true;
    // Default CKKS parameters: poly_modulus_degree = 8192, coeff_modulus = [60, 40, 40, 60]
    setupCKKS(8192, {60, 40, 40, 60});
  } else {
    // Fall back to BFV for integer operations
    useCKKS_ = false;
    // Default BFV parameters
    setupBFV(4096, {36, 36, 36}, 20);
  }
}

void SEALBackend::setupCKKS(std::size_t polyModulusDegree, std::vector<int> coeffModulusBits) {
  seal::EncryptionParameters parms(seal::scheme_type::ckks);
  parms.set_poly_modulus_degree(polyModulusDegree);
  
  // Create coefficient modulus
  parms.set_coeff_modulus(seal::CoeffModulus::Create(polyModulusDegree, coeffModulusBits));
  
  context_ = std::make_shared<seal::SEALContext>(parms);
  
  // Verify context is valid
  if (!context_->parameters_set()) {
    throw std::runtime_error("Invalid SEAL context parameters for CKKS");
  }
  
  // Create encoder and evaluator
  ckksEncoder_ = std::make_unique<seal::CKKSEncoder>(*context_);
  evaluator_ = std::make_unique<seal::Evaluator>(*context_);
  
  // Set scale for CKKS
  scale_ = pow(2.0, 40);
  
  LOG_MESSAGE("SEALBackend: Initialized CKKS with poly_modulus_degree=" << polyModulusDegree);
}

void SEALBackend::setupBFV(std::size_t polyModulusDegree, std::vector<int> coeffModulusBits, int plainModulusBits) {
  seal::EncryptionParameters parms(seal::scheme_type::bfv);
  parms.set_poly_modulus_degree(polyModulusDegree);
  parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(polyModulusDegree));
  parms.set_plain_modulus(seal::PlainModulus::Batching(polyModulusDegree, plainModulusBits));
  
  context_ = std::make_shared<seal::SEALContext>(parms);
  
  if (!context_->parameters_set()) {
    throw std::runtime_error("Invalid SEAL context parameters for BFV");
  }
  
  bfvEncoder_ = std::make_unique<seal::BatchEncoder>(*context_);
  evaluator_ = std::make_unique<seal::Evaluator>(*context_);
  
  LOG_MESSAGE("SEALBackend: Initialized BFV with poly_modulus_degree=" << polyModulusDegree);
}

void SEALBackend::generateKeys() {
  if (!context_) {
    throw std::runtime_error("SEALBackend: Context not initialized. Call initialize() first.");
  }
  
  seal::KeyGenerator keygen(*context_);
  publicKey_ = keygen.create_public_key();
  secretKey_ = keygen.secret_key();
  
  // Generate relinearization keys for multiplication
  relinKeys_ = keygen.create_relin_keys();
  
  // Create encryptor, decryptor
  encryptor_ = std::make_unique<seal::Encryptor>(*context_, publicKey_);
  decryptor_ = std::make_unique<seal::Decryptor>(*context_, secretKey_);
  
  keysGenerated_ = true;
  
  LOG_MESSAGE("SEALBackend: Keys generated successfully");
}

void SEALBackend::loadKeys(const std::string &publicKeyPath, const std::string &secretKeyPath) {
  if (!context_) {
    throw std::runtime_error("SEALBackend: Context not initialized. Call initialize() first.");
  }
  
  // Load public key
  std::ifstream pkFile(publicKeyPath, std::ios::binary);
  if (!pkFile.is_open()) {
    throw std::runtime_error("SEALBackend: Cannot open public key file: " + publicKeyPath);
  }
  publicKey_.load(*context_, pkFile);
  pkFile.close();
  
  // Load secret key
  std::ifstream skFile(secretKeyPath, std::ios::binary);
  if (!skFile.is_open()) {
    throw std::runtime_error("SEALBackend: Cannot open secret key file: " + secretKeyPath);
  }
  secretKey_.load(*context_, skFile);
  skFile.close();
  
  // Create encryptor and decryptor
  encryptor_ = std::make_unique<seal::Encryptor>(*context_, publicKey_);
  decryptor_ = std::make_unique<seal::Decryptor>(*context_, secretKey_);
  
  keysGenerated_ = true;
  
  LOG_MESSAGE("SEALBackend: Keys loaded from " << publicKeyPath << " and " << secretKeyPath);
}

void SEALBackend::saveKeys(const std::string &publicKeyPath, const std::string &secretKeyPath) {
  if (!keysGenerated_) {
    throw std::runtime_error("SEALBackend: Keys not generated. Call generateKeys() first.");
  }
  
  // Save public key
  std::ofstream pkFile(publicKeyPath, std::ios::binary);
  if (!pkFile.is_open()) {
    throw std::runtime_error("SEALBackend: Cannot create public key file: " + publicKeyPath);
  }
  publicKey_.save(pkFile);
  pkFile.close();
  
  // Save secret key
  std::ofstream skFile(secretKeyPath, std::ios::binary);
  if (!skFile.is_open()) {
    throw std::runtime_error("SEALBackend: Cannot create secret key file: " + secretKeyPath);
  }
  secretKey_.save(skFile);
  skFile.close();
  
  LOG_MESSAGE("SEALBackend: Keys saved to " << publicKeyPath << " and " << secretKeyPath);
}

seal::Ciphertext SEALBackend::encryptDouble(double value) {
  if (!encryptor_ || !ckksEncoder_) {
    throw std::runtime_error("SEALBackend: Not initialized for CKKS encryption");
  }
  
  seal::Plaintext plaintext;
  ckksEncoder_->encode(value, scale_, plaintext);
  
  seal::Ciphertext ciphertext;
  encryptor_->encrypt(plaintext, ciphertext);
  
  return ciphertext;
}

seal::Ciphertext SEALBackend::encryptInt(int64_t value) {
  if (!encryptor_ || !bfvEncoder_) {
    throw std::runtime_error("SEALBackend: Not initialized for BFV encryption");
  }
  
  std::vector<int64_t> pod_matrix(bfvEncoder_->slot_count(), 0);
  pod_matrix[0] = value;
  
  seal::Plaintext plaintext;
  bfvEncoder_->encode(pod_matrix, plaintext);
  
  seal::Ciphertext ciphertext;
  encryptor_->encrypt(plaintext, ciphertext);
  
  return ciphertext;
}

double SEALBackend::decryptDouble(const seal::Ciphertext &ciphertext) const {
  if (!decryptor_ || !ckksEncoder_) {
    throw std::runtime_error("SEALBackend: Not initialized for CKKS decryption");
  }
  
  seal::Plaintext plaintext;
  decryptor_->decrypt(ciphertext, plaintext);
  
  std::vector<double> result;
  ckksEncoder_->decode(plaintext, result);
  
  return result[0];
}

int64_t SEALBackend::decryptInt(const seal::Ciphertext &ciphertext) const {
  if (!decryptor_ || !bfvEncoder_) {
    throw std::runtime_error("SEALBackend: Not initialized for BFV decryption");
  }
  
  seal::Plaintext plaintext;
  decryptor_->decrypt(ciphertext, plaintext);
  
  std::vector<int64_t> result;
  bfvEncoder_->decode(plaintext, result);
  
  return result[0];
}

seal::Ciphertext SEALBackend::add(const seal::Ciphertext &a, const seal::Ciphertext &b) {
  if (!evaluator_) {
    throw std::runtime_error("SEALBackend: Evaluator not initialized");
  }
  
  seal::Ciphertext result;
  evaluator_->add(a, b, result);
  return result;
}

seal::Ciphertext SEALBackend::addPlain(const seal::Ciphertext &a, double value) {
  if (!evaluator_ || !ckksEncoder_) {
    throw std::runtime_error("SEALBackend: Not initialized for CKKS operations");
  }
  
  seal::Plaintext plaintext;
  ckksEncoder_->encode(value, scale_, plaintext);
  
  seal::Ciphertext result;
  evaluator_->add_plain(a, plaintext, result);
  return result;
}

seal::Ciphertext SEALBackend::multiply(const seal::Ciphertext &a, const seal::Ciphertext &b) {
  if (!evaluator_) {
    throw std::runtime_error("SEALBackend: Evaluator not initialized");
  }
  
  seal::Ciphertext result;
  evaluator_->multiply(a, b, result);
  
  // Relinearize to reduce ciphertext size
  evaluator_->relinearize_inplace(result, relinKeys_);
  
  // Rescale for CKKS
  if (useCKKS_) {
    evaluator_->rescale_to_next_inplace(result);
  }
  
  return result;
}

seal::Ciphertext SEALBackend::multiplyPlain(const seal::Ciphertext &a, double value) {
  if (!evaluator_ || !ckksEncoder_) {
    throw std::runtime_error("SEALBackend: Not initialized for CKKS operations");
  }
  
  seal::Plaintext plaintext;
  ckksEncoder_->encode(value, scale_, plaintext);
  
  seal::Ciphertext result;
  evaluator_->multiply_plain(a, plaintext, result);
  
  // Rescale for CKKS
  if (useCKKS_) {
    evaluator_->rescale_to_next_inplace(result);
  }
  
  return result;
}

seal::Ciphertext SEALBackend::square(const seal::Ciphertext &a) {
  if (!evaluator_) {
    throw std::runtime_error("SEALBackend: Evaluator not initialized");
  }
  
  seal::Ciphertext result;
  evaluator_->square(a, result);
  
  // Relinearize
  evaluator_->relinearize_inplace(result, relinKeys_);
  
  // Rescale for CKKS
  if (useCKKS_) {
    evaluator_->rescale_to_next_inplace(result);
  }
  
  return result;
}

} // namespace fhenomenon

