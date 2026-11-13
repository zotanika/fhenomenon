#pragma once

#include <cstdint>
#include <memory>
#include <random>

namespace fhenomenon::toyfhe {

enum class Encoding { Integer, FixedPoint };

struct Parameters {
  int64_t q = static_cast<int64_t>(1) << 58;     // Ciphertext modulus (must be a multiple of t)
  int64_t t = static_cast<int64_t>(1) << 32;     // Plaintext modulus
  int64_t scale = static_cast<int64_t>(1) << 16; // Fixed-point scale base
  // Fresh-encryption noise bound. Multiplication noise grows with message
  // magnitude (|m1|*e2 + |m2|*e1 must stay below delta/2 = q/(2t) for exact
  // integer decryption). With these defaults the budget covers chained
  // products of three-digit integers even in the worst case; larger messages
  // degrade gracefully into off-by-small errors, as in a real leveled scheme
  // without modulus switching.
  int64_t noise_bound = static_cast<int64_t>(1) << 3;
};

struct Ciphertext {
  int64_t c0 = 0;
  int64_t c1 = 0;
  int scale_power = 0;
  Encoding encoding = Encoding::Integer;

  std::shared_ptr<Ciphertext> clone() const { return std::make_shared<Ciphertext>(*this); }
};

// ToyFHE is a deliberately insecure single-slot toy scheme. Relinearization
// and rescaling decode with the secret key, which the Engine holds. It exists
// to make FHN semantics observable from a fresh clone — scale/level
// bookkeeping, HMULT = mult + relin + rescale — not to provide security.
class Engine {
  public:
  Engine();

  void initialize(const Parameters &params);
  void generateKeys();

  bool isInitialized() const { return initialized_; }
  bool areKeysGenerated() const { return keysGenerated_; }

  Ciphertext encryptInt(int64_t value) const;
  Ciphertext encryptDouble(double value) const;

  Ciphertext add(const Ciphertext &lhs, const Ciphertext &rhs) const;
  Ciphertext multiply(const Ciphertext &lhs, const Ciphertext &rhs) const;
  Ciphertext addPlain(const Ciphertext &cipher, double scalar) const;
  Ciphertext multiplyPlain(const Ciphertext &cipher, double scalar) const;

  int64_t decryptInt(const Ciphertext &cipher) const;
  double decryptDouble(const Ciphertext &cipher) const;

  private:
  Ciphertext encryptEncoded(int64_t message, Encoding encoding, int scalePower) const;
  Ciphertext encodeRaw(int64_t value, Encoding encoding, int scalePower) const;
  Ciphertext multiplyDecoded(const Ciphertext &cipher, int64_t factor, int64_t divisor, Encoding encoding,
                             int scalePower) const;
  int64_t decodeRaw(const Ciphertext &cipher) const;
  Ciphertext alignScale(const Ciphertext &cipher, int targetScale) const;
  Ciphertext multiplyPlainInternal(const Ciphertext &cipher, int64_t scalar, int scalePowerIncrease) const;
  int64_t sampleUniform() const;
  int64_t sampleNoise() const;
  static int64_t centeredMod(int64_t value, int64_t modulus);
  int64_t delta() const { return params_.q / params_.t; }
  long double scaleFactor(int scalePower) const;

  Parameters params_{};
  bool initialized_;
  bool keysGenerated_;
  int64_t secretKey_;
  mutable std::mt19937_64 rng_;
};

} // namespace fhenomenon::toyfhe
