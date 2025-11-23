#pragma once

#include <cstdint>
#include <memory>
#include <random>

namespace fhenomenon::toyfhe {

enum class Encoding { Integer, FixedPoint };

struct Parameters {
  int64_t q = static_cast<int64_t>(1) << 55;     // Ciphertext modulus
  int64_t t = static_cast<int64_t>(1) << 40;     // Plaintext modulus
  int64_t scale = static_cast<int64_t>(1) << 20; // Fixed-point scale base
  int64_t noise_bound = static_cast<int64_t>(1) << 8;
};

struct Ciphertext {
  int64_t c0 = 0;
  int64_t c1 = 0;
  int scale_power = 0;
  Encoding encoding = Encoding::Integer;

  std::shared_ptr<Ciphertext> clone() const { return std::make_shared<Ciphertext>(*this); }
};

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
