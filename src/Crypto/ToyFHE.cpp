#include "Crypto/ToyFHE.h"
#include "Utils/log.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fhenomenon::toyfhe {

namespace {
int64_t mod(int64_t value, int64_t modulus) {
  int64_t result = value % modulus;
  if (result < 0) {
    result += modulus;
  }
  return result;
}

#if defined(__SIZEOF_INT128__)
__extension__ int64_t mulMod(int64_t lhs, int64_t rhs, int64_t modulus) {
  using wide_int = __int128;
  wide_int wide = static_cast<wide_int>(lhs) * static_cast<wide_int>(rhs);
  int64_t result = static_cast<int64_t>(wide % modulus);
  if (result < 0) {
    result += modulus;
  }
  return result;
}
#else
int64_t mulMod(int64_t lhs, int64_t rhs, int64_t modulus) {
  lhs = mod(lhs, modulus);
  rhs = mod(rhs, modulus);
  int64_t result = 0;
  while (rhs > 0) {
    if (rhs & 1) {
      result = mod(result + lhs, modulus);
    }
    lhs = mod(lhs + lhs, modulus);
    rhs >>= 1;
  }
  return result;
}
#endif

bool isApproximatelyInteger(double scalar, long double tolerance = 1e-9L) {
  const long double scalar_ld = static_cast<long double>(scalar);
  const long double rounded_ld = static_cast<long double>(std::llround(scalar));
  return std::fabs(scalar_ld - rounded_ld) <= tolerance;
}
} // namespace

Engine::Engine()
  : initialized_(false), keysGenerated_(false), secretKey_(0), rng_(std::random_device{}()) {}

void Engine::initialize(const Parameters &params) {
  if (params.q <= params.t) {
    throw std::runtime_error("ToyFHE: ciphertext modulus q must be larger than plaintext modulus t");
  }
  if (params.scale <= 1) {
    throw std::runtime_error("ToyFHE: scale must be greater than 1");
  }
  params_ = params;
  initialized_ = true;
}

void Engine::generateKeys() {
  if (!initialized_) {
    throw std::runtime_error("ToyFHE: call initialize() before generateKeys()");
  }
  std::uniform_int_distribution<int64_t> dist(1, params_.t - 1);
  secretKey_ = dist(rng_);
  keysGenerated_ = true;
  LOG_MESSAGE("ToyFHE: Generated secret key " << secretKey_);
}

Ciphertext Engine::encryptEncoded(int64_t message, Encoding encoding, int scalePower) const {
  if (!keysGenerated_) {
    throw std::runtime_error("ToyFHE: keys not generated");
  }

  const int64_t plain = mod(message, params_.t);
  const int64_t scaled = mulMod(delta(), plain, params_.q);
  const int64_t a = sampleUniform();
  const int64_t e = sampleNoise();

  Ciphertext ciphertext;
  ciphertext.c1 = a;
  const int64_t secretProd = mulMod(a, secretKey_, params_.q);
  ciphertext.c0 = mod(scaled - secretProd + e, params_.q);
  ciphertext.scale_power = scalePower;
  ciphertext.encoding = encoding;
  return ciphertext;
}

Ciphertext Engine::encryptInt(int64_t value) const {
  return encryptEncoded(value, Encoding::Integer, 0);
}

Ciphertext Engine::encryptDouble(double value) const {
  const long double scaled = static_cast<long double>(value) * scaleFactor(1);
  const int64_t encoded = static_cast<int64_t>(std::llround(scaled));
  return encryptEncoded(encoded, Encoding::FixedPoint, 1);
}

Ciphertext Engine::alignScale(const Ciphertext &cipher, int targetScale) const {
  Ciphertext current = cipher;
  while (current.scale_power < targetScale) {
    current = multiplyPlainInternal(current, params_.scale, 1);
  }
  return current;
}

Ciphertext Engine::add(const Ciphertext &lhs, const Ciphertext &rhs) const {
  if (!keysGenerated_) {
    throw std::runtime_error("ToyFHE: keys not generated");
  }

  const int targetScale = std::max(lhs.scale_power, rhs.scale_power);
  Ciphertext left = alignScale(lhs, targetScale);
  Ciphertext right = alignScale(rhs, targetScale);

  Ciphertext result;
  result.c0 = mod(left.c0 + right.c0, params_.q);
  result.c1 = mod(left.c1 + right.c1, params_.q);
  result.scale_power = targetScale;
  result.encoding =
    (left.encoding == Encoding::FixedPoint || right.encoding == Encoding::FixedPoint) ? Encoding::FixedPoint
                                                                                      : Encoding::Integer;
  return result;
}

Ciphertext Engine::multiply(const Ciphertext &lhs, const Ciphertext &rhs) const {
  if (!keysGenerated_) {
    throw std::runtime_error("ToyFHE: keys not generated");
  }

  const int64_t c0 = mulMod(lhs.c0, rhs.c0, params_.q);
  const int64_t term1 = mulMod(lhs.c0, rhs.c1, params_.q);
  const int64_t term2 = mulMod(lhs.c1, rhs.c0, params_.q);
  const int64_t cross = mod(term1 + term2, params_.q);
  const int64_t c2 = mulMod(lhs.c1, rhs.c1, params_.q);

  Ciphertext result;
  result.c0 = c0;
  const int64_t relin = mulMod(c2, secretKey_, params_.q);
  result.c1 = mod(cross + relin, params_.q);
  result.scale_power = lhs.scale_power + rhs.scale_power;
  result.encoding =
    (lhs.encoding == Encoding::FixedPoint || rhs.encoding == Encoding::FixedPoint) ? Encoding::FixedPoint
                                                                                   : Encoding::Integer;
  return result;
}

Ciphertext Engine::multiplyPlainInternal(const Ciphertext &cipher, int64_t scalar, int scalePowerIncrease) const {
  Ciphertext result;
  result.c0 = mulMod(cipher.c0, scalar, params_.q);
  result.c1 = mulMod(cipher.c1, scalar, params_.q);
  result.scale_power = cipher.scale_power + scalePowerIncrease;
  result.encoding =
    (scalePowerIncrease > 0 || cipher.encoding == Encoding::FixedPoint) ? Encoding::FixedPoint : Encoding::Integer;
  return result;
}

Ciphertext Engine::addPlain(const Ciphertext &cipher, double scalar) const {
  if (!keysGenerated_) {
    throw std::runtime_error("ToyFHE: keys not generated");
  }

  Ciphertext result = cipher;
  const long double factor = scaleFactor(cipher.scale_power);
  const int64_t encodedScalar = static_cast<int64_t>(std::llround(static_cast<long double>(scalar) * factor));
  const int64_t plain = mod(encodedScalar, params_.t);
  const int64_t scaled = mulMod(delta(), plain, params_.q);
  result.c0 = mod(result.c0 + scaled, params_.q);
  if (cipher.encoding == Encoding::Integer && !isApproximatelyInteger(scalar)) {
    result.encoding = Encoding::FixedPoint;
  }
  return result;
}

Ciphertext Engine::multiplyPlain(const Ciphertext &cipher, double scalar) const {
  if (!keysGenerated_) {
    throw std::runtime_error("ToyFHE: keys not generated");
  }

  const bool scalarIsInteger = isApproximatelyInteger(scalar);

  if (scalarIsInteger && cipher.encoding == Encoding::Integer) {
    const int64_t factor = static_cast<int64_t>(std::llround(scalar));
    return multiplyPlainInternal(cipher, factor, 0);
  }

  const long double scaledScalar = static_cast<long double>(scalar) * scaleFactor(1);
  const int64_t encoded = static_cast<int64_t>(std::llround(scaledScalar));
  return multiplyPlainInternal(cipher, encoded, 1);
}

int64_t Engine::decodeRaw(const Ciphertext &cipher) const {
  const int64_t prod = mulMod(cipher.c1, secretKey_, params_.q);
  return mod(cipher.c0 + prod, params_.q);
}

int64_t Engine::decryptInt(const Ciphertext &cipher) const {
  if (!keysGenerated_) {
    throw std::runtime_error("ToyFHE: keys not generated");
  }

  const long double scaled = static_cast<long double>(decodeRaw(cipher)) / static_cast<long double>(delta());
  const int64_t rounded = static_cast<int64_t>(std::llround(scaled));
  return centeredMod(rounded, params_.t);
}

double Engine::decryptDouble(const Ciphertext &cipher) const {
  if (!keysGenerated_) {
    throw std::runtime_error("ToyFHE: keys not generated");
  }

  const int64_t decoded = decryptInt(cipher);
  const long double factor = scaleFactor(cipher.scale_power);
  return static_cast<double>(static_cast<long double>(decoded) / factor);
}

int64_t Engine::sampleUniform() const {
  std::uniform_int_distribution<int64_t> dist(0, params_.q - 1);
  return dist(rng_);
}

int64_t Engine::sampleNoise() const {
  std::uniform_int_distribution<int64_t> dist(-params_.noise_bound, params_.noise_bound);
  return dist(rng_);
}

int64_t Engine::centeredMod(int64_t value, int64_t modulus) {
  int64_t result = value % modulus;
  if (result < 0) {
    result += modulus;
  }
  if (result > modulus / 2) {
    result -= modulus;
  }
  return result;
}

long double Engine::scaleFactor(int scalePower) const {
  long double factor = 1.0L;
  for (int i = 0; i < scalePower; ++i) {
    factor *= static_cast<long double>(params_.scale);
  }
  return factor;
}

} // namespace fhenomenon::toyfhe
