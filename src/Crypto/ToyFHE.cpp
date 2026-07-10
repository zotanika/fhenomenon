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

Engine::Engine() : initialized_(false), keysGenerated_(false), secretKey_(0), rng_(std::random_device{}()) {}

void Engine::initialize(const Parameters &params) {
  if (params.q <= params.t) {
    throw std::runtime_error("ToyFHE: ciphertext modulus q must be larger than plaintext modulus t");
  }
  if (params.q % params.t != 0) {
    // Rescaling and decryption rely on delta*(m + k*t) == delta*m + k*q,
    // which requires q to be an exact multiple of t.
    throw std::runtime_error("ToyFHE: ciphertext modulus q must be a multiple of plaintext modulus t");
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

// Wrap an already delta-scaled value (in [0, q)) into a fresh ciphertext.
Ciphertext Engine::encodeRaw(int64_t value, Encoding encoding, int scalePower) const {
  const int64_t a = sampleUniform();
  const int64_t e = sampleNoise();

  Ciphertext ciphertext;
  ciphertext.c1 = a;
  const int64_t secretProd = mulMod(a, secretKey_, params_.q);
  ciphertext.c0 = mod(value - secretProd + e, params_.q);
  ciphertext.scale_power = scalePower;
  ciphertext.encoding = encoding;
  return ciphertext;
}

Ciphertext Engine::encryptEncoded(int64_t message, Encoding encoding, int scalePower) const {
  if (!keysGenerated_) {
    throw std::runtime_error("ToyFHE: keys not generated");
  }

  const int64_t plain = mod(message, params_.t);
  return encodeRaw(mulMod(delta(), plain, params_.q), encoding, scalePower);
}

Ciphertext Engine::encryptInt(int64_t value) const { return encryptEncoded(value, Encoding::Integer, 0); }

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
  result.encoding = (left.encoding == Encoding::FixedPoint || right.encoding == Encoding::FixedPoint)
                      ? Encoding::FixedPoint
                      : Encoding::Integer;
  return result;
}

// Core of ciphertext-ciphertext and ciphertext-scalar multiplication: decode
// the ciphertext, multiply by `factor`, divide by `divisor` rounding to
// nearest, and re-encode with a fresh mask. The 128-bit intermediate never
// wraps, so plaintext range constraints apply only to the final product.
// Decoding uses the secret key — the same toy-level cheat as the implicit
// relinearization this replaces; see the Engine comment in ToyFHE.h.
Ciphertext Engine::multiplyDecoded(const Ciphertext &cipher, int64_t factor, int64_t divisor, Encoding encoding,
                                   int scalePower) const {
  const int64_t p = centeredMod(decodeRaw(cipher), params_.q);

#if defined(__SIZEOF_INT128__)
  __extension__ using wide_int = __int128;
  const wide_int product = static_cast<wide_int>(p) * factor;
  wide_int scaled = product / divisor;
  const wide_int remainder = product % divisor;
  if (2 * remainder >= divisor) {
    ++scaled;
  } else if (2 * remainder <= -divisor) {
    --scaled;
  }
  // Reducing mod q after the division also folds the plaintext product back
  // into [0, t): delta * (m + k*t) = delta*m + k*q.
  const wide_int q_wide = params_.q;
  const int64_t value = static_cast<int64_t>(((scaled % q_wide) + q_wide) % q_wide);
#else
  // Without 128-bit integers, approximate with long double. Rounding error
  // grows with the product magnitude; small toy messages still decrypt
  // correctly.
  const long double approx =
    static_cast<long double>(p) * static_cast<long double>(factor) / static_cast<long double>(divisor);
  const long double reduced = std::fmod(approx, static_cast<long double>(params_.q));
  const int64_t value = mod(static_cast<int64_t>(std::llround(reduced)), params_.q);
#endif

  return encodeRaw(value, encoding, scalePower);
}

Ciphertext Engine::multiply(const Ciphertext &lhs, const Ciphertext &rhs) const {
  if (!keysGenerated_) {
    throw std::runtime_error("ToyFHE: keys not generated");
  }

  // Tensor product of the decoded values (delta*m + e each), divided once by
  // delta — the BFV rescale that keeps the result at delta*m — and, for
  // fixed-point operands, by `scale` until the mantissa is back at a single
  // scale level (the rescale-after-multiply discipline of CKKS). Every public
  // path keeps scale_power <= 1, so the divisor fits in 64 bits.
  const int64_t p2 = centeredMod(decodeRaw(rhs), params_.q);

  const Encoding encoding = (lhs.encoding == Encoding::FixedPoint || rhs.encoding == Encoding::FixedPoint)
                              ? Encoding::FixedPoint
                              : Encoding::Integer;

  int scalePower = lhs.scale_power + rhs.scale_power;
  int64_t divisor = delta();
  while (encoding == Encoding::FixedPoint && scalePower > 1) {
    divisor *= params_.scale;
    --scalePower;
  }

  return multiplyDecoded(lhs, p2, divisor, encoding, scalePower);
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

  // An integer-encoded ciphertext cannot absorb a fractional scalar at
  // scale_power 0 (the scalar would round to an integer); raise the
  // ciphertext to one scale level first.
  Ciphertext result = cipher;
  if (cipher.encoding == Encoding::Integer && !isApproximatelyInteger(scalar)) {
    result = alignScale(cipher, 1);
  }

  const long double factor = scaleFactor(result.scale_power);
  const int64_t encodedScalar = static_cast<int64_t>(std::llround(static_cast<long double>(scalar) * factor));
  const int64_t plain = mod(encodedScalar, params_.t);
  const int64_t scaled = mulMod(delta(), plain, params_.q);
  result.c0 = mod(result.c0 + scaled, params_.q);
  return result;
}

Ciphertext Engine::multiplyPlain(const Ciphertext &cipher, double scalar) const {
  if (!keysGenerated_) {
    throw std::runtime_error("ToyFHE: keys not generated");
  }

  // Integral scalars never consume a scale level, regardless of the
  // ciphertext encoding: the encoded mantissa is simply multiplied.
  if (isApproximatelyInteger(scalar)) {
    const int64_t factor = static_cast<int64_t>(std::llround(scalar));
    return multiplyPlainInternal(cipher, factor, 0);
  }

  // Fractional scalar: multiply the mantissa by the scale-encoded scalar and
  // rescale in the same exact step, so the scale^2 intermediate never
  // materializes (it would wrap mod t for any |value * scalar| >= 0.5).
  // Fixed-point inputs keep their scale level; integer inputs move to
  // fixed-point at one scale level.
  const long double scaledScalar = static_cast<long double>(scalar) * scaleFactor(1);
  const int64_t encoded = static_cast<int64_t>(std::llround(scaledScalar));
  if (cipher.scale_power >= 1) {
    return multiplyDecoded(cipher, encoded, params_.scale, Encoding::FixedPoint, cipher.scale_power);
  }
  return multiplyDecoded(cipher, encoded, 1, Encoding::FixedPoint, cipher.scale_power + 1);
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
