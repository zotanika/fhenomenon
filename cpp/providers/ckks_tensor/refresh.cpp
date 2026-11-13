#include "refresh.hpp"
#include "transforms.hpp"
#include <cmath>
#include <functional>
#include <stdexcept>

namespace fhenomenon::tensor_ckks {
namespace {
Value polynomial(Engine &engine, const Value &input, const std::vector<std::complex<double>> &coefficients) {
  // Balanced powers limit multiplication depth to ceil(log2 degree). Each
  // nonconstant coefficient consumes one further plaintext multiplication.
  std::vector<Value> powers(coefficients.size());
  powers.at(1) = input;
  std::function<Value(unsigned)> power = [&](unsigned n) -> Value {
    if (powers[n])
      return powers[n];
    const unsigned left = n / 2, right = n - left;
    powers[n] = engine.mul(power(left), power(right));
    return powers[n];
  };
  Value result;
  for (unsigned k = 1; k < coefficients.size(); ++k) {
    if (coefficients[k] == std::complex<double>{})
      continue;
    auto term = engine.plain(power(k), {coefficients[k]}, true);
    result = result ? engine.add(result, term) : term;
  }
  if (!result)
    throw std::invalid_argument("refresh polynomial has no variable term");
  if (coefficients[0] != std::complex<double>{})
    result = engine.plain(result, {coefficients[0]}, false);
  return result;
}

Value reduceChannel(Engine &engine, const Value &twice_channel, unsigned squares) {
  // twice_channel = 2*t. This one multiplication produces x=t/2^(d-2),
  // whose magnitude is at most one under the deterministic wrap bound.
  // Exp(i*pi*x/2) then needs d squarings to produce exp(2*pi*i*t).
  auto normalized = engine.plain(twice_channel, {std::ldexp(1.0, 1 - int(squares))}, true);
  auto exponential = polynomial(engine, normalized, refreshExponentialCoefficients());
  for (unsigned k = 0; k < squares; ++k)
    exponential = engine.mul(exponential, exponential);
  // -i*(z-conj(z)) = 2*sin(2*pi*t). The factor two is absorbed in the
  // arcsine coefficients, avoiding an additional plaintext scale level.
  auto twice_sine = engine.neg(engine.multiplyI(engine.sub(exponential, engine.conjugate(exponential))));
  return polynomial(engine, twice_sine, refreshArcsineCoefficients());
}
} // namespace

unsigned refreshSquarings(unsigned log_degree) {
  if (log_degree < 2 || log_degree > 30)
    throw std::invalid_argument("unsupported refresh degree");
  return log_degree + 2;
}
unsigned refreshDepth(unsigned log_degree) { return refreshSquarings(log_degree) + 14; }

std::vector<std::complex<double>> refreshExponentialCoefficients() {
  // Half-angle coefficients are <=pi/2, permitting a 62-bit working scale
  // in the signed 64-bit plaintext encoder. One further square compensates.
  const std::complex<double> imaginary_pi(0, std::acos(-1.0) / 2);
  std::vector<std::complex<double>> result(32);
  result[0] = 1;
  for (unsigned k = 1; k < result.size(); ++k)
    result[k] = result[k - 1] * imaginary_pi / double(k);
  return result;
}
std::vector<std::complex<double>> refreshArcsineCoefficients() {
  std::vector<std::complex<double>> result(16);
  double asin_coefficient = 1;
  const double two_pi = 2 * std::acos(-1.0);
  for (unsigned n = 0; n < 8; ++n) {
    const unsigned k = 2 * n + 1;
    result[k] = asin_coefficient / std::ldexp(two_pi, int(k));
    asin_coefficient *= double((2 * n + 1) * (2 * n + 1)) / double(2 * (n + 1) * (2 * n + 3));
  }
  return result;
}

std::int64_t refreshGain(const Engine &engine, double public_bound) {
  if (!std::isfinite(public_bound) || public_bound < 1)
    throw std::invalid_argument("refresh public bound must be finite and at least one");
  const double capacity = static_cast<double>(engine.bottomModulus()) / (16 * engine.valueScale(0));
  if (public_bound > capacity)
    throw std::invalid_argument("refresh public bound exceeds the declared fractional domain");
  int exponent = 0;
  std::frexp(capacity / public_bound, &exponent);
  if (exponent < 1 || exponent > 62)
    throw std::invalid_argument("refresh public amplification is outside exact integer range");
  const auto gain = std::int64_t{1} << (exponent - 1);
  if (double(gain) * public_bound > capacity)
    throw std::invalid_argument("refresh public amplification exceeded its domain");
  return gain;
}

Value refresh(Engine &engine, const Value &source, unsigned target, double public_bound) {
  if (engine.representation(source) != "slots")
    throw std::invalid_argument("refresh requires slot representation");
  const auto gain = refreshGain(engine, public_bound);
  const double q0 = static_cast<double>(engine.bottomModulus()), s0 = engine.valueScale(0);
  const unsigned depth = refreshDepth(engine.config().log_degree);
  if (target > engine.config().evaluation_top || target + depth >= engine.config().levels)
    throw std::invalid_argument("refresh target exceeds the prepared parameter chain");
  const unsigned top = target + depth;
  // Public exact amplification uses available modular range for small bounds.
  // It changes neither the deterministic wrap bound nor the multiplication
  // depth. The inverse factor is folded into the final public transform.
  auto amplified = engine.multiplyInteger(source, gain);
  auto bottom = engine.levelDown(amplified, 0);
  auto lifted = engine.liftBottom(bottom, top);
  // Relative trace kills non-subring coefficients. Dividing its surviving
  // integer coefficients by r preserves the integer modular-reduction lattice.
  const unsigned ratio = engine.physicalSlots() / engine.period(source);
  auto traced = traceToPeriod(engine, lifted);
  auto coefficients = coefficientToSlot(engine, traced, s0 / (q0 * ratio));
  auto conjugate = engine.conjugate(coefficients);
  auto twice_real = engine.add(coefficients, conjugate);
  auto twice_imaginary = engine.neg(engine.multiplyI(engine.sub(coefficients, conjugate)));
  const unsigned squares = refreshSquarings(engine.config().log_degree);
  auto real = reduceChannel(engine, twice_real, squares);
  auto imaginary = reduceChannel(engine, twice_imaginary, squares);
  auto reduced = engine.add(real, engine.multiplyI(imaginary));
  // The exact integer factor avoids encoding q0/S0 at the high working scale.
  const double restore = q0 / (s0 * gain);
  const auto integer_factor = std::int64_t(std::ldexp(1.0, int(std::floor(std::log2(restore)))));
  if (integer_factor < 1)
    throw std::invalid_argument("refresh bottom modulus is below its scale");
  reduced = engine.multiplyInteger(reduced, integer_factor);
  auto result = slotToCoefficient(engine, reduced, restore / double(integer_factor));
  if (engine.level(result) != target ||
      std::abs(engine.scale(result) - engine.valueScale(target)) > 1e-12 * engine.valueScale(target))
    throw std::runtime_error("refresh output violated its exact target/scale schedule");
  return result;
}
} // namespace fhenomenon::tensor_ckks
