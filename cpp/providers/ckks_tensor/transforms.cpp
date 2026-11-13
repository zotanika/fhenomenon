#include "transforms.hpp"
#include <cmath>
#include <stdexcept>

namespace fhenomenon::tensor_ckks {
namespace {
void validPeriod(unsigned m) {
  if (!m || (m & (m - 1)) || m > (1u << 16))
    throw std::invalid_argument("transform period must be a supported power of two");
}
} // namespace

ComplexMatrix coefficientToSlotMatrix(unsigned m) {
  validPeriod(m);
  ComplexMatrix result(m, std::vector<std::complex<double>>(m));
  const double pi = std::acos(-1.0);
  unsigned exponent = 1;
  for (unsigned j = 0; j < m; ++j) {
    for (unsigned k = 0; k < m; ++k)
      result[k][j] = std::polar(1.0 / m, -pi * double(exponent) * k / (2 * m));
    exponent = (exponent * 5) % (4 * m);
  }
  return result;
}
ComplexMatrix slotToCoefficientMatrix(unsigned m) {
  validPeriod(m);
  ComplexMatrix result(m, std::vector<std::complex<double>>(m));
  const double pi = std::acos(-1.0);
  unsigned exponent = 1;
  for (unsigned j = 0; j < m; ++j) {
    for (unsigned k = 0; k < m; ++k)
      result[j][k] = std::polar(1.0, pi * double(exponent) * k / (2 * m));
    exponent = (exponent * 5) % (4 * m);
  }
  return result;
}
Value linearTransform(Engine &engine, const Value &input, const ComplexMatrix &matrix, double multiplier) {
  const unsigned m = engine.period(input);
  validPeriod(m);
  if (matrix.size() != m || m > engine.physicalSlots() || !std::isfinite(multiplier) || !engine.level(input))
    throw std::invalid_argument("invalid encrypted linear transform");
  for (const auto &row : matrix) {
    if (row.size() != m)
      throw std::invalid_argument("linear transform matrix is not square");
    for (auto z : row)
      if (!std::isfinite(z.real()) || !std::isfinite(z.imag()))
        throw std::invalid_argument("nonfinite linear transform coefficient");
  }
  unsigned baby = 1;
  while (baby * baby < m)
    baby *= 2;
  std::vector<Value> rotations(baby);
  rotations[0] = input;
  for (unsigned i = 1; i < baby; ++i)
    rotations[i] = engine.rotate(input, i);
  Value result;
  for (unsigned giant = 0; giant < m; giant += baby) {
    Value group;
    for (unsigned i = 0; i < baby && giant + i < m; ++i) {
      const unsigned diagonal = giant + i;
      std::vector<std::complex<double>> weights(m);
      for (unsigned row = 0; row < m; ++row) {
        const unsigned original_row = (row + m - giant) % m;
        weights[row] = matrix[original_row][(original_row + diagonal) % m] * multiplier;
      }
      auto term = engine.plain(rotations[i], weights, true);
      group = group ? engine.add(group, term) : term;
    }
    if (giant)
      group = engine.rotate(group, giant);
    result = result ? engine.add(result, group) : group;
  }
  return result;
}
Value coefficientToSlot(Engine &engine, const Value &input, double multiplier) {
  return linearTransform(engine, input, coefficientToSlotMatrix(engine.period(input)), multiplier);
}
Value slotToCoefficient(Engine &engine, const Value &input, double multiplier) {
  return linearTransform(engine, input, slotToCoefficientMatrix(engine.period(input)), multiplier);
}
Value traceToPeriod(Engine &engine, const Value &input) {
  const unsigned m = engine.period(input), physical = engine.physicalSlots();
  validPeriod(m);
  if (m > physical || physical % m)
    throw std::invalid_argument("period does not divide physical slots");
  auto result = input;
  for (unsigned step = m; step < physical; step *= 2)
    result = engine.add(result, engine.rotate(result, step));
  return result;
}
} // namespace fhenomenon::tensor_ckks
