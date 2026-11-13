#include "transforms.hpp"
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fhenomenon::tensor_ckks;
namespace {
void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
std::vector<std::complex<double>> applyMatrix(const ComplexMatrix &matrix, const std::vector<std::complex<double>> &x) {
  std::vector<std::complex<double>> y(x.size());
  for (unsigned i = 0; i < x.size(); ++i)
    for (unsigned j = 0; j < x.size(); ++j)
      y[i] += matrix[i][j] * x[j];
  return y;
}
} // namespace
int main() {
  try {
    // These literal real-polynomial coefficients distinguish the native 5^j
    // slot order, the second coefficient half, conjugation and normalization.
    const double pi = std::acos(-1.0);
    for (unsigned m : {1u, 2u, 4u, 8u, 16u}) {
      std::vector<double> coefficients(2 * m);
      for (unsigned k = 0; k < 2 * m; ++k)
        coefficients[k] = (int(k % 7) - 3) * 0.125;
      std::vector<std::complex<double>> slots(m), packed(m);
      unsigned exponent = 1;
      for (unsigned j = 0; j < m; ++j) {
        const auto root = std::polar(1.0, pi * exponent / (2 * m));
        std::complex<double> power = 1;
        for (double coefficient : coefficients) {
          slots[j] += coefficient * power;
          power *= root;
        }
        exponent = (exponent * 5) % (4 * m);
        packed[j] = {coefficients[j], coefficients[j + m]};
      }
      auto got_coefficients = applyMatrix(coefficientToSlotMatrix(m), slots);
      auto got_slots = applyMatrix(slotToCoefficientMatrix(m), packed);
      for (unsigned j = 0; j < m; ++j) {
        require(std::abs(got_coefficients[j] - packed[j]) < 2e-12, "native coefficient order/normalization mismatch");
        require(std::abs(got_slots[j] - slots[j]) < 2e-12, "native canonical evaluation mismatch");
      }
    }
    for (unsigned bad : {0u, 3u, 6u}) {
      bool rejected = false;
      try {
        (void)coefficientToSlotMatrix(bad);
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      require(rejected, "invalid transform period accepted");
    }
    std::cout << "native canonical transform mathematics passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
