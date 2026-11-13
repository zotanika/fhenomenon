#include "refresh.hpp"
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>

using namespace fhenomenon::tensor_ckks;
int main() {
  try {
    double maximum = 0;
    // A missing 2*pi, wrong imaginary sign, omitted squaring, or arcsine
    // correction breaks reduction of integers plus these known fractions.
    for (unsigned log_degree : {10u, 17u}) {
      const unsigned squares = refreshSquarings(log_degree);
      if (refreshDepth(log_degree) != log_degree + 16)
        throw std::runtime_error("half-angle exponential depth is inconsistent");
      const double bound = std::ldexp(1., log_degree - 1) + 2;
      for (double integer : {-bound, -17., -1., 0., 1., 17., bound})
        for (double fraction : {-.061, -.02, -.0001, .0001, .02, .061}) {
          const double input = integer + fraction;
          const double normalized = std::ldexp(input, 2 - int(squares));
          auto exp_coefficients = refreshExponentialCoefficients();
          std::complex<double> exponential = 0;
          for (auto it = exp_coefficients.rbegin(); it != exp_coefficients.rend(); ++it)
            exponential = exponential * normalized + *it;
          for (unsigned k = 0; k < squares; ++k)
            exponential *= exponential;
          auto asin_coefficients = refreshArcsineCoefficients();
          std::complex<double> recovered = 0;
          for (auto it = asin_coefficients.rbegin(); it != asin_coefficients.rend(); ++it)
            recovered = recovered * (2 * exponential.imag()) + *it;
          const double error = std::abs(recovered.real() - fraction);
          maximum = std::max(maximum, error);
          if (error > 5e-8 || std::abs(recovered.imag()) > 1e-14)
            throw std::runtime_error("modular polynomial failed its declared fractional domain");
        }
    }
    std::cout << "native modular polynomial reference maximum error " << maximum << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
