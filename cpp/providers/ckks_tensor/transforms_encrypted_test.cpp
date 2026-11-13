#include "transforms.hpp"
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>

using namespace fhenomenon::tensor_ckks;
int main(int argc, char **argv) {
  try {
    Config cfg;
    cfg.log_degree = 10;
    cfg.test_only = true;
    cfg.logical_slots = 8;
    cfg.levels = 6;
    cfg.evaluation_top = 5;
    Engine engine(cfg);
    Client client(engine);
    if (argc == 2 && std::string(argv[1]) == "cuda")
      engine.enableCuda();
    else if (argc != 1)
      throw std::invalid_argument("expected optional cuda argument");
    const unsigned m = 8;
    const double pi = std::acos(-1.0);
    const std::vector<double> coefficients{.25, -.125, 0, .03125, 0, 0, 0, 0, .5, 0, 0, -.25, 0, 0, 0, 0};
    std::vector<std::complex<double>> slots(m), packed(m);
    unsigned exponent = 1;
    for (unsigned j = 0; j < m; ++j) {
      auto root = std::polar(1.0, pi * exponent / (2 * m));
      std::complex<double> power = 1;
      for (double c : coefficients) {
        slots[j] += c * power;
        power *= root;
      }
      exponent = (exponent * 5) % (4 * m);
      packed[j] = {coefficients[j], coefficients[j + m]};
    }
    auto input = client.encrypt(slots);
    auto transformed = coefficientToSlot(engine, input);
    auto roundtrip = slotToCoefficient(engine, transformed);
    if (client.decryptCalls() != 0 || engine.level(input) != 5 || engine.level(roundtrip) != 3)
      throw std::runtime_error("transform changed source or decrypted an intermediate");
    auto decoded = client.decrypt(transformed, 2), recovered = client.decrypt(roundtrip, 2);
    double maximum = 0;
    for (unsigned j = 0; j < m; ++j) {
      maximum = std::max(maximum, std::abs(decoded[j] - packed[j]));
      maximum = std::max(maximum, std::abs(recovered[j] - slots[j]));
    }
    if (maximum > 1e-6 || client.lastEgressModuli() != 1)
      throw std::runtime_error("encrypted transform ordering or roundtrip precision failed");
    std::cout << "native encrypted transforms passed; max error " << maximum << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
