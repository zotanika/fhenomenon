#include "refresh.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace fhenomenon::tensor_ckks;
int main(int argc, char **argv) {
  try {
    const unsigned work_bits = argc > 1 ? unsigned(std::stoul(argv[1])) : Config{}.bootstrap_prime_bits;
    const bool cuda = argc == 3 && std::string(argv[2]) == "cuda";
    if (argc > 3 || (argc == 3 && !cuda))
      throw std::invalid_argument("expected [work_bits [cuda]]");
    double max_absolute = 0, max_small_relative = 0, max_large_absolute = 0;
    for (unsigned trial = 0; trial < 3; ++trial) {
      Config cfg;
      cfg.log_degree = 10;
      cfg.test_only = true;
      cfg.logical_slots = 32;
      cfg.evaluation_top = 8;
      cfg.levels = 9 + refreshDepth(cfg.log_degree);
      cfg.bootstrap_prime_bits = work_bits;
      Engine engine(cfg);
      Client client(engine);
      if (cuda)
        engine.enableCuda();
      if (refreshGain(engine, 1) != 512 || refreshGain(engine, 1000) != 1)
        throw std::runtime_error("public amplification did not follow its magnitude contract");
      const double capacity = double(engine.bottomModulus()) / (16 * engine.valueScale(0));
      if (refreshGain(engine, capacity) != 1)
        throw std::runtime_error("public amplification rejected its exact domain edge");
      bool rejected = false;
      try {
        (void)refreshGain(engine, std::nextafter(capacity, std::numeric_limits<double>::infinity()));
      } catch (const std::invalid_argument &) {
        rejected = true;
      }
      if (!rejected)
        throw std::runtime_error("public amplification accepted the next value outside its domain");
      const std::vector<std::complex<double>> expected{1e-6, 1e-4, {-1e-6, 1e-6}, 0, 1, -1, {.6, .8}, -1e-4};
      auto value = client.encrypt(expected);
      value = refresh(engine, value, 8, 1);
      value = refresh(engine, value, 8, 1);
      if (client.decryptCalls())
        throw std::runtime_error("precision probe decrypted an intermediate");
      const auto actual = client.decrypt(value);
      for (unsigned j = 0; j < actual.size(); ++j) {
        const double error = std::abs(actual[j] - expected[j]);
        if (!std::isfinite(error))
          throw std::runtime_error("nonfinite encrypted precision probe");
        max_absolute = std::max(max_absolute, error);
        if (std::abs(expected[j]) > 0 && std::abs(expected[j]) <= 1e-4)
          max_small_relative = std::max(max_small_relative, error / std::abs(expected[j]));
      }
      std::vector<std::complex<double>> large(8, trial == 1 ? -1000. : 1000.);
      if (trial == 2)
        large = {1000., -1000., {600, 800}, {-600, -800}, 1000., -1000., {600, -800}, {-600, 800}};
      auto large_value = refresh(engine, client.encrypt(large), 8, 1000);
      auto large_actual = client.decrypt(large_value, 1000);
      for (unsigned j = 0; j < large_actual.size(); ++j)
        max_large_absolute = std::max(max_large_absolute, std::abs(large_actual[j] - large[j]));
    }
    std::cout << "work_bits=" << work_bits << " independent_keys=3 repeated_refreshes=2"
              << " maximum_absolute=" << max_absolute << " small_maximum_relative=" << max_small_relative
              << " B1000_maximum_absolute=" << max_large_absolute << '\n';
    if (max_absolute > 1e-8 || max_small_relative > 1e-3 || max_large_absolute > 1e-3)
      throw std::runtime_error("refresh precision probe exceeded the small-value precision contract");
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
