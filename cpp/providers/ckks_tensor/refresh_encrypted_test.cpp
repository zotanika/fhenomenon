#include "refresh.hpp"
#include <complex>
#include <iostream>
#include <stdexcept>

using namespace fhenomenon::tensor_ckks;
int main(int argc, char **argv) {
  try {
    Config cfg;
    cfg.log_degree = 10;
    cfg.test_only = true;
    cfg.logical_slots = 256;
    cfg.evaluation_top = 8;
    cfg.levels = 9 + refreshDepth(cfg.log_degree);
    Engine engine(cfg);
    Client client(engine);
    if (argc == 2 && std::string(argv[1]) == "cuda")
      engine.enableCuda();
    else if (argc != 1)
      throw std::invalid_argument("expected optional cuda argument");
    const std::vector<std::complex<double>> wanted{{.125, 0},  {-.25, .1}, {.375, -.2}, {-.0625, .4},
                                                   {.2, -.15}, {-.3, .25}, {.45, 0},    {-.5, -.125}};
    auto original = client.encrypt(wanted);
    auto bottom = engine.levelDown(original, 0);
    auto refreshed = refresh(engine, bottom, 2, 1);
    auto squared = engine.mul(refreshed, refreshed);
    auto refreshed_again = refresh(engine, squared, 8, 1);
    if (client.decryptCalls() != 0 || engine.level(original) != 8 || engine.level(bottom) != 0 ||
        engine.level(refreshed) != 2 || engine.level(refreshed_again) != 8)
      throw std::runtime_error("refresh level, immutability or secret-boundary contract failed");
    auto got = client.decrypt(refreshed), got_square = client.decrypt(refreshed_again);
    auto original_after = client.decrypt(original), bottom_after = client.decrypt(bottom);
    double maximum = 0;
    for (unsigned j = 0; j < wanted.size(); ++j) {
      maximum = std::max(maximum, std::abs(got[j] - wanted[j]));
      maximum = std::max(maximum, std::abs(got_square[j] - wanted[j] * wanted[j]));
      if (std::abs(original_after[j] - wanted[j]) > 1e-8 || std::abs(bottom_after[j] - wanted[j]) > 1e-8)
        throw std::runtime_error("refresh modified a retained source ciphertext");
    }
    if (maximum > 1e-4 || client.lastEgressModuli() != 1)
      throw std::runtime_error("actual native refresh did not preserve usable encrypted precision");
    double bounded_maximum = 0;
    for (unsigned period : {8u, 32u, 256u}) {
      // Constant 1000 places the full declared magnitude into one polynomial
      // coefficient, exercising the edge of the arcsine correction domain.
      std::vector<std::complex<double>> bounded(period, 1000.);
      auto encrypted = client.encrypt(bounded);
      const auto before = client.decryptCalls();
      auto output = refresh(engine, encrypted, 8, 1000);
      if (client.decryptCalls() != before || engine.level(encrypted) != 8 || engine.level(output) != 8)
        throw std::runtime_error("bounded refresh used a client or mutated its source");
      auto values = client.decrypt(output, 1000);
      for (unsigned j = 0; j < period; ++j)
        bounded_maximum = std::max(bounded_maximum, std::abs(values[j] - bounded[j]));
    }
    if (bounded_maximum > 1e-3)
      throw std::runtime_error("bounded refresh exceeded its numerical test tolerance");
    bool rejected = false;
    try {
      (void)refresh(engine, original, 8, 1e6);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!rejected)
      throw std::runtime_error("unsupported public magnitude was accepted");
    std::cout << "native encrypted refresh passed; max error " << maximum << ", B1000 periods8/32/256 max error "
              << bounded_maximum << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
