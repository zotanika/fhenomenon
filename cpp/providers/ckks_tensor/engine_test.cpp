#include "engine.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace fhenomenon::tensor_ckks;
void check(bool yes, const char *message) {
  if (!yes)
    throw std::runtime_error(message);
}
void close(Client &c, const Value &v, const std::vector<std::complex<double>> &expected, double tol = 2e-7) {
  auto actual = c.decrypt(v, 32.0);
  check(actual.size() == expected.size(), "length");
  for (size_t i = 0; i < actual.size(); ++i) {
    if (std::abs(actual[i] - expected[i]) > tol) {
      std::cerr << i << " actual " << actual[i] << " expected " << expected[i] << "\n";
      throw std::runtime_error("native arithmetic precision");
    }
  }
  check(c.lastEgressModuli() == 1, "minimum-prime egress");
}
int main(int argc, char **argv) {
  (void)argv;
  Config cfg;
  cfg.log_degree = 10;
  cfg.levels = 9;
  cfg.evaluation_top = 8;
  cfg.bottom_bits = 62;
  cfg.logical_slots = 4;
  cfg.test_only = true;
  Engine e(cfg);
  Client c(e);
  if (argc > 1)
    e.enableCuda();
  auto a = c.encrypt({{.1, .03}, {-.2, 0}, {.35, -.04}, {.7, .02}});
  auto b = c.encrypt({{.5, 0}, {.25, 0}, {-.5, 0}, {1, 0}});
  auto av = c.decrypt(a, 1.0), bv = c.decrypt(b, 1.0);
  close(c, a, av);
  auto product = e.mul(a, b);
  std::vector<std::complex<double>> expected(4);
  for (size_t i = 0; i < 4; ++i)
    expected[i] = av[i] * bv[i];
  close(c, product, expected);
  check(e.level(product) == 7, "multiply consumes one level");
  auto low = e.levelDown(b, 3), sum = e.add(a, low);
  for (size_t i = 0; i < 4; ++i)
    expected[i] = av[i] + bv[i];
  close(c, sum, expected);
  check(e.level(sum) == 3, "mixed-level arithmetic");
  close(c, e.rotate(a, 1), {av[1], av[2], av[3], av[0]});
  close(c, e.conjugate(a), {std::conj(av[0]), std::conj(av[1]), std::conj(av[2]), std::conj(av[3])});
  close(c, e.plain(a, {{.1, 0}, {2, 0}, {-.3, 0}, {.5, 0}}, true), {av[0] * .1, av[1] * 2., av[2] * -.3, av[3] * .5});
  close(c, e.multiplyI(a),
        {av[0] * std::complex<double>(0, 1), av[1] * std::complex<double>(0, 1), av[2] * std::complex<double>(0, 1),
         av[3] * std::complex<double>(0, 1)});
  close(c, e.add(a, c.encrypt({.25})), {av[0] + .25, av[1] + .25, av[2] + .25, av[3] + .25});
  check(e.level(a) == 8, "egress leaves input immutable");
  Engine other(cfg);
  Client otherClient(other);
  bool rejected = false;
  try {
    e.add(a, otherClient.encrypt({1., 2., 3., 4.}));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, "key contexts must not mix");
  rejected = false;
  try {
    e.mul(e.levelDown(a, 0), e.levelDown(b, 0));
  } catch (const std::exception &) {
    rejected = true;
  }
  check(rejected, "exhausted multiplication must reject");
  auto coeff = e.withRepresentation(a, "coefficients");
  rejected = false;
  try {
    e.add(coeff, a);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, "representation mismatch must reject");
  cfg.test_only = false;
  rejected = false;
  try {
    Engine insecure(cfg);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check(rejected, "small insecure profile needs explicit opt-in");
  if (argc == 1) {
    rejected = false;
    try {
      e.enableCuda();
    } catch (const std::exception &) {
      rejected = true;
    }
    check(rejected, "device selection after encryption must fail");
  }
  std::cout << "native engine arithmetic, identity, scale, egress checks passed\n";
}
