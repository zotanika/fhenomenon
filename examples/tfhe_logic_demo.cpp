#include "Backend/Backend.h"
#include "Compuon.h"
#include <cassert>
#include <iostream>

using namespace fhenomenon;

// DummyParameter needed because Parameter is abstract
class DummyParameter : public Parameter {
  public:
  void printParams() const override {}
  void init() override {}
};

int main() {
  try {
    std::cout << "Starting TFHE Logic Demo..." << std::endl;

    // Initialize TfheBackend
    auto &backend = Backend::getInstance("tfhe");

    DummyParameter params;

    // Test Bitwise AND
    {
      Compuon<int> a(5); // 0101
      Compuon<int> b(3); // 0011
      backend.transform(a, params);
      backend.transform(b, params);

      auto c = a & b; // Should be 0001 = 1
      int res = c.decrypt();
      std::cout << "5 & 3 = " << res << (res == 1 ? " [PASS]" : " [FAIL]") << std::endl;
    }

    // Test Bitwise OR
    {
      Compuon<int> a(5); // 0101
      Compuon<int> b(3); // 0011
      backend.transform(a, params);
      backend.transform(b, params);

      auto c = a | b; // Should be 0111 = 7
      int res = c.decrypt();
      std::cout << "5 | 3 = " << res << (res == 7 ? " [PASS]" : " [FAIL]") << std::endl;
    }

    // Test Bitwise XOR
    {
      Compuon<int> a(5); // 0101
      Compuon<int> b(3); // 0011
      backend.transform(a, params);
      backend.transform(b, params);

      auto c = a ^ b; // Should be 0110 = 6
      int res = c.decrypt();
      std::cout << "5 ^ 3 = " << res << (res == 6 ? " [PASS]" : " [FAIL]") << std::endl;
    }

    // Test Equality
    {
      Compuon<int> a(42);
      Compuon<int> b(42);
      Compuon<int> c(10);
      backend.transform(a, params);
      backend.transform(b, params);
      backend.transform(c, params);

      auto eq1 = (a == b);
      int res1 = eq1.decrypt();
      std::cout << "42 == 42 -> " << res1 << (res1 == 1 ? " [PASS]" : " [FAIL]") << std::endl;

      auto eq2 = (a == c);
      int res2 = eq2.decrypt();
      std::cout << "42 == 10 -> " << res2 << (res2 == 0 ? " [PASS]" : " [FAIL]") << std::endl;
    }

    // Test Less Than
    {
      Compuon<int> a(10);
      Compuon<int> b(20);
      backend.transform(a, params);
      backend.transform(b, params);

      auto lt = (a < b);
      int res = lt.decrypt();
      std::cout << "10 < 20 -> " << res << (res == 1 ? " [PASS]" : " [FAIL]") << std::endl;

      auto lt2 = (b < a);
      int res2 = lt2.decrypt();
      std::cout << "20 < 10 -> " << res2 << (res2 == 0 ? " [PASS]" : " [FAIL]") << std::endl;
    }

  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
