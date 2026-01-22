#include "Fhenomenon.h"

#if 1
// (FIXME) below headers should be removed later
#include "Parameter/ParameterGen.h"
#include "Profile.h"
#endif

#include <iostream>
#include <memory>

using namespace fhenomenon;

void performArithmeticDemo(const std::shared_ptr<Profile> &profile) {
  std::cout << "\n--- Arithmetic Demo ---\n";
  Compuon<int> a = 10;
  a.belong(profile);

  Compuon<int> b = 20;
  b.belong(profile);

  std::cout << "Initial values: a=" << a.decrypt() << ", b=" << b.decrypt() << std::endl;

  Compuon<int> c = a + 2;
  Compuon<int> d = c * b;
  c = c + 2;
  d = c * d;
  a = c + 1;

  std::cout << "Result A: " << a.getValue() << " (Decrypted: " << a.decrypt() << ")" << std::endl;
  std::cout << "Result B: " << b.getValue() << " (Decrypted: " << b.decrypt() << ")" << std::endl;
  std::cout << "Result C: " << c.getValue() << " (Decrypted: " << c.decrypt() << ")" << std::endl;
  std::cout << "Result D: " << d.getValue() << " (Decrypted: " << d.decrypt() << ")" << std::endl;
}

void performLogicDemo(const std::shared_ptr<Profile> &profile) {
  std::cout << "\n--- Logic Demo ---\n";
  try {
    // Test Bitwise AND
    {
      Compuon<int> a(5); // 0101
      a.belong(profile);
      Compuon<int> b(3); // 0011
      b.belong(profile);

      auto c = a & b; // Should be 0001 = 1
      int res = c.decrypt();
      std::cout << "5 & 3 = " << res << (res == 1 ? " [PASS]" : " [FAIL]") << std::endl;
    }

    // Test Bitwise OR
    {
      Compuon<int> a(5); // 0101
      a.belong(profile);
      Compuon<int> b(3); // 0011
      b.belong(profile);

      auto c = a | b; // Should be 0111 = 7
      int res = c.decrypt();
      std::cout << "5 | 3 = " << res << (res == 7 ? " [PASS]" : " [FAIL]") << std::endl;
    }

    // Test Bitwise XOR
    {
      Compuon<int> a(5); // 0101
      a.belong(profile);
      Compuon<int> b(3); // 0011
      b.belong(profile);

      auto c = a ^ b; // Should be 0110 = 6
      int res = c.decrypt();
      std::cout << "5 ^ 3 = " << res << (res == 6 ? " [PASS]" : " [FAIL]") << std::endl;
    }

    // Test Equality
    {
      Compuon<int> a(42);
      a.belong(profile);
      Compuon<int> b(42);
      b.belong(profile);
      Compuon<int> c(10);
      c.belong(profile);

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
      a.belong(profile);
      Compuon<int> b(20);
      b.belong(profile);

      auto lt = (a < b);
      int res = lt.decrypt();
      std::cout << "10 < 20 -> " << res << (res == 1 ? " [PASS]" : " [FAIL]") << std::endl;

      auto lt2 = (b < a);
      int res2 = lt2.decrypt();
      std::cout << "20 < 10 -> " << res2 << (res2 == 0 ? " [PASS]" : " [FAIL]") << std::endl;
    }

  } catch (const std::exception &e) {
    std::cout << "Logic operations skipped or failed (expected if backend does not support them): " << e.what()
              << std::endl;
  }
}

int main() {
  try {
    std::shared_ptr<Parameter> param = ParameterGen::createCKKSParam(CKKSParamPreset::FGb);
    std::shared_ptr<Profile> profile = Profile::createProfile(param);

    performArithmeticDemo(profile);
    performLogicDemo(profile);

  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
