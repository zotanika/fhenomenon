#include "Fhenomenon.h"

#if 1
// (FIXME) below headers should be removed later
#include "Parameter/ParameterGen.h"
#include "Profile.h"
#endif

using namespace fhenomenon;

int main() {
  try {
    std::shared_ptr<Parameter> param = ParameterGen::createCKKSParam(CKKSParamPreset::FGb);
    std::shared_ptr<Profile> profile = Profile::createProfile(param);
    Compuon<int> a = 10;
    a.belong(profile); // not default

    Compuon<int> b = 20;
    b.belong(profile); // not required

    Compuon<int> c = a + 2;
    Compuon<int> d = c * b;
    c = c + 2;
    d = c * d;
    a = c + 1;
    std::cout << "Result A: " << a.getValue() << std::endl;
    std::cout << "Result decrypted A: " << a.decrypt() << std::endl;
    std::cout << "Result B: " << b.getValue() << std::endl;
    std::cout << "Result decrypted B: " << b.decrypt() << std::endl;
    std::cout << "Result C: " << c.getValue() << std::endl;
    std::cout << "Result decrypted C: " << c.decrypt() << std::endl;
    std::cout << "Result D: " << d.getValue() << std::endl;
    std::cout << "Result decrypted D: " << d.decrypt() << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
