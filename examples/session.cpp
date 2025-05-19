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
    std::shared_ptr<Profile> profile = std::make_shared<Profile>();
    profile->setParam(param);

    auto session = Session::create(Backend::getInstance());
    

    Compuon<int> a = 0;
    a.belong(profile);
    std::cout <<"a's address: "<< &a << std::endl; 

    Compuon<int> b = 10;
    b.belong(profile);
    std::cout <<"b's address: "<< &b << std::endl; 

    Compuon<int> c = 0;
    c.belong(profile);
    std::cout <<"c's address: "<< &c << std::endl; 

    Compuon<int> d = 0;
    d.belong(profile);
    std::cout <<"d's address: "<< &d<< std::endl; 

    std::cout << "Input: " << a.getValue() << ", " << b.getValue() << ", " << c.getValue() << ", " << d.getValue()
              << std::endl;
    std::cout << "Input decrypted: " << a.decrypt() << ", " << b.decrypt() << ", " << c.decrypt() << ", " << d.decrypt()
              << std::endl;

    session->run([&]() {
      a = 7;
      b = b + 10;
      a = a + 2;
      b = b + 2;
      a = a + 2;
      a = a + 2;
      c = a * b;
      d = d + 2;    
      a = a + 2;
      a = a + 4;
      b = a + 2;
      d = b * a;   
      a = a + 2;
      a = a + 2;
      
    });

    std::cout << "Result: " << a.getValue() << ", " << b.getValue() << ", " << c.getValue() << ", " << d.getValue()
              << std::endl;
    std::cout << "Result decrypted: " << a.decrypt() << ", " << b.decrypt() << ", " << c.decrypt() << ", "
              << d.decrypt() << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
