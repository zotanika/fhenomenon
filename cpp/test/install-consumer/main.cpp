#include "Fhenomenon.h"
#include "Parameter/ParameterGen.h"
#include "Profile.h"

int main() {
  using namespace fhenomenon;
  auto profile = Profile::createProfile(ParameterGen::createCKKSParam(CKKSParamPreset::FGb));
  Fhenon<int> a = 2;
  Fhenon<int> b = 3;
  a.belong(profile);
  b.belong(profile);
  const auto sum = a + b;
  return sum.decrypt() == 5 ? 0 : 1;
}
