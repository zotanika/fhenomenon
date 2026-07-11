#include "Fhenomenon.h"
#include "Parameter/ParameterGen.h"
#include "Profile.h"

#include <gtest/gtest.h>

#include <utility>

using namespace fhenomenon;

// This binary must never create a Session: it pins the guarantee that every
// Fhenon operation is safe when Session::getSession() is null (the sanitizer
// build additionally catches any member call through the null session).
TEST(NoSessionTest, EagerOpsWithoutAnySession) {
  std::shared_ptr<Parameter> param = ParameterGen::createCKKSParam(CKKSParamPreset::FGb);
  auto profile = Profile::createProfile(param);

  Fhenon<int> a = 3;
  a.belong(profile);
  Fhenon<int> b = 4;
  b.belong(profile);

  Fhenon<int> c = a + b; // eager add + copy-construction of the result
  EXPECT_EQ(c.decrypt(), 7);

  c = 9; // scalar assignment
  EXPECT_EQ(c.decrypt(), 9);

  Fhenon<int> d = std::move(c); // move construction
  EXPECT_EQ(d.decrypt(), 9);

  Fhenon<int> e = d * 2; // scalar multiply
  EXPECT_EQ(e.decrypt(), 18);

  b = a; // copy assignment
  EXPECT_EQ(b.decrypt(), 3);
}
