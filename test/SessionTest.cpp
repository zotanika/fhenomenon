#include "Fhenomenon.h"
#include "Parameter/ParameterGen.h"
#include "Profile.h"

#include <gtest/gtest.h>

using namespace fhenomenon;

namespace {
std::shared_ptr<Profile> makeProfile() {
  std::shared_ptr<Parameter> param = ParameterGen::createCKKSParam(CKKSParamPreset::FGb);
  return Profile::createProfile(param);
}
} // namespace

// A variable written inside a session must be observed by later reads of the
// same variable within that session (read-after-write).
TEST(SessionTest, ReadAfterWriteInsideRun) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());

  Fhenon<int> a = 0;
  a.belong(profile);

  session->run([&]() {
    a = 7;
    a = a + 2;
  });

  EXPECT_EQ(a.decrypt(), 9);
}

TEST(SessionTest, ChainedIncrementsAccumulate) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());

  Fhenon<int> a = 0;
  a.belong(profile);

  session->run([&]() {
    a = 7;
    a = a + 2;
    a = a + 2;
    a = a + 2;
  });

  EXPECT_EQ(a.decrypt(), 13);
}

TEST(SessionTest, ProductsUseUpdatedOperands) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());

  Fhenon<int> a = 0;
  Fhenon<int> b = 10;
  Fhenon<int> c = 0;
  a.belong(profile);
  b.belong(profile);
  c.belong(profile);

  session->run([&]() {
    a = 7;
    a = a + 6;  // 13
    b = b + 10; // 20
    c = a * b;  // 260
    c = c + 2;  // 262
  });

  EXPECT_EQ(a.decrypt(), 13);
  EXPECT_EQ(b.decrypt(), 20);
  EXPECT_EQ(c.decrypt(), 262);
}

TEST(SessionTest, ScalarMultiplyChain) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());

  Fhenon<int> d = 3;
  d.belong(profile);

  session->run([&]() {
    d = d * 2;
    d = d * 2;
    d = d * 2;
  });

  EXPECT_EQ(d.decrypt(), 24);
}

// One session must support several run() calls: recording state is dropped
// between runs, and results carry over through the caller's variables.
TEST(SessionTest, SecondRunStartsFresh) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());

  Fhenon<int> a = 0;
  a.belong(profile);

  session->run([&]() {
    a = 5;
    a = a + 1;
  });
  EXPECT_EQ(a.decrypt(), 6);

  session->run([&]() { a = a + 2; });
  EXPECT_EQ(a.decrypt(), 8);
}

TEST(SessionTest, EmptyRunIsANoOp) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());

  Fhenon<int> a = 4;
  a.belong(profile);

  EXPECT_NO_THROW(session->run([&]() {}));
  EXPECT_EQ(a.decrypt(), 4);

  // The session must remain usable afterwards.
  session->run([&]() { a = a + 1; });
  EXPECT_EQ(a.decrypt(), 5);
}

// Outside run(), operators evaluate eagerly and assignment must carry the
// encrypted state, not just the plaintext mirror.
TEST(SessionTest, SessionlessAssignmentCarriesCiphertext) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());
  (void)session; // inactive: exercises the eager (sessionless) paths

  Fhenon<int> a = 10;
  a.belong(profile);
  Fhenon<int> b = 0;
  b.belong(profile);

  b = a + 5; // eager add, then copy-assignment of the result
  EXPECT_EQ(b.decrypt(), 15);

  b = 3; // scalar assignment must re-encrypt
  EXPECT_EQ(b.decrypt(), 3);

  Fhenon<int> c = a * b; // eager ct*ct multiply, copy-initialization
  EXPECT_EQ(c.decrypt(), 30);
}
