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

// An operand that was never belong()'d must fail loudly, not silently
// compute on an empty buffer (and must not be corrupted by write-back).
TEST(SessionTest, UnencryptedOperandThrows) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());

  Fhenon<int> a = 5; // deliberately no belong()
  Fhenon<int> b = 3;
  b.belong(profile);
  Fhenon<int> c = 0;
  c.belong(profile);

  EXPECT_THROW(session->run([&]() { c = a + b; }), std::runtime_error);

  // a must not have been stamped encrypted by a failed run.
  EXPECT_FALSE(a.isEncrypted_);
  EXPECT_EQ(a.decrypt(), 5);
}

// Moving a Fhenon must transfer the encrypted state to the new object and
// leave the moved-from shell unencrypted — not the other way around.
TEST(SessionTest, MoveConstructorCarriesEncryptedState) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());
  (void)session; // inactive: exercises the sessionless move path

  Fhenon<int> a = 10;
  a.belong(profile);

  Fhenon<int> b = std::move(a);
  EXPECT_TRUE(b.isEncrypted_);
  EXPECT_EQ(b.decrypt(), 10);

  // The moved-from shell must not keep serving the old ciphertext.
  EXPECT_FALSE(a.isEncrypted_);
}

// Copy-constructing from a session variable inside run() records against the
// source's entity and cannot deliver a result (the copy dies with the
// lambda); it must fail loudly instead of silently clobbering the source.
TEST(SessionTest, InSessionCopyOfSessionVariableThrows) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());

  Fhenon<int> a = 0;
  a.belong(profile);

  EXPECT_THROW(session->run([&]() {
    a = 4;
    Fhenon<int> b = a;
    b = b + 1;
  }),
               std::logic_error);

  // The failed run must not have leaked the increment into a.
  EXPECT_EQ(a.decrypt(), 0);
}

// A Fhenon declared inside the run() lambda dies before the recorded graph
// executes; evaluation must refuse to run instead of reading the dead stack.
TEST(SessionTest, LambdaLocalFhenonFailsLoudly) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());

  Fhenon<int> a = 0;
  a.belong(profile);

  try {
    session->run([&]() {
      Fhenon<int> t = 13;
      t.belong(profile);
      t = t + 1;
      a = t;
    });
    FAIL() << "run() must throw when a session-tracked variable dies before evaluation";
  } catch (const std::runtime_error &e) {
    // Must be the lifetime diagnostic, not a downstream symptom of reading
    // the dead object (e.g. "operand is not encrypted").
    EXPECT_NE(std::string(e.what()).find("died"), std::string::npos) << "actual message: " << e.what();
  }

  // The session must remain usable after the failed run.
  session->run([&]() { a = a + 2; });
  EXPECT_EQ(a.decrypt(), 2);
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

// Two entities can end the session bound to the SAME value id (b = a).
// Adoption of plan-allocated buffers must dedupe by id, or the two
// write-backs would double-free one buffer when both entities die.
TEST(SessionTest, AliasedWriteBackTargetsShareOneBuffer) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());

  Fhenon<int> a = 0;
  Fhenon<int> b = 0;
  a.belong(profile);
  b.belong(profile);

  session->run([&]() {
    a = a + 7;
    b = a; // b's latest binding is a's value id
  });

  EXPECT_EQ(a.decrypt(), 7);
  EXPECT_EQ(b.decrypt(), 7);
}
