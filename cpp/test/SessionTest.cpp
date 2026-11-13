#include "Fhenomenon.h"
#include "Parameter/ParameterGen.h"
#include "Profile.h"

#include <gtest/gtest.h>

#include <optional>
#include <unordered_set>

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

TEST(SessionTest, CompositeAddThenScalarMultiplyPreservesProfile) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());
  Fhenon<int> a = 2, b = 3, c = 0;
  a.belong(profile);
  b.belong(profile);
  c.belong(profile);
  session->run([&] { c = (a + b) * 2; });
  EXPECT_EQ(c.decrypt(), 10);
  EXPECT_EQ(c.getProfile(), profile);
}

TEST(SessionTest, CompositeMultiplyThenScalarAddPreservesProfile) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());
  Fhenon<int> a = 2, b = 3, c = 0;
  a.belong(profile);
  b.belong(profile);
  c.belong(profile);
  session->run([&] { c = (a * b) + 3; });
  EXPECT_EQ(c.decrypt(), 9);
}

TEST(SessionTest, SharedOwnedUserValuesAndDistinctOutputsSurvive) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());
  auto a = std::make_shared<Fhenon<int>>(2);
  auto b = std::make_shared<Fhenon<int>>(0);
  Fhenon<int> c = 0, d = 0;
  a->belong(profile);
  b->belong(profile);
  c.belong(profile);
  d.belong(profile);
  session->run([&] {
    *b = *a + 3;
    c = *b;
    *b = *b + 4;
    d = *b;
  });
  EXPECT_EQ(a->decrypt(), 2);
  EXPECT_EQ(b->decrypt(), 9);
  EXPECT_EQ(c.decrypt(), 5);
  EXPECT_EQ(d.decrypt(), 9);
}

TEST(SessionTest, ReturnedIntermediateRetainedOutsideRunIsWrittenBack) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());
  Fhenon<int> a = 2, b = 3, c = 0;
  a.belong(profile);
  b.belong(profile);
  c.belong(profile);
  std::optional<Fhenon<int>> retained;
  session->run([&] {
    retained.emplace(a + b);
    c = *retained + b;
  });
  ASSERT_TRUE(retained.has_value());
  EXPECT_EQ(retained->decrypt(), 5);
  EXPECT_EQ(c.decrypt(), 8);
}

TEST(SessionTest, OlderSessionBecomesCurrentForItsRun) {
  auto profile = makeProfile();
  auto older = Session::create(Backend::getInstance());
  auto newer = Session::create(Backend::getInstance());
  Fhenon<int> a = 2;
  a.belong(profile);
  older->run([&] {
    EXPECT_TRUE(Session::isRecording());
    EXPECT_EQ(Session::getSession(), older);
    a = a + 3;
  });
  EXPECT_EQ(a.decrypt(), 5);
  EXPECT_FALSE(Session::isRecording());
  newer->run([&] { a = a + 1; });
  EXPECT_EQ(a.decrypt(), 6);
}

TEST(SessionTest, NestedRunRejectsBeforeChangingOuterRecording) {
  auto profile = makeProfile();
  auto outer = Session::create(Backend::getInstance());
  auto inner = Session::create(Backend::getInstance());
  Fhenon<int> a = 2;
  a.belong(profile);
  outer->run([&] {
    a = a + 1;
    EXPECT_THROW(outer->run([&] { a = a + 100; }), std::logic_error);
    EXPECT_THROW(inner->run([&] { a = a + 100; }), std::logic_error);
    EXPECT_TRUE(Session::isRecording());
    EXPECT_EQ(Session::getSession(), outer);
    a = a + 3;
  });
  EXPECT_EQ(a.decrypt(), 6);
}

TEST(SessionTest, CreatingSessionDuringRecordingCannotReplaceOuter) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());
  Fhenon<int> a = 2;
  a.belong(profile);
  session->run([&] {
    a = a + 1;
    EXPECT_THROW(Session::create(Backend::getInstance()), std::logic_error);
    EXPECT_EQ(Session::getSession(), session);
    a = a + 3;
  });
  EXPECT_EQ(a.decrypt(), 6);
}

TEST(SessionTest, DecryptDuringRecordingRejectsAndSessionRecovers) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());
  Fhenon<int> a = 2;
  a.belong(profile);
  EXPECT_THROW(session->run([&] {
    a = a + 3;
    (void)a.decrypt();
  }),
               std::logic_error);
  EXPECT_EQ(a.decrypt(), 2);
  session->run([&] { a = a + 1; });
  EXPECT_EQ(a.decrypt(), 3);
}

TEST(SessionTest, SharedOwnedUserCopyHasSameRecordingRulesAsStackValue) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());
  auto a = std::make_shared<Fhenon<int>>(2);
  a->belong(profile);
  EXPECT_THROW(session->run([&] {
    Fhenon<int> copy = *a;
    copy = copy + 3;
  }),
               std::logic_error);
  EXPECT_EQ(a->decrypt(), 2);
}

TEST(SessionTest, UnsupportedFloatingRecordingRejectsBeforeExecution) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());
  Fhenon<int> a = 2;
  Fhenon<double> d = 3.5;
  Fhenon<float> f = 4.5f;
  a.belong(profile);
  d.belong(profile);
  f.belong(profile);
  auto reject = [&](auto &value) {
    try {
      session->run([&] {
        a = a + 1;
        value = value + value;
      });
      FAIL() << "unsupported recording must throw";
    } catch (const std::logic_error &e) {
      EXPECT_NE(std::string(e.what()).find("int"), std::string::npos);
    }
  };
  reject(d);
  reject(f);
  EXPECT_EQ(a.decrypt(), 2);
  EXPECT_DOUBLE_EQ(d.decrypt(), 3.5);
  EXPECT_FLOAT_EQ(f.decrypt(), 4.5f);
}

TEST(SessionTest, RepeatedDoublingDoesNotPrintAutomatically) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());
  Fhenon<int> a = 1;
  a.belong(profile);
  testing::internal::CaptureStdout();
  session->run([&] {
    for (int i = 0; i < 12; ++i)
      a = a + a;
  });
  auto output = testing::internal::GetCapturedStdout();
  EXPECT_TRUE(output.empty()) << "unexpected output bytes: " << output.size();
  EXPECT_EQ(a.decrypt(), 4096);
}

namespace {
// Observe real executor buffer lifetimes without changing the kernel or
// lowering path. Only plan-created buffers are counted; caller inputs stay owned.
struct ScopedAllocationProbe {
  FhnRuntime &runtime;
  inline static ScopedAllocationProbe *active = nullptr;
  inline static FhnBufferAllocFn allocate = nullptr;
  inline static FhnBufferFreeFn release = nullptr;
  std::unordered_set<FhnBuffer *> live;
  size_t peak = 0, allocations = 0, frees = 0;

  explicit ScopedAllocationProbe(FhnRuntime &value) : runtime(value) {
    allocate = runtime.buffer_alloc;
    release = runtime.buffer_free;
    active = this;
    runtime.buffer_alloc = [](FhnBackendCtx *ctx) {
      auto *buffer = allocate(ctx);
      if (active && buffer) {
        active->live.insert(buffer);
        ++active->allocations;
        active->peak = std::max(active->peak, active->live.size());
      }
      return buffer;
    };
    runtime.buffer_free = [](FhnBackendCtx *ctx, FhnBuffer *buffer) {
      if (active && active->live.erase(buffer))
        ++active->frees;
      release(ctx, buffer);
    };
  }
  ~ScopedAllocationProbe() {
    runtime.buffer_alloc = allocate;
    runtime.buffer_free = release;
    active = nullptr;
  }
};
} // namespace

TEST(SessionTest, TwelveInternalScalarAddResultsAreFreedAfterLastUse) {
  auto profile = makeProfile();
  auto &backend = Backend::getInstance();
  auto session = Session::create(backend);
  Fhenon<int> a = 1;
  a.belong(profile);
  ASSERT_NE(backend.fhnRuntime(), nullptr);
  ScopedAllocationProbe probe(*const_cast<FhnRuntime *>(backend.fhnRuntime()));
  session->run([&] {
    for (int i = 0; i < 12; ++i)
      a = a + 1;
  });
  EXPECT_EQ(a.decrypt(), 13);
  EXPECT_EQ(probe.allocations, 12u);
  EXPECT_LE(probe.peak, 2u); // previous result + next result, plus the caller-owned input
  EXPECT_EQ(probe.live.size(), 1u);
  EXPECT_EQ(probe.frees, 11u);
}

TEST(SessionTest, RetainedCopiesOfAnExpressionHaveIndependentAssignments) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());
  Fhenon<int> a = 2, b = 3, c = 0;
  a.belong(profile);
  b.belong(profile);
  c.belong(profile);
  std::optional<Fhenon<int>> first, second;
  session->run([&] {
    first.emplace(a + b);
    second.emplace(*first);
    *first = *first + 1;
    c = *second;
  });
  EXPECT_EQ(first->decrypt(), 6);
  EXPECT_EQ(second->decrypt(), 5);
  EXPECT_EQ(c.decrypt(), 5);
}

namespace {
// Use the real builtin primitives while deliberately selecting the legacy
// scheduler path, as a backend without an FHN runtime does.
class LegacyBuiltinBackend final : public Backend {
  const Backend &delegate_ = Backend::getInstance();

  public:
  BackendType getBackendType() const override { return delegate_.getBackendType(); }
  void transform(FhenonBase &value, const Parameter &param) const override { delegate_.transform(value, param); }
  std::shared_ptr<FhenonBase> add(const FhenonBase &a, const FhenonBase &b) const override {
    return delegate_.add(a, b);
  }
  std::shared_ptr<FhenonBase> multiply(const FhenonBase &a, const FhenonBase &b) const override {
    return delegate_.multiply(a, b);
  }
  std::any decrypt(const FhenonBase &value) const override { return delegate_.decrypt(value); }
  std::shared_ptr<FhenonBase> bitAnd(const FhenonBase &a, const FhenonBase &b) const override {
    return delegate_.bitAnd(a, b);
  }
  std::shared_ptr<FhenonBase> bitOr(const FhenonBase &a, const FhenonBase &b) const override {
    return delegate_.bitOr(a, b);
  }
  std::shared_ptr<FhenonBase> bitXor(const FhenonBase &a, const FhenonBase &b) const override {
    return delegate_.bitXor(a, b);
  }
  std::shared_ptr<FhenonBase> compareEq(const FhenonBase &a, const FhenonBase &b) const override {
    return delegate_.compareEq(a, b);
  }
  std::shared_ptr<FhenonBase> compareLt(const FhenonBase &a, const FhenonBase &b) const override {
    return delegate_.compareLt(a, b);
  }
  std::shared_ptr<FhenonBase> compareLe(const FhenonBase &a, const FhenonBase &b) const override {
    return delegate_.compareLe(a, b);
  }
};
} // namespace

TEST(SessionTest, LegacyExecutionWritesBackRetainedExpressionCopies) {
  auto profile = makeProfile();
  LegacyBuiltinBackend backend;
  auto session = Session::create(backend);
  Fhenon<int> a = 2, b = 3, c = 0;
  a.belong(profile);
  b.belong(profile);
  c.belong(profile);
  std::optional<Fhenon<int>> retained;
  session->run([&] {
    retained.emplace(a + b);
    c = *retained * 2;
  });
  EXPECT_EQ(c.decrypt(), 10);
  EXPECT_EQ(retained->decrypt(), 5);
}

TEST(SessionTest, AssignmentFromUnencryptedInputDoesNotUseDestinationCiphertext) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());
  Fhenon<int> source = 2, destination = 9;
  destination.belong(profile);
  EXPECT_THROW(session->run([&] { destination = source; }), std::runtime_error);
  EXPECT_FALSE(source.isEncrypted_);
  EXPECT_EQ(source.decrypt(), 2);
  EXPECT_EQ(destination.decrypt(), 9);
}

TEST(SessionTest, SharedCiphertextInputsRemainUsableAcrossRuns) {
  auto profile = makeProfile();
  auto session = Session::create(Backend::getInstance());
  Fhenon<int> a = 2, b = 0, c = 0;
  a.belong(profile);
  b.belong(profile);
  c.belong(profile);
  session->run([&] { b = a; });
  ASSERT_NO_THROW(session->run([&] { c = a + b; }));
  EXPECT_EQ(c.decrypt(), 4);
  ASSERT_NO_THROW(session->run([&] {
    a = a + 1;
    c = a + b;
  }));
  EXPECT_EQ(a.decrypt(), 3);
  EXPECT_EQ(b.decrypt(), 2);
  EXPECT_EQ(c.decrypt(), 5);
  Fhenon<int> copy = b;
  ASSERT_NO_THROW(session->run([&] { c = b + copy; }));
  EXPECT_EQ(c.decrypt(), 4);
}

TEST(SessionTest, LegacyAssignmentRejectsUnencryptedInputBeforeChangingDestination) {
  auto profile = makeProfile();
  LegacyBuiltinBackend backend;
  auto session = Session::create(backend);
  Fhenon<int> source = 2, destination = 9;
  destination.belong(profile);
  EXPECT_THROW(session->run([&] { destination = source; }), std::runtime_error);
  EXPECT_EQ(destination.getValue(), 9);
  EXPECT_EQ(destination.decrypt(), 9);
  EXPECT_FALSE(source.isEncrypted_);
}
