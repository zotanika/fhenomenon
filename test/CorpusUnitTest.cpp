#include "corpus_backend.h"
#include "corpus_builder.h"
#include "corpus_oracle.h"

#include <gtest/gtest.h>

using namespace fhenomenon::corpus;

TEST(CorpusOracle, ElementwiseArithmetic) {
  ShapeBuilder b;
  const uint32_t x = b.input({1, 2, 3, 4});
  const uint32_t y = b.input({10, 20, 30, 40});
  const uint32_t s = b.cc(FHN_ADD_CC, x, y);
  const uint32_t d = b.cc(FHN_SUB_CC, y, x);
  const uint32_t p = b.cc(FHN_MULT_CC, x, y);
  const uint32_t n = b.un(FHN_NEGATE, x);
  Shape shape = b.finish("t", "test", 4, 1, {s, d, p, n});

  auto vals = evaluate(*shape.program, shape.inputs);
  ASSERT_TRUE(vals.has_value());
  EXPECT_EQ(vals->at(s), (Slots{11, 22, 33, 44}));
  EXPECT_EQ(vals->at(d), (Slots{9, 18, 27, 36}));
  EXPECT_EQ(vals->at(p), (Slots{10, 40, 90, 160}));
  EXPECT_EQ(vals->at(n), (Slots{-1, -2, -3, -4}));
}

TEST(CorpusOracle, ScalarOpsUseFparams) {
  ShapeBuilder b;
  const uint32_t x = b.input({5, -3});
  const uint32_t a = b.cs(FHN_ADD_CS, x, 7);
  const uint32_t m = b.cs(FHN_MULT_CS, x, -2);
  Shape shape = b.finish("t", "test", 2, 0, {a, m});

  auto vals = evaluate(*shape.program, shape.inputs);
  ASSERT_TRUE(vals.has_value());
  EXPECT_EQ(vals->at(a), (Slots{12, 4}));
  EXPECT_EQ(vals->at(m), (Slots{-10, 6}));
}

TEST(CorpusOracle, RotateIsCyclicSignedLeft) {
  ShapeBuilder b;
  const uint32_t x = b.input({1, 2, 3, 4});
  const uint32_t l = b.rot(x, 1);  // positive = left
  const uint32_t r = b.rot(x, -1); // negative = right
  const uint32_t w = b.rot(x, 5);  // wraps mod slot count
  Shape shape = b.finish("t", "test", 4, 0, {l, r, w});

  auto vals = evaluate(*shape.program, shape.inputs);
  ASSERT_TRUE(vals.has_value());
  EXPECT_EQ(vals->at(l), (Slots{2, 3, 4, 1}));
  EXPECT_EQ(vals->at(r), (Slots{4, 1, 2, 3}));
  EXPECT_EQ(vals->at(w), (Slots{2, 3, 4, 1}));
}

TEST(CorpusOracle, BooleanAndComparisonOps) {
  ShapeBuilder b;
  const uint32_t x = b.input({3, 5, 7});
  const uint32_t y = b.input({3, 9, 2});
  const uint32_t eq = b.cc(FHN_EQ, x, y);
  const uint32_t lt = b.cc(FHN_LT, x, y);
  const uint32_t le = b.cc(FHN_LE, x, y);
  const uint32_t an = b.cc(FHN_AND, eq, le);
  Shape shape = b.finish("t", "test", 3, 0, {eq, lt, le, an});

  auto vals = evaluate(*shape.program, shape.inputs);
  ASSERT_TRUE(vals.has_value());
  EXPECT_EQ(vals->at(eq), (Slots{1, 0, 0}));
  EXPECT_EQ(vals->at(lt), (Slots{0, 1, 0}));
  EXPECT_EQ(vals->at(le), (Slots{1, 1, 0}));
  EXPECT_EQ(vals->at(an), (Slots{1, 0, 0}));
}

TEST(CorpusOracle, FusedOpcodeIsRejected) {
  ShapeBuilder b;
  const uint32_t x = b.input({1});
  const uint32_t y = b.input({2});
  const uint32_t h = b.cc(FHN_HMULT, x, y); // corpus generators never emit fused ops
  Shape shape = b.finish("t", "test", 1, 1, {h});
  EXPECT_FALSE(evaluate(*shape.program, shape.inputs).has_value());
}

TEST(CorpusOracle, SlotSizeMismatchIsRejected) {
  ShapeBuilder b;
  const uint32_t x = b.input({1, 2});
  const uint32_t y = b.input({1, 2, 3});
  const uint32_t s = b.cc(FHN_ADD_CC, x, y);
  Shape shape = b.finish("t", "test", 2, 0, {s});
  EXPECT_FALSE(evaluate(*shape.program, shape.inputs).has_value());
}

static std::string getTestLibPath() {
#ifdef __APPLE__
  return std::string(TEST_LIB_DIR) + "/libtoyfhe_fhn.dylib";
#else
  return std::string(TEST_LIB_DIR) + "/libtoyfhe_fhn.so";
#endif
}

TEST(CorpusBackend, LoadsToyFheAndResolvesDataPlane) {
  std::string error;
  auto backend = CorpusBackend::load(getTestLibPath(), "toyfhe_", &error);
  ASSERT_TRUE(backend.has_value()) << error;
  EXPECT_NE(backend->ctx(), nullptr);
  EXPECT_NE(backend->kernels(), nullptr);
  EXPECT_NE(backend->encryptI64(), nullptr);
  EXPECT_NE(backend->decryptI64(), nullptr);
  // ToyFHE exports no movement hooks.
  EXPECT_EQ(backend->prefetch(), nullptr);
  EXPECT_EQ(backend->evict(), nullptr);

  // Round-trip one value through the data plane.
  FhnBuffer *buf = backend->bufferAlloc()(backend->ctx());
  ASSERT_NE(buf, nullptr);
  ASSERT_EQ(backend->encryptI64()(backend->ctx(), buf, 42), 0);
  int64_t out = 0;
  ASSERT_EQ(backend->decryptI64()(backend->ctx(), buf, &out), 0);
  EXPECT_EQ(out, 42);
  backend->bufferFree()(backend->ctx(), buf);
}

TEST(CorpusBackend, MissingLibraryReportsError) {
  std::string error;
  auto backend = CorpusBackend::load("/nonexistent/libnope.so", "toyfhe_", &error);
  EXPECT_FALSE(backend.has_value());
  EXPECT_FALSE(error.empty());
}
