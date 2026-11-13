#include "Crypto/ToyFHE.h"

#include <gtest/gtest.h>
#include <stdexcept>

using namespace fhenomenon::toyfhe;

class ToyFheEngineTest : public ::testing::Test {
  protected:
  void SetUp() override {
    engine_.initialize(Parameters{});
    engine_.generateKeys();
  }
  Engine engine_;
};

TEST_F(ToyFheEngineTest, MultiplyIntegersExactly) {
  const auto product = engine_.multiply(engine_.encryptInt(2), engine_.encryptInt(3));
  EXPECT_EQ(engine_.decryptInt(product), 6);
  EXPECT_EQ(product.scale_power, 0);
  EXPECT_EQ(product.encoding, Encoding::Integer);

  EXPECT_EQ(engine_.decryptInt(engine_.multiply(engine_.encryptInt(13), engine_.encryptInt(20))), 260);
  EXPECT_EQ(engine_.decryptInt(engine_.multiply(engine_.encryptInt(-4), engine_.encryptInt(5))), -20);
  EXPECT_EQ(engine_.decryptInt(engine_.multiply(engine_.encryptInt(0), engine_.encryptInt(123))), 0);
}

TEST_F(ToyFheEngineTest, MultiplyChainsStayExact) {
  // ((2 * 3) + 4) * 5 = 50
  auto t = engine_.multiply(engine_.encryptInt(2), engine_.encryptInt(3));
  t = engine_.add(t, engine_.encryptInt(4));
  t = engine_.multiply(t, engine_.encryptInt(5));
  EXPECT_EQ(engine_.decryptInt(t), 50);

  // The documented worst-case budget: chained products of three-digit
  // integers stay exact (999^3 noise ~2.4e7 < delta/2 ~3.4e7).
  auto big = engine_.multiply(engine_.encryptInt(999), engine_.encryptInt(999));
  big = engine_.multiply(big, engine_.encryptInt(999));
  EXPECT_EQ(engine_.decryptInt(big), 997002999);
}

TEST_F(ToyFheEngineTest, AddPlainFractionalScalarOnIntegerCiphertext) {
  // An integer ciphertext must be raised to a scale level before absorbing a
  // fractional scalar, or the scalar rounds away.
  const auto sum = engine_.addPlain(engine_.encryptInt(10), 0.5);
  EXPECT_EQ(sum.encoding, Encoding::FixedPoint);
  EXPECT_NEAR(engine_.decryptDouble(sum), 10.5, 1e-3);

  const auto fixed = engine_.addPlain(engine_.encryptDouble(1.25), 2.25);
  EXPECT_NEAR(engine_.decryptDouble(fixed), 3.5, 1e-3);
}

TEST_F(ToyFheEngineTest, RejectsNonDivisibleModuli) {
  Parameters bad;
  bad.q = (static_cast<int64_t>(1) << 58) + (static_cast<int64_t>(1) << 20);
  Engine engine;
  EXPECT_THROW(engine.initialize(bad), std::runtime_error);
}

TEST_F(ToyFheEngineTest, MultiplyFixedPointRescalesToOneLevel) {
  const auto product = engine_.multiply(engine_.encryptDouble(3.0), engine_.encryptDouble(2.0));
  EXPECT_EQ(product.scale_power, 1);
  EXPECT_EQ(product.encoding, Encoding::FixedPoint);
  EXPECT_NEAR(engine_.decryptDouble(product), 6.0, 1e-3);

  const auto frac = engine_.multiply(engine_.encryptDouble(1.5), engine_.encryptDouble(2.5));
  EXPECT_NEAR(engine_.decryptDouble(frac), 3.75, 1e-3);
}

TEST_F(ToyFheEngineTest, MultiplyMixedEncodings) {
  const auto product = engine_.multiply(engine_.encryptDouble(2.5), engine_.encryptInt(4));
  EXPECT_EQ(product.scale_power, 1);
  EXPECT_NEAR(engine_.decryptDouble(product), 10.0, 1e-3);
}

TEST_F(ToyFheEngineTest, MultiplyPlainIntegralScalarKeepsScaleLevel) {
  const auto ct = engine_.encryptDouble(3.0);
  const auto product = engine_.multiplyPlain(ct, 2.0);
  EXPECT_EQ(product.scale_power, ct.scale_power);
  EXPECT_NEAR(engine_.decryptDouble(product), 6.0, 1e-3);
}

TEST_F(ToyFheEngineTest, MultiplyPlainFractionalScalarRescales) {
  const auto product = engine_.multiplyPlain(engine_.encryptDouble(3.0), 0.5);
  EXPECT_EQ(product.scale_power, 1);
  EXPECT_NEAR(engine_.decryptDouble(product), 1.5, 1e-3);

  // Repeated fractional multiplies must not exhaust the t = scale^2 headroom.
  auto ct = engine_.encryptDouble(64.0);
  for (int i = 0; i < 4; ++i) {
    ct = engine_.multiplyPlain(ct, 0.5);
  }
  EXPECT_NEAR(engine_.decryptDouble(ct), 4.0, 1e-3);
}
