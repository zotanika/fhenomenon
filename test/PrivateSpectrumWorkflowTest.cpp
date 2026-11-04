#include "Workflow.h"
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace private_spectrum;
using namespace fhenomenon::runtime;

TEST(PrivateSpectrumWorkflow, AccumulatesComplexProductsInAllModes) {
  const auto frames = makeFrames(4);
  const auto expected = cleartextSpectrum(frames);
  for (auto mode : {Mode::Tile, Mode::Pair, Mode::Native}) {
    SCOPED_TRACE(modeName(mode));
    Runtime runtime({0, 8192});
    Client owner(runtime, FHN_NATIVE_TEST_LIBRARY);
    EncryptedMonitor monitor(runtime, owner.computeServices(), FHN_NATIVE_TEST_LIBRARY, mode);
    std::uint64_t previousPreparation = 0;
    for (const auto &frame : frames) {
      const auto evidence = monitor.append(owner.encrypt(frame));
      if (mode != Mode::Native) {
        EXPECT_NE(evidence.prepared, previousPreparation);
        EXPECT_EQ(evidence.prepared, evidence.reused);
        previousPreparation = evidence.prepared;
      }
      EXPECT_EQ(runtime.stats().live_resources, 7u);
    }
    EXPECT_LT(maxComplexError(owner.decrypt(monitor.result()), expected), 1e-5);
    bool tileSelected = false;
    for (const auto &event : runtime.trace())
      tileSelected |= event.kind == TraceKind::Selected && event.recipe == "contrib.spectral.scope.pair-tile";
    EXPECT_EQ(tileSelected, mode == Mode::Tile);
  }
  EXPECT_NEAR(diagnostic(expected, 1, 0)->phase, -0.65, 1e-12);
  EXPECT_GT(diagnostic(expected, 1, 0)->coherence, 0.99);
  EXPECT_LT(diagnostic(expected, 2, 0)->coherence, 0.5);
}

TEST(PrivateSpectrumWorkflow, RejectsForeignAndMissingInputsWithoutChangingState) {
  Runtime runtime;
  Client owner(runtime, FHN_NATIVE_TEST_LIBRARY);
  EncryptedMonitor monitor(runtime, owner.computeServices(), FHN_NATIVE_TEST_LIBRARY, Mode::Tile);
  const auto frames = makeFrames(2);
  monitor.append(owner.encrypt(frames[0]));
  const auto before = monitor.result().cross[1].identity();
  Runtime foreignRuntime;
  Client foreign(foreignRuntime, FHN_NATIVE_TEST_LIBRARY);
  auto bad = owner.encrypt(frames[1]);
  bad[3] = foreign.encrypt(frames[1])[3];
  EXPECT_THROW(monitor.append(bad), std::invalid_argument);
  bad[3] = {};
  EXPECT_THROW(monitor.append(bad), std::invalid_argument);
  EXPECT_EQ(monitor.result().cross[1].identity(), before);
  EXPECT_EQ(monitor.frames(), 1u);
}

TEST(PrivateSpectrumWorkflow, StopsAtTheFiniteWindowLimit) {
  Runtime runtime;
  Client owner(runtime, FHN_NATIVE_TEST_LIBRARY);
  EncryptedMonitor monitor(runtime, owner.computeServices(), FHN_NATIVE_TEST_LIBRARY, Mode::Native);
  EXPECT_THROW(monitor.result(), std::logic_error);
  const auto frames = makeFrames(maxFrames);
  for (const auto &frame : frames)
    monitor.append(owner.encrypt(frame));
  EXPECT_THROW(monitor.append(owner.encrypt(frames.front())), std::invalid_argument);
  EXPECT_EQ(monitor.frames(), maxFrames);
  EXPECT_EQ(runtime.stats().live_resources, 7u);
  EXPECT_LT(maxComplexError(owner.decrypt(monitor.result()), cleartextSpectrum(frames)), tolerance);
}

TEST(PrivateSpectrumWorkflow, RejectsUnavailableDiagnosticsAndNonfiniteResults) {
  Spectrum data{};
  EXPECT_FALSE(diagnostic(data, 1, 0));
  data.power[0][0] = 1;
  data.power[1][0] = 1;
  EXPECT_FALSE(diagnostic(data, 1, 0));
  data.cross[1][0] = {0, std::numeric_limits<double>::infinity()};
  EXPECT_FALSE(diagnostic(data, 1, 0));
  EXPECT_THROW(maxComplexError(data, Spectrum{}), std::runtime_error);
  data.cross[1][0] = 1.001;
  EXPECT_FALSE(diagnostic(data, 1, 0));
  data.cross[1][0] = 1 + tolerance / 4;
  ASSERT_TRUE(diagnostic(data, 1, 0));
  EXPECT_DOUBLE_EQ(diagnostic(data, 1, 0)->coherence, 1);
  data.power[1][0] = {1, 0.01};
  EXPECT_FALSE(diagnostic(data, 1, 0));
  EXPECT_THROW(makeFrames(17), std::invalid_argument);
  EXPECT_THROW(makeFrames(1), std::invalid_argument);
}
} // namespace
