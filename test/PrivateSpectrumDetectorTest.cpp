#include "Detector.h"
#include <gtest/gtest.h>
#include <cmath>
#include <limits>

namespace {
using namespace private_spectrum;

// Unit power per frame, perfectly coherent at every bin. Expected decisions
// below are literal: a factor-four gain for reference + two peers must persist.
Spectrum unitSpectrum(unsigned frames = 2) {
  Spectrum result;
  for (unsigned sensor = 0; sensor < sensorCount; ++sensor)
    for (unsigned bin = 0; bin < binCount; ++bin) {
      result.power[sensor][bin] = frames;
      result.cross[sensor][bin] = frames;
    }
  return result;
}

Spectrum raised(unsigned bin = 3, std::initializer_list<unsigned> sensors = {0, 1, 3},
                double gain = 4, double coherence = 1, unsigned frames = 2) {
  auto result = unitSpectrum(frames);
  for (auto sensor : sensors)
    result.power[sensor][bin] = frames * gain;
  for (unsigned sensor = 0; sensor < sensorCount; ++sensor) {
    const double magnitude = std::sqrt(result.power[sensor][bin].real() * result.power[0][bin].real());
    result.cross[sensor][bin] = sensor == 0 ? std::complex<double>{magnitude, 0}
                                         : std::complex<double>{0, magnitude * std::sqrt(coherence)};
  }
  return result;
}

TEST(PrivateSpectrumDetector, RequiresTwoConsecutiveSharedWindows) {
  JointDetector detector(unitSpectrum(), 2);
  EXPECT_EQ(detector.observe(raised(), 2), Alarm::Quiet);
  EXPECT_EQ(detector.observe(raised(), 2), Alarm::SharedAnomaly);
  EXPECT_EQ(detector.observe(raised(), 2), Alarm::SharedAnomaly);
  EXPECT_EQ(detector.observe(unitSpectrum(), 2), Alarm::Quiet);
  EXPECT_EQ(detector.observe(raised(), 2), Alarm::Quiet);
}

TEST(PrivateSpectrumDetector, StableCoherenceAndLocalPowerAreQuiet) {
  for (const auto &observation : {unitSpectrum(), raised(3, {1}), raised(3, {0, 1})}) {
    JointDetector detector(unitSpectrum(), 2);
    EXPECT_EQ(detector.observe(observation, 2), Alarm::Quiet);
    EXPECT_EQ(detector.observe(observation, 2), Alarm::Quiet);
  }
}

TEST(PrivateSpectrumDetector, RequiresReferenceParticipation) {
  JointDetector detector(unitSpectrum(), 2);
  EXPECT_EQ(detector.observe(raised(3, {1, 2, 3}), 2), Alarm::Quiet);
  EXPECT_EQ(detector.observe(raised(3, {1, 2, 3}), 2), Alarm::Quiet);
}

TEST(PrivateSpectrumDetector, TracksPersistenceAtTheSameBin) {
  JointDetector detector(unitSpectrum(), 2);
  EXPECT_EQ(detector.observe(raised(0), 2), Alarm::Quiet);
  EXPECT_EQ(detector.observe(raised(7), 2), Alarm::Quiet);
  EXPECT_EQ(detector.observe(raised(0), 2), Alarm::Quiet);
  EXPECT_EQ(detector.observe(raised(0), 2), Alarm::SharedAnomaly);
}

TEST(PrivateSpectrumDetector, ATransientDoesNotCarryAcrossANormalWindow) {
  JointDetector detector(unitSpectrum(), 2);
  EXPECT_EQ(detector.observe(raised(), 2), Alarm::Quiet);
  EXPECT_EQ(detector.observe(unitSpectrum(), 2), Alarm::Quiet);
  EXPECT_EQ(detector.observe(raised(), 2), Alarm::Quiet);
}

TEST(PrivateSpectrumDetector, NormalizesPowerByEachWindowFrameCount) {
  JointDetector detector(unitSpectrum(2), 2);
  EXPECT_EQ(detector.observe(unitSpectrum(16), 16), Alarm::Quiet);
  EXPECT_EQ(detector.observe(raised(3, {0, 1, 3}, 4, 1, 16), 16), Alarm::Quiet);
  EXPECT_EQ(detector.observe(raised(3, {0, 1, 3}, 4, 1, 4), 4), Alarm::SharedAnomaly);
}

TEST(PrivateSpectrumDetector, ZeroCrossWithPositivePowerIsValidIncoherence) {
  for (double coherence : {0.0, 0.5}) {
    JointDetector detector(unitSpectrum(), 2);
    auto observation = raised(3, {0, 1, 3}, 4, coherence);
    EXPECT_EQ(detector.observe(observation, 2), Alarm::Quiet);
    EXPECT_EQ(detector.observe(observation, 2), Alarm::Quiet);
  }
}

TEST(PrivateSpectrumDetector, MissingOptionalParticipantDoesNotPreventQuorum) {
  auto baseline = unitSpectrum();
  auto observation = raised();
  for (unsigned bin = 0; bin < binCount; ++bin) {
    baseline.power[2][bin] = observation.power[2][bin] = 0;
    baseline.cross[2][bin] = observation.cross[2][bin] = 0;
  }
  JointDetector detector(baseline, 2);
  EXPECT_EQ(detector.observe(observation, 2), Alarm::Quiet);
  EXPECT_EQ(detector.observe(observation, 2), Alarm::SharedAnomaly);
}

TEST(PrivateSpectrumDetector, AllZeroEnergyIsQuiet) {
  JointDetector detector(Spectrum{}, 2);
  EXPECT_EQ(detector.observe(Spectrum{}, 2), Alarm::Quiet);
  EXPECT_EQ(detector.observe(Spectrum{}, 2), Alarm::Quiet);
}

TEST(PrivateSpectrumDetector, NewlyPresentEnergyHasNoUsableBaseline) {
  JointDetector detector(Spectrum{}, 2);
  EXPECT_EQ(detector.observe(raised(), 2), Alarm::Indeterminate);
  EXPECT_EQ(detector.observe(raised(), 2), Alarm::Indeterminate);
}

TEST(PrivateSpectrumDetector, GainMarginCannotConfirmOrSilentlyPreservePersistence) {
  for (double gain : {2.20, 2.25, 2.30}) {
    SCOPED_TRACE(gain);
    JointDetector detector(unitSpectrum(), 2);
    EXPECT_EQ(detector.observe(raised(), 2), Alarm::Quiet);
    EXPECT_EQ(detector.observe(raised(3, {0, 1, 3}, gain), 2), Alarm::Indeterminate);
    EXPECT_EQ(detector.observe(raised(), 2), Alarm::Indeterminate);
    EXPECT_EQ(detector.observe(raised(), 2), Alarm::SharedAnomaly);
  }
}

TEST(PrivateSpectrumDetector, CoherenceMarginIsIndeterminate) {
  for (double coherence : {0.78, 0.8, 0.82}) {
    SCOPED_TRACE(coherence);
    JointDetector detector(unitSpectrum(), 2);
    EXPECT_EQ(detector.observe(raised(3, {0, 1, 3}, 4, coherence), 2), Alarm::Indeterminate);
    EXPECT_EQ(detector.observe(raised(), 2), Alarm::Indeterminate);
    EXPECT_EQ(detector.observe(raised(), 2), Alarm::SharedAnomaly);
  }
}

TEST(PrivateSpectrumDetector, ValuesClearlyOutsideMarginsHaveDefiniteDecisions) {
  for (const auto &observation : {raised(3, {0, 1, 3}, 2.19), raised(3, {0, 1, 3}, 4, 0.77)}) {
    JointDetector detector(unitSpectrum(), 2);
    EXPECT_EQ(detector.observe(observation, 2), Alarm::Quiet);
    EXPECT_EQ(detector.observe(observation, 2), Alarm::Quiet);
  }
  JointDetector detector(unitSpectrum(), 2);
  auto observation = raised(3, {0, 1, 3}, 2.31, 0.83);
  EXPECT_EQ(detector.observe(observation, 2), Alarm::Quiet);
  EXPECT_EQ(detector.observe(observation, 2), Alarm::SharedAnomaly);
}

TEST(PrivateSpectrumDetector, DefiniteNegativeResetsUncertainPersistence) {
  JointDetector detector(unitSpectrum(), 2);
  EXPECT_EQ(detector.observe(raised(3, {0, 1, 3}, 2.25), 2), Alarm::Indeterminate);
  EXPECT_EQ(detector.observe(unitSpectrum(), 2), Alarm::Quiet);
  EXPECT_EQ(detector.observe(raised(), 2), Alarm::Quiet);
}

TEST(PrivateSpectrumDetector, InvalidParticipantCannotSupplyTheRequiredQuorum) {
  for (int kind = 0; kind < 7; ++kind) {
    SCOPED_TRACE(kind);
    auto observation = raised();
    if (kind == 0)
      observation.power[1][3] = -1;
    if (kind == 1)
      observation.power[1][3] = {8, 0.01};
    if (kind == 2)
      observation.power[1][3] = std::numeric_limits<double>::infinity();
    if (kind == 3)
      observation.cross[1][3] = {0, std::numeric_limits<double>::quiet_NaN()};
    if (kind == 4)
      observation.cross[1][3] = {0, 9}; // Coherence > 1 is physically inconsistent.
    if (kind == 5)
      observation.power[0][3] = -1;
    if (kind == 6)
      observation.cross[0][3] = {8, 1}; // Reference cross must equal its own power.
    JointDetector detector(unitSpectrum(), 2);
    EXPECT_EQ(detector.observe(raised(), 2), Alarm::Quiet);
    EXPECT_EQ(detector.observe(observation, 2), Alarm::Indeterminate);
    EXPECT_EQ(detector.observe(raised(), 2), Alarm::Indeterminate);
    EXPECT_EQ(detector.observe(raised(), 2), Alarm::SharedAnomaly);
  }
}

TEST(PrivateSpectrumDetector, InvalidOptionalParticipantDoesNotVetoKnownQuorum) {
  auto observation = raised();
  observation.power[2][3] = std::numeric_limits<double>::quiet_NaN();
  JointDetector detector(unitSpectrum(), 2);
  EXPECT_EQ(detector.observe(observation, 2), Alarm::Quiet);
  EXPECT_EQ(detector.observe(observation, 2), Alarm::SharedAnomaly);
}

TEST(PrivateSpectrumDetector, InvalidBaselineCannotProduceAnAlarm) {
  for (double power : {-1.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    auto baseline = unitSpectrum();
    baseline.power[0][3] = power;
    JointDetector detector(baseline, 2);
    EXPECT_EQ(detector.observe(raised(), 2), Alarm::Indeterminate);
    EXPECT_EQ(detector.observe(raised(), 2), Alarm::Indeterminate);
  }
}

TEST(PrivateSpectrumDetector, ConfirmedSharedBinWinsOverAnUncertainOtherBin) {
  auto observation = raised();
  observation.power[0][7] = std::numeric_limits<double>::quiet_NaN();
  observation.power[1][7] = observation.power[3][7] = 8;
  observation.cross[1][7] = observation.cross[3][7] = 8;
  JointDetector detector(unitSpectrum(), 2);
  EXPECT_EQ(detector.observe(observation, 2), Alarm::Indeterminate);
  EXPECT_EQ(detector.observe(observation, 2), Alarm::SharedAnomaly);
}

TEST(PrivateSpectrumDetector, AvoidsIntermediateOverflowForLargeFiniteSpectra) {
  auto baseline = unitSpectrum();
  auto observation = raised();
  for (unsigned sensor = 0; sensor < sensorCount; ++sensor)
    for (unsigned bin = 0; bin < binCount; ++bin) {
      baseline.power[sensor][bin] *= 1e200;
      baseline.cross[sensor][bin] *= 1e200;
      observation.power[sensor][bin] *= 1e200;
      observation.cross[sensor][bin] *= 1e200;
    }
  JointDetector detector(baseline, 2);
  EXPECT_EQ(detector.observe(observation, 2), Alarm::Quiet);
  EXPECT_EQ(detector.observe(observation, 2), Alarm::SharedAnomaly);
}

TEST(PrivateSpectrumDetector, RejectsOutOfRangeWindowSizesWithoutAcceptingAWindow) {
  EXPECT_THROW(JointDetector(unitSpectrum(), 1), std::invalid_argument);
  EXPECT_THROW(JointDetector(unitSpectrum(), 17), std::invalid_argument);
  JointDetector detector(unitSpectrum(), 2);
  EXPECT_EQ(detector.observe(raised(), 2), Alarm::Quiet);
  EXPECT_THROW(detector.observe(raised(), 0), std::invalid_argument);
  EXPECT_EQ(detector.observe(raised(), 2), Alarm::SharedAnomaly);
  EXPECT_THROW(detector.observe(raised(), 17), std::invalid_argument);
}

TEST(PrivateSpectrumFixture, ScenariosHaveLiteralExpectedVerdictsAcrossWindowSizes) {
  for (unsigned frames : {2u, 4u, 16u})
    for (Scenario scenario : {Scenario::Normal, Scenario::LocalDisturbance, Scenario::SharedOscillation}) {
      SCOPED_TRACE(frames);
      SCOPED_TRACE(scenarioName(scenario));
      const auto data = makeScenario(scenario, frames);
      ASSERT_EQ(data.baseline.size(), frames);
      ASSERT_EQ(data.observations[0].size(), frames);
      ASSERT_EQ(data.observations[1].size(), frames);
      JointDetector detector(cleartextSpectrum(data.baseline), frames);
      EXPECT_EQ(detector.observe(cleartextSpectrum(data.observations[0]), frames), Alarm::Quiet);
      EXPECT_EQ(detector.observe(cleartextSpectrum(data.observations[1]), frames),
                scenario == Scenario::SharedOscillation ? Alarm::SharedAnomaly : Alarm::Quiet);
    }
}

TEST(PrivateSpectrumFixture, OnlySpecifiedSensorAndBinAmplitudesChange) {
  const auto original = makeFrames(4);
  for (Scenario scenario : {Scenario::Normal, Scenario::LocalDisturbance, Scenario::SharedOscillation}) {
    const auto data = makeScenario(scenario, 4);
    EXPECT_EQ(data.baseline, original);
    for (const auto &window : data.observations)
      for (unsigned frame = 0; frame < 4; ++frame)
        for (unsigned sensor = 0; sensor < sensorCount; ++sensor)
          for (unsigned bin = 0; bin < binCount; ++bin) {
            const bool boosted = bin == 3 &&
                                 ((scenario == Scenario::LocalDisturbance && sensor == 1) ||
                                  (scenario == Scenario::SharedOscillation && sensor != 2));
            EXPECT_EQ(window[frame][sensor][bin], original[frame][sensor][bin] * (boosted ? 2.0 : 1.0));
          }
  }
  EXPECT_THROW(makeScenario(Scenario::Normal, 1), std::invalid_argument);
  EXPECT_THROW(makeScenario(Scenario::Normal, 17), std::invalid_argument);
  EXPECT_THROW(makeScenario(static_cast<Scenario>(99), 4), std::invalid_argument);
}
} // namespace
