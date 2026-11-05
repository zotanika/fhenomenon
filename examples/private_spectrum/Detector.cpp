#include "Detector.h"
#include <cmath>
#include <stdexcept>

namespace private_spectrum {
namespace {
enum class Evidence { Negative, Positive, Uncertain };
constexpr double powerFloor = 1e-8;
constexpr double gainThreshold = 2.25;
constexpr double gainMargin = 0.05;
constexpr double coherenceThreshold = 0.8;
constexpr double coherenceMargin = 0.02;

void validateFrames(unsigned frames) {
  if (frames < minFrames || frames > maxFrames)
    throw std::invalid_argument("frame count must be between 2 and 16");
}

bool validPower(std::complex<double> power) {
  return std::isfinite(power.real()) && std::isfinite(power.imag()) && power.real() >= 0 &&
         std::abs(power.imag()) <= tolerance;
}

Evidence aboveThreshold(double value, double threshold, double margin) {
  // Include both boundary values in the uncertainty band, allowing for a few
  // floating-point rounding steps in normalization and magnitude calculation.
  constexpr double rounding = 1e-12;
  if (value > threshold + margin + rounding)
    return Evidence::Positive;
  if (value < threshold - margin - rounding)
    return Evidence::Negative;
  return Evidence::Uncertain;
}

Evidence gainEvidence(std::complex<double> baseline, unsigned baselineFrames, std::complex<double> current,
                      unsigned frames) {
  if (!validPower(baseline) || !validPower(current))
    return Evidence::Uncertain;
  const double before = baseline.real() / baselineFrames;
  const double now = current.real() / frames;
  if (now <= powerFloor)
    return Evidence::Negative;
  if (before <= powerFloor)
    return Evidence::Uncertain;
  // Infinite gain from division of finite, nonnegative powers is definitely
  // above the threshold; it does not imply nonfinite input evidence.
  return aboveThreshold(now / before, gainThreshold, gainMargin);
}

Evidence coherenceEvidence(const Spectrum &window, unsigned sensor, unsigned bin) {
  const auto power = window.power[sensor][bin];
  const auto reference = window.power[0][bin];
  const auto cross = window.cross[sensor][bin];
  if (!validPower(power) || !validPower(reference) || !std::isfinite(cross.real()) || !std::isfinite(cross.imag()))
    return Evidence::Uncertain;
  if (power.real() <= powerFloor || reference.real() <= powerFloor)
    return Evidence::Negative;
  // Dividing before squaring avoids overflowing the product of large powers.
  // A zero cross sum with positive powers is valid zero coherence.
  const double normalized = std::abs(cross) / std::sqrt(power.real()) / std::sqrt(reference.real());
  const double coherence = normalized * normalized;
  if (!std::isfinite(coherence) || coherence > 1 + tolerance)
    return Evidence::Uncertain;
  return aboveThreshold(coherence, coherenceThreshold, coherenceMargin);
}

Evidence both(Evidence a, Evidence b) {
  if (a == Evidence::Negative || b == Evidence::Negative)
    return Evidence::Negative;
  if (a == Evidence::Positive && b == Evidence::Positive)
    return Evidence::Positive;
  return Evidence::Uncertain;
}

Evidence binEvidence(const Spectrum &baseline, unsigned baselineFrames, const Spectrum &window, unsigned frames,
                     unsigned bin) {
  auto reference = gainEvidence(baseline.power[0][bin], baselineFrames, window.power[0][bin], frames);
  // Its self cross spectrum equals its power; do not accept a malformed anchor.
  const auto referenceCross = window.cross[0][bin];
  const auto referencePower = window.power[0][bin];
  if (!std::isfinite(referenceCross.real()) || !std::isfinite(referenceCross.imag()) || !validPower(referencePower) ||
      std::abs(referenceCross - referencePower) > tolerance)
    reference = Evidence::Uncertain;
  if (reference == Evidence::Negative)
    return Evidence::Negative;

  unsigned positive = 0;
  unsigned uncertain = 0;
  for (unsigned sensor = 1; sensor < sensorCount; ++sensor) {
    const auto participation =
      both(gainEvidence(baseline.power[sensor][bin], baselineFrames, window.power[sensor][bin], frames),
           coherenceEvidence(window, sensor, bin));
    positive += participation == Evidence::Positive;
    uncertain += participation == Evidence::Uncertain;
  }
  if (positive + uncertain < 2)
    return Evidence::Negative;
  if (reference == Evidence::Positive && positive >= 2)
    return Evidence::Positive;
  return Evidence::Uncertain;
}
} // namespace

const char *alarmName(Alarm alarm) {
  switch (alarm) {
  case Alarm::Quiet:
    return "quiet";
  case Alarm::SharedAnomaly:
    return "shared-anomaly";
  case Alarm::Indeterminate:
    return "indeterminate";
  }
  throw std::invalid_argument("unknown spectrum alarm");
}

JointDetector::JointDetector(const Spectrum &baseline, unsigned baselineFrames)
  : baseline_(baseline), baselineFrames_(baselineFrames) {
  validateFrames(baselineFrames);
}

Alarm JointDetector::observe(const Spectrum &window, unsigned frames) {
  validateFrames(frames);
  bool shared = false;
  bool uncertain = false;
  for (unsigned bin = 0; bin < binCount; ++bin) {
    const auto evidence = binEvidence(baseline_, baselineFrames_, window, frames, bin);
    shared |= evidence == Evidence::Positive && previousPositive_[bin];
    uncertain |= evidence == Evidence::Uncertain || (evidence == Evidence::Positive && previousUncertain_[bin]);
    previousPositive_[bin] = evidence == Evidence::Positive;
    previousUncertain_[bin] = evidence == Evidence::Uncertain;
  }
  if (shared)
    return Alarm::SharedAnomaly;
  return uncertain ? Alarm::Indeterminate : Alarm::Quiet;
}
} // namespace private_spectrum
