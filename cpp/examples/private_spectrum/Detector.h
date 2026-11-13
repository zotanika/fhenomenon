#pragma once

#include "Fixture.h"

namespace private_spectrum {
enum class Alarm { Quiet, SharedAnomaly, Indeterminate };
const char *alarmName(Alarm);

// Trusted key-holder rule for the synthetic example, not a calibrated detector.
// Baseline and observation sums are normalized by their window frame counts.
class JointDetector {
  public:
  JointDetector(const Spectrum &baseline, unsigned baselineFrames);
  Alarm observe(const Spectrum &window, unsigned frames);

  private:
  Spectrum baseline_;
  unsigned baselineFrames_;
  std::array<bool, binCount> previousPositive_{};
  std::array<bool, binCount> previousUncertain_{};
};
} // namespace private_spectrum
