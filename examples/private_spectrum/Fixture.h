#pragma once

#include <array>
#include <complex>
#include <optional>
#include <vector>

namespace private_spectrum {
inline constexpr unsigned sensorCount = 4;
inline constexpr unsigned binCount = 8;
inline constexpr unsigned minFrames = 2;
inline constexpr unsigned maxFrames = 16;
inline constexpr double tolerance = 1e-5;

using Bins = std::array<std::complex<double>, binCount>;
using Frame = std::array<Bins, sensorCount>;
struct Spectrum {
  Frame cross{};
  Frame power{};
};
struct Diagnostic {
  double phase;
  double coherence;
};

enum class Scenario { Normal, LocalDisturbance, SharedOscillation };
const char *scenarioName(Scenario);
inline constexpr std::array<const char *, sensorCount> contributorNames{
  "Simulated operator A (reference)", "Simulated operator B", "Simulated operator C", "Simulated operator D"};
struct ScenarioFrames {
  std::vector<Frame> baseline;
  std::array<std::vector<Frame>, 2> observations;
};

std::vector<Frame> makeFrames(unsigned count);
ScenarioFrames makeScenario(Scenario, unsigned count);
Spectrum cleartextSpectrum(const std::vector<Frame> &);
double maxComplexError(const Spectrum &, const Spectrum &);
std::optional<Diagnostic> diagnostic(const Spectrum &, unsigned sensor, unsigned bin);
} // namespace private_spectrum
