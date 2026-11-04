#include "Fixture.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace private_spectrum {
std::vector<Frame> makeFrames(unsigned count) {
  if (count < minFrames || count > maxFrames)
    throw std::invalid_argument("frame count must be between 2 and 16");
  std::vector<Frame> frames(count);
  constexpr std::array<double, 4> jitter{0.0, 2.4, -2.4, 1.1};
  for (unsigned frame = 0; frame < count; ++frame) {
    for (unsigned bin = 0; bin < binCount; ++bin) {
      const double angle = 0.26 + 0.21 * bin + 0.31 * frame;
      const double amplitude = 1.15 + 0.08 * bin;
      frames[frame][0][bin] = std::polar(amplitude, angle);
      frames[frame][1][bin] = std::polar(0.9 * amplitude, angle - 0.65 + 0.04 * bin);
      frames[frame][2][bin] = std::polar(0.85 * amplitude, angle + jitter[frame % jitter.size()]);
      frames[frame][3][bin] = std::polar(0.65 * amplitude + 0.07 * (frame % 3), angle + 0.95);
    }
  }
  return frames;
}
Spectrum cleartextSpectrum(const std::vector<Frame> &frames) {
  Spectrum result;
  for (const auto &frame : frames)
    for (unsigned sensor = 0; sensor < sensorCount; ++sensor)
      for (unsigned bin = 0; bin < binCount; ++bin) {
        const auto sample = frame[sensor][bin];
        result.cross[sensor][bin] += sample * std::conj(frame[0][bin]);
        result.power[sensor][bin] += std::norm(sample);
      }
  return result;
}
double maxComplexError(const Spectrum &a, const Spectrum &b) {
  double largest = 0;
  for (unsigned sensor = 0; sensor < sensorCount; ++sensor)
    for (unsigned bin = 0; bin < binCount; ++bin) {
      const double cross = std::abs(a.cross[sensor][bin] - b.cross[sensor][bin]);
      const double power = std::abs(a.power[sensor][bin] - b.power[sensor][bin]);
      if (!std::isfinite(cross) || !std::isfinite(power))
        throw std::runtime_error("nonfinite spectrum error");
      largest = std::max({largest, cross, power});
    }
  return largest;
}
std::optional<Diagnostic> diagnostic(const Spectrum &data, unsigned sensor, unsigned bin) {
  constexpr double floor = 1e-8;
  const auto power = data.power.at(sensor).at(bin);
  const auto reference = data.power[0].at(bin);
  const auto cross = data.cross.at(sensor).at(bin);
  const double denominator = power.real() * reference.real();
  const double magnitude = std::abs(cross);
  if (!std::isfinite(denominator) || !std::isfinite(magnitude) || !std::isfinite(power.imag()) ||
      !std::isfinite(reference.imag()) || power.real() <= floor || reference.real() <= floor ||
      denominator <= floor || magnitude <= floor || std::abs(power.imag()) > tolerance ||
      std::abs(reference.imag()) > tolerance)
    return std::nullopt;
  const double coherence = std::norm(cross) / denominator;
  if (!std::isfinite(coherence) || coherence > 1 + tolerance)
    return std::nullopt;
  return Diagnostic{std::arg(cross), std::min(coherence, 1.0)};
}
} // namespace private_spectrum
