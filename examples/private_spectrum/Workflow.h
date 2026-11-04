#pragma once

#include "FHN/contrib/SpectralKernels.h"
#include "Fixture.h"
#include <memory>

namespace fhenomenon {
class Session;
}

namespace private_spectrum {
using fhenomenon::runtime::NativeRuntime;
using fhenomenon::runtime::NativeServices;
using fhenomenon::runtime::Runtime;
using fhenomenon::runtime::Value;

using EncryptedFrame = std::array<Value, sensorCount>;
struct EncryptedSpectrum {
  EncryptedFrame cross;
  EncryptedFrame power;
};

enum class Mode { Tile, Pair, Native };
const char *modeName(Mode);

struct PreparationEvidence {
  std::uint64_t prepared = 0;
  std::uint64_t reused = 0;
};

// Owner holds the secret key and converts between selected local FFT bins and
// ciphertexts. The Runtime must outlive both roles and the client must remain
// alive while the evaluator uses the context's evaluation keys.
class Client {
  public:
  Client(Runtime &, const std::string &path);
  ~Client();
  NativeServices computeServices() const;
  EncryptedFrame encrypt(const Frame &);
  Spectrum decrypt(const EncryptedSpectrum &);

  private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Evaluator sees ciphertext and public metadata in this in-process simulation.
class EncryptedMonitor {
  public:
  EncryptedMonitor(Runtime &, NativeServices, const std::string &path, Mode);
  PreparationEvidence append(const EncryptedFrame &);
  const EncryptedSpectrum &result() const;
  unsigned frames() const { return frames_; }

  private:
  Runtime &runtime_;
  NativeServices services_;
  NativeRuntime native_;
  fhenomenon::contrib::SpectralRuntime spectral_;
  Mode mode_;
  std::shared_ptr<fhenomenon::Session> session_;
  EncryptedSpectrum accumulated_;
  unsigned frames_ = 0;
  double inputScale_ = 0;
};

void run(const std::string &path, unsigned frames, Mode mode, bool compare);
} // namespace private_spectrum
