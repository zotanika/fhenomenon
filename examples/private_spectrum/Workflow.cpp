#include "Workflow.h"
#include "Session/Session.h"
#include <chrono>
#include <iostream>
#include <set>
#include <tuple>

namespace private_spectrum {
using namespace fhenomenon::runtime;

const char *modeName(Mode mode) {
  switch (mode) {
    case Mode::Tile: return "pair tile";
    case Mode::Pair: return "ordinary pair recipes";
    case Mode::Native: return "native operations";
  }
  throw std::invalid_argument("invalid monitor mode");
}

EncryptedMonitor::EncryptedMonitor(Runtime &runtime, NativeServices services, const std::string &path, Mode mode)
  : runtime_(runtime), services_(std::move(services)), native_(runtime, services_), spectral_(runtime), mode_(mode) {
  modeName(mode_);
  if (mode_ != Mode::Native)
    fhenomenon::contrib::registerSpectralKernels(runtime_, services_, path);
  ScopeOptions options;
  options.use_kernels = mode_ == Mode::Tile;
  session_ = fhenomenon::Session::create(runtime_, options);
}

PreparationEvidence EncryptedMonitor::append(const EncryptedFrame &frame) {
  if (frames_ == maxFrames)
    throw std::invalid_argument("monitor window is full");
  const auto freshLevel = services_.fresh_level(native_.context());
  double frameScale = inputScale_;
  for (const auto &sensor : frame) {
    if (!runtime_.owns(sensor))
      throw std::invalid_argument("frame ciphertext does not belong to this runtime");
    FhnCkksBufferInfo info{};
    try {
      if (services_.buffer_info(native_.context(), native_.buffer(sensor), &info))
        throw std::invalid_argument("invalid frame ciphertext");
    } catch (const Error &e) {
      throw std::invalid_argument(e.what());
    }
    if (info.logical_slots != binCount || info.level != freshLevel || info.components != 2 ||
        !std::isfinite(info.scale) || info.scale <= 0 || (frameScale && frameScale != info.scale))
      throw std::invalid_argument("incompatible fresh frame ciphertext layout");
    frameScale = info.scale;
  }

  const auto cross = [&](Scope &scope, ScopeValue sensor, ScopeValue conjugate, ScopeValue preparation) {
    if (mode_ == Mode::Native)
      return native_.op(scope, FHN_HMULT, {sensor, conjugate});
    return spectral_.pair(scope, sensor, preparation);
  };
  const auto power = [&](Scope &scope, ScopeValue sensor) {
    auto conjugate = native_.op(scope, FHN_CONJUGATE, {sensor});
    return native_.op(scope, FHN_HMULT, {sensor, conjugate});
  };
  const auto add = [&](Scope &scope, ScopeValue product, const Value &previous) {
    if (!previous)
      return product;
    return native_.op(scope, FHN_ADD_CC, {scope.input(previous), product});
  };

  // The reference auto-spectrum is both sensor 0's cross-spectrum and power.
  // Capture adjacent cross requests before emitting native power and ADD.
  auto first = session_->run([&](Scope &scope) {
    const auto reference = scope.input(frame[0]);
    const auto sensor = scope.input(frame[1]);
    ScopeValue conjugate, preparation;
    if (mode_ == Mode::Native)
      conjugate = native_.op(scope, FHN_CONJUGATE, {reference});
    else
      std::tie(conjugate, preparation) = spectral_.prepareReference(scope, reference);

    const auto referencePower = cross(scope, reference, conjugate, preparation);
    const auto sensorCross = cross(scope, sensor, conjugate, preparation);
    const auto sensorPower = power(scope, sensor);
    ScopeValues exports{conjugate, referencePower, sensorCross,
                        add(scope, referencePower, accumulated_.power[0]),
                        add(scope, sensorPower, accumulated_.power[1])};
    if (mode_ != Mode::Native)
      exports.push_back(preparation);
    return exports;
  });
  const auto &conjugate = first[0];
  const auto &sensorCross = first[2];
  const Value preparation = mode_ == Mode::Native ? Value{} : first[5];
  PreparationEvidence evidence;
  if (preparation)
    evidence.prepared = preparation.identity();

  // The same frame-local preparation crosses the scope boundary. The exported
  // sensor-1 cross product is an ordinary ciphertext consumed by native ADD.
  auto second = session_->run([&](Scope &scope) {
    const auto third = scope.input(frame[2]);
    const auto fourth = scope.input(frame[3]);
    const auto refConjugate = scope.input(conjugate);
    ScopeValue refPreparation;
    if (preparation) {
      refPreparation = scope.input(preparation);
      evidence.reused = preparation.identity();
    }
    const auto thirdCross = cross(scope, third, refConjugate, refPreparation);
    const auto fourthCross = cross(scope, fourth, refConjugate, refPreparation);
    const auto thirdPower = power(scope, third);
    const auto fourthPower = power(scope, fourth);
    return ScopeValues{add(scope, scope.input(sensorCross), accumulated_.cross[1]),
                       add(scope, thirdCross, accumulated_.cross[2]),
                       add(scope, fourthCross, accumulated_.cross[3]),
                       add(scope, thirdPower, accumulated_.power[2]),
                       add(scope, fourthPower, accumulated_.power[3])};
  });

  // Commit the new state only after both scopes succeed. The ADD outputs
  // retain no inputs, so prior frames and preparations can be released.
  accumulated_ = {{first[3], second[0], second[1], second[2]}, {first[3], first[4], second[3], second[4]}};
  inputScale_ = frameScale;
  ++frames_;
  return evidence;
}

const EncryptedSpectrum &EncryptedMonitor::result() const {
  if (!frames_)
    throw std::logic_error("monitor has no frames");
  return accumulated_;
}

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

Spectrum runMode(const std::string &path, const std::vector<Frame> &frames, const Spectrum &expected, Mode mode) {
  Runtime runtime({0, 8192});
  auto start = Clock::now();
  Client owner(runtime, path);
  const auto clientSetupMs = elapsed(start);
  EncryptedMonitor monitor(runtime, owner.computeServices(), path, mode);

  double encryptMs = 0;
  double evaluatorMs = 0;
  std::vector<PreparationEvidence> evidence;
  for (const auto &frame : frames) {
    start = Clock::now();
    const auto encrypted = owner.encrypt(frame);
    encryptMs += elapsed(start);
    start = Clock::now();
    const auto reuse = monitor.append(encrypted);
    evaluatorMs += elapsed(start);
    evidence.push_back(reuse);
  }
  start = Clock::now();
  const auto result = owner.decrypt(monitor.result());
  const auto decryptMs = elapsed(start);
  const auto error = maxComplexError(result, expected);
  if (error > tolerance)
    throw std::runtime_error("decrypted spectra differ from the complex oracle");

  std::cout << "\n" << modeName(mode) << "\n"
            << "  Evaluator append time " << evaluatorMs << " ms\n"
            << "  Client setup " << clientSetupMs << " ms, encrypt " << encryptMs << " ms, decrypt "
            << decryptMs << " ms\n"
            << "  Max complex error " << error << "\n"
            << "  Peak accounted bytes " << runtime.stats().peak_live_bytes << " (excludes keys and scratch)\n";
  for (unsigned i = 0; i < evidence.size(); ++i)
    if (evidence[i].prepared)
      std::cout << "  Frame " << i + 1 << " preparation " << evidence[i].prepared
                << " imported as " << evidence[i].reused << "\n";
  std::set<std::string> seen;
  for (const auto &event : runtime.trace()) {
    if (event.kind != TraceKind::Selected || (event.operation == "scope.span" && event.recipe == "runtime.submit"))
      continue;
    const auto description = event.operation == "scope.span" ? event.recipe + " " + event.detail : event.recipe;
    if (seen.insert(description).second)
      std::cout << "  Selected " << description << "\n";
  }
  return result;
}
} // namespace

void run(const std::string &path, unsigned frameCount, Mode mode, bool compare) {
  const auto frames = makeFrames(frameCount);
  const auto expected = cleartextSpectrum(frames);
  std::cout << "Monitoring " << frameCount << " frames, " << sensorCount << " sensors, " << binCount
            << " selected bins per sensor\n";
  const auto result = runMode(path, frames, expected, mode);
  if (compare) {
    for (const auto baseline : {Mode::Tile, Mode::Pair, Mode::Native}) {
      if (baseline == mode)
        continue;
      const auto other = runMode(path, frames, expected, baseline);
      if (maxComplexError(result, other) > 2 * tolerance)
        throw std::runtime_error("evaluator modes disagree");
    }
  }
  std::cout << "\nDecrypted diagnostics at selected bin 0\n";
  for (unsigned sensor = 0; sensor < sensorCount; ++sensor) {
    std::cout << "  Sensor " << sensor << "  ";
    if (const auto value = diagnostic(result, sensor, 0))
      std::cout << "phase " << value->phase << " rad, coherence " << value->coherence << "\n";
    else
      std::cout << "unavailable\n";
  }
}
} // namespace private_spectrum
