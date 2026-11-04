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

void EncryptedMonitor::resetWindow() {
  accumulated_ = {};
  frames_ = 0;
  inputScale_ = 0;
}

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

using WindowAlarms = std::array<Alarm, 2>;
struct TrustedScenario {
  Scenario scenario;
  ScenarioFrames frames;
  Spectrum baseline;
  std::array<Spectrum, 2> observations;
  WindowAlarms expected;
};

// The synthetic fixture and its cleartext oracle belong to the trusted test
// harness. The evaluator never receives the cleartext spectra or detector state.
std::vector<TrustedScenario> trustedScenarios(const std::string &selection, unsigned frames) {
  std::vector<Scenario> selected;
  for (const auto scenario : {Scenario::Normal, Scenario::LocalDisturbance, Scenario::SharedOscillation})
    if (selection == "all" || selection == scenarioName(scenario))
      selected.push_back(scenario);
  if (selected.empty())
    throw std::invalid_argument("--scenario requires all, normal, local, or shared");

  std::vector<TrustedScenario> result;
  for (const auto scenario : selected) {
    TrustedScenario item{scenario, makeScenario(scenario, frames), {}, {}, {Alarm::Quiet, Alarm::Quiet}};
    item.baseline = cleartextSpectrum(item.frames.baseline);
    if (scenario == Scenario::SharedOscillation)
      item.expected[1] = Alarm::SharedAnomaly;
    JointDetector oracle(item.baseline, frames);
    for (unsigned window = 0; window < item.observations.size(); ++window) {
      item.observations[window] = cleartextSpectrum(item.frames.observations[window]);
      if (oracle.observe(item.observations[window], frames) != item.expected[window])
        throw std::runtime_error("cleartext detector differs from the literal scenario expectation");
    }
    result.push_back(std::move(item));
  }
  return result;
}

struct ExecutionEvidence {
  double encryptMs = 0;
  double evaluatorMs = 0;
  double trustedMs = 0;
  unsigned preparations = 0;
  PreparationEvidence firstPreparation;
  std::uint64_t previousPreparation = 0;
};

void evaluateWindow(Runtime &runtime, Client &owner, EncryptedMonitor &monitor, const std::vector<Frame> &frames,
                    const Spectrum &expected, ExecutionEvidence &evidence) {
  monitor.resetWindow();
  if (runtime.stats().live_resources != 0)
    throw std::runtime_error("previous window retained ciphertexts after reset");
  for (const auto &frame : frames) {
    auto start = Clock::now();
    const auto encrypted = owner.encrypt(frame);
    evidence.encryptMs += elapsed(start);
    start = Clock::now();
    const auto reuse = monitor.append(encrypted);
    evidence.evaluatorMs += elapsed(start);
    if (reuse.prepared) {
      if (reuse.prepared != reuse.reused || reuse.prepared == evidence.previousPreparation)
        throw std::runtime_error("frame preparation was not freshly created and reused across scopes");
      if (!evidence.preparations)
        evidence.firstPreparation = reuse;
      ++evidence.preparations;
      evidence.previousPreparation = reuse.prepared;
    }
  }
  if (runtime.stats().live_resources != 7)
    throw std::runtime_error("completed window must retain seven ciphertexts");
  const auto start = Clock::now();
  const auto error = owner.verify(monitor.result(), expected);
  evidence.trustedMs += elapsed(start);
  if (error > tolerance)
    throw std::runtime_error("decrypted spectra differ from the complex oracle");
}

void printExecutionEvidence(const Runtime &runtime, const ExecutionEvidence &evidence, double clientSetupMs) {
  std::cout << "  Evaluator append time " << evidence.evaluatorMs << " ms\n"
            << "  Client setup " << clientSetupMs << " ms, encrypt " << evidence.encryptMs
            << " ms, trusted verification/verdict " << evidence.trustedMs << " ms\n"
            << "  Retained ciphertexts: 7 per completed window, 0 after reset\n"
            << "  Peak accounted bytes " << runtime.stats().peak_live_bytes << " (excludes keys and scratch)\n";
  if (evidence.preparations)
    std::cout << "  Reference preparations " << evidence.preparations << ", all reused across scopes"
              << " (first " << evidence.firstPreparation.prepared << " imported as "
              << evidence.firstPreparation.reused << ")\n";
  std::set<std::string> seen;
  for (const auto &event : runtime.trace()) {
    if (event.kind != TraceKind::Selected || (event.operation == "scope.span" && event.recipe == "runtime.submit"))
      continue;
    const auto description = event.operation == "scope.span" ? event.recipe + " " + event.detail : event.recipe;
    if (seen.insert(description).second)
      std::cout << "  Selected " << description << "\n";
  }
}

std::vector<WindowAlarms> runMode(const std::string &path, const std::vector<TrustedScenario> &scenarios, Mode mode) {
  Runtime runtime({0, 8192});
  auto start = Clock::now();
  Client owner(runtime, path);
  const auto clientSetupMs = elapsed(start);
  EncryptedMonitor monitor(runtime, owner.computeServices(), path, mode);
  ExecutionEvidence evidence;
  std::vector<WindowAlarms> outcomes;
  std::cout << "\n" << modeName(mode) << "\n";
  for (const auto &scenario : scenarios) {
    std::cout << "  Scenario " << scenarioName(scenario.scenario) << "\n";
    evaluateWindow(runtime, owner, monitor, scenario.frames.baseline, scenario.baseline, evidence);
    start = Clock::now();
    owner.startMonitoring(monitor.result(), monitor.frames());
    evidence.trustedMs += elapsed(start);
    WindowAlarms alarms;
    for (unsigned window = 0; window < alarms.size(); ++window) {
      evaluateWindow(runtime, owner, monitor, scenario.frames.observations[window], scenario.observations[window],
                     evidence);
      start = Clock::now();
      alarms[window] = owner.observe(monitor.result(), monitor.frames());
      evidence.trustedMs += elapsed(start);
      if (alarms[window] != scenario.expected[window])
        throw std::runtime_error("encrypted verdict differs from the cleartext and literal scenario expectations");
      std::cout << "    Window " << window + 1 << " alarm: " << alarmName(alarms[window]) << "\n";
    }
    std::cout << "    Final alarm: " << alarmName(alarms.back()) << "\n";
    outcomes.push_back(alarms);
  }
  monitor.resetWindow();
  if (runtime.stats().live_resources != 0)
    throw std::runtime_error("last window retained ciphertexts after reset");
  printExecutionEvidence(runtime, evidence, clientSetupMs);
  return outcomes;
}
} // namespace

void run(const std::string &path, unsigned frameCount, Mode mode, bool compare, const std::string &selection) {
  const auto scenarios = trustedScenarios(selection, frameCount);
  std::cout << "Synthetic joint monitoring: " << frameCount << " frames per window, " << binCount
            << " selected bins per sensor\n"
            << "Each scenario uses a baseline and two observation windows.\n";
  for (const auto name : contributorNames)
    std::cout << "  " << name << ", one sensor\n";
  const auto result = runMode(path, scenarios, mode);
  if (compare) {
    for (const auto otherMode : {Mode::Tile, Mode::Pair, Mode::Native}) {
      if (otherMode != mode && runMode(path, scenarios, otherMode) != result)
        throw std::runtime_error("evaluator modes disagree on alarms");
    }
    // Each mode checks every complex bin against the same oracle, bounding any
    // pairwise spectral difference by twice the verification tolerance.
    std::cout << "\nAll modes agree on alarms and pass every complex-bin oracle check.\n";
  }
}
} // namespace private_spectrum
