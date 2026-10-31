#include "FHN/FhnDefaultExecutor.h"
#include "corpus_backend.h"
#include "corpus_circuit.h"
#include "corpus_oracle.h"
#include "corpus_shapes.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fhenomenon;
using namespace fhenomenon::corpus;
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
struct Options {
  std::string backend, baseline, prefix = "ckks_", shape, config;
  uint32_t warmup = 2, reps = 7;
  uint64_t workspace = 0;
  double atol = 1e-4, rtol = 1e-6;
  bool required = false, list = false;
};
bool unsignedNumber(const char *s, uint64_t &n) {
  if (!s[0] || !std::all_of(s, s + std::strlen(s), [](char c) { return c >= '0' && c <= '9'; }))
    return false;
  char *end = nullptr;
  errno = 0;
  n = std::strtoull(s, &end, 10);
  return errno == 0 && end != s && *end == 0;
}
bool tolerance(const char *s, double &n) {
  char *end = nullptr;
  errno = 0;
  n = std::strtod(s, &end);
  return errno == 0 && end != s && *end == 0 && std::isfinite(n) && n >= 0;
}
void usage() {
  std::fprintf(stderr, "usage: fhn-bench-circuit --backend LIB [--prefix ckks_] [--shape NAME]\n"
                       "  [--baseline LIB] [--warmup N] [--reps N] [--workspace-bytes N]\n"
                       "  [--atol N] [--rtol N] [--config JSON_FILE] [--require-execution] [--list]\n");
}
bool parse(int argc, char **argv, Options &o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--require-execution")
      o.required = true;
    else if (a == "--list")
      o.list = true;
    else {
      if (++i == argc)
        return false;
      if (a == "--backend")
        o.backend = argv[i];
      else if (a == "--baseline")
        o.baseline = argv[i];
      else if (a == "--prefix")
        o.prefix = argv[i];
      else if (a == "--shape")
        o.shape = argv[i];
      else if (a == "--config") {
        std::ifstream input(argv[i]);
        if (!input)
          return false;
        o.config.assign(std::istreambuf_iterator<char>(input), {});
        if (o.config.empty() || o.config.find('\0') != std::string::npos)
          return false;
      } else if (a == "--atol") {
        if (!tolerance(argv[i], o.atol))
          return false;
      } else if (a == "--rtol") {
        if (!tolerance(argv[i], o.rtol))
          return false;
      } else if (a == "--warmup" || a == "--reps" || a == "--workspace-bytes") {
        uint64_t n = 0;
        if (!unsignedNumber(argv[i], n))
          return false;
        if (a == "--workspace-bytes")
          o.workspace = n;
        else {
          if (n > 100000 || (a == "--reps" && n == 0))
            return false;
          (a == "--warmup" ? o.warmup : o.reps) = static_cast<uint32_t>(n);
        }
      } else
        return false;
    }
  }
  return o.list || !o.backend.empty();
}

// All setup, allocation, trusted-client calls and decryption are outside the
// timed evaluation. Each fixture owns its plugin until every buffer/plan dies.
class Fixture {
  public:
  Fixture(CorpusBackend backend, const Shape &shape) : backend_(std::move(backend)), shape_(shape) {}
  ~Fixture() {
    if (executable_)
      backend_.executableFree()(executable_);
    for (auto *b : owned_)
      backend_.bufferFree()(backend_.ctx(), b);
  }
  Fixture(const Fixture &) = delete;
  Fixture &operator=(const Fixture &) = delete;
  void inputs() {
    const auto start = Clock::now();
    for (uint32_t i = 0; i < shape_.program->num_inputs; ++i) {
      const auto &slots = shape_.inputs.at(shape_.program->input_ids[i]);
      FhnBuffer *b = allocate();
      std::vector<double> values(slots.begin(), slots.end());
      if (backend_.encryptPacked(b, values.data(), shape_.slot_count) != 0)
        throw std::runtime_error("input encryption failed");
      inputs_.push_back(b);
    }
    encrypt_ms = elapsed(start);
  }
  bool compile(const Circuit &circuit, uint64_t limit, std::string &reason) {
    if (!backend_.hasCircuit()) {
      reason = "backend has no circuit export group";
      return false;
    }
    const FhnCircuit view = circuit.view();
    FhnCompileOptions opts{sizeof(FhnCompileOptions), limit};
    char error[512]{};
    auto start = Clock::now();
    executable_ = backend_.compileCircuit(&view, inputs_.data(), static_cast<uint32_t>(inputs_.size()), &opts, error,
                                          sizeof(error));
    compile_ms = elapsed(start);
    if (!executable_) {
      reason = error[0] ? error : "compile failed without a diagnostic";
      return false;
    }
    start = Clock::now();
    if (backend_.prepareExecutable(executable_) != 0)
      throw std::runtime_error("evaluation-key preparation failed");
    prepare_ms = elapsed(start);
    stats.struct_size = sizeof(stats);
    if (backend_.executableStats()(executable_, &stats) != 0)
      throw std::runtime_error("executable statistics failed");
    for (size_t i = 0; i < shape_.output_ids.size(); ++i)
      outputs_.push_back(allocate());
    return true;
  }
  bool legacy(std::string &reason) {
    auto start = Clock::now();
    if (backend_.prepare(shape_.program.get()) != 0) {
      reason = "legacy key preparation failed";
      return false;
    }
    prepare_ms = elapsed(start);
    uint32_t max_id = 0;
    for (auto [id, values] : shape_.inputs)
      max_id = std::max(max_id, id);
    for (uint32_t i = 0; i < shape_.program->num_instructions; ++i)
      max_id = std::max(max_id, shape_.program->instructions[i].result_id);
    legacy_buffers_.resize(static_cast<size_t>(max_id) + 1, nullptr);
    for (uint32_t i = 0; i < inputs_.size(); ++i)
      legacy_buffers_[shape_.program->input_ids[i]] = const_cast<FhnBuffer *>(inputs_[i]);
    for (uint32_t i = 0; i < shape_.program->num_instructions; ++i) {
      auto id = shape_.program->instructions[i].result_id;
      legacy_buffers_[id] = allocate();
    }
    for (auto id : shape_.output_ids)
      outputs_.push_back(legacy_buffers_.at(id));
    legacy_executor_ = std::make_unique<FhnDefaultExecutor>(backend_.kernels());
    if (run() != 0) {
      reason = "legacy instruction execution failed";
      return false;
    }
    return true;
  }
  int run() {
    if (executable_)
      return backend_.runExecutable()(executable_, inputs_.data(), static_cast<uint32_t>(inputs_.size()),
                                      outputs_.data(), static_cast<uint32_t>(outputs_.size()));
    return legacy_executor_->execute(backend_.ctx(), shape_.program.get(), legacy_buffers_.data());
  }
  void verify(double atol, double rtol) {
    const auto expected = evaluate(*shape_.program, shape_.inputs);
    if (!expected)
      throw std::runtime_error("independent plaintext oracle failed");
    for (size_t i = 0; i < outputs_.size(); ++i) {
      std::vector<double> actual(shape_.slot_count);
      const auto decrypt_start = Clock::now();
      const int decrypt_rc = backend_.decryptPacked(outputs_[i], actual.data(), shape_.slot_count);
      decrypt_ms += elapsed(decrypt_start);
      if (decrypt_rc != 0)
        throw std::runtime_error("output decryption failed");
      for (uint32_t j = 0; j < shape_.slot_count; ++j) {
        const double want = static_cast<double>(expected->at(shape_.output_ids[i])[j]);
        const double err = std::abs(actual[j] - want);
        max_error = std::max(max_error, err);
        if (!std::isfinite(actual[j]) || err > atol + rtol * std::abs(want)) {
          char message[256];
          std::snprintf(message, sizeof(message), "output %u slot %u got %.17g want %.17g (error %.4g)",
                        shape_.output_ids[i], j, actual[j], want, err);
          throw std::runtime_error(message);
        }
      }
    }
  }
  double timedRun() {
    const auto start = Clock::now();
    const int rc = run();
    const double ms = elapsed(start);
    if (rc != 0)
      throw std::runtime_error("evaluation failed");
    return ms;
  }
  uint64_t externalBytes() const {
    if (!backend_.levelBytes() || !backend_.freshLevel())
      return 0;
    return owned_.size() * backend_.levelBytes()(backend_.ctx(), backend_.freshLevel()(backend_.ctx()));
  }
  FhnExecutableStats stats{};
  double context_ms = 0, encrypt_ms = 0, compile_ms = 0, prepare_ms = 0, decrypt_ms = 0, max_error = 0;

  private:
  FhnBuffer *allocate() {
    auto *b = backend_.bufferAlloc()(backend_.ctx());
    if (!b)
      throw std::runtime_error("buffer allocation failed");
    try {
      owned_.push_back(b);
    } catch (...) {
      backend_.bufferFree()(backend_.ctx(), b);
      throw;
    }
    return b;
  }
  CorpusBackend backend_;
  const Shape &shape_;
  FhnExecutable *executable_ = nullptr;
  std::vector<FhnBuffer *> owned_, outputs_, legacy_buffers_;
  std::vector<const FhnBuffer *> inputs_;
  std::unique_ptr<FhnDefaultExecutor> legacy_executor_;
};
struct Distribution {
  double minimum, median, maximum;
  explicit Distribution(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    minimum = values.front();
    maximum = values.back();
    const size_t n = values.size();
    median = n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
  }
};
} // namespace

int main(int argc, char **argv) {
  Options options;
  if (!parse(argc, argv, options)) {
    usage();
    return 2;
  }
  auto shapes = allShapes();
  if (!options.shape.empty() &&
      std::none_of(shapes.begin(), shapes.end(), [&](const Shape &s) { return s.name == options.shape; })) {
    std::fprintf(stderr, "unknown shape: %s\n", options.shape.c_str());
    return 2;
  }
  if (options.list) {
    for (const auto &s : shapes)
      std::printf("%s\n", s.name.c_str());
    return 0;
  }
  // Load each image once for the benchmark lifetime; each shape still owns a
  // fresh context/client. Repeated unload/reload also triggers a vtable ODR
  // registration failure in Apple ASan even for a standalone trivial dylib.
  auto retainLibrary = [](const std::string &path) {
    return std::shared_ptr<void>(path.empty() ? nullptr : dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL), [](void *p) {
      if (p)
        dlclose(p);
    });
  };
  auto current_library = retainLibrary(options.backend);
  auto baseline_library = retainLibrary(options.baseline);
  if (!current_library || (!options.baseline.empty() && !baseline_library)) {
    std::fprintf(stderr, "could not load benchmark backend library\n");
    return 1;
  }
  unsigned executed = 0, unsupported = 0, failed = 0;
  for (const auto &shape : shapes) {
    if (!options.shape.empty() && shape.name != options.shape)
      continue;
    try {
      std::string reason;
      auto circuit = importCircuit(*shape.program, &reason);
      if (!circuit) {
        ++unsupported;
        std::printf("UNSUPPORTED %s: %s\n", shape.name.c_str(), reason.c_str());
        continue;
      }
      const auto context_start = Clock::now();
      auto backend = CorpusBackend::load(options.backend, options.prefix, &reason,
                                         options.config.empty() ? nullptr : options.config.c_str());
      if (!backend)
        throw std::runtime_error(reason);
      const double context_ms = elapsed(context_start);
      Fixture current(std::move(*backend), shape);
      current.context_ms = context_ms;
      current.inputs();
      if (!current.compile(*circuit, options.workspace, reason)) {
        ++unsupported;
        std::printf("UNSUPPORTED %s: %s\n", shape.name.c_str(), reason.c_str());
        continue;
      }
      if (current.run() != 0)
        throw std::runtime_error("initial circuit execution failed");
      current.verify(options.atol, options.rtol);
      std::unique_ptr<Fixture> baseline;
      if (!options.baseline.empty()) {
        auto old = CorpusBackend::load(options.baseline, options.prefix, &reason,
                                       options.config.empty() ? nullptr : options.config.c_str());
        if (!old)
          throw std::runtime_error("baseline load: " + reason);
        baseline = std::make_unique<Fixture>(std::move(*old), shape);
        baseline->inputs();
        if (!baseline->legacy(reason)) {
          std::printf("BASELINE_UNAVAILABLE %s: %s\n", shape.name.c_str(), reason.c_str());
          baseline.reset();
        } else
          baseline->verify(options.atol, options.rtol);
      }
      std::vector<double> samples, old_samples;
      samples.reserve(options.reps);
      old_samples.reserve(options.reps);
      for (uint32_t i = 0; i < options.warmup + options.reps; ++i) {
        double now = 0, old = 0;
        // Alternate order to reduce one-sided thermal/scheduling bias.
        if (baseline && i % 2 == 0)
          old = baseline->timedRun();
        now = current.timedRun();
        if (baseline && i % 2 != 0)
          old = baseline->timedRun();
        if (i >= options.warmup) {
          samples.push_back(now);
          if (baseline)
            old_samples.push_back(old);
        }
      }
      current.verify(options.atol, options.rtol);
      if (baseline)
        baseline->verify(options.atol, options.rtol);
      const Distribution timing(samples);
      const auto &s = current.stats;
      std::printf("PASS %s nodes=%u regions=%u linear=%u products=%u rotations=%u slots=%u "
                  "storage_bytes=%llu scratch_bytes=%llu io_buffer_bytes=%llu "
                  "context_ms=%.4f encrypt_ms=%.4f compile_ms=%.4f prepare_ms=%.4f decrypt_verify_ms=%.4f "
                  "min_ms=%.4f median_ms=%.4f max_ms=%.4f max_error=%.5g",
                  shape.name.c_str(), s.source_nodes, s.regions, s.linear_regions, s.product_regions,
                  s.rotation_regions, s.storage_slots, static_cast<unsigned long long>(s.storage_bytes),
                  static_cast<unsigned long long>(s.scratch_bytes),
                  static_cast<unsigned long long>(current.externalBytes()), current.context_ms, current.encrypt_ms,
                  current.compile_ms, current.prepare_ms, current.decrypt_ms, timing.minimum, timing.median,
                  timing.maximum, current.max_error);
      if (baseline) {
        const Distribution before(old_samples);
        std::printf(" baseline_min_ms=%.4f baseline_median_ms=%.4f baseline_max_ms=%.4f speedup=%.4f "
                    "baseline_max_error=%.5g baseline_all_buffer_bytes=%llu",
                    before.minimum, before.median, before.maximum, before.median / timing.median, baseline->max_error,
                    static_cast<unsigned long long>(baseline->externalBytes()));
      }
      std::puts("");
      ++executed;
    } catch (const std::exception &e) {
      ++failed;
      std::fprintf(stderr, "FAIL %s: %s\n", shape.name.c_str(), e.what());
    }
    std::fflush(stdout);
  }
  std::printf("executed=%u unsupported=%u failed=%u\n", executed, unsupported, failed);
  return failed || !executed || (options.required && unsupported) ? 1 : 0;
}
