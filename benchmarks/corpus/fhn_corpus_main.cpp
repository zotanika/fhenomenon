#include "FHN/FhnDefaultExecutor.h"
#include "FHN/FhnMovementPlan.h"
#include "corpus_backend.h"
#include "corpus_oracle.h"
#include "corpus_shapes.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

using namespace fhenomenon;
using namespace fhenomenon::corpus;

namespace {

uint64_t transfers(const FhnMovementPlan::Stats &s) {
  return static_cast<uint64_t>(s.prefetch_count) + static_cast<uint64_t>(s.evict_count);
}

uint32_t maxWorkingSet(const FhnProgram &program) {
  uint32_t max_ws = 0;
  for (uint32_t i = 0; i < program.num_instructions; ++i) {
    const FhnInstruction &inst = program.instructions[i];
    std::vector<uint32_t> ids{inst.result_id};
    for (int j = 0; j < 4; ++j)
      if (inst.operands[j] != 0)
        ids.push_back(inst.operands[j]);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    max_ws = std::max(max_ws, static_cast<uint32_t>(ids.size()));
  }
  return max_ws;
}

void usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s [--backend <lib.so>] [--prefix <sym, default toyfhe_>] "
               "[--shape <name>] [--max-depth <N>] [--list] [--budget-bytes <N|min>]\n"
               "       [--config <json-file>] [--atol <N>] [--rtol <N>] "
               "[--require-execution] [--compare-fallback]\n",
               argv0);
}

bool parseUnsigned(const char *text, uint64_t &value) {
  if (!text[0] || !std::all_of(text, text + std::strlen(text), [](char c) { return c >= '0' && c <= '9'; }))
    return false;
  char *end = nullptr;
  errno = 0;
  value = std::strtoull(text, &end, 10);
  return errno == 0 && end != text && *end == '\0';
}

bool parseTolerance(const char *text, double &value) {
  char *end = nullptr;
  errno = 0;
  value = std::strtod(text, &end);
  return errno == 0 && end != text && *end == '\0' && std::isfinite(value) && value >= 0;
}

bool fused(FhnOpCode op) { return op >= FHN_HMULT && op <= FHN_MAD; }

bool executable(const FhnDefaultExecutor &executor, FhnOpCode op) {
  if (executor.supports(op))
    return true;
  switch (op) {
  case FHN_HMULT:
    return executor.supports(FHN_MULT_CC) && executor.supports(FHN_RELINEARIZE) && executor.supports(FHN_RESCALE);
  case FHN_HROT:
    return executor.supports(FHN_ROTATE);
  case FHN_HROT_ADD:
    return (executor.supports(FHN_HROT) || executor.supports(FHN_ROTATE)) && executor.supports(FHN_ADD_CC);
  case FHN_HCONJ_ADD:
    return executor.supports(FHN_CONJUGATE) && executor.supports(FHN_ADD_CC);
  case FHN_MAD:
    return executor.supports(FHN_MULT_CS) && executor.supports(FHN_ADD_CC);
  default:
    return false;
  }
}

using Outputs = std::map<uint32_t, std::vector<double>>;

bool closeEnough(double actual, double expected, double atol, double rtol) {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::abs(actual - expected) <= atol + rtol * std::abs(expected);
}

// Each path gets fresh encrypted inputs. All buffers remain behind the C ABI.
std::optional<Outputs> executeShape(const Shape &shape, const CorpusBackend &backend, FhnKernelTable *table,
                                    uint64_t budget, const FhnLevelModel *model, double atol, double rtol) {
  std::vector<uint32_t> pinned(shape.program->input_ids, shape.program->input_ids + shape.program->num_inputs);
  pinned.insert(pinned.end(), shape.output_ids.begin(), shape.output_ids.end());
  auto plan = FhnMovementPlan::analyze(*shape.program, pinned, budget, FhnEvictionPolicy::Belady, model);
  if (!plan)
    return std::nullopt;
  uint32_t max_id = 0;
  for (const auto &[id, values] : shape.inputs)
    max_id = std::max(max_id, id);
  for (uint32_t i = 0; i < shape.program->num_instructions; ++i)
    max_id = std::max(max_id, shape.program->instructions[i].result_id);
  std::vector<FhnBuffer *> buffers(max_id + 1, nullptr);
  const bool integer_scalar = shape.slot_count == 1 && backend.encryptI64();
  const bool double_scalar = shape.slot_count == 1 && !integer_scalar && backend.encryptF64();
  bool ok = backend.prepare(shape.program.get()) == 0;
  if (!ok) return std::nullopt;
  for (const auto &[id, values] : shape.inputs) {
    buffers[id] = backend.bufferAlloc()(backend.ctx());
    if (!buffers[id]) {
      ok = false;
      break;
    }
    std::vector<double> input(values.begin(), values.end());
    const int rc = integer_scalar ? backend.encryptI64()(backend.ctx(), buffers[id], values[0])
                   : double_scalar
                     ? backend.encryptF64()(backend.ctx(), buffers[id], input[0])
                     : backend.encryptPacked(buffers[id], input.data(), shape.slot_count);
    if (rc != 0) {
      ok = false;
      break;
    }
  }
  FhnDefaultExecutor executor(table);
  FhnMovementHooks hooks{backend.ctx(), backend.bufferAlloc(), backend.bufferFree(), backend.prefetch(),
                         backend.evict()};
  if (ok && executor.execute(hooks, shape.program.get(), buffers.data(), *plan) != 0)
    ok = false;
  const auto expected = evaluate(*shape.program, shape.inputs);
  if (!expected)
    ok = false;
  Outputs outputs;
  if (ok)
    for (uint32_t id : shape.output_ids) {
      auto &values = outputs[id];
      values.resize(shape.slot_count);
      int64_t integer = 0;
      const int rc = integer_scalar ? backend.decryptI64()(backend.ctx(), buffers[id], &integer)
                     : double_scalar
                       ? backend.decryptF64()(backend.ctx(), buffers[id], values.data())
                       : backend.decryptPacked(buffers[id], values.data(), shape.slot_count);
      if (integer_scalar)
        values[0] = static_cast<double>(integer);
      if (rc != 0) {
        ok = false;
        break;
      }
      for (uint32_t slot = 0; slot < shape.slot_count; ++slot) {
        const int64_t want = expected->at(id)[slot];
        if ((integer_scalar && integer != want) ||
            (!integer_scalar && !closeEnough(values[slot], static_cast<double>(want), atol, rtol))) {
          std::fprintf(stderr, "FAIL %s: output %u slot %u got %.17g want %" PRId64 "\n", shape.name.c_str(), id, slot,
                       values[slot], want);
          ok = false;
          break; // One concrete mismatch per output is enough to diagnose failure.
        }
      }
    }
  for (auto *buffer : buffers)
    if (buffer)
      backend.bufferFree()(backend.ctx(), buffer);
  if (!ok)
    return std::nullopt;
  return outputs;
}

// Queries a backend's level model once via its trio getters. Requires the
// caller to have already confirmed all three are non-null (all-or-nothing).
FhnLevelModel queryLevelModel(const CorpusBackend &backend) {
  FhnLevelModel model;
  model.fresh_level = backend.freshLevel()(backend.ctx());
  // A full-trio backend can declare a nonsensical fresh_level (negative, or
  // absurdly large). Guard here, before the bytes loop, so a huge value
  // can't hang the loop and a negative one doesn't reach the call site's
  // min-indexing before analyze()'s own model validation gets a chance to
  // reject it — that ordering previously segfaulted.
  if (model.fresh_level < 0 || model.fresh_level > 4096)
    return model;
  for (int64_t l = 0; l <= model.fresh_level; ++l)
    model.bytes_by_level.push_back(backend.levelBytes()(backend.ctx(), l));
  for (int op = FHN_NOP; op < FHN_OPCODE_COUNT; ++op)
    model.effects[op] = backend.opcodeLevelEffect()(backend.ctx(), static_cast<FhnOpCode>(op));
  return model;
}

} // namespace

int main(int argc, char **argv) {
  std::string backend_path;
  std::string prefix = "toyfhe_";
  std::string only_shape;
  uint32_t max_depth = UINT32_MAX;
  bool list_only = false;
  bool budget_bytes_given = false;
  bool budget_bytes_min = false;
  uint64_t budget_bytes = 0;
  std::string config_path;
  bool require_execution = false, compare_fallback = false;
  double atol = 1e-6, rtol = 1e-6;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
      backend_path = argv[++i];
    } else if (std::strcmp(argv[i], "--prefix") == 0 && i + 1 < argc) {
      prefix = argv[++i];
    } else if (std::strcmp(argv[i], "--shape") == 0 && i + 1 < argc) {
      only_shape = argv[++i];
    } else if (std::strcmp(argv[i], "--max-depth") == 0 && i + 1 < argc) {
      uint64_t depth = 0;
      if (!parseUnsigned(argv[++i], depth) || depth > UINT32_MAX) {
        usage(argv[0]);
        return 2;
      }
      max_depth = static_cast<uint32_t>(depth);
    } else if (std::strcmp(argv[i], "--list") == 0) {
      list_only = true;
    } else if (std::strcmp(argv[i], "--budget-bytes") == 0 && i + 1 < argc) {
      budget_bytes_given = true;
      const char *arg = argv[++i];
      if (std::strcmp(arg, "min") == 0)
        budget_bytes_min = true;
      else if (!parseUnsigned(arg, budget_bytes)) {
        usage(argv[0]);
        return 2;
      }
    } else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      config_path = argv[++i];
    } else if (std::strcmp(argv[i], "--atol") == 0 && i + 1 < argc) {
      if (!parseTolerance(argv[++i], atol)) {
        usage(argv[0]);
        return 2;
      }
    } else if (std::strcmp(argv[i], "--rtol") == 0 && i + 1 < argc) {
      if (!parseTolerance(argv[++i], rtol)) {
        usage(argv[0]);
        return 2;
      }
    } else if (std::strcmp(argv[i], "--require-execution") == 0) {
      require_execution = true;
    } else if (std::strcmp(argv[i], "--compare-fallback") == 0) {
      compare_fallback = true;
    } else {
      usage(argv[0]);
      return 2;
    }
  }

  auto shapes = allShapes();
  if (!only_shape.empty()) {
    bool exists = false;
    for (const auto &s : shapes)
      exists = exists || s.name == only_shape;
    if (!exists) {
      std::fprintf(stderr, "error: unknown shape '%s' (use --list)\n", only_shape.c_str());
      return 2;
    }
  }

  if (list_only) {
    if (require_execution) {
      std::fprintf(stderr, "error: --list cannot require execution\n");
      return 2;
    }
    for (const auto &s : shapes)
      std::printf("%-14s slots=%-3u depth=%-2u insts=%-4u %s\n", s.name.c_str(), s.slot_count, s.ct_mult_depth,
                  s.program->num_instructions, s.axis.c_str());
    return 0;
  }

  std::string config;
  if (!config_path.empty()) {
    std::ifstream file(config_path);
    if (!file) {
      std::fprintf(stderr, "error: cannot read config file\n");
      return 2;
    }
    config.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (config.empty() || config.find('\0') != std::string::npos) {
      std::fprintf(stderr, "error: empty config or embedded NUL\n");
      return 2;
    }
    if (backend_path.empty()) {
      std::fprintf(stderr, "error: --config requires --backend\n");
      return 2;
    }
  }
  std::optional<CorpusBackend> backend;
  if (!backend_path.empty()) {
    std::string error;
    backend = CorpusBackend::load(backend_path, prefix, &error, config_path.empty() ? nullptr : config.c_str());
    if (!backend) {
      std::fprintf(stderr, "error: cannot load backend: %s\n", error.c_str());
      return 1;
    }
  }

  FhnLevelModel model;
  if (budget_bytes_given) {
    if (!backend || !backend->freshLevel() || !backend->levelBytes() || !backend->opcodeLevelEffect()) {
      std::fprintf(stderr, "error: --budget-bytes requires a backend with a level model\n");
      return 2;
    }
    model = queryLevelModel(*backend);
    // A negative or absurdly large fresh_level, or an empty level table
    // (the queryLevelModel guard above returns early on either), must not
    // reach the min-indexing below: that's the segfault this guard closes.
    if (model.fresh_level < 0 || model.fresh_level > 4096 || model.bytes_by_level.empty()) {
      std::fprintf(stderr, "error: backend declared an invalid level model (fresh_level out of range)\n");
      return 2;
    }
    // min = 3 * bytes at the fresh level: the corpus generators are all
    // unary/binary primitives, so no instruction's working set (result +
    // operands) ever exceeds 3 buffers — this is the smallest budget every
    // shape can feasibly plan under.
    if (budget_bytes_min) {
      const uint64_t bytes = model.bytes_by_level[static_cast<size_t>(model.fresh_level)];
      if (bytes > UINT64_MAX / 3) {
        std::fprintf(stderr, "error: budget overflow\n");
        return 2;
      }
      budget_bytes = 3 * bytes;
    }
  }

  bool failed = false;
  uint32_t planned = 0, executed = 0, skipped = 0, failures = 0;
  std::printf("Planning estimates only: scheduled transfers exclude input normalization and backend scratch.\n");
  uint64_t total_belady = 0;
  uint64_t total_lru = 0;
  std::vector<double> shape_savings;
  std::printf("%-14s %6s %-11s | %8s %14s %14s %8s\n", "shape", "budget", "point", "hw", "belady p/e", "lru p/e",
              "saved");

  for (const auto &shape : shapes) {
    if (!only_shape.empty() && shape.name != only_shape)
      continue;

    bool shape_failed = false;
    auto unlimited = FhnMovementPlan::analyze(*shape.program, shape.output_ids, 0);
    if (!unlimited) {
      std::fprintf(stderr, "FAIL %s: movement analysis rejected the program\n", shape.name.c_str());
      failed = true;
      ++failures;
      continue;
    }
    ++planned;
    const uint32_t hw = unlimited->stats().high_water;
    const uint32_t b_min = maxWorkingSet(*shape.program);
    const uint32_t b_mid = std::max(b_min, static_cast<uint32_t>((hw * 6 + 9) / 10));
    std::vector<uint32_t> points{b_min, b_mid, hw};
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    bool counted = false;

    for (uint32_t budget : points) {
      auto belady = FhnMovementPlan::analyze(*shape.program, shape.output_ids, budget, FhnEvictionPolicy::Belady);
      auto lru = FhnMovementPlan::analyze(*shape.program, shape.output_ids, budget, FhnEvictionPolicy::Lru);
      if (!belady || !lru) {
        std::fprintf(stderr, "FAIL %s: budget %u infeasible\n", shape.name.c_str(), budget);
        shape_failed = true;
        continue;
      }
      if (budget == hw && belady->stats().evict_count != 0) {
        std::fprintf(stderr, "FAIL %s: evictions at the high-water budget\n", shape.name.c_str());
        shape_failed = true;
      }
      const uint64_t tb = transfers(belady->stats());
      const uint64_t tl = transfers(lru->stats());
      const double saved =
        tl == 0 ? 0.0 : 100.0 * (static_cast<double>(tl) - static_cast<double>(tb)) / static_cast<double>(tl);
      std::string tag;
      if (budget == b_min)
        tag += "min";
      if (budget == b_mid) {
        if (!tag.empty())
          tag += "=";
        tag += "mid";
      }
      if (budget == hw) {
        if (!tag.empty())
          tag += "=";
        tag += "hw";
      }
      std::printf("%-14s %6u %-11s | %8u %7u/%-6u %7u/%-6u %7.1f%%\n", shape.name.c_str(), budget, tag.c_str(), hw,
                  belady->stats().prefetch_count, belady->stats().evict_count, lru->stats().prefetch_count,
                  lru->stats().evict_count, saved);
      if (budget == b_mid && !counted) {
        total_belady += tb;
        total_lru += tl;
        shape_savings.push_back(saved);
        counted = true;
      }
    }

    // Byte-mode report: hw_bytes from an unlimited byte-mode analysis, then
    // both policies at budget_bytes. Pinned set matches the slot sweep
    // above (outputs only) — not the execution pass's inputs∪outputs.
    if (budget_bytes_given) {
      auto hw_bytes_plan =
        FhnMovementPlan::analyze(*shape.program, shape.output_ids, 0, FhnEvictionPolicy::Belady, &model);
      auto belady_bytes =
        FhnMovementPlan::analyze(*shape.program, shape.output_ids, budget_bytes, FhnEvictionPolicy::Belady, &model);
      auto lru_bytes =
        FhnMovementPlan::analyze(*shape.program, shape.output_ids, budget_bytes, FhnEvictionPolicy::Lru, &model);
      if (!hw_bytes_plan || !belady_bytes || !lru_bytes) {
        std::fprintf(stderr, "FAIL %s: byte-budget infeasible/model invalid\n", shape.name.c_str());
        shape_failed = true;
      } else {
        std::printf("planned-model-bytes[%s]: hw_bytes=%" PRIu64
                    " belady p/e=%u/%u lru p/e=%u/%u (budget-bytes %" PRIu64 ")\n",
                    shape.name.c_str(), hw_bytes_plan->stats().high_water_bytes, belady_bytes->stats().prefetch_count,
                    belady_bytes->stats().evict_count, lru_bytes->stats().prefetch_count,
                    lru_bytes->stats().evict_count, budget_bytes);
      }
    }

    if (shape_failed) {
      failed = true;
      ++failures;
      continue;
    }
    std::string reason;
    std::vector<FhnKernelEntry> primitives;
    if (!backend)
      reason = "no backend supplied";
    else if (shape.ct_mult_depth > max_depth)
      reason = "exceeds --max-depth";
    else if (!backend->hasVector() && (shape.slot_count != 1 || (!backend->encryptI64() && !backend->encryptF64())))
      reason = "backend lacks the required packed/scalar data-plane pair";
    if (reason.empty()) {
      FhnDefaultExecutor normal(backend->kernels());
      bool has_fused = false;
      for (uint32_t i = 0; i < backend->kernels()->num_kernels; ++i) {
        const auto &entry = backend->kernels()->kernels[i];
        if (!fused(entry.opcode))
          primitives.push_back(entry);
      }
      FhnKernelTable primitive_table{static_cast<uint32_t>(primitives.size()), primitives.data()};
      FhnDefaultExecutor fallback(&primitive_table);
      for (uint32_t i = 0; i < shape.program->num_instructions; ++i) {
        const auto op = shape.program->instructions[i].opcode;
        has_fused = has_fused || fused(op);
        if (!executable(normal, op))
          reason = "backend lacks required operations";
        if (compare_fallback && (!executable(fallback, op) || (fused(op) && !normal.supports(op))))
          reason = "native fused kernel or complete fallback unavailable";
      }
      if (compare_fallback && !has_fused)
        reason = "shape contains no fused operation to compare";
    }
    if (!reason.empty()) {
      std::printf("SKIP %s: %s\n", shape.name.c_str(), reason.c_str());
      ++skipped;
      continue;
    }
    const uint64_t execution_budget = budget_bytes_given ? budget_bytes : b_mid;
    const FhnLevelModel *execution_model = budget_bytes_given ? &model : nullptr;
    const auto actual =
      executeShape(shape, *backend, backend->kernels(), execution_budget, execution_model, atol, rtol);
    bool exec_ok = actual.has_value();
    if (exec_ok && compare_fallback) {
      FhnKernelTable primitive_table{static_cast<uint32_t>(primitives.size()), primitives.data()};
      const auto fallback =
        executeShape(shape, *backend, &primitive_table, execution_budget, execution_model, atol, rtol);
      exec_ok = fallback.has_value();
      if (fallback)
        for (const auto &[id, values] : *actual)
          for (size_t slot = 0; slot < values.size(); ++slot)
            if (!closeEnough(fallback->at(id)[slot], values[slot], atol, rtol))
              exec_ok = false;
    }
    if (!exec_ok) {
      std::fprintf(stderr, "FAIL %s: execution/verification failed\n", shape.name.c_str());
      failed = true;
      ++failures;
    } else {
      ++executed;
      std::printf("EXECUTED %s: verified all %u slots%s\n", shape.name.c_str(), shape.slot_count,
                  compare_fallback ? ", native and primitive fallback agree" : "");
    }
  }

  if (total_lru > 0) {
    std::printf("planned aggregate @B_mid: belady=%" PRIu64 " lru=%" PRIu64 " transfers -> belady saves %.1f%%\n",
                total_belady, total_lru,
                100.0 * (static_cast<double>(total_lru) - static_cast<double>(total_belady)) /
                  static_cast<double>(total_lru));
    std::vector<double> sorted_savings = shape_savings;
    std::sort(sorted_savings.begin(), sorted_savings.end());
    const size_t n = sorted_savings.size();
    const double median =
      (n % 2 == 1) ? sorted_savings[n / 2] : (sorted_savings[n / 2 - 1] + sorted_savings[n / 2]) / 2.0;
    std::printf("planned median per-shape savings @B_mid: %.1f%%\n", median);
  }
  std::printf("summary: planned=%u executed=%u skipped=%u failed=%u\n", planned, executed, skipped, failures);
  if (require_execution && (skipped != 0 || executed == 0)) {
    std::fprintf(stderr, "error: required execution was incomplete\n");
    failed = true;
  }
  return failed ? 1 : 0;
}
