#include "FHN/FhnDefaultExecutor.h"
#include "FHN/FhnMovementPlan.h"
#include "corpus_backend.h"
#include "corpus_oracle.h"
#include "corpus_shapes.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
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
               "[--shape <name>] [--max-depth <N>] [--list] [--budget-bytes <N|min>]\n",
               argv0);
}

// Queries a backend's level model once via its trio getters. Requires the
// caller to have already confirmed all three are non-null (all-or-nothing).
FhnLevelModel queryLevelModel(const CorpusBackend &backend) {
  FhnLevelModel model;
  model.fresh_level = backend.freshLevel()(backend.ctx());
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

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
      backend_path = argv[++i];
    } else if (std::strcmp(argv[i], "--prefix") == 0 && i + 1 < argc) {
      prefix = argv[++i];
    } else if (std::strcmp(argv[i], "--shape") == 0 && i + 1 < argc) {
      only_shape = argv[++i];
    } else if (std::strcmp(argv[i], "--max-depth") == 0 && i + 1 < argc) {
      max_depth = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (std::strcmp(argv[i], "--list") == 0) {
      list_only = true;
    } else if (std::strcmp(argv[i], "--budget-bytes") == 0 && i + 1 < argc) {
      budget_bytes_given = true;
      const char *arg = argv[++i];
      if (std::strcmp(arg, "min") == 0)
        budget_bytes_min = true;
      else
        budget_bytes = std::strtoull(arg, nullptr, 10);
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
    for (const auto &s : shapes)
      std::printf("%-14s slots=%-3u depth=%-2u insts=%-4u %s\n", s.name.c_str(), s.slot_count, s.ct_mult_depth,
                  s.program->num_instructions, s.axis.c_str());
    return 0;
  }

  std::optional<CorpusBackend> backend;
  if (!backend_path.empty()) {
    std::string error;
    backend = CorpusBackend::load(backend_path, prefix, &error);
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
    // min = 3 * bytes at the fresh level: the corpus generators are all
    // unary/binary primitives, so no instruction's working set (result +
    // operands) ever exceeds 3 buffers — this is the smallest budget every
    // shape can feasibly plan under.
    if (budget_bytes_min)
      budget_bytes = 3 * model.bytes_by_level[static_cast<size_t>(model.fresh_level)];
  }

  bool failed = false;
  uint64_t total_belady = 0;
  uint64_t total_lru = 0;
  std::vector<double> shape_savings;
  std::printf("%-14s %6s %-11s | %8s %14s %14s %8s\n", "shape", "budget", "point", "hw", "belady p/e", "lru p/e",
              "saved");

  for (const auto &shape : shapes) {
    if (!only_shape.empty() && shape.name != only_shape)
      continue;

    auto unlimited = FhnMovementPlan::analyze(*shape.program, shape.output_ids, 0);
    if (!unlimited) {
      std::fprintf(stderr, "FAIL %s: movement analysis rejected the program\n", shape.name.c_str());
      failed = true;
      continue;
    }
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
        failed = true;
        continue;
      }
      if (budget == hw && belady->stats().evict_count != 0) {
        std::fprintf(stderr, "FAIL %s: evictions at the high-water budget\n", shape.name.c_str());
        failed = true;
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
        failed = true;
      } else {
        std::printf("movement-bytes[%s]: hw_bytes=%" PRIu64 " belady p/e=%u/%u lru p/e=%u/%u (budget-bytes %" PRIu64
                    ")\n",
                    shape.name.c_str(), hw_bytes_plan->stats().high_water_bytes, belady_bytes->stats().prefetch_count,
                    belady_bytes->stats().evict_count, lru_bytes->stats().prefetch_count,
                    lru_bytes->stats().evict_count, budget_bytes);
      }
    }

    // Execution pass: backend loaded, opcodes supported, single-slot
    // instantiation, and within the operator-declared exactness depth.
    if (backend && shape.slot_count == 1 && shape.ct_mult_depth <= max_depth) {
      FhnDefaultExecutor executor(backend->kernels());
      bool supported = true;
      for (uint32_t i = 0; i < shape.program->num_instructions; ++i)
        supported = supported && executor.supports(shape.program->instructions[i].opcode);
      if (supported) {
        std::vector<uint32_t> pinned(shape.program->input_ids, shape.program->input_ids + shape.program->num_inputs);
        pinned.insert(pinned.end(), shape.output_ids.begin(), shape.output_ids.end());
        auto plan = FhnMovementPlan::analyze(*shape.program, pinned, b_mid, FhnEvictionPolicy::Belady);
        if (!plan) {
          std::fprintf(stderr, "FAIL %s: execution plan infeasible at B_mid\n", shape.name.c_str());
          failed = true;
          continue;
        }

        uint32_t max_id = 0;
        for (uint32_t i = 0; i < shape.program->num_instructions; ++i)
          max_id = std::max(max_id, shape.program->instructions[i].result_id);
        for (uint32_t i = 0; i < shape.program->num_inputs; ++i)
          max_id = std::max(max_id, shape.program->input_ids[i]);

        std::vector<FhnBuffer *> buffers(max_id + 1, nullptr);
        bool exec_ok = true;
        for (const auto &[id, slots] : shape.inputs) {
          buffers[id] = backend->bufferAlloc()(backend->ctx());
          if (!buffers[id] || backend->encryptI64()(backend->ctx(), buffers[id], slots[0]) != 0) {
            exec_ok = false;
            break;
          }
        }

        FhnMovementHooks hooks{backend->ctx(), backend->bufferAlloc(), backend->bufferFree(), backend->prefetch(),
                               backend->evict()};
        if (exec_ok && executor.execute(hooks, shape.program.get(), buffers.data(), *plan) != 0)
          exec_ok = false;

        auto expected = evaluate(*shape.program, shape.inputs);
        if (exec_ok && expected) {
          for (uint32_t out : shape.output_ids) {
            int64_t got = 0;
            if (backend->decryptI64()(backend->ctx(), buffers[out], &got) != 0 || got != expected->at(out)[0]) {
              std::fprintf(stderr, "FAIL %s: output id %u got %" PRId64 " want %" PRId64 "\n", shape.name.c_str(), out,
                           got, expected->at(out)[0]);
              exec_ok = false;
            }
          }
        } else if (exec_ok) {
          exec_ok = false;
        }

        for (uint32_t id = 1; id <= max_id; ++id)
          if (buffers[id])
            backend->bufferFree()(backend->ctx(), buffers[id]);

        if (!exec_ok) {
          std::fprintf(stderr, "FAIL %s: execution/verification failed\n", shape.name.c_str());
          failed = true;
        } else {
          std::printf("%-14s executed and verified on backend\n", shape.name.c_str());
        }
      }
    }
  }

  if (total_lru > 0) {
    std::printf("aggregate @B_mid: belady=%" PRIu64 " lru=%" PRIu64 " transfers -> belady saves %.1f%%\n", total_belady,
                total_lru,
                100.0 * (static_cast<double>(total_lru) - static_cast<double>(total_belady)) /
                  static_cast<double>(total_lru));
    std::vector<double> sorted_savings = shape_savings;
    std::sort(sorted_savings.begin(), sorted_savings.end());
    const size_t n = sorted_savings.size();
    const double median =
      (n % 2 == 1) ? sorted_savings[n / 2] : (sorted_savings[n / 2 - 1] + sorted_savings[n / 2]) / 2.0;
    std::printf("median per-shape savings @B_mid: %.1f%%\n", median);
  }
  return failed ? 1 : 0;
}
