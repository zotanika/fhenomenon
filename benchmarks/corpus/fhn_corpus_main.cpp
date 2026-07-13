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
               "[--shape <name>] [--max-depth <N>] [--list]\n",
               argv0);
}

} // namespace

int main(int argc, char **argv) {
  std::string backend_path;
  std::string prefix = "toyfhe_";
  std::string only_shape;
  uint32_t max_depth = UINT32_MAX;
  bool list_only = false;

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
    } else {
      usage(argv[0]);
      return 2;
    }
  }

  auto shapes = allShapes();
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

  bool failed = false;
  uint64_t total_belady = 0;
  uint64_t total_lru = 0;
  std::printf("%-14s %6s | %8s %14s %14s %8s\n", "shape", "budget", "hw", "belady p/e", "lru p/e", "saved");

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
    const uint32_t points[3] = {b_min, b_mid, hw};
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
      std::printf("%-14s %6u | %8u %7u/%-6u %7u/%-6u %7.1f%%\n", shape.name.c_str(), budget, hw,
                  belady->stats().prefetch_count, belady->stats().evict_count, lru->stats().prefetch_count,
                  lru->stats().evict_count, saved);
      if (budget == b_mid && !counted) {
        total_belady += tb;
        total_lru += tl;
        counted = true;
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
  }
  return failed ? 1 : 0;
}
