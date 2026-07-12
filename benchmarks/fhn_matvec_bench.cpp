// fhn-bench-matvec — the flagship FHN benchmark.
//
// ONE FhnProgram computes an encrypted matrix-vector product M·v, executed
// two ways:
//   fused:      dispatched over the backend's full kernel table, so
//               FHN_HMULT and FHN_HROT_ADD hit the fused kernels directly;
//   decomposed: dispatched over a host-side filtered copy of the same table
//               with the fused entries removed, so the default executor's
//               decomposition rules run instead
//               (HMULT -> MULT_CC + RELINEARIZE + RESCALE,
//                HROT_ADD -> ROTATE + ADD_CC).
//
// Per row i (SSA ids throughout):
//   p   = HMULT(row_i, v)                       slot-wise products
//   t   = HROT_ADD(t, t)  d = n/2, n/4, ..., 1  rotate-and-add reduction
// after which every slot of the final t holds dot(row_i, v).
//
// Noise-budget headroom: entries are drawn from [0, 9], so a product slot is
// at most 81 and max |dot| = n * 81. ToyFHE decrypts integers exactly while
// |noise| < q/(2t) = 2^25. Per-product noise is bounded by
// |m1|*e2 + |m2|*e1 <= 2 * 81 * 8 = 1296 and the reduction sums n of them,
// so total noise stays near n * 1300 — for n <= 512 that is < 7e5, far below
// 2^25 ~ 3.4e7.

#include "FHN/FhnDefaultExecutor.h"
#include "FHN/FhnMovementPlan.h"
#include "FHN/ToyFheKernels.h"
#include "FHN/fhn_program.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct TimingStats {
  double median_ms = 0.0;
  double min_ms = 0.0;
};

// Warmup once, then time `reps` executions of executor.execute only (inputs
// are already encrypted; result buffers are simply overwritten every run).
TimingStats time_path(const char *label, fhenomenon::FhnDefaultExecutor &executor, FhnBackendCtx *ctx,
                      const FhnProgram *prog, FhnBuffer **bufs, uint32_t reps) {
  if (executor.execute(ctx, prog, bufs) != 0) {
    std::fprintf(stderr, "FATAL: warmup execution failed on the %s path\n", label);
    std::exit(1);
  }
  std::vector<double> samples;
  samples.reserve(reps);
  for (uint32_t r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = executor.execute(ctx, prog, bufs);
    const auto t1 = std::chrono::steady_clock::now();
    if (rc != 0) {
      std::fprintf(stderr, "FATAL: timed execution failed on the %s path (rep %u)\n", label, r);
      std::exit(1);
    }
    samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  TimingStats stats;
  stats.min_ms = samples.front();
  const std::size_t mid = samples.size() / 2;
  stats.median_ms = (samples.size() % 2 != 0) ? samples[mid] : 0.5 * (samples[mid - 1] + samples[mid]);
  return stats;
}

// Decrypt slot 0 of every program output and compare against plaintext M·v.
bool check_outputs(const char *label, FhnBackendCtx *ctx, const FhnProgram *prog, FhnBuffer **bufs, uint32_t n,
                   const std::vector<int64_t> &expected) {
  std::vector<int64_t> slots(n, 0);
  for (uint32_t i = 0; i < prog->num_outputs; ++i) {
    if (toyfhe_fhn_decrypt_vec_i64(ctx, bufs[prog->output_ids[i]], slots.data(), n) != 0) {
      std::fprintf(stderr, "FATAL [%s]: failed to decrypt output row %u\n", label, i);
      return false;
    }
    if (slots[0] != expected[i]) {
      std::fprintf(stderr, "FATAL [%s]: row %u slot 0 decrypted to %lld, expected %lld\n", label, i,
                   static_cast<long long>(slots[0]), static_cast<long long>(expected[i]));
      return false;
    }
  }
  return true;
}

void usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s [--n <size, power of two, default 64>] [--reps <default 5>] "
               "[--budget <max resident buffers, default 0 = unlimited>]\n",
               argv0);
}

// Analyze the movement plan for `prog`, pinning its outputs, and print a
// stats line tagged with `label` (the same tag used for the timing rows).
// Reporting only: exits the process on infeasible-budget analysis failure so
// the caller need not check a return value.
void report_movement_plan(const char *label, const FhnProgram *prog, uint32_t budget) {
  std::vector<uint32_t> pinned(prog->output_ids, prog->output_ids + prog->num_outputs);
  const auto plan = fhenomenon::FhnMovementPlan::analyze(*prog, pinned, budget);
  if (!plan) {
    std::fprintf(stderr, "error: movement analysis failed (budget %u infeasible?)\n", budget);
    std::exit(1);
  }
  std::printf("movement[%s]: high_water=%u allocs=%u prefetches=%u evicts=%u (budget %u)\n", label,
              plan->stats().high_water, plan->stats().alloc_count, plan->stats().prefetch_count,
              plan->stats().evict_count, budget);
}

} // namespace

int main(int argc, char **argv) {
  uint32_t n = 64;
  uint32_t reps = 5;
  uint32_t budget = 0;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
      n = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
      reps = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (std::strcmp(argv[i], "--budget") == 0 && i + 1 < argc) {
      budget = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else {
      usage(argv[0]);
      return 1;
    }
  }
  if (n == 0 || (n & (n - 1)) != 0) {
    std::fprintf(stderr, "error: --n must be a power of two (got %u)\n", n);
    return 1;
  }
  if (reps == 0) {
    std::fprintf(stderr, "error: --reps must be >= 1\n");
    return 1;
  }

  uint32_t log2n = 0;
  while ((1u << log2n) < n) {
    ++log2n;
  }

  // --- Deterministic data: n x n matrix M and vector v, entries in [0, 9].
  std::mt19937 rng(20260710u);
  std::uniform_int_distribution<int64_t> dist(0, 9);
  std::vector<std::vector<int64_t>> M(n, std::vector<int64_t>(n, 0));
  std::vector<int64_t> v(n, 0);
  for (uint32_t i = 0; i < n; ++i) {
    v[i] = dist(rng);
    for (uint32_t j = 0; j < n; ++j) {
      M[i][j] = dist(rng);
    }
  }
  std::vector<int64_t> expected(n, 0);
  for (uint32_t i = 0; i < n; ++i) {
    int64_t dot = 0;
    for (uint32_t j = 0; j < n; ++j) {
      dot += M[i][j] * v[j];
    }
    expected[i] = dot;
  }

  // --- Backend.
  FhnBackendInfo *info = toyfhe_fhn_get_info();
  FhnBackendCtx *ctx = toyfhe_fhn_create(nullptr);
  FhnKernelTable *full_table = toyfhe_fhn_get_kernels(ctx);

  // --- Build ONE program: inputs are ids 1..n (rows) and n+1 (v).
  const uint32_t v_id = n + 1;
  const uint32_t num_instructions = n * (1 + log2n);
  FhnProgram *prog = fhn_program_alloc(num_instructions, n + 1, n);
  if (prog == nullptr) {
    std::fprintf(stderr, "FATAL: program allocation failed\n");
    return 1;
  }
  for (uint32_t i = 0; i < n; ++i) {
    prog->input_ids[i] = i + 1;
  }
  prog->input_ids[n] = v_id;

  uint32_t next_id = v_id + 1;
  uint32_t inst_idx = 0;
  for (uint32_t row = 0; row < n; ++row) {
    uint32_t t_id = next_id++;
    FhnInstruction &mul = prog->instructions[inst_idx++];
    mul.opcode = FHN_HMULT;
    mul.result_id = t_id;
    mul.operands[0] = row + 1;
    mul.operands[1] = v_id;
    for (uint32_t d = n / 2; d >= 1; d /= 2) {
      const uint32_t r_id = next_id++;
      FhnInstruction &red = prog->instructions[inst_idx++];
      red.opcode = FHN_HROT_ADD;
      red.result_id = r_id;
      red.operands[0] = t_id;
      red.operands[1] = t_id;
      red.params[0] = static_cast<int64_t>(d);
      t_id = r_id;
    }
    prog->output_ids[row] = t_id;
  }

  // --- Provision buffers and encrypt the inputs (excluded from timing).
  const uint32_t num_buffers = next_id;
  std::vector<FhnBuffer *> bufs(num_buffers, nullptr);
  for (uint32_t i = 0; i < num_buffers; ++i) {
    bufs[i] = toyfhe_fhn_buffer_alloc(ctx);
  }
  for (uint32_t row = 0; row < n; ++row) {
    if (toyfhe_fhn_encrypt_vec_i64(ctx, bufs[row + 1], M[row].data(), n) != 0) {
      std::fprintf(stderr, "FATAL: failed to encrypt row %u\n", row);
      return 1;
    }
  }
  if (toyfhe_fhn_encrypt_vec_i64(ctx, bufs[v_id], v.data(), n) != 0) {
    std::fprintf(stderr, "FATAL: failed to encrypt v\n");
    return 1;
  }

  // --- Two executors over the same backend.
  // (a) fused: the backend's full kernel table.
  fhenomenon::FhnDefaultExecutor fused_executor(full_table);
  // (b) decomposed: a host-side filtered copy that drops the fused opcodes
  // (FHN_HMULT, FHN_HROT_ADD) so the executor's decomposition rules run.
  // The filtered storage must stay alive as long as the executor.
  std::vector<FhnKernelEntry> primitive_entries;
  for (uint32_t i = 0; i < full_table->num_kernels; ++i) {
    const FhnKernelEntry &entry = full_table->kernels[i];
    if (entry.opcode == FHN_HMULT || entry.opcode == FHN_HROT_ADD) {
      continue;
    }
    primitive_entries.push_back(entry);
  }
  FhnKernelTable primitive_table = {static_cast<uint32_t>(primitive_entries.size()), primitive_entries.data()};
  fhenomenon::FhnDefaultExecutor decomposed_executor(&primitive_table);

  // --- Correctness on BOTH paths before timing anything.
  if (fused_executor.execute(ctx, prog, bufs.data()) != 0 ||
      !check_outputs("fused", ctx, prog, bufs.data(), n, expected)) {
    std::fprintf(stderr, "FATAL: fused path failed correctness check\n");
    return 1;
  }
  if (decomposed_executor.execute(ctx, prog, bufs.data()) != 0 ||
      !check_outputs("decomposed", ctx, prog, bufs.data(), n, expected)) {
    std::fprintf(stderr, "FATAL: decomposed path failed correctness check\n");
    return 1;
  }

  // --- Movement-plan stats (reporting only; execution above already ran
  // through the executors' own paths). Both variants dispatch the same
  // FhnProgram IR — fused vs decomposed is purely a kernel-table/executor
  // distinction, not a program transformation — so both calls analyze the
  // same `prog`, tagged with the labels used for the timing rows below.
  report_movement_plan("fused", prog, budget);
  report_movement_plan("decomposed", prog, budget);

  // --- Timing.
  const TimingStats fused = time_path("fused", fused_executor, ctx, prog, bufs.data(), reps);
  const TimingStats decomposed = time_path("decomposed", decomposed_executor, ctx, prog, bufs.data(), reps);

  // --- Instruction counts: fused = program length; decomposed expands
  // HMULT -> 3 (MULT_CC + RELINEARIZE + RESCALE) and HROT_ADD -> 2
  // (ROTATE + ADD_CC).
  const uint64_t fused_count = prog->num_instructions;
  uint64_t decomposed_count = 0;
  for (uint32_t i = 0; i < prog->num_instructions; ++i) {
    switch (prog->instructions[i].opcode) {
    case FHN_HMULT:
      decomposed_count += 3;
      break;
    case FHN_HROT_ADD:
      decomposed_count += 2;
      break;
    default:
      decomposed_count += 1;
      break;
    }
  }

  // --- Report.
  std::printf("# FHN encrypted matvec benchmark (fused vs decomposed)\n\n");
  std::printf("backend: %s %s | n = %u | reps = %u\n\n", info->name, info->version, n, reps);
  std::printf("ToyFHE is a CPU reference toy: fused-vs-decomposed deltas here reflect executor dispatch and "
              "memory-pass overhead only. Real backend numbers (GPU kernel-launch and key-switch fusion) come "
              "from running the same program on a hardware backend.\n\n");
  std::printf("| path | instructions dispatched | median ms | min ms | speedup |\n");
  std::printf("|------|------------------------:|----------:|-------:|--------:|\n");
  std::printf("| fused | %llu | %.3f | %.3f | %.2fx |\n", static_cast<unsigned long long>(fused_count), fused.median_ms,
              fused.min_ms, decomposed.median_ms / fused.median_ms);
  std::printf("| decomposed | %llu | %.3f | %.3f | 1.00x |\n", static_cast<unsigned long long>(decomposed_count),
              decomposed.median_ms, decomposed.min_ms);

  // --- Cleanup.
  for (uint32_t i = 0; i < num_buffers; ++i) {
    toyfhe_fhn_buffer_free(ctx, bufs[i]);
  }
  fhn_program_free(prog);
  toyfhe_fhn_destroy(ctx);
  return 0;
}
