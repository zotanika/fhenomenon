# FHN Real-Workload Corpus (`fhn-corpus`) — Design

Date: 2026-07-13
Status: approved (design discussion in session; ordering, backend strategy,
and shape list each confirmed explicitly)

## Premise

The movement pass's unit tests prove *correctness* (Belady behaves per
spec); they do not prove *usefulness* (how much transfer traffic exact
future knowledge saves on realistic workload shapes). This corpus supplies
the evidence: 12 FhnProgram generators spanning distinct dataflow axes,
planned under Belady AND an LRU baseline, reported as transfer savings —
plus execution oracles on whatever FHN backend is loaded at runtime.

The corpus is deliberately IR-level and backend-pluggable:

- ToyFHE is demoted, not deleted: it remains the CI-executable oracle and
  the dlopen fixture, but nothing in the corpus design depends on its
  limits (single slot, no rotate). Oracles are multi-slot from day one.
- Cheddar (GPU) is a drop-in: the same binary takes `--backend <lib.so>`,
  so GPU execution runs on a GPU machine without code changes.
- The TFHE-rs ("Concrete") builtin cannot execute FhnPrograms today (its
  build skips the FhnRuntime entirely); porting it onto the FHN ABI is a
  separate backlog item, after which boolean shapes gain an executor.

## Goals

- 12 workload shapes with distinct reuse-distance/fan-out/depth profiles.
- Belady-vs-LRU comparison with a budget sweep per shape, and one headline
  aggregate: "Belady saves N% of transfers vs LRU across the corpus."
- Execution verification against plaintext oracles on any loaded backend
  that supports a shape's opcodes; plan statistics for every shape
  regardless of executability.
- CI registration with ToyFHE as the backend.

## Non-goals (v1)

- Byte-accurate budgets or wall-clock transfer measurements (needs a real
  device backend + kernel size metadata; separate roadmap items).
- TFHE-rs FHN port (separate work item; boolean shapes stay plan-only).
- Fhenon-API-level examples (blocked on the frontend rotate API; the
  corpus builds FhnPrograms directly).
- Automated GPU CI (Cheddar runs are a manual step on a GPU machine).

## Component 1 — `EvictionPolicy` in `FhnMovementPlan`

```cpp
enum class FhnEvictionPolicy { Belady, Lru };

static std::optional<FhnMovementPlan> analyze(const FhnProgram &program,
                                              const std::vector<uint32_t> &pinned,
                                              uint32_t device_budget = 0,
                                              FhnEvictionPolicy policy = FhnEvictionPolicy::Belady);
```

LRU exists ONLY as a benchmarking baseline (documented in the header): the
simulation, liveness, alloc/free/prefetch scheduling, and infeasibility
rules are shared; only victim selection differs. LRU victim = resident
buffer (outside the current working set) whose most recent touch
(def, prefetch, or use as operand) is oldest; ties break on lower id.
The simulation must track last-touch positions for this. Belady remains
the default everywhere; Session and the executor never pass `Lru`.

Unit tests: on the existing Belady-discriminator program, `Lru` must evict
b2 where `Belady` evicts a1 (the same program pins both policies); an
LRU-mode plan must still execute correctly through the plan-aware executor
(movement placement differs, results don't).

## Component 2 — Corpus structure

New directory `benchmarks/corpus/`, one binary target `fhn-corpus`:

- `corpus_shapes.h/.cpp` — the 12 generators. Each shape is a struct:
  `{ std::string name; uint32_t slot_count; FhnProgramPtr program;
     std::map<uint32_t, std::vector<int64_t>> inputs;  // id -> slot values
     std::map<uint32_t, std::vector<int64_t>> expected; // output id -> slots }`
  where `FhnProgramPtr` is the owning
  `std::unique_ptr<FhnProgram, decltype(&fhn_program_free)>` alias and
  `expected` is computed by running the oracle over the program.
- `corpus_oracle.h/.cpp` — plaintext evaluator over `std::vector<int64_t>`
  slots: elementwise ADD/SUB/MULT (CC/CS), cyclic ROTATE by params[0]
  (positive = left), NEGATE; comparisons (EQ/LT/LE) elementwise over arbitrary
  int64 slots producing 0/1, and AND/OR/XOR elementwise over 0/1 slots. Fused opcodes are not emitted by generators (executors
  decompose primitives anyway; the corpus measures movement, not fusion).
- `corpus_backend.h/.cpp` — a minimal, self-contained dlopen loader over
  the public C ABI only (`fhn_get_abi_version` handshake, create/destroy,
  get_kernels, buffer alloc/free, encrypt/decrypt i64, optional
  prefetch/evict). Deliberately independent of the C++ Backend hierarchy:
  it doubles as a documentation-by-example ABI consumer. Reuses
  `FhnDefaultExecutor` for dispatch.
- `fhn_corpus_main.cpp` — CLI: `--backend <lib.so>` (optional; no backend
  = plan-only mode), `--prefix <sym>` (default `toyfhe_`), `--shape <name>`
  (filter), `--list`.

## Component 3 — The 12 shapes

Sizes chosen so every shape has ≥ 30 instructions and a live-set profile
that makes eviction meaningful. Each generator documents which axis it
stresses. Slot counts are per-shape; `rot` marks rotate usage.

| # | name | axis stressed | sketch |
|---|------|---------------|--------|
| 1 | `matvec` (rot) | fan-out reuse of one input ct | 64×64 diag-packed matvec: rotate input, MULT_CS row, accumulate |
| 2 | `matmul-tile` (rot) | chained matvecs, deep+wide | 2 chained 32×32 tiles |
| 3 | `horner15` | long sequential chain, CS-heavy | degree-15 Horner: x reused every step |
| 4 | `reduce-tree` (rot) | log-depth fan-in | rotate-add reduction over 64 slots |
| 5 | `wide-front` | max simultaneous live set | 24 independent a·b+c triples, then pairwise sum |
| 6 | `logreg` (rot) | composite: dot + poly | dot(w,x) via rot-reduce, then degree-3 sigmoid approx |
| 7 | `conv1d` (rot) | overlapping window reuse | 5-tap stencil over 64 slots, taps as CS scalars |
| 8 | `stats` (rot) | multi-output | mean and variance of 64 slots (sum tree + square-sum tree) |
| 9 | `psi-eq` | boolean tree (plan-only today) | 32 pairwise EQ, AND-tree to one bit |
| 10 | `iter-update` | rebinding-heavy chain | x = a·x + b unrolled ×24 |
| 11 | `weighted-sum` | shared-scalar CS-heavy | Σ wᵢ·xᵢ over 24 inputs, weights as fparams |
| 12 | `diamond` | fork-join reuse, LRU-adversarial | 8-level diamond DAG, both forks reuse the root |

Each shape records its ciphertext-multiplication depth (`ct_mult_depth`):
ToyFHE's noise budget only guarantees exactness up to ~3 chained ct*ct
multiplies, so the runner takes `--max-depth <N>` (an operator-supplied
exactness bound for the loaded backend; unlimited when absent) and skips
execution — not planning — of deeper shapes. With `--max-depth 3`, ToyFHE
executes the no-rotate, non-boolean subset {wide-front, iter-update,
weighted-sum, diamond} at `slot_count = 1`; `horner15` (depth 15) is
planned everywhere but executed only on backends with rescaling headroom.
Rotate shapes execute on backends whose kernel table supports FHN_ROTATE
(Cheddar); boolean shapes execute once a TFHE-class backend lands.
Generators keep plaintext magnitudes small enough that int64 oracles never
overflow and executable shapes stay within ToyFHE's plaintext modulus. Executability is decided at runtime by checking
the shape's opcode set against the loaded executor's `supports()`;
generators emit primitive opcodes only (no fused composites), so
decomposition coverage is never involved and the check is exact. Nothing
is hardcoded per backend.

## Component 4 — Runner semantics

Per shape:
1. Generate program + plaintext inputs; compute expected outputs with the
   oracle.
2. Plan-stats pass (always): analyze with budget sweep at three points —
   `B_min` = the program's largest single-instruction working set (the
   smallest feasible budget), `B_mid` = max(B_min, ceil(0.6 × liveness
   high-water)), `B_hw` = liveness high-water (expect zero evictions —
   a per-shape sanity check the runner asserts). At each point run BOTH
   policies and record evict/prefetch counts.
3. Execution pass (when a backend is loaded and supports the opcodes):
   encrypt inputs, `analyze(..., budget = B_mid, Belady)`, execute through
   the plan-aware executor, decrypt outputs, compare to oracle exactly
   (int64). Any mismatch or executor failure → report FAIL, exit nonzero
   at the end.
4. Report one table row per shape × budget point: transfers (prefetch +
   evict) under Belady vs LRU and savings %. Footer: aggregate transfers
   at B_mid across all shapes and the headline savings percentage.

Determinism: fixed-seed value generation; identical output across runs.

## Component 5 — CI

`test/CMakeLists.txt` registers `FhnCorpusTest` the same way it registers
`FhnMatvecBenchTest`: `add_test(... COMMAND fhn-corpus --backend
$<TARGET_FILE:toyfhe_fhn> --prefix toyfhe_ --max-depth 3)` with a dependency
on the `toyfhe_fhn` target. It exercises: all 12 plan-stat
sweeps + LRU/Belady comparison + the 4 depth-safe executable shapes
end-to-end.

## Verification summary

- New unit tests for `FhnEvictionPolicy::Lru` (discriminator flip +
  execution equivalence).
- Corpus self-checks: B_hw zero-eviction assertion per shape; oracle
  comparison on executable shapes; deterministic re-run.
- Existing suite untouched and green; sanitizers clean on `fhn-corpus`
  with the ToyFHE backend.
- Cheddar: same binary on a GPU machine (manual, documented in the corpus
  README section of the report output or --help).

## Risks / honest limits

- Slot semantics for CKKS backends (Cheddar) are approximate; v1's exact
  int64 compare will need a tolerance mode when GPU execution is first
  attempted (documented, deferred until that machine time exists).
- LRU is one baseline; it is the standard one for cache replacement, but
  transfer-optimal claims are relative to it, not to all possible
  heuristics.
