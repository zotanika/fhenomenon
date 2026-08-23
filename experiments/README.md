# GPU experiments — Cheddar backend on the DGX Spark

A running record of GPU experiments that exercise the **Cheddar-FHE** CKKS
backend through Fhenomenon's FHN ABI, on the **DGX Spark** (`spark-0faa`).

Why this exists: Cheddar is an accelerator backend with hard constraints
(scheme, parameters, device residency). Whether a given fhenomenon + cheddar
pair even builds — let alone runs correctly and fast — depends on the exact
machine, CUDA version, and GPU arch. This directory keeps that history so a
result is reproducible and attributable months later, instead of living in
shell scrollback.

## The machine

| | |
|---|---|
| Host | `spark-0faa` |
| GPU | NVIDIA **GB10** (Grace-Blackwell), compute capability **12.1** → `sm_121` |
| Memory | 122 GiB unified (LPDDR5X) |
| CPU | 20 cores, aarch64 |
| CUDA | 13.0 |
| Toolchain | GCC 13.3, CMake 3.28 |

## Layout

```
experiments/
  README.md          ← this file (methodology + machine)
  JOURNAL.md         ← human-readable notebook, newest entries appended
  log.jsonl          ← append-only machine-readable record, one JSON per run
  build-cheddar.sh   ← reproducible libcheddar.so build for this box
  capture-env.sh     ← emits the env/toolchain/git fingerprint as JSON
  run.sh             ← runs one experiment and records it into the two logs
  runs/<id>/         ← full stdout for each run (tracked; this is the evidence trail)
```

Only `.gpu-deps/` (regenerable build deps) is git-ignored; the run logs are kept.

## Reproducing from scratch

```bash
git submodule update --init refs/cheddar-fhe        # pinned cheddar commit
experiments/build-cheddar.sh                         # libtommath + spdlog + libcheddar.so (sm_121)
cmake -S . -B build -DBUILD_TEST=ON -DBUILD_DOCUMENTATION=OFF
cmake --build build --target cheddar_fhn FhnCheddarGpuTest fhn-corpus -j
```

The build script is idempotent and lands machine-local deps in `.gpu-deps/`
(git-ignored). Set `FORCE=1` to rebuild deps, `CUDA_ARCH=<nn>` to retarget.

### Spark-specific porting notes

Two upstream assumptions do not hold on this box and the build script handles both:

1. **CUDA arch.** Cheddar hardcodes `CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86 89 90`.
   CUDA 13 dropped `sm_60/61/70`, and GB10 is `sm_121` (absent from the list).
   The script patches the submodule's `CMakeLists.txt` to `sm_121`. This dirties
   the submodule working tree by design — it is not committed.
2. **Deps without root.** No passwordless sudo here, so `libtommath` is built
   from source (static, `-fPIC`) and `spdlog` headers are cloned to
   `/tmp/spdlog` (the path the fhenomenon-side CMake expects).

RMM is pinned to `branch-22.12` (Nov 2022). It builds under CUDA 13 / Thrust on
aarch64 with only `-Wdeprecated-declarations` noise about `*4` vector types — no
hard errors. That compatibility is the main thing to re-check on any toolchain bump.

## Running an experiment

```bash
experiments/run.sh gpu-test                       # FhnCheddarGpuTest: CKKS correctness + micro-latency
experiments/run.sh corpus --max-depth 6           # 12-shape workload corpus, executed+verified on GPU
NOTE="baseline after driver bump" experiments/run.sh gpu-test
```

Each run appends one line to `log.jsonl` and one entry to `JOURNAL.md`, and
saves full stdout under `runs/<id>/`. Records carry the env fingerprint and
both git SHAs, so you can diff results against a known source state.

## The two experiment surfaces

- **`gpu-test`** — `FhnCheddarGpuTest` (ctest). Confirms the backend loads via
  `dlopen`, advertises the CKKS op set, and that add / hmult / scalar ops
  encrypt→compute→decrypt correctly on the GPU. Reports one-time context+keygen
  cost and per-op pipeline latency.
- **`corpus`** — `fhn-corpus --backend libcheddar_fhn.so`. Runs the 12-shape
  real-workload corpus through the GPU backend, verifying each shape's decrypted
  output against the plaintext oracle and reporting Belady-vs-LRU data-movement
  savings under byte budgets. **Not yet runnable on Cheddar** — see below.

## Next experiment (scoped): corpus on the GPU backend

Recorded as BLOCKED on 2026-07-15. The corpus verify path is exact-integer and
Cheddar is approximate CKKS, so three things are needed, in order:

1. **i64 adapter in the Cheddar backend** — export `fhn_encrypt_i64` /
   `fhn_decrypt_i64` wrapping the existing f64 data plane; decrypt rounds to
   nearest integer and tolerates CKKS noise. Without these two symbols
   `CorpusBackend` rejects the library ("missing required fhn_* symbols").
2. **`--backend-config <path>` passthrough in `fhn-corpus`** — today it calls
   `fhn_create(nullptr)`; Cheddar requires a param config, so the corpus must
   forward one instead of hardcoding null.
3. **A Spark param set** covering depth up to 15 (`horner15`) and the rotation
   keys each shape needs at its own slot count (1 / 32 / 64).

Only after all three can shapes execute + verify on the GPU. The movement
analysis (Belady vs LRU) additionally wants Cheddar's level model
(`fhn_fresh_level` / `fhn_level_bytes` / `fhn_opcode_level_effect`), currently
not exported.
