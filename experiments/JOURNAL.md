# GPU experiment journal — Cheddar backend on the DGX Spark

Newest entries are appended at the bottom by `experiments/run.sh`. The bootstrap
entry below was recorded by hand; everything after it is tool-generated.

See [`README.md`](README.md) for the machine spec and methodology.

---

### 2026-07-15 — bootstrap — Cheddar GPU backend builds & links on GB10

First time the Cheddar backend has been built on this Spark. This is the
baseline: everything downstream depends on it.

- **fhenomenon** `cfec1b6` · **cheddar** `c428a13` (`feat/fhn-kernels`)
- machine: `spark-0faa`, NVIDIA GB10 (`sm_121`), CUDA 13.0, GCC 13.3, aarch64
- **Result: SUCCESS.** `libcheddar.so` (9.2 MB) built for `sm_121`; the
  fhenomenon side auto-detected it and built `cheddar_fhn` (0.67 MB),
  `FhnCheddarGpuTest`, and `fhn-corpus`.
- Porting deltas needed for this box (all handled by `build-cheddar.sh`):
  - patched Cheddar's CUDA arch list `60 61 70 75 80 86 89 90` → `121`
    (CUDA 13 dropped `sm_60/61/70`; GB10 needs `sm_121`);
  - built `libtommath` v1.3.0 from source (static, `-fPIC`) — no sudo for apt;
  - cloned `spdlog` v1.11.0 headers to `/tmp/spdlog`.
- RMM `branch-22.12` compiled under CUDA 13 / Thrust on aarch64 with only
  `-Wdeprecated-declarations` warnings about `ulong4`/`double4`-style vector
  types. **No hard errors** — the main compatibility risk to re-check on any
  CUDA/driver bump.
- build logs: [`runs/cheddar-build/`](runs/cheddar-build/), [`runs/fhn-build/`](runs/fhn-build/)

### 2026-07-15T06:03:07Z — `gpu-test` — **PASS** (1.57s)

- run id: `20260715-060307-gpu-test`  ·  fhn `cfec1b6`  ·  cheddar `c428a13`
- cmd: `ctest --test-dir /home/zotanika/workspace/fhenomenon/build -R FhnCheddarGpuTest -V`
- context+keygen: 1062 ms (one-time)
- ADD_CC: 10 µs · HMULT: 163 µs · rotate-pair: 152 µs
- tests passed: 10
- note: baseline: first tracked GPU run on GB10
- log: [`experiments/runs/20260715-060307-gpu-test/output.log`](runs/20260715-060307-gpu-test/output.log)

### 2026-07-15T06:07:24Z — `corpus` — **BLOCKED** (frontier, not a regression)

- run id: `20260715-060724-corpus`  ·  fhn `cfec1b6`  ·  cheddar `c428a13`
- cmd: `fhn-corpus --backend libcheddar_fhn.so --prefix "" --shape weighted-sum`
- **Root cause (precise):** the corpus verify path uses an **integer data plane**
  (`fhn_encrypt_i64` / `fhn_decrypt_i64`) because its oracle checks exact integer
  outputs. Cheddar is CKKS (approximate reals) and exports only the **`f64`** data
  plane (`fhn_encrypt_f64` / `fhn_decrypt_f64`). `CorpusBackend` rejects the library
  up front: "missing required fhn_* symbols".
- Two further blockers sit behind that one, in order:
  1. `fhn-corpus` hardcodes `create(nullptr)`; Cheddar's `fhn_create` **requires** a
     param config (`log_degree`, `main_primes`, `level_config`, `rotation_keys`).
  2. Executing/verifying the 12 shapes needs a Cheddar param set covering depth up
     to **15** (`horner15`) and per-shape rotation keys at each shape's slot count.
- **Next experiment (scoped):** add an i64 adapter to the Cheddar backend
  (`fhn_encrypt_i64`/`fhn_decrypt_i64` wrapping the f64 plane, decrypt rounds to
  nearest with a noise tolerance), add a `--backend-config` passthrough to
  `fhn-corpus`, and author a Spark param set. Only then can shapes execute+verify
  on GPU. See [`README.md`](README.md) "next experiment".
- log: [`experiments/runs/20260715-060724-corpus/output.log`](runs/20260715-060724-corpus/output.log)
