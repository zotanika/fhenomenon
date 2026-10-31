# Native CKKS Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** A genuine CPU RNS-CKKS backend executes encrypted packed circuits without the evaluator possessing a secret key.
**Architecture:** Complete L2 bindings, exact L1 CRT, explicit L4 evaluation and L5 host crypto; integrate L6 with a separate optional client handle. Preserve the audited runtime and ABI1.
**Tech Stack:** C++17, existing NTT/BConv/Arena, GoogleTest, CMake, OS entropy; no new third-party dependency.
**Spec:** docs/superpowers/specs/2026-10-28-native-ckks-execution-design.md.

## Global Constraints

- Original writable checkout /Users/sunchuljung/repo/zotanika/fhenomenon; attached worktree is unusable. Preserve prior unstaged audit fixes and existing user untracked files. No commits/push/merge.
- Keep layer isolation and no secret types in L4. No plaintext-assisted evaluation or secret-key paths in FhnBackendCtx.
- Hybrid/dnum, real centered rescale, full-active-Q decoding, periodic power-of-two slots, positive-left rotation.
- C++17, ABI1 vtable unchanged, optional additive client exports, old backends compatible.
- Fail closed for entropy, invalid metadata, keys, capacity and scratch. Publish numerical/parameter limits; no security/performance claims without evidence.
- The exact cross-task interface contract is in /private/tmp/fhenomenon-native-20261028/interfaces.md; preserve an in-repo copy with implementation docs before delivery.

## Review Focus

- In-place results, partial storage overlaps and shrinking levels must not corrupt live inputs (Tasks1/2).
- Approximate BConv must not masquerade as exact centered CRT (Tasks1/2/3).
- Lower-level digits and original fullQ offset of P key limbs must agree (Tasks2/3).
- Key lifetime and preparation cannot expose a secret to evaluation or create keys inside kernels (Tasks3/4).
- Genuine random ciphertexts must pass deep/two-level and native/fallback checks, including all logical slots (Tasks3/4/5).

### Task1: Storage bindings and exact CRT
**Files:** src/CKKS/storage/{Storage.cpp,include/CKKS/Storage.h}, src/CKKS/tables/{CrtTables.cpp,include/CKKS/CrtTables.h}, their layer CMake; test/CKKS/{StorageTest,CrtTest}.cpp.
**Interfaces:** Fixed aligned PolySlab; mutable/const polynomial and ciphertext views with explicit capacity/stride; L2 EvaluationKeyView; CrtTables mixed-radix/centeredMod/centeredValue as shared contract.
- [x] Write alignment/capacity/stride/alias/constness/move regressions; write CRT exact-sign/cross-prime/negative-roundtrip tests from independent integer calculations.
- [x] Observe RED with minimal stubs, implement, run isolated targets and sanitizer boundary probes.
- [x] Independent review and handoff exact headers to Tasks2/3.

### Task2: Evaluation tables and real homomorphic operations
**Files:** src/CKKS/eval/{EvalTables.cpp,Eval.cpp,include/CKKS/EvalTables.h,include/CKKS/Eval.h,CMakeLists.txt}, test/CKKS/EvalTest.cpp.
**Interfaces:** Immutable EvaluationTables; free eval add/subtract/negate/tensor/relinearize/rescale/rotate/hmult/scalar/levelDown plus explicit scratchBytes; L2 data-only key view. No Keys dependency.
- [x] RED independent ring/CRT/Galois oracles, invalid key/basis/scale/scratch, partial and exact alias cases.
- [x] Implement hybrid projector switching and exact centered ModDown; actual tensor3→relin2→rescale chain; signed rotations.
- [x] Assert zero heap allocation during successful kernels, scratch bounds, lower-level key projection; isolated layer tests and review.

### Task3: Host encoding, keys and public encryption
**Files:** src/CKKS/keys/{Keys.cpp,Encoding.cpp,include/CKKS/Keys.h,include/CKKS/Encoding.h,CMakeLists.txt}, test/CKKS/KeysTest.cpp.
**Interfaces:** Injected RandomSource/SystemRandom, owning secret/public/evaluation keys; public-key-only encrypt, decrypt; canonical encode/decode; host scratch bound.
- [x] RED literal fractional encoding/periodic slots, full-Q CRT decode, random ciphertext distinction, wrong-key mismatch, RNG failure, actual two-level arithmetic and rotations using L4.
- [x] Implement OS entropy, unbiased sampling and consistent RNS small polynomials; projector key generation and independent public encryption.
- [x] Validate secret cleanup, key expiry interfaces, multi-context coexistence, precision/error bounds; review.

### Task4: Native backend and trusted-client ABI
**Files:** src/CKKS/backend/*; include/FHN/fhn_backend_api.h; benchmarks/corpus/*; root/shared CMake; test native integration targets.
**Interfaces:** ckks_ exports, separate FhnClientCtx with optional prepare/program, client vector encryption/decryption; compute context only weak evaluation-key references. Scalar old backend compatibility retained.
- [x] RED loader/client lifetime, missing prepared keys, invalid/default profiles, level model and footprint queries, native packed required execution.
- [x] Implement adapter, default and explicit config validation, program-derived key preparation, truthful scratch/reserved/live sizes.
- [x] Connect corpus and install/build variants; document runnable real-backend command.

### Task5: Integrated validation and delivery
- [x] Run packed8x8 and explicit two-level encrypted circuit, native/fallback comparison, host/eval separation and allocation checks.
- [x] Run whole Debug and strict ASan/UBSan/noLTO suite, test-off build/install consumer, layer violations and preserved audit regressions.
- [x] Independent Astra math/security-boundary and whole-change review; fix material findings and recheck.
- [x] Update native handoff/status and deliver genuine implementation with exact tested limits and no unmeasured speed claims.


## Completion — 2026-10-28

All five tasks completed in the original writable checkout. Final Debug and
ASan/UBSan suites: 29/29 targets each. Release without tests/LTO and installed
native client execution passed. Independent reviews found two bugs (active-Q
encoding range and preparation retry after allocation failure), both fixed and
reverified with permanent regressions. Detailed counts, evidence and remaining
scope are in [validation record](../../native-ckks-validation-2026-10-28.md).
No commits or publishing were performed. Prior audit changes and user files
were preserved. The native FHN path is usable; C++ frontend client integration,
bootstrap, GPU, security hardening and performance comparison are not claimed.
