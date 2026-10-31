# Circuit Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkboxes for tracking.

**Goal:** Execute more existing encrypted apps faster through reusable semantic circuits, measured on unchanged native CKKS parameters and plaintext oracles.

**Architecture:** A new optional C circuit API describes value operations. The native compiler lowers affine/quadratic regions and rotations into reusable liveness-sized storage and trusted internal kernels. Existing opcode semantics remain the comparison path.

**Tech Stack:** C++17, C ABI, native RNS-CKKS, CMake, GoogleTest; no new external dependency.

**Spec:** ../specs/2026-10-28-circuit-execution-design.md and ../specs/2026-10-28-circuit-execution-interfaces.md (exact cross-task signatures).

## Global Constraints

- Work in /Users/sunchuljung/repo/zotanika/fhenomenon; preserve existing edits and user-owned examples. Git metadata is read-only; do not commit.
- Default N8192, Q approximately 50/40/40 bits, P approximately 60 bits, scale 2^40; atol=1e-4, rtol=1e-6.
- No secret key/client backpointer in executable/evaluator; immutable evaluation keys are weakly owned and leased per run.
- No allocation during prepared run; all output publication occurs after successful evaluation. Input/output alias is allowed.
- Workspace limit counts reserved internal slot and scratch storage, not keys/tables/external input/output allocations.
- Same app semantics, including encrypted weights. No plaintext fallback, security-parameter weakening, or scale metadata relabeling.
- Parallel implementation has exclusive file ownership. Root owns shared CMake, API and corpus integration. No concurrent benchmark during builds/tests.

## Review Focus

- Malformed graphs, duplicate IDs/outputs and unknown enum values must fail before output publication (Task 2).
- Changed source arrays, rebound/aliased inputs and output aliasing must preserve copied-plan semantics (Task 2).
- Key expiration and failed preparation must not leave a seemingly valid executable or modify outputs (Task 2).
- Integer overflow, nonfinite constants and mismatched scale joins must reject explicitly (Tasks 1, 2).
- Benchmark skips, failed verification and plaintext-only apps must not appear as successful encrypted execution (Task 3).

### Task 1: Exact region arithmetic — crypto agent

**Files:** src/CKKS/eval/*; test/CKKS/RegionTest.cpp; test/CKKS/EvalTest.cpp.
**Interfaces:** All region types/functions in cross-task contract, published as CKKS/detail/Region.h; keep checked eval primitives for differential reference.

- [x] Add tests for linear integer/constant evaluation, product sums and rotate-add; run against stubs to establish RED.
- [x] Preserve source digit NTT limbs and implement exact combined Pq reduction; compare all residues against separate tensor/relinearize/rescale over signs, levels and keys.
- [x] Implement region functions without repeated full key/input scans, heap allocations or overlapping live source/destination writes.
- [x] Verify guards, int64/nonfinite boundaries, scratch failures and prepared noheap calls with CkksRegionTest and CkksEvalTest.
- [x] Independent review of arithmetic and invariants; fix verified findings.

### Task 2: Reusable native compiler — compiler agent

**Files:** src/CKKS/backend/*; test/CKKS/CircuitTest.cpp.
**Interfaces:** Exact five ckks_fhn_* executable exports and public circuit structs from cross-task contract; consume Task 1 region API.

- [x] Publish native Circuit.h and add RED tests for integer scaling with no level loss, quadratic sums and rebound execution.
- [x] Validate and copy graph/signatures; collect bounded affine/quadratic forms, fused rotate-add regions and explicit level transitions.
- [x] Allocate reusable slots by live intervals, scratch and fixed run metadata; expose actual reserved bytes and reject insufficient budget at compile time.
- [x] Prepare only derived key requirements, bind immutable key views once and lease per run; evaluate all regions then publish outputs.
- [x] Verify source mutation, invalid/duplicate IDs and enum, scale/constant overflow, wrong context/shape, aliases, expired keys, prepare retry, failure atomicity and noheap repeated run.
- [x] Independent review of compiler and lifetime/boundary invariants; fix verified findings.

### Task 3: Corpus integration and measured app results — root + workload test agent

**Files:** include/FHN/fhn_circuit_api.h; benchmarks/corpus/corpus_circuit.{h,cpp}; benchmarks/corpus/corpus_backend.{h,cpp}; benchmarks/fhn_circuit_bench.cpp; benchmarks/CMakeLists.txt; test/CorpusCircuitTest.cpp; test/CorpusCircuitVerificationTest.cmake; test/CMakeLists.txt; test/CKKS/CMakeLists.txt; docs.
**Interfaces:** corpus::Circuit owns node/input/output vectors and view(); importCircuit(const FhnProgram&,std::string*) returns optional<Circuit>. CorpusBackend resolves circuit exports as an optional complete group. New fhn-bench-circuit command consumes native C ABI.

- [x] Publish public C-compatible structs/function typedefs and importer stub; workload agent establishes RED import tests.
- [x] Implement explicit corpus semantic importer, including HROT_ADD expansion; reject physical-only and Boolean operations with diagnostics.
- [x] Add optional loader exports and reusable benchmark fixture, keeping keygen/encryption/compile/prepare outside evaluation timing and verifying every output slot.
- [x] Test 10 expected native apps, 3 explicit unsupported cases, CLI error paths and require-execution behavior.
- [x] Build Debug, ASan/UBSan and Release; run full CTest suite plus installed public circuit consumer.
- [x] In a quiet window, alternate preserved baseline and compiled evaluation for shared apps; report distributions, setup costs, accuracy and reserved workspace. New coverage is not a speedup claim.
- [x] Independent integration review; update native docs, README and measured validation report with limitations.


## Completion evidence

All tasks completed in the working checkout; no commits because Git metadata is
read-only. Final Debug and ASan/UBSan suites: 33/33 CTest targets and 328
GoogleTests. Release/test-off and installed public API consumers pass. Both
independent Astra reviews are clean after corrections. Final paired app runs
execute 10/13; six shared apps improve in both rounds.
[Full measurements and scope](../../circuit-execution-validation-2026-10-28.md).

Performance feedback produced one additional exact refinement: build the
Galois permutation map once per rotation in caller scratch, avoiding repeated
component/limb index calculation. Differential/noheap/guard tests and scoped
independent review cover this final change. No unresolved findings remain.
