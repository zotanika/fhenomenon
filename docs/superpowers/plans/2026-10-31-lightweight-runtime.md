# Lightweight Runtime Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans. Track completion below and in the task ledger.

**Goal:** Immediately execute contributed recipes, retain typed intermediate
resources across commands, and attach the existing compiler only to explicit
regions, demonstrated with genuine CKKS state reuse.

**Architecture:** A small C++17 runtime owns immutable resource identities,
dependencies, tracked memory and immediate recipe dispatch. Native CKKS adapters
provide direct execution, reusable rotation preparation and optional circuit
regions. The core does not know arithmetic or build an optimizing graph.

**Tech Stack:** Existing C++17/CMake/GoogleTest, native CKKS C API. No new external dependencies.

**Spec:** `docs/superpowers/specs/2026-10-31-lightweight-runtime-design.md`.

## Global constraints

- Preserve existing physical FHN semantics and native tests; no plaintext fallback.
- No whole-program graph search or mandatory compilation before submission.
- Compute providers have no client or secret-key pointer.
- Work explicitly in `/Users/sunchuljung/repo/zotanika/fhenomenon`; retain prior uncommitted work. Do not write Git metadata or commit.
- User-owned `.motif/` and extra examples remain unchanged.
- Test builds use separate directories when run by independent workers.

## Review focus

- Same-shaped re-encryption must not reuse input-dependent preparation.
- Failure after allocating one of several outputs must release unpublished outputs.
- Handle copies/dependencies must not double-count or prematurely release memory.
- Key expiry and context mismatches must reject before public output mutation.
- Failed/oversized optional compilation must leave immediate execution usable.

## Task 1 — Immediate generic runtime

Files: `include/FHN/Runtime.h`, `src/FHN/Runtime.cpp`, `test/RuntimeTest.cpp`.

Interface in namespace `fhenomenon::runtime`: `Value` immutable shared resource
handle; `Values = std::vector<Value>`; `ResourceKind {Value, Artifact}`;
`Request {std::string operation; Values inputs; std::any attributes;}`;
`Recipe {std::string operation, name; int priority; accepts(Request,string&)
-> bool; execute(Runtime&,Request)->Values;}`.
`Runtime::addRecipe`, `Runtime::submit`, `Runtime::makeValue(format,kind,bytes,
factory->std::any,dependencies)`, `Runtime::stats`, `Runtime::trace`.
`Value` exposes identity, format, kind, bytes, const payload and dependencies.
`RuntimeOptions` supplies `max_live_bytes` (0 unlimited) and `max_trace_events`
(default 256, 0 disables). Errors distinguish unsupported, invalid request,
budget and execution failure. Freeze exact header with root before integration.

- [x] Add focused failing behavior tests, then minimal implementation.
- [x] Check immediate behavior, recipe selection, non-retried failure,
  multi-output lifetime, budget-before-factory, cross-runtime rejection,
  dependency accounting and trace limit.
- [x] Build focused tests; independent review.

## Task 2 — Native reusable rotation source

Files: native eval region implementation/header, backend `PreparedRotation.cpp`,
`include/FHN/fhn_prepared_api.h`, native `CKKS/PreparedRotation.h`, plugin retention,
`test/CKKS/PreparedRotationTest.cpp`, native layer CMake/test additions.

Public optional exports: opaque `FhnRotationPreparation`;
`ckks_fhn_prepare_rotation_source(ctx,const buffer*) -> preparation*`;
`ckks_fhn_rotation_preparation_bytes(const preparation*) -> uint64_t`;
`ckks_fhn_rotation_preparation_required_bytes(ctx,const buffer*) -> uint64_t`
(nonallocating reservation estimate);
`ckks_fhn_rotate_prepared(ctx,out,const preparation*,int64_t rotation) -> int`;
`ckks_fhn_rotation_preparation_free(preparation*)`.
Preparation owns a canonical source snapshot and coefficient-domain source
component, has no secret/client access, and requires its context to outlive it.
Exact API details frozen with root; expose C typedefs for dynamic loading.

- [x] Add differential tests and observe missing behavior fail.
- [x] Reuse source inverse transforms without changing per-rotation conversion
  semantics; guard scratch, key/context lifetime and transactional publication.
- [x] Verify same-input/key exact results at all levels, signed/identity rotations,
  snapshot semantics and failures; Debug and ASan/UBSan.
- [x] Independent arithmetic/API review.

## Task 3 — Native recipes and optional regions

Files: `include/FHN/NativeRuntime.h`, `src/FHN/NativeRuntime.cpp`,
`test/NativeRuntimeTest.cpp`, root integration CMake and install consumer.

Consumes generic runtime + native C API. Compute-only `NativeRuntime` adapter
owns or leases a native backend context. Trusted callers import freshly encrypted
buffers and separately prepare keys. Registers immediate physical-op recipe,
rotation preparation/consumer recipes and explicit compiled-region invocation.
No native private header is exposed to the generic runtime or contributor code.
Region node limits are checked before calling compiler; workspace limit uses
existing compile options. Expose compiled handle for separate client preparation.

- [x] Pin eager execution without compiler, state reuse, stale-source rejection,
  mixed eager/compiled flows, shared/multiple outputs and failed-region isolation.
- [x] Implement adapter with strong native-context ownership in resource payloads.
- [x] Keep exact reserved resource statistics and document excluded scratch/keys.
- [x] Validate installed consumer and separate contribution translation unit.

## Task 4 — Existing workload proof and documentation

Files: runtime benchmark/integration using existing corpus, runtime documentation,
README and superseded architecture references.

- [x] Execute unchanged `stats` and `matvec-fused` with all-slot oracle checks.
- [x] Demonstrate one input preparation consumed by multiple rotation commands;
  report setup/evaluation/tracked bytes separately and compare current path.
- [x] Run full Debug and sanitizer CTest, Release/test-off build and installed API
  check; final independent review. Do not time while other builds/tests run.
- [x] Record measured results and limits, update design status honestly.
